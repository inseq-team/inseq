"""HuggingFace Seq2seq model."""
import logging
from abc import abstractmethod
from typing import Dict, List, NoReturn, Optional, Tuple, Union

import torch
from torch import long
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutput, ModelOutput, Seq2SeqLMOutput

from ..attr.attribution_decorators import batched
from ..data import BatchEncoding
from ..utils import check_device
from ..utils.typing import (
    EmbeddingsTensor,
    IdsTensor,
    LogitsTensor,
    MultiLayerEmbeddingsTensor,
    MultiLayerMultiUnitScoreTensor,
    OneOrMoreIdSequences,
    OneOrMoreTokenSequences,
    TextInput,
    VocabularyEmbeddingsTensor,
)
from .attribution_model import AttributionModel
from .decoder_only import DecoderOnlyAttributionModel
from .encoder_decoder import EncoderDecoderAttributionModel
from .model_decorators import unhooked

logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Update if other model types are added
SUPPORTED_AUTOCLASSES = [AutoModelForSeq2SeqLM, AutoModelForCausalLM]


class HuggingfaceModel(AttributionModel):
    """Model wrapper for any ForCausalLM and ForConditionalGeneration model on the HuggingFace Hub used to enable
    feature attribution. Corresponds to AutoModelForCausalLM and AutoModelForSeq2SeqLM auto classes.

    Attributes:
        _autoclass (:obj:`Type[transformers.AutoModel`]): The HuggingFace model class to use for initialization.
            Must be defined in subclasses.
        model (:obj:`transformers.AutoModelForSeq2SeqLM` or :obj:`transformers.AutoModelForSeq2SeqLM`):
            the model on which attribution is performed.
        tokenizer (:obj:`transformers.AutoTokenizer`): the tokenizer associated to the model.
        device (:obj:`str`): the device on which the model is run.
        encoder_int_embeds (:obj:`captum.InterpretableEmbeddingBase`): the interpretable embedding layer for the
            encoder, used for layer attribution methods in Captum.
        decoder_int_embeds (:obj:`captum.InterpretableEmbeddingBase`): the interpretable embedding layer for the
            decoder, used for layer attribution methods in Captum.
        embed_scale (:obj:`float`, *optional*): scale factor for embeddings.
        tokenizer_name (:obj:`str`, *optional*): The name of the tokenizer in the Huggingface Hub.
            Default: use model name.
    """

    _autoclass = None

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        attribution_method: Optional[str] = None,
        tokenizer: Union[str, PreTrainedTokenizerBase, None] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        """AttributionModel subclass for Huggingface-compatible models.

        Args:
            model (:obj:`str` or :obj:`transformers.PreTrainedModel`): the name of the model in the
                Huggingface Hub or path to folder containing local model files.
            attribution_method (str, optional): The attribution method to use.
                Passing it here reduces overhead on attribute call, since it is already
                initialized.
            tokenizer (:obj:`str` or :obj:`transformers.PreTrainedTokenizerBase`, optional): the name of the tokenizer
                in the Huggingface Hub or path to folder containing local tokenizer files.
                Default: use model name.
            device (str, optional): the Torch device on which the model is run.
            **kwargs: additional arguments for the model and the tokenizer.
        """
        super().__init__(**kwargs)
        if self._autoclass is None or self._autoclass not in SUPPORTED_AUTOCLASSES:
            raise ValueError(
                f"Invalid autoclass {self._autoclass}. Must be one of {[x.__name__ for x in SUPPORTED_AUTOCLASSES]}."
            )
        model_args = kwargs.pop("model_args", {})
        model_kwargs = kwargs.pop("model_kwargs", {})
        if isinstance(model, PreTrainedModel):
            self.model = model
        else:
            if "output_attentions" not in model_kwargs:
                model_kwargs["output_attentions"] = True

            self.model = self._autoclass.from_pretrained(model, *model_args, **model_kwargs)
        self.model_name = self.model.config.name_or_path
        self.tokenizer_name = tokenizer if isinstance(tokenizer, str) else None
        if tokenizer is None:
            tokenizer = model if isinstance(model, str) else self.model_name
            if not tokenizer:
                raise ValueError(
                    "Unspecified tokenizer for model loaded from scratch. Use explicit identifier as tokenizer=<ID>"
                    "during model loading."
                )
        tokenizer_inputs = kwargs.pop("tokenizer_inputs", {})
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})

        if isinstance(tokenizer, PreTrainedTokenizerBase):
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, *tokenizer_inputs, **tokenizer_kwargs)
        if self.model.config.pad_token_id is not None:
            self.pad_token = self.tokenizer.convert_ids_to_tokens(self.model.config.pad_token_id)
            self.tokenizer.pad_token = self.pad_token
        self.bos_token_id = getattr(self.model.config, "decoder_start_token_id", None)
        if self.bos_token_id is None:
            self.bos_token_id = self.model.config.bos_token_id
        self.bos_token = self.tokenizer.convert_ids_to_tokens(self.bos_token_id)
        self.eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if self.eos_token_id is None:
            self.eos_token_id = self.tokenizer.pad_token_id
        if self.tokenizer.unk_token_id is None:
            self.tokenizer.unk_token_id = self.tokenizer.pad_token_id
        self.embed_scale = 1.0
        self.encoder_int_embeds = None
        self.decoder_int_embeds = None
        self.is_encoder_decoder = self.model.config.is_encoder_decoder
        self.configure_embeddings_scale()
        self.setup(device, attribution_method, **kwargs)

    @staticmethod
    def load(
        model: Union[str, PreTrainedModel],
        attribution_method: Optional[str] = None,
        tokenizer: Union[str, PreTrainedTokenizerBase, None] = None,
        device: str = None,
        **kwargs,
    ) -> "HuggingfaceModel":
        """Loads a HuggingFace model and tokenizer and wraps them in the appropriate AttributionModel."""
        if isinstance(model, str):
            is_encoder_decoder = AutoConfig.from_pretrained(model).is_encoder_decoder
        else:
            is_encoder_decoder = model.config.is_encoder_decoder
        if is_encoder_decoder:
            return HuggingfaceEncoderDecoderModel(model, attribution_method, tokenizer, device, **kwargs)
        else:
            return HuggingfaceDecoderOnlyModel(model, attribution_method, tokenizer, device, **kwargs)

    @AttributionModel.device.setter
    def device(self, new_device: str) -> None:
        check_device(new_device)
        self._device = new_device
        is_loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        is_loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        is_quantized = is_loaded_in_8bit or is_loaded_in_4bit

        # Enable compatibility with 8bit models
        if self.model:
            if not is_quantized:
                self.model.to(self._device)
            else:
                mode = "8bit" if is_loaded_in_8bit else "4bit"
                logger.warning(
                    f"The model is loaded in {mode} mode. The device cannot be changed after loading the model."
                )

    @abstractmethod
    def configure_embeddings_scale(self) -> None:
        """Configure the scale factor for embeddings."""
        pass

    @property
    def info(self) -> Dict[str, str]:
        dic_info: Dict[str, str] = super().info
        extra_info = {
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_class": self.tokenizer.__class__.__name__,
        }
        dic_info.update(extra_info)
        return dic_info

    @unhooked
    @batched
    def generate(
        self,
        inputs: Union[TextInput, BatchEncoding],
        return_generation_output: bool = False,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[List[str], Tuple[List[str], ModelOutput]]:
        """Wrapper of model.generate to handle tokenization and decoding.

        Args:
            inputs (`Union[TextInput, BatchEncoding]`):
                Inputs to be provided to the model for generation.
            return_generation_output (`bool`, *optional*, defaults to False):
                If true, generation outputs are returned alongside the generated text.

        Returns:
            `Union[List[str], Tuple[List[str], ModelOutput]]`: Generated text or a tuple of generated text and
            generation outputs.
        """
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and len(inputs) > 0 and all(isinstance(x, str) for x in inputs)
        ):
            inputs = self.encode(inputs)
        inputs = inputs.to(self.device)
        generation_out = self.model.generate(
            inputs=inputs.input_ids,
            return_dict_in_generate=True,
            **kwargs,
        )
        sequences = generation_out.sequences
        texts = self.decode(ids=sequences, skip_special_tokens=skip_special_tokens)
        if return_generation_output:
            return texts, generation_out
        return texts

    @staticmethod
    def output2logits(forward_output: Union[Seq2SeqLMOutput, CausalLMOutput]) -> LogitsTensor:
        # Full logits for last position of every sentence:
        # (batch_size, tgt_seq_len, vocab_size) => (batch_size, vocab_size)
        return forward_output.logits[:, -1, :].squeeze(1)

    def encode(
        self,
        texts: TextInput,
        as_targets: bool = False,
        return_baseline: bool = False,
        include_eos_baseline: bool = False,
        max_input_length: int = 512,
        add_bos_token: bool = True,
        add_special_tokens: bool = True,
    ) -> BatchEncoding:
        """Encode one or multiple texts, producing a BatchEncoding.

        Args:
            texts (str or list of str): the texts to tokenize.
            return_baseline (bool, optional): if True, baseline token ids are returned.

        Returns:
            BatchEncoding: contains ids and attention masks.
        """
        if as_targets and not self.is_encoder_decoder:
            raise ValueError("Decoder-only models should use tokenization as source only.")
        max_length = self.tokenizer.max_len_single_sentence
        # Some tokenizer have weird values for max_len_single_sentence
        # Cap length with max_model_input_sizes instead
        if max_length > 1e6:
            if hasattr(self.tokenizer, "max_model_input_sizes") and self.tokenizer.max_model_input_sizes:
                max_length = max(v for _, v in self.tokenizer.max_model_input_sizes.items())
            else:
                max_length = max_input_length
        batch = self.tokenizer(
            text=texts if not as_targets else None,
            text_target=texts if as_targets else None,
            add_special_tokens=add_special_tokens,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        baseline_ids = None
        if return_baseline:
            if include_eos_baseline:
                baseline_ids = torch.ones_like(batch["input_ids"]).long() * self.tokenizer.unk_token_id
            else:
                baseline_ids_non_eos = batch["input_ids"].ne(self.eos_token_id).long() * self.tokenizer.unk_token_id
                baseline_ids_eos = batch["input_ids"].eq(self.eos_token_id).long() * self.eos_token_id
                baseline_ids = baseline_ids_non_eos + baseline_ids_eos
        # We prepend a BOS token only when tokenizing target texts.
        if as_targets and self.is_encoder_decoder and add_bos_token:
            ones_mask = torch.ones((batch["input_ids"].shape[0], 1), device=self.device, dtype=long)
            batch["attention_mask"] = torch.cat((ones_mask, batch["attention_mask"]), dim=1)
            bos_ids = ones_mask * self.bos_token_id
            batch["input_ids"] = torch.cat((bos_ids, batch["input_ids"]), dim=1)
            if return_baseline:
                baseline_ids = torch.cat((bos_ids, baseline_ids), dim=1)
        return BatchEncoding(
            input_ids=batch["input_ids"],
            input_tokens=[self.tokenizer.convert_ids_to_tokens(x) for x in batch["input_ids"]],
            attention_mask=batch["attention_mask"],
            baseline_ids=baseline_ids,
        )

    def decode(
        self,
        ids: Union[List[int], List[List[int]], IdsTensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def embed_ids(self, ids: IdsTensor, as_targets: bool = False) -> EmbeddingsTensor:
        if as_targets and not self.is_encoder_decoder:
            raise ValueError("Decoder-only models should use tokenization as source only.")
        if self.encoder_int_embeds is not None and not as_targets:
            embeddings = self.encoder_int_embeds.indices_to_embeddings(ids)
        elif self.decoder_int_embeds is not None and as_targets:
            embeddings = self.decoder_int_embeds.indices_to_embeddings(ids)
        else:
            embeddings = self.get_embedding_layer()(ids)
        return embeddings * self.embed_scale

    def convert_ids_to_tokens(
        self, ids: IdsTensor, skip_special_tokens: Optional[bool] = True
    ) -> OneOrMoreTokenSequences:
        if ids.ndim < 2:
            return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
        return [
            self.tokenizer.convert_ids_to_tokens(id_slice, skip_special_tokens=skip_special_tokens) for id_slice in ids
        ]

    def convert_tokens_to_ids(self, tokens: TextInput) -> OneOrMoreIdSequences:
        if isinstance(tokens[0], str):
            return self.tokenizer.convert_tokens_to_ids(tokens)
        return [self.tokenizer.convert_tokens_to_ids(token_slice) for token_slice in tokens]

    def convert_tokens_to_string(
        self,
        tokens: OneOrMoreTokenSequences,
        skip_special_tokens: bool = True,
        as_targets: bool = False,
    ) -> TextInput:
        if isinstance(tokens, list) and len(tokens) == 0:
            return ""
        elif isinstance(tokens[0], str):
            tmp_decode_state = self.tokenizer._decode_use_source_tokenizer
            self.tokenizer._decode_use_source_tokenizer = not as_targets
            out_strings = self.tokenizer.convert_tokens_to_string(
                tokens if not skip_special_tokens else [t for t in tokens if t not in self.special_tokens]
            )
            self.tokenizer._decode_use_source_tokenizer = tmp_decode_state
            return out_strings
        return [self.convert_tokens_to_string(token_slice, skip_special_tokens, as_targets) for token_slice in tokens]

    def convert_string_to_tokens(
        self,
        text: TextInput,
        skip_special_tokens: bool = True,
        as_targets: bool = False,
    ) -> OneOrMoreTokenSequences:
        if isinstance(text, str):
            ids = self.tokenizer(
                text=text if not as_targets else None,
                text_target=text if as_targets else None,
                add_special_tokens=not skip_special_tokens,
            )["input_ids"]
            return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens)
        return [self.convert_string_to_tokens(t, skip_special_tokens, as_targets) for t in text]

    def clean_tokens(
        self,
        tokens: OneOrMoreTokenSequences,
        skip_special_tokens: bool = False,
        as_targets: bool = False,
    ) -> OneOrMoreTokenSequences:
        """Cleans special characters from tokens.

        Args:
            tokens (`OneOrMoreTokenSequences`):
                A list containing one or more lists of tokens.
            skip_special_tokens (`bool`, *optional*, defaults to True):
                If true, special tokens are skipped.
            as_targets (`bool`, *optional*, defaults to False):
                If true, a target tokenizer is used to clean the tokens.

        Returns:
            `OneOrMoreTokenSequences`: A list containing one or more lists of cleaned tokens.
        """
        if isinstance(tokens, list) and len(tokens) == 0:
            return []
        elif isinstance(tokens[0], str):
            clean_tokens = []
            for tok in tokens:
                clean_tok = self.convert_tokens_to_string(
                    [tok], skip_special_tokens=skip_special_tokens, as_targets=as_targets
                )
                if clean_tok:
                    clean_tokens.append(clean_tok)
                elif tok:
                    clean_tokens.append(" ")
            return clean_tokens
        return [self.clean_tokens(token_seq, skip_special_tokens, as_targets) for token_seq in tokens]

    @property
    def special_tokens(self) -> List[str]:
        return self.tokenizer.all_special_tokens

    @property
    def special_tokens_ids(self) -> List[int]:
        return self.tokenizer.all_special_ids

    @property
    def vocabulary_embeddings(self) -> VocabularyEmbeddingsTensor:
        return self.get_embedding_layer().weight

    def get_embedding_layer(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()


class HuggingfaceEncoderDecoderModel(HuggingfaceModel, EncoderDecoderAttributionModel):
    """Model wrapper for any ForConditionalGeneration model on the HuggingFace Hub used to enable
    feature attribution. Corresponds to AutoModelForSeq2SeqLM auto classes in HF transformers.

    Attributes:
        model (::obj:`transformers.AutoModelForSeq2SeqLM`):
            the model on which attribution is performed.
    """

    _autoclass = AutoModelForSeq2SeqLM

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        attribution_method: Optional[str] = None,
        tokenizer: Union[str, PreTrainedTokenizerBase, None] = None,
        device: str = None,
        **kwargs,
    ) -> NoReturn:
        super().__init__(model, attribution_method, tokenizer, device, **kwargs)

    def configure_embeddings_scale(self):
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        if hasattr(encoder, "embed_scale"):
            self.embed_scale = encoder.embed_scale
        if hasattr(decoder, "embed_scale") and decoder.embed_scale != self.embed_scale:
            raise ValueError("Different encoder and decoder embed scales are not supported")

    def get_encoder(self) -> torch.nn.Module:
        return self.model.get_encoder()

    def get_decoder(self) -> torch.nn.Module:
        return self.model.get_decoder()

    @staticmethod
    def get_attentions_dict(
        output: Seq2SeqLMOutput,
    ) -> Dict[str, MultiLayerMultiUnitScoreTensor]:
        if output.encoder_attentions is None or output.decoder_attentions is None:
            raise ValueError("Model does not support attribution relying on attention outputs.")
        return {
            "encoder_self_attentions": torch.stack(output.encoder_attentions, dim=1),
            "decoder_self_attentions": torch.stack(output.decoder_attentions, dim=1),
            "cross_attentions": torch.stack(output.cross_attentions, dim=1),
        }

    @staticmethod
    def get_hidden_states_dict(output: Seq2SeqLMOutput) -> Dict[str, MultiLayerEmbeddingsTensor]:
        return {
            "encoder_hidden_states": torch.stack(output.encoder_hidden_states, dim=1),
            "decoder_hidden_states": torch.stack(output.decoder_hidden_states, dim=1),
        }


class HuggingfaceDecoderOnlyModel(HuggingfaceModel, DecoderOnlyAttributionModel):
    """Model wrapper for any ForCausalLM or LMHead model on the HuggingFace Hub used to enable
    feature attribution. Corresponds to AutoModelForCausalLM auto classes in HF transformers.

    Attributes:
        model (::obj:`transformers.AutoModelForCausalLM`):
            the model on which attribution is performed.
    """

    _autoclass = AutoModelForCausalLM

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        attribution_method: Optional[str] = None,
        tokenizer: Union[str, PreTrainedTokenizerBase, None] = None,
        device: str = None,
        **kwargs,
    ) -> NoReturn:
        super().__init__(model, attribution_method, tokenizer, device, **kwargs)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.pad_token is None:
            self.pad_token = self.tokenizer.bos_token
            self.tokenizer.pad_token = self.tokenizer.bos_token

    def configure_embeddings_scale(self):
        if hasattr(self.model, "embed_scale"):
            self.embed_scale = self.model.embed_scale

    @staticmethod
    def get_attentions_dict(output: CausalLMOutput) -> Dict[str, MultiLayerMultiUnitScoreTensor]:
        if output.attentions is None:
            raise ValueError("Model does not support attribution relying on attention outputs.")
        return {
            "decoder_self_attentions": torch.stack(output.attentions, dim=1),
        }

    @staticmethod
    def get_hidden_states_dict(output: CausalLMOutput) -> Dict[str, MultiLayerEmbeddingsTensor]:
        return {
            "decoder_hidden_states": torch.stack(output.hidden_states, dim=1),
        }
