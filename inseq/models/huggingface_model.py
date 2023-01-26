""" HuggingFace Seq2seq model """
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
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import CausalLMOutput, ModelOutput, Seq2SeqLMOutput

from ..data import BatchEncoding
from ..utils.typing import (
    EmbeddingsTensor,
    FullLogitsTensor,
    IdsTensor,
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
        tokenizer (AutoTokenizer): the tokenizer associated to the model.
        device (str): the device on which the model is run (CPU or GPU).
        encoder_int_embeds (InterpretableEmbeddingBase): the interpretable embedding layer for the encoder, used for
            layer attribution methods in Captum.
        decoder_int_embeds (InterpretableEmbeddingBase): the interpretable embedding layer for the decoder, used for
            layer attribution methods in Captum.
        embed_scale (float, optional): scale factor for embeddings.
    """

    _autoclass = None

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        attribution_method: Optional[str] = None,
        tokenizer: Union[str, PreTrainedTokenizer, None] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the AttributionModel with a Huggingface-compatible model.
        Performs the setup for model and embeddings.

        Args:
            model (:obj:`str` or :obj:`transformers.PreTrainedModel`): the name of the model in the
                Huggingface Hub or path to folder containing local model files.
            tokenizer (:obj:`str` or :obj:`transformers.PreTrainedTokenizerBase`, optional): the name of the tokenizer
                in the Huggingface Hub or path to folder containing local tokenizer files.
                Default: use model name.
            attribution_method (str, optional): The attribution method to use.
                Passing it here reduces overhead on attribute call, since it is already
                initialized.
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

        if isinstance(tokenizer, PreTrainedTokenizer):
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, *tokenizer_inputs, **tokenizer_kwargs)
        if self.model.config.pad_token_id is not None:
            self.pad_token = self.tokenizer.convert_ids_to_tokens(self.model.config.pad_token_id)
            self.tokenizer.pad_token = self.pad_token
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
        tokenizer: Union[str, PreTrainedTokenizer, None] = None,
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
    def generate(
        self,
        inputs: Union[TextInput, BatchEncoding],
        return_generation_output: bool = False,
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
            isinstance(inputs, list) and len(inputs) > 0 and all([isinstance(x, str) for x in inputs])
        ):
            inputs = self.encode(inputs)
        inputs = inputs.to(self.device)
        generation_out = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            return_dict_in_generate=True,
            **kwargs,
        )
        texts = self.tokenizer.batch_decode(
            generation_out.sequences,
            skip_special_tokens=True,
        )
        if return_generation_output:
            return texts, generation_out
        return texts

    @staticmethod
    def output2logits(forward_output: Union[Seq2SeqLMOutput, CausalLMOutput]) -> FullLogitsTensor:
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
    ) -> BatchEncoding:
        """Encode one or multiple texts, producing a BatchEncoding

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
            add_special_tokens=True,
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
                baseline_ids = batch["input_ids"].ne(self.eos_token_id).long() * self.tokenizer.unk_token_id
        # We prepend a BOS token only when tokenizing target texts.
        if as_targets and self.is_encoder_decoder:
            ones_mask = torch.ones((batch["input_ids"].shape[0], 1), device=self.device, dtype=long)
            batch["attention_mask"] = torch.cat((ones_mask, batch["attention_mask"]), dim=1)
            bos_ids = ones_mask * self.model.config.decoder_start_token_id
            batch["input_ids"] = torch.cat((bos_ids, batch["input_ids"]), dim=1)
            if return_baseline:
                baseline_ids = torch.cat((bos_ids, baseline_ids), dim=1)
        return BatchEncoding(
            input_ids=batch["input_ids"],
            input_tokens=[self.tokenizer.convert_ids_to_tokens(x) for x in batch["input_ids"]],
            attention_mask=batch["attention_mask"],
            baseline_ids=baseline_ids,
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
        if len(ids.shape) < 2:
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
            )["input_ids"]
            return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens)
        return [self.convert_string_to_tokens(t, skip_special_tokens, as_targets) for t in text]

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
        tokenizer: Union[str, PreTrainedTokenizer, None] = None,
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
        tokenizer: Union[str, PreTrainedTokenizer, None] = None,
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
