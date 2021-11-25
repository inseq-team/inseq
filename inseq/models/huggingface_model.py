""" HuggingFace Seq2seq model """
from typing import List, Literal, NoReturn, Optional, Tuple, Union, overload

import logging
import warnings

import torch
from captum.attr import (
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)
from torch import long
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.generation_utils import (
    BeamSampleOutput,
    BeamSearchOutput,
    GreedySearchOutput,
    SampleOutput,
)

from ..data import BatchEncoding
from ..utils import optional, pretty_tensor
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
from .model_decorators import unhooked

logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)

GenerationOutput = Union[
    GreedySearchOutput,
    SampleOutput,
    BeamSearchOutput,
    BeamSampleOutput,
    torch.LongTensor,
]


class HuggingfaceModel(AttributionModel):
    """Performs  attribution for any seq2seq model in the HuggingFace Hub.

    Attributes:
        model (AutoModelForSeq2SeqLM): the seq2seq model on which
            attribution is performed.
        tokenizer (AutoTokenizer): the tokenizer associated to the model.
        device (torch.device): the device on which the model is run (CPU or GPU).
        pad_id (int): the id of the pad token.
        eos_id (int): the id of the end of sequence token.
        bos_id (int): the id of the beginning of sequence token.
        encoder_int_embeds (InterpretableEmbeddingBase): the interpretable embedding
            layer for the encoder.
        decoder_int_embeds (InterpretableEmbeddingBase): the interpretable embedding
            layer for the decoder.
        encoder_embed_scale (float, optional): scale factor for encoder embeddings.
        decoder_embed_scale (float, optional): scale factor for decoder embeddings.
    """

    def __init__(
        self,
        model_name_or_path: str,
        attribution_method: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        **kwargs,
    ) -> NoReturn:
        """
        Initialize the AttributionModel with a Huggingface-compatible seq2seq model.
        Performs the setup for model and embeddings.

        Args:
            model_name_or_path (str): the name of the model in the
                Huggingface Hub or path to folder containing local model files.
            tokenizer_name_or_path (str, optional): the name of the tokenizer in the
                Huggingface Hub or path to folder containing local tokenizer files.
                Default: use model_name_or_path value.
            attribution_method (str, optional): The attribution method to use.
                Passing it here reduces overhead on attribute call, since it is already
                initialized.
            **kwargs: additional arguments for the model and the tokenizer.
        """
        if not tokenizer_name_or_path:
            tokenizer_name_or_path = model_name_or_path
        model_args = kwargs.pop("model_args", {})
        model_kwargs = kwargs.pop("model_kwargs", {})
        tokenizer_inputs = kwargs.pop("tokenizer_inputs", {})
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, *model_args, **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, *tokenizer_inputs, **tokenizer_kwargs
        )
        self.model_name = self.model.config.name_or_path
        self.pad_id = self.model.config.pad_token_id
        self.eos_id = self.model.config.eos_token_id
        self.bos_id = self.model.config.decoder_start_token_id
        self.pad_token = self.tokenizer.convert_ids_to_tokens(self.pad_id)
        self.bos_token = self.tokenizer.convert_ids_to_tokens(self.bos_id)
        self.encoder_embed_scale = 1.0
        self.decoder_embed_scale = 1.0
        super().__init__(attribution_method, **kwargs)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        return HuggingfaceModel(model_name_or_path, **kwargs)

    def setup(self, **kwargs) -> NoReturn:
        super().setup(**kwargs)
        self.configure_embeddings_scale()

    def configure_embeddings_scale(self):
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        if hasattr(encoder, "embed_scale"):
            self.encoder_embed_scale = encoder.embed_scale
        if hasattr(decoder, "embed_scale"):
            self.decoder_embed_scale = decoder.embed_scale

    def encode_texts(
        self,
        texts: TextInput,
        as_targets: Optional[bool] = False,
        prepend_bos_token: Optional[bool] = True,
        return_baseline: Optional[bool] = False,
    ) -> BatchEncoding:
        """Encode one or multiple texts, producing a BatchEncoding

        Args:
            texts (str or list of str): the texts to tokenize.
            return_baseline (bool, optional): if True, baseline token ids are returned.

        Returns:
            BatchEncoding: contains ids and attention masks.
        """
        with optional(as_targets, self.tokenizer.as_target_tokenizer()):
            batch = self.tokenizer(
                texts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=(
                    self.tokenizer.max_len_single_sentence
                    if self.tokenizer.max_len_single_sentence < 1e6
                    else max(
                        [v for _, v in self.tokenizer.max_model_input_sizes.items()]
                    )
                ),
                return_tensors="pt",
            )
        baseline_ids = None
        if return_baseline:
            baseline_ids = batch["input_ids"].ne(self.eos_id).long() * self.pad_id
        # We prepend a BOS token only when tokenizing target texts.
        if as_targets and prepend_bos_token:
            ones_mask = torch.ones(
                (batch["input_ids"].shape[0], 1), device=self.device, dtype=long
            )
            batch["attention_mask"] = torch.cat(
                (ones_mask, batch["attention_mask"]), dim=1
            )
            bos_ids = ones_mask * self.bos_id
            batch["input_ids"] = torch.cat((bos_ids, batch["input_ids"]), dim=1)
            if return_baseline:
                baseline_ids = torch.cat((bos_ids, baseline_ids), dim=1)
        return BatchEncoding(
            input_ids=batch["input_ids"].to(self.device),
            input_tokens=[
                self.tokenizer.convert_ids_to_tokens(x) for x in batch["input_ids"]
            ],
            attention_mask=batch["attention_mask"].to(self.device),
            baseline_ids=baseline_ids,
        )

    def convert_ids_to_tokens(
        self, ids: IdsTensor, skip_special_tokens: Optional[bool] = True
    ) -> OneOrMoreTokenSequences:
        if len(ids.shape) < 2:
            return self.tokenizer.convert_ids_to_tokens(
                ids, skip_special_tokens=skip_special_tokens
            )
        return [
            self.tokenizer.convert_ids_to_tokens(
                id_slice, skip_special_tokens=skip_special_tokens
            )
            for id_slice in ids
        ]

    def convert_tokens_to_ids(self, tokens: TextInput) -> OneOrMoreIdSequences:
        if isinstance(tokens[0], str):
            return self.tokenizer.convert_tokens_to_ids(tokens)
        return [
            self.tokenizer.convert_tokens_to_ids(token_slice) for token_slice in tokens
        ]

    def convert_tokens_to_string(
        self,
        tokens: OneOrMoreTokenSequences,
        skip_special_tokens: bool = True,
        as_targets: bool = True,
    ) -> TextInput:
        if isinstance(tokens, list) and len(tokens) == 0:
            return ""
        elif isinstance(tokens[0], str):
            tmp_decode_state = self.tokenizer._decode_use_source_tokenizer
            self.tokenizer._decode_use_source_tokenizer = not as_targets
            out_strings = self.tokenizer.convert_tokens_to_string(
                tokens
                if not skip_special_tokens
                else [t for t in tokens if t not in self.special_tokens]
            )
            self.tokenizer._decode_use_source_tokenizer = tmp_decode_state
            return out_strings
        return [
            self.convert_tokens_to_string(token_slice, skip_special_tokens, as_targets)
            for token_slice in tokens
        ]

    def convert_string_to_tokens(
        self,
        text: TextInput,
        skip_special_tokens: bool = True,
        as_targets: bool = False,
    ) -> OneOrMoreTokenSequences:
        if isinstance(text, str):
            with optional(as_targets, self.tokenizer.as_target_tokenizer()):
                ids = self.tokenizer(text)["input_ids"]
            return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens)
        return [
            self.convert_string_to_tokens(t, skip_special_tokens, as_targets)
            for t in text
        ]

    @property
    def special_tokens(self) -> List[str]:
        return list(self.tokenizer.special_tokens_map.values())

    @property
    def special_tokens_ids(self) -> List[int]:
        return self.convert_tokens_to_ids(self.special_tokens)

    @property
    def vocabulary_embeddings(self) -> VocabularyEmbeddingsTensor:
        return self.model.get_encoder().embed_tokens.weight

    def encoder_embed(self, ids: IdsTensor) -> EmbeddingsTensor:
        if self.encoder_int_embeds:
            embeddings = self.encoder_int_embeds.indices_to_embeddings(ids)
            return embeddings * self.encoder_embed_scale
        else:
            embeddings = self.model.get_input_embeddings()
            return embeddings(ids) * self.encoder_embed_scale

    def decoder_embed(self, ids: IdsTensor) -> EmbeddingsTensor:
        if self.decoder_int_embeds:
            embeddings = self.decoder_int_embeds.indices_to_embeddings(ids)
            return embeddings * self.decoder_embed_scale
        else:
            embeddings = self.model.get_decoder().get_input_embeddings()
            return embeddings(ids) * self.encoder_embed_scale

    @overload
    @unhooked
    def generate(
        self,
        encodings: BatchEncoding,
        return_generation_output: Literal[False] = False,
        **kwargs,
    ) -> List[str]:
        ...

    @overload
    @unhooked
    def generate(
        self,
        encodings: BatchEncoding,
        return_generation_output: Literal[True],
        **kwargs,
    ) -> Tuple[List[str], GenerationOutput]:
        ...

    @unhooked
    def generate(
        self,
        encodings: BatchEncoding,
        return_generation_output: Optional[bool] = False,
        **kwargs,
    ) -> Union[List[str], Tuple[List[str], GenerationOutput]]:
        generation_out = self.model.generate(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask,
            return_dict_in_generate=True,
            **kwargs,
        )
        texts = self.tokenizer.batch_decode(
            generation_out.sequences,
            skip_special_tokens=True,
        )
        texts = texts[0] if len(texts) == 1 else texts
        if return_generation_output:
            return texts, generation_out
        return texts

    def score_func(
        self,
        encoder_tensors: Optional[Union[IdsTensor, EmbeddingsTensor]] = None,
        encoder_attention_mask: Optional[IdsTensor] = None,
        decoder_tensors: Optional[Union[IdsTensor, EmbeddingsTensor]] = None,
        decoder_attention_mask: Optional[IdsTensor] = None,
        compute_embeddings: Optional[bool] = True,
    ) -> FullLogitsTensor:
        output = self.model(
            input_ids=encoder_tensors if compute_embeddings else None,
            inputs_embeds=encoder_tensors if not compute_embeddings else None,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_tensors if compute_embeddings else None,
            decoder_inputs_embeds=decoder_tensors if not compute_embeddings else None,
            decoder_attention_mask=decoder_attention_mask,
        )
        # Full logits for last position of every sentence:
        # (batch_size, tgt_seq_len, vocab_size) => (batch_size, vocab_size)
        logits = output.logits[:, -1, :].squeeze(1)
        logger.debug(f"logits: {pretty_tensor(logits)}")
        return logits

    def get_embedding_layer(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()

    def configure_interpretable_embeddings(self) -> None:
        """Configure the model with interpretable embeddings for gradient attribution."""
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                self.encoder_int_embeds = configure_interpretable_embedding_layer(
                    encoder, "embed_tokens"
                )
                self.decoder_int_embeds = configure_interpretable_embedding_layer(
                    decoder, "embed_tokens"
                )
            except AssertionError:
                logger.warn(
                    "Interpretable embeddings were already configured for layer embed_tokens"
                )

    def remove_interpretable_embeddings(self) -> None:
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        if not self.encoder_int_embeds or not self.encoder_int_embeds:
            logger.warn(
                "Cannot remove interpretable embedding wrapper."
                " No interpretable embedding layer was configured."
            )
        else:
            remove_interpretable_embedding_layer(encoder, self.encoder_int_embeds)
            remove_interpretable_embedding_layer(decoder, self.decoder_int_embeds)
            self.encoder_int_embeds, self.decoder_int_embeds = None, None
