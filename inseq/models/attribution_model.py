import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch

from ..attr import STEP_SCORES_MAP
from ..attr.feat import FeatureAttribution, extract_args, join_token_ids
from ..data import (
    BatchEncoding,
    DecoderOnlyBatch,
    EncoderDecoderBatch,
    FeatureAttributionOutput,
    FeatureAttributionStepOutput,
)
from ..utils import MissingAttributionMethodError, check_device, format_input_texts, get_default_device, isnotebook
from ..utils.typing import (
    EmbeddingsTensor,
    ExpandedTargetIdsTensor,
    FullLogitsTensor,
    IdsTensor,
    OneOrMoreIdSequences,
    OneOrMoreTokenSequences,
    SingleScorePerStepTensor,
    TargetIdsTensor,
    TextInput,
    TextSequences,
    TokenWithId,
    VocabularyEmbeddingsTensor,
)
from .model_decorators import unhooked

ModelOutput = TypeVar("ModelOutput")


logger = logging.getLogger(__name__)


class AttributionModel(ABC, torch.nn.Module):
    # Default arguments for custom attributed functions
    # in the AttributionModel.forward method.
    _DEFAULT_ATTRIBUTED_FN_ARGS = [
        "attribution_model",
        "forward_output",
        "encoder_input_ids",
        "decoder_input_ids",
        "encoder_input_embeds",
        "decoder_input_embeds",
        "target_ids",
        "encoder_attention_mask",
        "decoder_attention_mask",
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__()
        if not hasattr(self, "model"):
            self.model = None
            self.model_name = None
            self.is_encoder_decoder = True
        self.pad_token = None
        self.embed_scale = None
        self._device = None
        self.attribution_method = None
        self.is_hooked = False
        self._default_attributed_fn_id = "probability"

    @property
    def device(self) -> Optional[str]:
        return self._device

    @device.setter
    def device(self, new_device: str) -> None:
        check_device(new_device)
        self._device = new_device
        if self.model:
            self.model.to(self._device)

    def setup(self, device: Optional[str] = None, attribution_method: Optional[str] = None, **kwargs) -> None:
        """Move the model to device and in eval mode."""
        self.device = device if device is not None else get_default_device()
        if self.model:
            self.model.eval()
            self.model.zero_grad()
            self.attribution_method = self.get_attribution_method(attribution_method, **kwargs)

    @property
    def default_attributed_fn_id(self) -> str:
        return self._default_attributed_fn_id

    @default_attributed_fn_id.setter
    def set_attributed_fn(self, fn: str):
        if fn not in STEP_SCORES_MAP:
            raise ValueError(f"Unknown function: {fn}. Register custom functions with inseq.register_step_score")
        self._default_attributed_fn_id = fn

    @property
    def info(self) -> Dict[Optional[str], Optional[str]]:
        return {
            "model_name": self.model_name,
            "model_class": self.model.__class__.__name__ if self.model is not None else None,
        }

    def get_attribution_method(
        self,
        method: Optional[str] = None,
        override_default_attribution: Optional[bool] = False,
        **kwargs,
    ) -> FeatureAttribution:
        # No method present -> missing method error
        if not method:
            if not self.attribution_method:
                raise MissingAttributionMethodError()
        else:
            if self.attribution_method:
                self.attribution_method.unhook()
            # If either the default method is missing or the override is set,
            # set the default method to the given method
            if override_default_attribution or not self.attribution_method:
                self.attribution_method = FeatureAttribution.load(method, attribution_model=self, **kwargs)
            # Temporarily use the current method without overriding the default
            else:
                return FeatureAttribution.load(method, attribution_model=self, **kwargs)
        return self.attribution_method

    def get_attributed_fn(
        self, attributed_fn: Union[str, Callable[..., SingleScorePerStepTensor], None] = None
    ) -> Callable[..., SingleScorePerStepTensor]:
        if attributed_fn is None:
            attributed_fn = self.default_attributed_fn_id
        if isinstance(attributed_fn, str):
            if attributed_fn not in STEP_SCORES_MAP:
                raise ValueError(
                    f"Unknown function: {attributed_fn}. Register custom functions with inseq.register_step_score"
                )
            attributed_fn = STEP_SCORES_MAP[attributed_fn]
        return attributed_fn

    def attribute(
        self,
        input_texts: TextInput,
        generated_texts: Optional[TextInput] = None,
        method: Optional[str] = None,
        override_default_attribution: Optional[bool] = False,
        attr_pos_start: Optional[int] = None,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        output_step_attributions: bool = False,
        attribute_target: bool = False,
        step_scores: List[str] = [],
        include_eos_baseline: bool = False,
        attributed_fn: Union[str, Callable[..., SingleScorePerStepTensor], None] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> FeatureAttributionOutput:
        """Perform attribution for one or multiple texts."""
        if not input_texts:
            raise ValueError("At least one text must be provided to perform attribution.")
        if attribute_target and not self.is_encoder_decoder:
            logger.warning("attribute_target parameter is set to True, but will be ignored (not an encoder-decoder).")
        original_device = self.device
        if device is not None:
            self.device = device
        input_texts, generated_texts = format_input_texts(input_texts, generated_texts)
        if batch_size is not None:
            n_batches = len(input_texts) // batch_size + ((len(input_texts) % batch_size) > 0)
            logger.info(f"Splitting input texts into {n_batches} batches of size {batch_size}.")
        constrained_decoding = generated_texts is not None
        orig_input_texts = input_texts
        # If constrained decoding is not enabled, we need to generate the
        # generated texts from the input texts.
        generation_args = kwargs.pop("generation_args", {})
        if constrained_decoding and generation_args:
            logger.warning(
                f"Generation arguments {generation_args} are provided, but constrained decoding is enabled. "
                "Generation arguments will be ignored."
            )
        if not constrained_decoding:
            encoded_input = self.encode(input_texts, return_baseline=True, include_eos_baseline=include_eos_baseline)
            generated_texts = self.generate(encoded_input, return_generation_output=False, **generation_args)
        logger.debug(f"reference_texts={generated_texts}")
        attribution_method = self.get_attribution_method(method, override_default_attribution)
        attributed_fn = self.get_attributed_fn(attributed_fn)
        attribution_args, attributed_fn_args, step_scores_args = extract_args(
            attribution_method, attributed_fn, step_scores, default_args=self._DEFAULT_ATTRIBUTED_FN_ARGS, **kwargs
        )
        if isnotebook():
            logger.debug("Pretty progress currently not supported in notebooks, falling back to tqdm.")
            pretty_progress = False
        if not self.is_encoder_decoder:
            assert all(
                generated_texts[idx].startswith(input_texts[idx]) for idx in range(len(input_texts))
            ), "Forced generations with decoder-only models must start with the input texts."
            if constrained_decoding and len(input_texts) > 1:
                logger.info(
                    "Batched constrained decoding is currently not supported for decoder-only models."
                    " Using batch size of 1."
                )
                batch_size = 1
            if len(input_texts) > 1 and (attr_pos_start is not None or attr_pos_end is not None):
                logger.info(
                    "Custom attribution positions are currently not supported when batching generations for"
                    " decoder-only models. Using batch size of 1."
                )
                batch_size = 1
        attribution_outputs = attribution_method.prepare_and_attribute(
            input_texts,
            generated_texts,
            batch_size=batch_size,
            attr_pos_start=attr_pos_start,
            attr_pos_end=attr_pos_end,
            show_progress=show_progress,
            pretty_progress=pretty_progress,
            output_step_attributions=output_step_attributions,
            attribute_target=attribute_target,
            step_scores=step_scores,
            include_eos_baseline=include_eos_baseline,
            attributed_fn=attributed_fn,
            attribution_args=attribution_args,
            attributed_fn_args=attributed_fn_args,
            step_scores_args=step_scores_args,
        )
        attribution_output = FeatureAttributionOutput.merge_attributions(attribution_outputs)
        attribution_output.info["input_texts"] = orig_input_texts
        attribution_output.info["generated_texts"] = (
            [generated_texts] if isinstance(generated_texts, str) else generated_texts
        )
        attribution_output.info["generation_args"] = generation_args
        attribution_output.info["constrained_decoding"] = constrained_decoding
        if device and original_device:
            self.device = original_device
        return attribution_output

    def embed(self, inputs: Union[TextInput, IdsTensor], as_targets: bool = False):
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and len(inputs) > 0 and all([isinstance(x, str) for x in inputs])
        ):
            batch = self.encode(inputs, as_targets)
            inputs = batch.input_ids
        return self.embed_ids(inputs, as_targets=as_targets)

    def tokenize_with_ids(
        self, inputs: TextInput, as_targets: bool = False, skip_special_tokens: bool = True
    ) -> List[List[TokenWithId]]:
        tokenized_sentences = self.convert_string_to_tokens(
            inputs, as_targets=as_targets, skip_special_tokens=skip_special_tokens
        )
        ids_sentences = self.convert_tokens_to_ids(tokenized_sentences)
        return join_token_ids(tokenized_sentences, ids_sentences)

    # Framework-specific methods

    @unhooked
    @abstractmethod
    def generate(
        self,
        encodings: Union[TextInput, BatchEncoding],
        return_generation_output: Optional[bool] = False,
        **kwargs,
    ) -> Union[List[str], Tuple[List[str], Any]]:
        pass

    @staticmethod
    @abstractmethod
    def output2logits(forward_output) -> FullLogitsTensor:
        pass

    @abstractmethod
    def encode(
        self,
        texts: TextInput,
        as_targets: bool = False,
        return_baseline: bool = False,
        include_eos_baseline: bool = False,
    ) -> BatchEncoding:
        pass

    @abstractmethod
    def embed_ids(self, ids: IdsTensor, as_targets: bool = False) -> EmbeddingsTensor:
        pass

    @abstractmethod
    def convert_ids_to_tokens(
        self, ids: torch.Tensor, skip_special_tokens: Optional[bool] = True
    ) -> OneOrMoreTokenSequences:
        pass

    @abstractmethod
    def convert_tokens_to_ids(
        self,
        tokens: Union[List[str], List[List[str]]],
    ) -> OneOrMoreIdSequences:
        pass

    @abstractmethod
    def convert_tokens_to_string(
        self,
        tokens: OneOrMoreTokenSequences,
        skip_special_tokens: Optional[bool] = True,
        as_targets: bool = False,
    ) -> TextInput:
        pass

    @abstractmethod
    def convert_string_to_tokens(
        self,
        text: TextInput,
        skip_special_tokens: bool = True,
        as_targets: bool = False,
    ) -> OneOrMoreTokenSequences:
        pass

    @property
    @abstractmethod
    def special_tokens(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def special_tokens_ids(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def vocabulary_embeddings(self) -> VocabularyEmbeddingsTensor:
        pass

    @abstractmethod
    def get_embedding_layer(self) -> torch.nn.Module:
        pass

    def configure_interpretable_embeddings(self, **kwargs) -> None:
        """Configure the model with interpretable embeddings for gradient attribution.

        This method needs to be defined for models that cannot receive embeddings directly from their
        forward method parameters, to allow the usage of interpretable embeddings as surrogate for
        feature attribution methods. Model that support precomputed embedding inputs by design can
        skip this method.
        """
        pass

    def remove_interpretable_embeddings(self, **kwargs) -> None:
        """Removes interpretable embeddings used for gradient attribution.

        If the configure_interpretable_embeddings method is defined, this method needs to be defined
        to allow restoring original embeddings for the model. This is required for methods using the
        decorator @unhooked since they require the original model capabilities.
        """
        pass

    # Architecture-specific methods

    @abstractmethod
    def prepare_inputs_for_attribution(
        self,
        inputs: Any,
        include_eos_baseline: bool = False,
        use_layer_attribution: bool = False,
    ) -> Union[DecoderOnlyBatch, EncoderDecoderBatch]:
        pass

    @staticmethod
    @abstractmethod
    def format_forward_args(
        inputs: Union[DecoderOnlyBatch, EncoderDecoderBatch],
        use_embeddings: bool = True,
    ) -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def format_attribution_args(
        batch: Union[DecoderOnlyBatch, EncoderDecoderBatch],
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attributed_fn_args: Dict[str, Any] = {},
        attribute_batch_ids: bool = False,
        forward_batch_embeds: bool = True,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Tuple[Union[IdsTensor, EmbeddingsTensor, None], ...]]:
        pass

    @abstractmethod
    def get_text_sequences(self, batch: Union[DecoderOnlyBatch, EncoderDecoderBatch]) -> TextSequences:
        pass

    @abstractmethod
    def get_forward_output(
        self,
        **kwargs,
    ) -> ModelOutput:
        pass

    @staticmethod
    @abstractmethod
    def enrich_step_output(
        step_output: FeatureAttributionStepOutput,
        batch: Union[DecoderOnlyBatch, EncoderDecoderBatch],
        target_tokens: OneOrMoreTokenSequences,
        target_ids: TargetIdsTensor,
    ) -> FeatureAttributionStepOutput:
        pass

    @abstractmethod
    def format_step_function_args(
        self,
        forward_output: ModelOutput,
        target_ids: ExpandedTargetIdsTensor,
        encoder_input_ids: Optional[IdsTensor] = None,
        decoder_input_ids: Optional[IdsTensor] = None,
        encoder_input_embeds: Optional[EmbeddingsTensor] = None,
        decoder_input_embeds: Optional[EmbeddingsTensor] = None,
        encoder_attention_mask: Optional[IdsTensor] = None,
        decoder_attention_mask: Optional[IdsTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def forward(
        self,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attributed_fn_argnames: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> FullLogitsTensor:
        pass
