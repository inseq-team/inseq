from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import logging
from abc import ABC, abstractmethod

import torch
from rich.status import Status

from ..attr import STEP_SCORES_MAP
from ..attr.feat.feature_attribution import FeatureAttribution
from ..data import BatchEncoding, FeatureAttributionOutput
from ..utils import MissingAttributionMethodError, extract_signature_args, format_input_texts, isnotebook
from ..utils.typing import (
    EmbeddingsTensor,
    FullLogitsTensor,
    IdsTensor,
    ModelIdentifier,
    OneOrMoreIdSequences,
    OneOrMoreTokenSequences,
    SingleScorePerStepTensor,
    TextInput,
    VocabularyEmbeddingsTensor,
)
from .model_decorators import unhooked


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

    def __init__(self, attribution_method: Optional[str] = None, device: str = None, **kwargs) -> None:
        super().__init__()
        if not hasattr(self, "model"):
            self.model = None
            self.model_name = None
        self._device = None
        self.attribution_method = None
        self.is_hooked = False
        self._default_attributed_fn_id = "probability"

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, new_device: str) -> None:
        if "cuda" in new_device and not torch.cuda.is_available():
            raise torch.cuda.CudaError("Cannot use CUDA device, CUDA is not available.")
        self._device = new_device
        if self.model:
            self.model.to(self._device)

    def setup(self, device: str = None, attribution_method: str = None) -> None:
        """Move the model to device and in eval mode."""
        if device is not None:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.model:
            self.model.eval()
            self.model.zero_grad()
            self.attribution_method = self.get_attribution_method(attribution_method)

    @property
    def default_attributed_fn_id(self) -> str:
        return self._default_attributed_fn_id

    @default_attributed_fn_id.setter
    def set_attributed_fn(self, fn: str):
        if fn not in STEP_SCORES_MAP:
            raise ValueError(f"Unknown function: {fn}. Register custom functions with inseq.register_step_score")
        self._default_attributed_fn_id = fn

    @staticmethod
    def load(
        model_name_or_path: ModelIdentifier,
        attribution_method: Optional[str] = None,
        **kwargs,
    ):
        return load_model(model_name_or_path, attribution_method, **kwargs)

    def get_attribution_method(
        self,
        method: Optional[str] = None,
        override_default_attribution: Optional[bool] = False,
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
                self.attribution_method = FeatureAttribution.load(method, attribution_model=self)
            # Temporarily use the current method without overriding the default
            else:
                return FeatureAttribution.load(method, attribution_model=self)
        return self.attribution_method

    def get_attributed_fn(
        self, attributed_fn: Union[str, Callable[..., SingleScorePerStepTensor], None] = None
    ) -> Callable[..., SingleScorePerStepTensor]:
        if attributed_fn is None:
            attributed_fn = self.default_attributed_fn_id
        if isinstance(attributed_fn, str):
            if attributed_fn not in STEP_SCORES_MAP:
                raise ValueError(
                    f"Unknown function: {attributed_fn}." "Register custom functions with inseq.register_step_score"
                )
            attributed_fn = STEP_SCORES_MAP[attributed_fn]
        return attributed_fn

    def extract_args(
        self,
        attribution_method: "FeatureAttribution",
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        step_scores: List[str],
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        attribution_args = kwargs.pop("attribution_args", {})
        attributed_fn_args = kwargs.pop("attributed_fn_args", {})
        step_scores_args = kwargs.pop("step_scores_args", {})
        extra_attribution_args, attribution_unused_args = attribution_method.get_attribution_args(**kwargs)
        extra_attributed_fn_args, attributed_fn_unused_args = extract_signature_args(
            kwargs, attributed_fn, exclude_args=self._DEFAULT_ATTRIBUTED_FN_ARGS, return_remaining=True
        )
        extra_step_scores_args = {}
        for step_score in step_scores:
            if step_score not in STEP_SCORES_MAP:
                raise AttributeError(
                    f"Step score {step_score} not found. Available step scores are: "
                    f"{', '.join([x for x in STEP_SCORES_MAP.keys()])}. Use the inseq.register_step_score"
                    f"function to register a custom step score."
                )
            extra_step_scores_args.update(
                **extract_signature_args(
                    kwargs,
                    STEP_SCORES_MAP[step_score],
                    exclude_args=self._DEFAULT_ATTRIBUTED_FN_ARGS,
                    return_remaining=False,
                )
            )
        step_scores_unused_args = {k: v for k, v in kwargs.items() if k not in extra_step_scores_args}
        unused_args = {
            k: v
            for k, v in kwargs.items()
            if k in attribution_unused_args.keys() & attributed_fn_unused_args.keys() & step_scores_unused_args.keys()
        }
        if unused_args:
            logger.warning(f"Unused arguments during attribution: {list(unused_args.keys())}")
        attribution_args.update(extra_attribution_args)
        attributed_fn_args.update(extra_attributed_fn_args)
        step_scores_args.update(extra_step_scores_args)
        return attribution_args, attributed_fn_args, step_scores_args

    def attribute(
        self,
        input_texts: TextInput,
        generated_texts: Optional[TextInput] = None,
        method: Optional[str] = None,
        override_default_attribution: Optional[bool] = False,
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        output_step_attributions: bool = False,
        attribute_target: bool = False,
        step_scores: List[str] = [],
        include_eos_baseline: bool = False,
        prepend_bos_token: bool = True,
        attributed_fn: Union[str, Callable[..., SingleScorePerStepTensor], None] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> FeatureAttributionOutput:
        """Perform attribution for one or multiple texts."""
        if not input_texts:
            raise ValueError("At least one text must be provided to perform attribution.")
        if device is not None:
            original_device = self.device
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
        if not constrained_decoding:
            input_texts = self.encode(input_texts, return_baseline=True, include_eos_baseline=include_eos_baseline)
            generated_texts = self.generate(input_texts, return_generation_output=False, **generation_args)
        logger.debug(f"reference_texts={generated_texts}")
        attribution_method = self.get_attribution_method(method, override_default_attribution)
        attributed_fn = self.get_attributed_fn(attributed_fn)
        attribution_args, attributed_fn_args, step_scores_args = self.extract_args(
            attribution_method, attributed_fn, step_scores, **kwargs
        )
        if isnotebook():
            logger.debug("Pretty progress currently not supported in notebooks, falling back to tqdm.")
            pretty_progress = False
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
            prepend_bos_token=prepend_bos_token,
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
        if device is not None:
            self.device = original_device
        return attribution_output

    def embed(self, inputs: Union[TextInput, IdsTensor], as_targets: bool = False):
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and len(inputs) > 0 and all([isinstance(x, str) for x in inputs])
        ):
            batch = self.encode(inputs, as_targets)
            inputs = batch.input_ids
        if as_targets:
            return self.decoder_embed_ids(inputs)
        return self.encoder_embed_ids(inputs)

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
    def encode(self, texts: TextInput, as_targets: Optional[bool] = False, *args, **kwargs) -> BatchEncoding:
        pass

    @abstractmethod
    def encoder_embed_ids(self, ids: IdsTensor) -> EmbeddingsTensor:
        pass

    @abstractmethod
    def decoder_embed_ids(self, ids: IdsTensor) -> EmbeddingsTensor:
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
    ) -> TextInput:
        pass

    @abstractmethod
    def convert_string_to_tokens(
        self, text: TextInput, skip_special_tokens: Optional[bool] = True
    ) -> OneOrMoreTokenSequences:
        pass

    @property
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


def load_model(
    model_name_or_path: ModelIdentifier,
    attribution_method: Optional[str] = None,
    **kwargs,
):
    from .huggingface_model import HuggingfaceModel

    from_hf = kwargs.pop("from_hf", None)
    desc_id = ", ".join(model_name_or_path) if isinstance(model_name_or_path, tuple) else model_name_or_path
    desc = f"Loading {desc_id}" + (
        f" with {attribution_method} method..." if attribution_method else " without methods..."
    )
    with Status(desc):
        if from_hf:
            return HuggingfaceModel(model_name_or_path, attribution_method, **kwargs)
        else:  # Default behavior is using Huggingface
            return HuggingfaceModel(model_name_or_path, attribution_method, **kwargs)
