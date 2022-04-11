from typing import Any, List, Optional, Tuple, Union

import logging
from abc import ABC, abstractmethod

import torch
from rich.status import Status

from ..attr.feat.feature_attribution import FeatureAttribution
from ..data import BatchEncoding, FeatureAttributionOutput
from ..utils import MissingAttributionMethodError, format_input_texts, isnotebook
from ..utils.typing import (
    EmbeddingsTensor,
    IdsTensor,
    ModelIdentifier,
    OneOrMoreIdSequences,
    OneOrMoreTokenSequences,
    TextInput,
    VocabularyEmbeddingsTensor,
)
from .model_decorators import unhooked


logger = logging.getLogger(__name__)


class AttributionModel(ABC, torch.nn.Module):
    def __init__(self, attribution_method: Optional[str] = None, device: str = None, **kwargs) -> None:
        super().__init__()
        if not hasattr(self, "model"):
            self.model = None
            self.model_name = None
        self._device = None
        self.attribution_method = None
        self.is_hooked = False

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
        output_step_probabilities: bool = False,
        include_eos_baseline: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ) -> FeatureAttributionOutput:
        """Perform attribution for one or multiple texts."""
        if not input_texts:
            raise ValueError("At least one text must be provided to perform attribution.")
        if device is not None:
            original_device = self.device
            self.device = device
        input_texts, generated_texts = format_input_texts(input_texts, generated_texts)
        constrained_decoding = generated_texts is not None
        orig_input_texts = input_texts
        if not constrained_decoding:
            input_texts = self.encode(input_texts, return_baseline=True, include_eos_baseline=include_eos_baseline)
            generation_args = kwargs.pop("generation_args", {})
            generated_texts = self.generate(input_texts, return_generation_output=False, **generation_args)
        logger.debug(f"reference_texts={generated_texts}")
        attribution_method = self.get_attribution_method(method, override_default_attribution)
        attribution_args = kwargs.pop("attribution_args", {})
        extra_attribution_args, unused_args = attribution_method.get_attribution_args(**kwargs)
        if unused_args:
            logger.warning(f"Unused arguments during attribution: {unused_args}")
        attribution_args.update(extra_attribution_args)
        if isnotebook():
            logger.debug("Pretty progress currently not supported in notebooks, falling back to tqdm.")
            pretty_progress = False
        attribution_output = attribution_method.prepare_and_attribute(
            input_texts,
            generated_texts,
            attr_pos_start=attr_pos_start,
            attr_pos_end=attr_pos_end,
            show_progress=show_progress,
            pretty_progress=pretty_progress,
            output_step_attributions=output_step_attributions,
            attribute_target=attribute_target,
            output_step_probabilities=output_step_probabilities,
            include_eos_baseline=include_eos_baseline,
            **attribution_args,
        )
        attribution_output.info["input_texts"] = orig_input_texts
        attribution_output.info["generated_texts"] = (
            [generated_texts] if isinstance(generated_texts, str) else generated_texts
        )
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

    @abstractmethod
    def encode(self, texts: TextInput, as_targets: Optional[bool] = False, *args) -> BatchEncoding:
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
