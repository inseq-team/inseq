from typing import Any, List, Optional, Sequence, Tuple, Union, overload

import logging
from abc import ABC, abstractmethod

import torch

from ..attr.feat.feature_attribution import FeatureAttribution
from ..data import BatchEncoding, FeatureAttributionSequenceOutput, OneOrMoreFeatureAttributionSequenceOutputs
from ..data.viz import LoadingMessage
from ..utils import LengthMismatchError, MissingAttributionMethodError, isnotebook
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


class AttributionModel(ABC):
    def __init__(self, attribution_method: Optional[str] = None, **kwargs) -> None:
        if not hasattr(self, "model"):
            self.model = None
            self.model_name = None
        self.attribution_method = None
        self.is_hooked = False
        self.setup(**kwargs)
        self.attribution_method = self.get_attribution_method(attribution_method)

    def setup(self, **kwargs) -> None:
        """Move the model to device and in eval mode."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.model:
            self.model.to(self.device)
            self.model.eval()
            self.model.zero_grad()

    @staticmethod
    def load(
        model_name_or_path: ModelIdentifier,
        attribution_method: Optional[str] = None,
        **kwargs,
    ):
        return load(model_name_or_path, attribution_method, **kwargs)

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

    def format_input_texts(
        self,
        texts: TextInput,
        ref_texts: Optional[TextInput] = None,
    ) -> Tuple[List[str], List[str]]:
        texts = [texts] if isinstance(texts, str) else texts
        reference_texts = [ref_texts] if isinstance(ref_texts, str) else ref_texts
        if reference_texts and len(texts) != len(reference_texts):
            raise LengthMismatchError(
                "Length mismatch for texts and reference_texts."
                "Input length: {}, reference length: {} ".format(len(texts), len(reference_texts))
            )
        return texts, reference_texts

    @overload
    def attribute(
        self,
        texts: str,
        reference_texts: Optional[TextInput] = None,
        method: Optional[str] = None,
        override_default_method: Optional[bool] = False,
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        **kwargs,
    ) -> FeatureAttributionSequenceOutput:
        ...

    @overload
    def attribute(
        self,
        texts: Sequence[str],
        reference_texts: Optional[TextInput] = None,
        method: Optional[str] = None,
        override_default_method: Optional[bool] = False,
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        **kwargs,
    ) -> List[FeatureAttributionSequenceOutput]:
        ...

    def attribute(
        self,
        texts: TextInput,
        reference_texts: Optional[TextInput] = None,
        method: Optional[str] = None,
        override_default_attribution: Optional[bool] = False,
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        output_step_attributions: bool = False,
        **kwargs,
    ) -> OneOrMoreFeatureAttributionSequenceOutputs:
        """Perform attribution for one or multiple texts."""
        if not texts:
            return []
        texts, reference_texts = self.format_input_texts(texts, reference_texts)
        if not reference_texts:
            texts = self.encode_texts(texts, return_baseline=True)
            generation_args = kwargs.pop("generation_args", {})
            reference_texts = self.generate(texts, return_generation_output=False, **generation_args)
        logger.debug(f"reference_texts={reference_texts}")
        attribution_method = self.get_attribution_method(method, override_default_attribution)
        attribution_args = kwargs.pop("attribution_args", {})
        attribution_args.update(attribution_method.get_attribution_args(**kwargs))
        if isnotebook():
            logger.debug("Pretty progress currently not supported in notebooks, falling back to tqdm.")
            pretty_progress = False
        return attribution_method.prepare_and_attribute(
            texts,
            reference_texts,
            attr_pos_start=attr_pos_start,
            attr_pos_end=attr_pos_end,
            show_progress=show_progress,
            pretty_progress=pretty_progress,
            output_step_attributions=output_step_attributions,
            **attribution_args,
        )

    def embed(self, inputs: Union[TextInput, IdsTensor], as_targets: bool = False):
        if isinstance(inputs, str) or (isinstance(inputs, list) and inputs[0] == str):
            batch = self.encode_texts(inputs, as_targets)
            inputs = batch.input_ids
        if as_targets:
            return self.decoder_embed_ids(inputs)
        return self.encoder_embed_ids(inputs)

    @abstractmethod
    def score_func(self, **kwargs) -> torch.Tensor:
        pass

    @unhooked
    @abstractmethod
    def generate(
        self,
        encodings: BatchEncoding,
        return_generation_output: Optional[bool] = False,
        **kwargs,
    ) -> Union[List[str], Tuple[List[str], Any]]:
        pass

    @abstractmethod
    def encode_texts(self, texts: TextInput, as_targets: Optional[bool] = False, *args) -> BatchEncoding:
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


class HookableModelWrapper(torch.nn.Module):
    """Module to wrap the AttributionModel class
    Used in methods requiring a nn.Module instead of a forward_func (e.g. DeepLIFT)
    """

    def __init__(self, attribution_model: AttributionModel):
        super().__init__()
        self.model = attribution_model.model
        self.model.zero_grad()
        self.forward = attribution_model.score_func


def load(
    model_name_or_path: ModelIdentifier,
    attribution_method: Optional[str] = None,
    **kwargs,
):
    from .huggingface_model import HuggingfaceModel

    from_hf = kwargs.pop("from_hf", None)
    verbose = kwargs.get("verbose", True)
    desc_id = ", ".join(model_name_or_path) if isinstance(model_name_or_path, tuple) else model_name_or_path
    desc = f"Loading {desc_id}" + (
        f" with {attribution_method} method..." if attribution_method else " without methods..."
    )
    with LoadingMessage(desc, verbose=verbose):
        if from_hf:
            return HuggingfaceModel(model_name_or_path, attribution_method, **kwargs)
        else:  # Default behavior is using Huggingface
            return HuggingfaceModel(model_name_or_path, attribution_method, **kwargs)
