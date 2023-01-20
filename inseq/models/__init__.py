import logging
from typing import List, Optional, Union

from rich.status import Status

from ..utils import isnotebook, optional
from ..utils.typing import ModelClass, ModelIdentifier
from .attribution_model import AttributionModel
from .decoder_only import DecoderOnlyAttributionModel
from .encoder_decoder import EncoderDecoderAttributionModel
from .huggingface_model import HuggingfaceDecoderOnlyModel, HuggingfaceEncoderDecoderModel, HuggingfaceModel

logger = logging.getLogger(__name__)

FRAMEWORKS_MAP = {
    "hf_transformers": HuggingfaceModel,
}


def load_model(
    model: Union[ModelIdentifier, ModelClass],
    attribution_method: Optional[str] = None,
    framework: str = "hf_transformers",
    **kwargs,
) -> AttributionModel:
    """Factory function to load a model with or without attribution methods.

    Args:
        model (`Union[ModelIdentifier, ModelClass]`):
            Either a model identifier (e.g. `gpt2` in HF transformers) or an instance of a model class supported by the
            selected modeling framework.
        attribution_method (`Optional[str]`, *optional*, defaults to None):
            Identifier for the attribution method to use. If `None`, the model will be loaded without any attribution
            methods, which can be added during attribution.
        framework (`str`, *optional*, defaults to "hf_transformers"):
            The framework to use for loading the model. Currently, only HF transformers is supported.

    Returns:
        `AttributionModel`: An instance of one of `AttributionModel` children classes matching the selected framework
        and model architecture.
    """
    model_name = model if isinstance(model, str) else "model"
    method_desc = f"with {attribution_method} method..." if attribution_method else " without attribution methods..."
    load_msg = f"Loading {model_name} {method_desc}"
    with optional(not isnotebook(), Status(load_msg), logger.info, msg=load_msg):
        return FRAMEWORKS_MAP[framework].load(model, attribution_method, **kwargs)


def list_supported_frameworks() -> List[str]:
    """
    Lists identifiers for all available frameworks. These can be used to load models with the `framework` argument in
    the :meth:`~inseq.load_model` function.
    """
    return list(FRAMEWORKS_MAP.keys())


__all__ = [
    "AttributionModel",
    "HuggingfaceModel",
    "HuggingfaceEncoderDecoderModel",
    "HuggingfaceDecoderOnlyModel",
    "DecoderOnlyAttributionModel",
    "EncoderDecoderAttributionModel",
    "load_model",
    "list_supported_frameworks",
]
