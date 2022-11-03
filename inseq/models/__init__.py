from typing import List, Optional, Union

import logging

from rich.status import Status

from ..utils import isnotebook, optional
from ..utils.typing import ModelClass, ModelIdentifier
from .attribution_model import AttributionModel
from .huggingface_model import HuggingfaceModel


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
    model_name = model if isinstance(model, str) else "model"
    method_desc = f"with {attribution_method} method..." if attribution_method else " without attribution methods..."
    load_msg = f"Loading {model_name} {method_desc}"
    with optional(not isnotebook(), Status(load_msg), logger.info, msg=load_msg):
        return FRAMEWORKS_MAP[framework].load(model, attribution_method, **kwargs)


def list_supported_frameworks() -> List[str]:
    """
    Lists identifiers for all available step scores.
    """
    return list(FRAMEWORKS_MAP.keys())


__all__ = [
    "AttributionModel",
    "HuggingfaceModel",
    "load_model",
    "list_supported_frameworks",
]
