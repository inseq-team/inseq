"""Interpretability for Sequence-to-sequence models 🔍"""

from importlib import metadata as importlib_metadata

from .attr import list_feature_attribution_methods
from .data import load_attributions, save_attributions, show_attributions
from .models import AttributionModel, load_model


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()

__all__ = [
    "AttributionModel",
    "load_model",
    "show_attributions",
    "save_attributions",
    "load_attributions",
    "list_feature_attribution_methods",
    "version",
]
