"""Interpretability for Sequence-to-sequence models 🔍"""

import sys
from importlib import metadata as importlib_metadata

from .data import show_attributions
from .models import AttributionModel, load


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()

__all__ = ["AttributionModel", "load", "show_attributions", "version"]
