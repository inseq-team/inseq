"""Interpretability for Sequence Generation Models 🔍."""

from .attr import list_feature_attribution_methods, list_step_functions, register_step_function
from .data import (
    FeatureAttributionOutput,
    list_aggregation_functions,
    list_aggregators,
    merge_attributions,
    show_attributions,
)
from .models import AttributionModel, list_supported_frameworks, load_model, register_model_config
from .utils.id_utils import explain


def get_version() -> str:
    """Returns the current version of the Inseq library."""
    try:
        import pkg_resources

        return pkg_resources.get_distribution("inseq").version
    except pkg_resources.DistributionNotFound:
        return "unknown"


__all__ = [
    "AttributionModel",
    "FeatureAttributionOutput",
    "load_model",
    "explain",
    "show_attributions",
    "list_feature_attribution_methods",
    "list_aggregators",
    "list_aggregation_functions",
    "list_step_functions",
    "list_supported_frameworks",
    "register_step_function",
    "register_model_config",
    "merge_attributions",
]
