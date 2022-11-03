"""Interpretability for Sequence-to-sequence models 🔍"""

from .attr import list_feature_attribution_methods, list_step_scores, register_step_score
from .data import FeatureAttributionOutput, show_attributions
from .models import AttributionModel, list_supported_frameworks, load_model


def get_version() -> str:
    try:
        import pkg_resources

        return pkg_resources.get_distribution("inseq").version
    except pkg_resources.DistributionNotFound:
        return "unknown"


__all__ = [
    "AttributionModel",
    "FeatureAttributionOutput",
    "load_model",
    "show_attributions",
    "list_feature_attribution_methods",
    "list_step_scores",
    "list_supported_frameworks",
    "register_step_score",
]
