from ..attr import STEP_SCORES_MAP, FeatureAttribution
from ..data import AggregationFunction, Aggregator

INSEQ_ID_MAP = {
    **FeatureAttribution.available_classes(),
    **Aggregator.available_classes(),
    **AggregationFunction.available_classes(),
    **STEP_SCORES_MAP,
}


def explain(id: str) -> None:
    """Given an identifier, prints a short explanation of what it represents in the Inseq library. Identifiers are
    used for attribution methods, aggregators, aggregation functions, and step functions.

    Example: `explain("attention")` prints the documentation for the attention attribution method.
    """
    if id not in INSEQ_ID_MAP:
        raise ValueError(f"Unknown identifier: {id}")
    doc = INSEQ_ID_MAP[id].__doc__
    if doc is None:
        print(
            "No documentation is available for this identifier. Please refer to the Inseq documentation for more "
            "information, or raise an issue on https://github.com/inseq-team/inseq/issues to request documentation "
            "for this identifier."
        )
    print(doc)
