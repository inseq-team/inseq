from typing import Any, Tuple

from .registry import get_available_methods


class InseqDeprecationWarning(UserWarning):
    """Special deprecation warning because the built-in one is ignored by default"""

    def __init__(self, msg):
        super().__init__(msg)


class UnknownAttributionMethodError(Exception):
    """Raised when an attribution method is not valid"""

    UNKNOWN_ATTRIBUTION_METHOD_MSG = (
        "Unknown attribution method: {attribution_method}.\nAvailable methods: {available_methods}"
    )

    def __init__(
        self,
        method_name: str,
        msg: str = UNKNOWN_ATTRIBUTION_METHOD_MSG,
        *args: Tuple[Any],
    ) -> None:
        from inseq.attr import FeatureAttribution

        msg = msg.format(
            attribution_method=method_name,
            available_methods=", ".join(get_available_methods(FeatureAttribution)),
        )
        super().__init__(msg, *args)


class MissingAttributionMethodError(Exception):
    """Raised when an attribution method is not found"""

    MISSING_ATTRIBUTION_METHOD_MSG = (
        "Attribution methods is not set. "
        "You can either define it permanently when instancing the AttributionModel, "
        "or pass it to the attribute method.\nAvailable methods: {available_methods}"
    )

    def __init__(self, msg: str = MISSING_ATTRIBUTION_METHOD_MSG, *args: Tuple[Any]) -> None:
        from inseq.attr import FeatureAttribution

        msg = msg.format(available_methods=", ".join(get_available_methods(FeatureAttribution)))
        super().__init__(msg, *args)


class LengthMismatchError(Exception):
    """Raised when lengths do not match"""

    pass
