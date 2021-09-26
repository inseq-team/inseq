from .registry import get_available_methods


class UnknownAttributionMethodError(Exception):
    """Raised when an attribution method is not valid"""

    UNKNOWN_ATTRIBUTION_METHOD_MSG = (
        "Unknown attribution method: {attribution_method}.\n"
        "Available methods: {available_methods}"
    )

    def __init__(self, method_name: str, msg=UNKNOWN_ATTRIBUTION_METHOD_MSG, *args):
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

    def __init__(self, msg=MISSING_ATTRIBUTION_METHOD_MSG, *args):
        from inseq.attr import FeatureAttribution

        msg = msg.format(
            available_methods=", ".join(get_available_methods(FeatureAttribution))
        )
        super().__init__(msg, *args)


class LengthMismatchError(Exception):
    """Raised when lengths do not match"""

    pass
