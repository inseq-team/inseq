from importlib.util import find_spec

_ipywidgets_available = find_spec("ipywidgets") is not None
_scikitlearn_available = find_spec("sklearn") is not None
_transformers_available = find_spec("transformers") is not None
_sentencepiece_available = find_spec("sentencepiece") is not None and find_spec("protobuf") is not None
_datasets_available = find_spec("datasets") is not None
_captum_available = find_spec("captum") is not None
_joblib_available = find_spec("joblib") is not None


def is_ipywidgets_available():
    return _ipywidgets_available


def is_scikitlearn_available():
    return _scikitlearn_available


def is_transformers_available():
    return _transformers_available


def is_sentencepiece_available():
    return _sentencepiece_available


def is_datasets_available():
    return _datasets_available


def is_captum_available():
    return _captum_available


def is_joblib_available():
    return _joblib_available
