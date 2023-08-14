import pytest

import inseq
from inseq.attr.feat.feature_attribution import DummyAttribution
from inseq.models import HuggingfaceDecoderOnlyModel
from inseq.models.model_config import MODEL_CONFIGS, ModelConfig


class MockModelConfig(ModelConfig):
    def __init__(self):
        super().__init__(**{field: "test" for field in ModelConfig.__dataclass_fields__.keys()})


class MockRequireConfigAttribution(DummyAttribution):
    """Mock attribution requiring model config."""

    method_name = "mock_require_config"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model, hook_to_model=False)
        self.use_model_config = True
        self.hook(**kwargs)


def test_missing_model_config_error():
    del MODEL_CONFIGS["GPT2LMHeadModel"]
    with pytest.raises(ValueError):
        inseq.load_model("hf-internal-testing/tiny-random-GPT2LMHeadModel", "mock_require_config")
    MODEL_CONFIGS["GPT2LMHeadModel"] = MockModelConfig()
    model = inseq.load_model("hf-internal-testing/tiny-random-GPT2LMHeadModel", "mock_require_config")
    assert isinstance(model, HuggingfaceDecoderOnlyModel)
    assert isinstance(model.attribution_method, MockRequireConfigAttribution)
