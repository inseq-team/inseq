import pytest

import inseq
from inseq.attr.feat.perturbation_attribution import ValueZeroingAttribution
from inseq.models import HuggingfaceDecoderOnlyModel
from inseq.models.model_config import MODEL_CONFIGS, ModelConfig


class MockModelConfig(ModelConfig):
    def __init__(self):
        super().__init__(**{field: "test" for field in ModelConfig.__dataclass_fields__.keys()})


def test_missing_model_config_error():
    del MODEL_CONFIGS["GPT2LMHeadModel"]
    with pytest.raises(ValueError):
        inseq.load_model("hf-internal-testing/tiny-random-GPT2LMHeadModel", "value_zeroing")
    MODEL_CONFIGS["GPT2LMHeadModel"] = MockModelConfig()
    model = inseq.load_model("hf-internal-testing/tiny-random-GPT2LMHeadModel", "value_zeroing")
    assert isinstance(model, HuggingfaceDecoderOnlyModel)
    assert isinstance(model.attribution_method, ValueZeroingAttribution)
