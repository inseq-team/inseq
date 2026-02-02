import warnings

import pytest

import inseq
from inseq.attr.feat.feature_attribution import DummyAttribution
from inseq.models import HuggingfaceDecoderOnlyModel
from inseq.models.model_config import (
    DEFAULT_DECODER_ONLY_CONFIG,
    DEFAULT_ENCODER_DECODER_CONFIG,
    MODEL_CONFIGS,
    ModelConfig,
    get_model_config,
)


class MockModelConfig(ModelConfig):
    def __init__(self):
        super().__init__(**dict.fromkeys(ModelConfig.__dataclass_fields__.keys(), "test"))


class MockRequireConfigAttribution(DummyAttribution):
    """Mock attribution requiring model config."""

    method_name = "mock_require_config"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model, hook_to_model=False)
        self.use_model_config = True
        self.hook(**kwargs)


def test_missing_model_config_warning():
    """Test that a warning is raised when model config is not found, and default config is used."""
    del MODEL_CONFIGS["GPT2LMHeadModel"]
    with pytest.warns(UserWarning, match="Model configuration for 'GPT2LMHeadModel' not found"):
        model = inseq.load_model("hf-internal-testing/tiny-random-GPT2LMHeadModel", "mock_require_config")
    # Check that default config was used
    assert model.config.self_attention_module == DEFAULT_DECODER_ONLY_CONFIG.self_attention_module
    assert model.config.value_vector == DEFAULT_DECODER_ONLY_CONFIG.value_vector
    # Restore config for other tests
    MODEL_CONFIGS["GPT2LMHeadModel"] = MockModelConfig()
    assert isinstance(model, HuggingfaceDecoderOnlyModel)
    assert isinstance(model.attribution_method, MockRequireConfigAttribution)


def test_default_decoder_only_config():
    """Test default decoder-only configuration values."""
    with pytest.warns(UserWarning, match="Using default decoder-only configuration"):
        config = get_model_config("UnknownDecoderModel", is_encoder_decoder=False)
    assert config.self_attention_module == "attn"
    assert config.value_vector == "value"
    assert config.cross_attention_module is None


def test_default_encoder_decoder_config():
    """Test default encoder-decoder configuration values."""
    with pytest.warns(UserWarning, match="Using default encoder-decoder configuration"):
        config = get_model_config("UnknownEncoderDecoderModel", is_encoder_decoder=True)
    assert config.self_attention_module == "self_attn"
    assert config.value_vector == "value"
    assert config.cross_attention_module == "cross_attention"


def test_known_model_no_warning():
    """Test that no warning is raised for known models."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = get_model_config("LlamaForCausalLM")
    assert config.self_attention_module == "self_attn"
    assert config.value_vector == "value_states"
