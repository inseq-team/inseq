import pytest
from pytest import fixture

from inseq import AttributionModel, FeatureAttributionOutput


@fixture(scope="session")
def saliency_mt_model():
    return AttributionModel.load("Helsinki-NLP/opus-mt-en-it", "saliency", device="cpu")


@pytest.mark.skip("TODO fix this test")
def test_save_load_attribution(tmp_path, saliency_mt_model):
    out_path = tmp_path / "tmp_attr.json"
    out = saliency_mt_model.attribute("This is a test.", device="cpu")
    out.save(out_path)
    loaded_out = FeatureAttributionOutput.load(out_path)
    assert out == loaded_out
