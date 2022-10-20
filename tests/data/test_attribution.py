from pytest import fixture

from inseq import FeatureAttributionOutput, load_model


@fixture(scope="session")
def saliency_mt_model():
    return load_model("Helsinki-NLP/opus-mt-en-it", "saliency", device="cpu")


def test_save_load_attribution(tmp_path, saliency_mt_model):
    out_path = tmp_path / "tmp_attr.json"
    out = saliency_mt_model.attribute("This is a test.", device="cpu")
    out.save(out_path)
    loaded_out = FeatureAttributionOutput.load(out_path)
    assert out == loaded_out
