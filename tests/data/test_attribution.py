import torch
from pytest import fixture

from inseq import FeatureAttributionOutput, load_model


@fixture(scope="session")
def saliency_mt_model():
    return load_model("Helsinki-NLP/opus-mt-en-it", "saliency", device="cpu")


def test_save_load_attribution(tmp_path, saliency_mt_model):
    out_path = tmp_path / "tmp_attr.json"
    out = saliency_mt_model.attribute("This is a test.", device="cpu", show_progress=False)
    out.save(out_path)
    loaded_out = FeatureAttributionOutput.load(out_path)
    assert out == loaded_out


def test_save_load_attribution_split(tmp_path, saliency_mt_model):
    out_path = tmp_path / "tmp_attr.json"
    out = saliency_mt_model.attribute(["This is a test.", "sequence number two"], device="cpu", show_progress=False)
    out.save(out_path, split_sequences=True)
    out_path_1 = tmp_path / "tmp_attr_1.json"
    loaded_out = FeatureAttributionOutput.load(out_path_1)
    assert torch.allclose(
        out.sequence_attributions[1].source_attributions, loaded_out.sequence_attributions[0].source_attributions
    )


def test_save_load_attribution_compressed(tmp_path, saliency_mt_model):
    out_path = tmp_path / "tmp_attr_compress.json.gz"
    out = saliency_mt_model.attribute("This is a test.", device="cpu", show_progress=False)
    out.save(out_path, compress=True)
    loaded_out = FeatureAttributionOutput.load(out_path, decompress=True)
    assert out == loaded_out
