import pytest

import inseq
from inseq.data import FeatureAttributionSequenceOutput


@pytest.fixture(scope="session")
def ig_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "integrated_gradients")


@pytest.mark.slow
@pytest.mark.require_gpu
@pytest.mark.parametrize(
    ("texts", "reference_texts"),
    [
        ("Hello world!", None),
        ("Hello world!", "Buongiorno mondo!"),
    ],
)
def test_cuda_attribution_consistency(texts, reference_texts, ig_mt_model):
    cpu_out = ig_mt_model.attribute(texts, reference_texts, show_progress=False, device="cpu")
    gpu_out = ig_mt_model.attribute(texts, reference_texts, show_progress=False, device="cuda:0")
    assert isinstance(cpu_out, FeatureAttributionSequenceOutput)
    assert isinstance(gpu_out, FeatureAttributionSequenceOutput)
    assert all([tok_cpu == tok_gpu for tok_cpu, tok_gpu in zip(cpu_out.target_tokens, gpu_out.target_tokens)])
    attr_score_matches = [
        abs(el_cpu - el_gpu) < 5e-3
        for cpu_attr, gpu_attr in zip(cpu_out.source_attributions, gpu_out.source_attributions)
        for el_cpu, el_gpu in zip(cpu_attr, gpu_attr)
    ]
    assert all(attr_score_matches)
