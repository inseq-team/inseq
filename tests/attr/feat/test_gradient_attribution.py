import pytest

import inseq
from inseq.data import FeatureAttributionSequenceOutput


@pytest.fixture
def saliency_mt_model():
    return inseq.load("Helsinki-NLP/opus-mt-en-it", "saliency")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("texts", "reference_texts"),
    [
        ("Hello world!", None),
        ("Hello world!", "Buongiorno mondo!"),
    ],
)
def test_gradient_attribution(texts, reference_texts, saliency_mt_model):
    attribution = saliency_mt_model.attribute(texts, reference_texts)
    assert isinstance(attribution, FeatureAttributionSequenceOutput)
