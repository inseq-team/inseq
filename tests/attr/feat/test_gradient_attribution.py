import pytest

import inseq
from inseq.data import FeatureAttributionSequenceOutput


@pytest.mark.parametrize(
    ("texts", "reference_texts"),
    [
        ("Hello world, today is a good day!", None),
        ("Hello world, today is a good day!", "Ciao mondo, oggi è una bella giornata!"),
    ],
)
def test_gradient_attribution(texts, reference_texts):
    model = inseq.load("Helsinki-NLP/opus-mt-en-it", "integrated_gradients")
    attribution = model.attribute(texts, reference_texts, n_steps=5)
    assert isinstance(attribution, FeatureAttributionSequenceOutput)
