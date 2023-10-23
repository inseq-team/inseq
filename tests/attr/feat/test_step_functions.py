from pytest import fixture

import inseq
from inseq.models import DecoderOnlyAttributionModel, EncoderDecoderAttributionModel


@fixture(scope="session")
def saliency_gpt2():
    return inseq.load_model("distilgpt2", "saliency")


@fixture(scope="session")
def saliency_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency")


def test_contrast_prob_consistency_decoder(saliency_gpt2: DecoderOnlyAttributionModel):
    out_contrast = saliency_gpt2.attribute(
        " the manager opened",
        " the manager opened her own restaurant.",
        attribute_target=True,
        step_scores=["contrast_prob"],
        contrast_targets="After returning to her hometown, the manager opened her own restaurant.",
    )
    contrast_prob = out_contrast.sequence_attributions[0].step_scores["contrast_prob"]
    out_regular = saliency_gpt2.attribute(
        "After returning to her hometown, the manager opened",
        "After returning to her hometown, the manager opened her own restaurant.",
        attribute_target=True,
        step_scores=["probability"],
    )
    regular_prob = out_regular.sequence_attributions[0].step_scores["probability"]
    assert all(c == r for c, r in zip(contrast_prob, regular_prob))


def test_contrast_prob_consistency_enc_dec(saliency_mt_model: EncoderDecoderAttributionModel):
    out_contrast = saliency_mt_model.attribute(
        "she started working as a cook in London.",
        "ha iniziato a lavorare come cuoca a Londra.",
        attribute_target=True,
        step_scores=["contrast_prob"],
        contrast_sources="After finishing her studies, she started working as a cook in London.",
        contrast_targets="Dopo aver terminato gli studi, ha iniziato a lavorare come cuoca a Londra.",
    )
    contrast_prob = out_contrast.sequence_attributions[0].step_scores["contrast_prob"]
    out_regular = saliency_mt_model.attribute(
        "After finishing her studies, she started working as a cook in London.",
        "Dopo aver terminato gli studi, ha iniziato a lavorare come cuoca a Londra.",
        attribute_target=True,
        step_scores=["probability"],
    )
    regular_prob = out_regular.sequence_attributions[0].step_scores["probability"]
    assert all(c == r for c, r in zip(contrast_prob, regular_prob[-len(contrast_prob) :]))
