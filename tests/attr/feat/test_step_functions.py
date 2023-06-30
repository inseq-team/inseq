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
        contrast_target_prefixes="After returning to her hometown,",
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
        " started working as a cook in London.",
        " ha iniziato a lavorare come cuoca a Londra.",
        attribute_target=True,
        step_scores=["contrast_prob"],
        contrast_sources="After finishing her studies, she started working as a cook in London.",
        contrast_target_prefixes="Dopo aver terminato gli studi,",
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


def test_contrast_prob_diff_contrast_targets_auto_align_seq2seq(saliency_mt_model: EncoderDecoderAttributionModel):
    out = saliency_mt_model.attribute(
        (
            " UN peacekeepers, whom arrived in Haiti after the 2010 earthquake, are being blamed for the spread of the"
            " disease which started near the troop's encampment."
        ),
        (
            "I soldati della pace dell'ONU, che sono arrivati ad Haiti dopo il terremoto del 2010, sono stati"
            " incolpati per la diffusione della malattia che è iniziata vicino al campo delle truppe."
        ),
        attributed_fn="contrast_prob_diff",
        step_scores=["contrast_prob_diff"],
        contrast_targets=(
            "Le forze di pace delle Nazioni Unite, arrivate ad Haiti dopo il terremoto del 2010, sono state accusate"
            " di aver diffuso la malattia iniziata nei pressi dell'accampamento delle truppe."
        ),
        contrast_targets_alignments="auto",
    )
    contrast_targets = [
        "▁Le → ▁I",
        "▁forze → ▁soldati",
        "▁di → ▁della",
        "▁pace",
        "▁delle → ▁dell",
        "▁delle → '",
        "▁Nazioni → ONU",
        ",",
        "▁arriva → ▁che",
        "te → ▁sono",
        "▁arriva → ▁arrivati",
        "▁ad",
        "▁Haiti",
        "▁dopo",
        "▁il",
        "▁terremoto",
        "▁del",
        "▁2010,",
        "▁sono",
        "▁state → ▁stati",
        "▁accusa → ▁in",
        "te → col",
        "▁accusa → pati",
        "▁di → ▁per",
        "▁aver → ▁la",
        "▁diffuso → ▁diffusione",
        "▁la → ▁della",
        "▁malattia",
        "▁iniziata → ▁che",
        "▁dell → ▁è",
        "▁iniziata",
        "▁pressi → ▁vicino",
        "▁nei → ▁al",
        "acca → ▁campo",
        "▁delle",
        "▁truppe",
        ".",
        "</s>",
    ]
    assert [t.token for t in out[0].target] == contrast_targets


def test_contrast_prob_diff_contrast_targets_auto_align_gpt(saliency_gpt2: DecoderOnlyAttributionModel):
    out = saliency_gpt2.attribute(
        "",
        "UN peacekeepers were deployed in the region.",
        attributed_fn="contrast_prob_diff",
        contrast_targets="<|endoftext|> UN peacekeepers were sent to the war-torn region.",
        contrast_targets_alignments="auto",
        step_scores=["contrast_prob_diff"],
    )
    contrast_targets = [
        "<|endoftext|>",
        "ĠUN",
        "Ġpeace",
        "keepers",
        "Ġwere",
        "Ġsent → Ġdeployed",
        "Ġto → Ġin",
        "Ġthe",
        "Ġregion",
        ".",
    ]
    assert [t.token for t in out[0].target] == contrast_targets
