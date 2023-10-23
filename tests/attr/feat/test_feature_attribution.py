import torch
from pytest import fixture

import inseq
from inseq.models import HuggingfaceDecoderOnlyModel, HuggingfaceEncoderDecoderModel


@fixture(scope="session")
def saliency_mt_model_larger() -> HuggingfaceEncoderDecoderModel:
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency")


@fixture(scope="session")
def saliency_gpt_model_larger() -> HuggingfaceDecoderOnlyModel:
    return inseq.load_model("gpt2", "saliency")


@fixture(scope="session")
def saliency_mt_model() -> HuggingfaceEncoderDecoderModel:
    return inseq.load_model("hf-internal-testing/tiny-random-MarianMTModel", "saliency")


@fixture(scope="session")
def saliency_gpt_model() -> HuggingfaceDecoderOnlyModel:
    return inseq.load_model("hf-internal-testing/tiny-random-GPT2LMHeadModel", "saliency")


def test_contrastive_attribution_seq2seq(saliency_mt_model_larger: HuggingfaceEncoderDecoderModel):
    """Runs a contrastive feature attribution using the method relying on logits difference
    introduced by [Yin and Neubig '22](https://arxiv.org/pdf/2202.10419.pdf), taking advantage of
    the custom feature attribution target function module.
    """
    # Perform the contrastive attribution:
    # Regular (forced) target -> "Non posso crederci."
    # Contrastive target      -> "Non posso crederlo."
    # contrast_ids & contrast_attention_mask are kwargs defined in the function definition
    out = saliency_mt_model_larger.attribute(
        "I can't believe it",
        "Non posso crederci.",
        attributed_fn="contrast_prob_diff",
        contrast_targets="Non posso crederlo.",
        show_progress=False,
    )
    attribution_scores = out.sequence_attributions[0].source_attributions

    # Since the two target strings are identical for the first three tokens (Non posso creder)
    # the resulting contrastive source attributions should be all 0
    assert attribution_scores[:, :3].sum().eq(0)

    # Starting at the following token in which they differ, scores should diverge
    assert not attribution_scores[:, :4].sum().eq(0)


def test_contrastive_attribution_gpt(saliency_gpt_model: HuggingfaceDecoderOnlyModel):
    out = saliency_gpt_model.attribute(
        "The female student didn't participate because",
        "The female student didn't participate because she was sick.",
        attributed_fn="contrast_prob_diff",
        contrast_targets="The female student didn't participate because he was sick.",
        show_progress=False,
    )
    attribution_scores = out.sequence_attributions[0].target_attributions
    assert attribution_scores.shape == torch.Size([23, 5, 32])


def test_contrastive_attribution_seq2seq_alignments(saliency_mt_model_larger: HuggingfaceEncoderDecoderModel):
    aligned = {
        "src": "UN peacekeepers",
        "orig_tgt": "I soldati della pace ONU",
        "contrast_tgt": "Le forze militari di pace delle Nazioni Unite",
        "alignments": [[(0, 0), (1, 1), (2, 2), (3, 4), (4, 5), (5, 7), (6, 9)]],
        "aligned_tgts": ["▁Le → ▁I", "▁forze → ▁soldati", "▁di → ▁della", "▁pace", "▁Nazioni → ▁ONU", "</s>"],
    }
    out = saliency_mt_model_larger.attribute(
        aligned["src"],
        aligned["orig_tgt"],
        attributed_fn="contrast_prob_diff",
        step_scores=["contrast_prob_diff"],
        contrast_targets=aligned["contrast_tgt"],
        contrast_targets_alignments=aligned["alignments"],
        show_progress=False,
    )
    # Check tokens are aligned as expected
    assert [t.token for t in out[0].target] == aligned["aligned_tgts"]

    # Check that a single list of alignments is correctly processed
    out_single_list = saliency_mt_model_larger.attribute(
        aligned["src"],
        aligned["orig_tgt"],
        attributed_fn="contrast_prob_diff",
        step_scores=["contrast_prob_diff"],
        contrast_targets=aligned["contrast_tgt"],
        contrast_targets_alignments=aligned["alignments"][0],
        attribute_target=True,
        show_progress=False,
    )
    assert out[0].target == out_single_list[0].target
    assert torch.allclose(
        out[0].source_attributions,
        out_single_list[0].source_attributions,
        atol=8e-2,
    )


def test_mcd_weighted_attribution_seq2seq(saliency_mt_model):
    """Runs a MCD-weighted feature attribution taking advantage of
    the custom feature attribution target function module.
    """
    out = saliency_mt_model.attribute(
        "Hello ladies and badgers!",
        attributed_fn="mc_dropout_prob_avg",
        n_mcd_steps=3,
        show_progress=False,
    )
    attribution_scores = out.sequence_attributions[0].source_attributions
    assert isinstance(attribution_scores, torch.Tensor)


def test_mcd_weighted_attribution_gpt(saliency_gpt_model):
    """Runs a MCD-weighted feature attribution taking advantage of
    the custom feature attribution target function module.
    """
    out = saliency_gpt_model.attribute(
        "Hello ladies and badgers!",
        attributed_fn="mc_dropout_prob_avg",
        n_mcd_steps=3,
        generation_args={"max_new_tokens": 3},
        show_progress=False,
    )
    attribution_scores = out.sequence_attributions[0].target_attributions
    assert isinstance(attribution_scores, torch.Tensor)
