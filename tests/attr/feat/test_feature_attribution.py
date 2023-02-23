import torch
from pytest import fixture
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

import inseq
from inseq.models import HuggingfaceDecoderOnlyModel, HuggingfaceEncoderDecoderModel


@fixture(scope="session")
def saliency_mt_model() -> HuggingfaceEncoderDecoderModel:
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency")


@fixture(scope="session")
def saliency_gpt_model() -> HuggingfaceDecoderOnlyModel:
    return inseq.load_model("gpt2", "saliency")


@fixture(scope="session")
def auxiliary_saliency_mt_model():
    return AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")


@fixture(scope="session")
def auxiliary_saliency_gpt_model():
    return AutoModelForCausalLM.from_pretrained("gpt2")


def test_contrastive_attribution_seq2seq(saliency_mt_model: HuggingfaceEncoderDecoderModel):
    """Runs a contrastive feature attribution using the method relying on logits difference
    introduced by [Yin and Neubig '22](https://arxiv.org/pdf/2202.10419.pdf), taking advantage of
    the custom feature attribution target function module.
    """
    # Pre-compute ids and attention map for the contrastive target
    contrast = saliency_mt_model.encode("Non posso crederlo.", as_targets=True)

    # Perform the contrastive attribution:
    # Regular (forced) target -> "Non posso crederci."
    # Contrastive target      -> "Non posso crederlo."
    # contrast_ids & contrast_attention_mask are kwargs defined in the function definition
    out = saliency_mt_model.attribute(
        "I can't believe it",
        "Non posso crederci.",
        attributed_fn="contrast_prob_diff",
        contrast_ids=contrast.input_ids,
        contrast_attention_mask=contrast.attention_mask,
        show_progress=False,
    )
    attribution_scores = out.sequence_attributions[0].source_attributions

    # Since the two target strings are identical for the first three tokens (Non posso creder)
    # the resulting contrastive source attributions should be all 0
    assert attribution_scores[:, :3].sum().eq(0)

    # Starting at the following token in which they differ, scores should diverge
    assert not attribution_scores[:, :4].sum().eq(0)


def test_contrastive_attribution_gpt(saliency_gpt_model: HuggingfaceDecoderOnlyModel):
    contrast = saliency_gpt_model.encode("The female student didn't participate because he was sick.")
    out = saliency_gpt_model.attribute(
        "The female student didn't participate because",
        "The female student didn't participate because she was sick.",
        attributed_fn="contrast_prob_diff",
        contrast_ids=contrast.input_ids,
        contrast_attention_mask=contrast.attention_mask,
        show_progress=False,
    )
    attribution_scores = out.sequence_attributions[0].target_attributions
    assert attribution_scores.shape == torch.Size([11, 4, 768])


def test_mcd_weighted_attribution_seq2seq(saliency_mt_model, auxiliary_saliency_mt_model):
    """Runs a MCD-weighted feature attribution taking advantage of
    the custom feature attribution target function module.
    """
    out = saliency_mt_model.attribute(
        "Hello ladies and badgers!",
        attributed_fn="mc_dropout_prob_avg",
        attributed_fn_args={"n_mcd_steps": 5, "aux_model": auxiliary_saliency_mt_model.to(saliency_mt_model.device)},
        show_progress=False,
    )
    attribution_scores = out.sequence_attributions[0].source_attributions
    assert isinstance(attribution_scores, torch.Tensor)


def test_mcd_weighted_attribution_gpt(saliency_gpt_model, auxiliary_saliency_gpt_model):
    """Runs a MCD-weighted feature attribution taking advantage of
    the custom feature attribution target function module.
    """
    out = saliency_gpt_model.attribute(
        "Hello ladies and badgers!",
        attributed_fn="mc_dropout_prob_avg",
        attributed_fn_args={"n_mcd_steps": 5, "aux_model": auxiliary_saliency_gpt_model.to(saliency_gpt_model.device)},
        generation_args={"max_new_tokens": 5},
        show_progress=False,
    )
    attribution_scores = out.sequence_attributions[0].target_attributions
    assert isinstance(attribution_scores, torch.Tensor)
