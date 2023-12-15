import torch
from pytest import fixture

import inseq
from inseq.attr.step_functions import StepFunctionArgs, _get_contrast_inputs, probability_fn
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


def attr_prob_diff_fn(
    args: StepFunctionArgs,
    contrast_targets,
    contrast_targets_alignments=None,
    logprob: bool = False,
):
    model_probs = probability_fn(args, logprob=logprob)
    c_out = _get_contrast_inputs(
        args,
        contrast_targets=contrast_targets,
        contrast_targets_alignments=contrast_targets_alignments,
        return_contrastive_target_ids=True,
    )
    args.target_ids = c_out.target_ids
    contrast_probs = probability_fn(args, logprob=logprob)
    return model_probs - contrast_probs


def test_contrast_attribute_target_only_enc_dec(saliency_mt_model: EncoderDecoderAttributionModel):
    inseq.register_step_function(fn=attr_prob_diff_fn, identifier="attr_prob_diff", overwrite=True)
    src = "The nurse was tired and went home."
    tgt = "L'infermiere era stanco e andò a casa."
    contrast_tgt = "L'infermiera era stanca e andò a casa."
    out_explicit_logit_prob_diff = saliency_mt_model.attribute(
        src,
        tgt,
        contrast_targets=contrast_tgt,
        attributed_fn="attr_prob_diff",
        step_scores=["attr_prob_diff", "contrast_prob_diff"],
        attribute_target=True,
    )
    out_default_prob_diff = saliency_mt_model.attribute(
        src,
        tgt,
        contrast_targets=contrast_tgt,
        attributed_fn="contrast_prob_diff",
        step_scores=["contrast_prob_diff"],
        attribute_target=True,
    )
    assert torch.allclose(
        out_explicit_logit_prob_diff[0].step_scores["contrast_prob_diff"],
        out_default_prob_diff[0].step_scores["contrast_prob_diff"],
    )
    assert torch.allclose(
        out_explicit_logit_prob_diff[0].source_attributions,
        out_default_prob_diff[0].source_attributions,
    )
    assert torch.allclose(
        out_explicit_logit_prob_diff[0].target_attributions,
        out_default_prob_diff[0].target_attributions,
        equal_nan=True,
    )
    out_contrast_force_inputs_prob_diff = saliency_mt_model.attribute(
        src,
        tgt,
        contrast_targets=contrast_tgt,
        attributed_fn="contrast_prob_diff",
        step_scores=["contrast_prob_diff"],
        attribute_target=True,
        contrast_force_inputs=True,
    )
    assert not torch.allclose(
        out_explicit_logit_prob_diff[0].source_attributions,
        out_contrast_force_inputs_prob_diff[0].source_attributions,
    )
    assert not torch.allclose(
        out_explicit_logit_prob_diff[0].target_attributions,
        out_contrast_force_inputs_prob_diff[0].target_attributions,
        equal_nan=True,
    )
    assert torch.allclose(
        out_explicit_logit_prob_diff[0].step_scores["contrast_prob_diff"],
        out_default_prob_diff[0].step_scores["contrast_prob_diff"],
    )
