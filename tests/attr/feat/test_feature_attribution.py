from typing import Any, Optional

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from pytest import fixture

import inseq
from inseq.attr.feat.internals_attribution import InternalsAttributionRegistry
from inseq.data import MultiDimensionalFeatureAttributionStepOutput
from inseq.models import HuggingfaceDecoderOnlyModel, HuggingfaceEncoderDecoderModel
from inseq.utils.typing import InseqAttribution, MultiLayerMultiUnitScoreTensor


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
        "aligned_tgts": ["<pad>", "▁Le → ▁I", "▁forze → ▁soldati", "▁di → ▁della", "▁pace", "▁Nazioni → ▁ONU", "</s>"],
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


class MultiStepAttentionWeights(InseqAttribution):
    """Variant of the AttentionWeights class with is_final_step_method = False.
    As a result, the attention matrix is computed and sliced at every generation step.
    We define it here to test consistency with the final step method.
    """

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: TensorOrTupleOfTensorsGeneric,
        encoder_self_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
        decoder_self_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
        cross_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
    ) -> MultiDimensionalFeatureAttributionStepOutput:
        # We adopt the format [batch_size, sequence_length, num_layers, num_heads]
        # for consistency with other multi-unit methods (e.g. gradient attribution)
        decoder_self_attentions = decoder_self_attentions[..., -1, :].to("cpu").clone().permute(0, 3, 1, 2)
        if self.forward_func.is_encoder_decoder:
            sequence_scores = {}
            if len(inputs) > 1:
                target_attributions = decoder_self_attentions
            else:
                target_attributions = None
                sequence_scores["decoder_self_attentions"] = decoder_self_attentions
            sequence_scores["encoder_self_attentions"] = (
                encoder_self_attentions.to("cpu").clone().permute(0, 4, 3, 1, 2)
            )
            return MultiDimensionalFeatureAttributionStepOutput(
                source_attributions=cross_attentions[..., -1, :].to("cpu").clone().permute(0, 3, 1, 2),
                target_attributions=target_attributions,
                sequence_scores=sequence_scores,
                _num_dimensions=2,  # num_layers, num_heads
            )
        else:
            return MultiDimensionalFeatureAttributionStepOutput(
                source_attributions=None,
                target_attributions=decoder_self_attentions,
                _num_dimensions=2,  # num_layers, num_heads
            )


class MultiStepAttentionWeightsAttribution(InternalsAttributionRegistry):
    """Variant of the basic attention attribution method computing attention weights at every generation step."""

    method_name = "per_step_attention"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        # Attention weights will be passed to the attribute_step method
        self.use_attention_weights = True
        # Does not rely on predicted output (i.e. decoding strategy agnostic)
        self.use_predicted_target = False
        self.method = MultiStepAttentionWeights(attribution_model)

    def attribute_step(
        self,
        attribute_fn_main_args: dict[str, Any],
        attribution_args: dict[str, Any],
    ) -> MultiDimensionalFeatureAttributionStepOutput:
        return self.method.attribute(**attribute_fn_main_args, **attribution_args)


def test_seq2seq_final_step_per_step_conformity(saliency_mt_model_larger: HuggingfaceEncoderDecoderModel):
    out_per_step = saliency_mt_model_larger.attribute(
        "Hello ladies and badgers!",
        method="per_step_attention",
        attribute_target=True,
        show_progress=False,
        output_step_attributions=True,
    )
    out_final_step = saliency_mt_model_larger.attribute(
        "Hello ladies and badgers!",
        method="attention",
        attribute_target=True,
        show_progress=False,
        output_step_attributions=True,
    )
    assert out_per_step[0] == out_final_step[0]


def test_gpt_final_step_per_step_conformity(saliency_gpt_model: HuggingfaceDecoderOnlyModel):
    out_per_step = saliency_gpt_model.attribute(
        "Hello ladies and badgers!",
        method="per_step_attention",
        show_progress=False,
        output_step_attributions=True,
    )
    out_final_step = saliency_gpt_model.attribute(
        "Hello ladies and badgers!",
        method="attention",
        show_progress=False,
        output_step_attributions=True,
    )
    assert out_per_step[0] == out_final_step[0]


# Batching for Seq2Seq models is not supported when using is_final_step methods
# Passing several sentences will attributed them one by one under the hood
# def test_seq2seq_multi_step_attention_weights_batched_full_match(saliency_mt_model: HuggingfaceEncoderDecoderModel):


def test_gpt_multi_step_attention_weights_batched_full_match(saliency_gpt_model_larger: HuggingfaceDecoderOnlyModel):
    out_per_step = saliency_gpt_model_larger.attribute(
        ["Hello world!", "Colorless green ideas sleep furiously."],
        method="per_step_attention",
        show_progress=False,
    )
    out_final_step = saliency_gpt_model_larger.attribute(
        ["Hello world!", "Colorless green ideas sleep furiously."],
        method="attention",
        show_progress=False,
    )
    for i in range(2):
        assert out_per_step[i].target_attributions.shape == out_final_step[i].target_attributions.shape
        assert torch.allclose(
            out_per_step[i].target_attributions, out_final_step[i].target_attributions, equal_nan=True, atol=1e-5
        )
