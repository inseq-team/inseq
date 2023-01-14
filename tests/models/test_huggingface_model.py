"""
TODO: Skipping layer attribution when attributing target and the DIG method
since it is bugged is not very elegant, this will need to be refactored.
"""

import json
import os

import pytest
import torch
from pytest import fixture, mark

import inseq
from inseq import list_feature_attribution_methods
from inseq.data import FeatureAttributionOutput, FeatureAttributionSequenceOutput
from inseq.utils import get_default_device


EXAMPLES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../fixtures/huggingface_model.json")
EXAMPLES = json.load(open(EXAMPLES_FILE))

USE_REFERENCE_TEXT = [True, False]
ATTRIBUTE_TARGET = [True, False]
RETURN_CONVERGENCE_DELTA = [True, False]
STEP_SCORES = [[], ["probability"]]
ATTRIBUTION_METHODS = list_feature_attribution_methods()

ATTENTION_IDX = [-2, [0, 5, 1], (1, -2), None]
ATTENTION_AGGREGATE_FN = ["average", None]


@fixture(scope="session")
def saliency_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency")


@fixture(scope="session")
def saliency_gpt2_model():
    return inseq.load_model("gpt2", "saliency")


@mark.slow
@mark.require_cuda_gpu
@mark.parametrize(("texts", "reference_texts"), EXAMPLES["short_texts"])
@mark.parametrize("attribute_target", ATTRIBUTE_TARGET)
def test_cuda_attribution_consistency_seq2seq(texts, reference_texts, attribute_target, saliency_mt_model):
    out = {}
    for device in ["cpu", "cuda"]:
        out[device] = saliency_mt_model.attribute(
            texts, reference_texts, show_progress=False, attribute_target=attribute_target, device=device
        )
        assert isinstance(out[device], FeatureAttributionOutput)
        assert isinstance(out[device].sequence_attributions[0], FeatureAttributionSequenceOutput)
    for out_cpu, out_gpu in zip(out["cpu"].sequence_attributions, out["cuda"].sequence_attributions):
        assert all([tok_cpu == tok_gpu for tok_cpu, tok_gpu in zip(out_cpu.target, out_gpu.target)])
        attr_score_matches = [
            torch.allclose(cpu_attr, gpu_attr, atol=1e-3)
            for cpu_attr, gpu_attr in zip(out_cpu.source_attributions, out_gpu.source_attributions)
        ]
        assert all(attr_score_matches)


@mark.slow
@mark.parametrize("attribution_method", ATTRIBUTION_METHODS)
@mark.parametrize("use_reference", USE_REFERENCE_TEXT)
@mark.parametrize("attribute_target", ATTRIBUTE_TARGET)
def test_batched_attribution_consistency_seq2seq(
    attribution_method, use_reference, attribute_target, saliency_mt_model
):
    if attribution_method == "discretized_integrated_gradients":
        pytest.skip("discretized_integrated_gradients currently unsupported")
    if attribution_method.startswith("layer_") and attribute_target:
        pytest.skip("Layer attribution methods do not support attribute_target=True")
    texts_single, reference_single = EXAMPLES["texts"][0]
    texts_batch, reference_batch = EXAMPLES["texts"][1]
    if not use_reference:
        reference_single, reference_batch = None, None
    out_single = saliency_mt_model.attribute(
        texts_single,
        reference_single,
        show_progress=False,
        attribute_target=attribute_target,
        device=get_default_device(),
        method=attribution_method,
    )
    out_batch = saliency_mt_model.attribute(
        texts_batch,
        reference_batch,
        show_progress=False,
        attribute_target=attribute_target,
        device=get_default_device(),
        method=attribution_method,
    )
    assert torch.allclose(
        out_single.sequence_attributions[0].source_attributions,
        out_batch.sequence_attributions[0].source_attributions,
        atol=8e-2,
    )
    if attribute_target:
        assert torch.allclose(
            out_single.sequence_attributions[0].target_attributions,
            out_batch.sequence_attributions[0].target_attributions,
            atol=8e-2,
            equal_nan=True,
        )


@mark.slow
@mark.parametrize(("texts", "reference_texts"), EXAMPLES["texts"])
@mark.parametrize("attribution_method", ATTRIBUTION_METHODS)
@mark.parametrize("use_reference", USE_REFERENCE_TEXT)
@mark.parametrize("attribute_target", ATTRIBUTE_TARGET)
@mark.parametrize("return_convergence_delta", RETURN_CONVERGENCE_DELTA)
@mark.parametrize("step_scores", STEP_SCORES)
def test_attribute_seq2seq(
    texts,
    reference_texts,
    attribution_method,
    use_reference,
    attribute_target,
    return_convergence_delta,
    step_scores,
    saliency_mt_model,
):
    if attribution_method == "discretized_integrated_gradients":
        pytest.skip("discretized_integrated_gradients currently unsupported")
    if attribution_method.startswith("layer_") and attribute_target:
        pytest.skip("Layer attribution methods do not support attribute_target=True")
    if not use_reference:
        reference_texts = None
    out = saliency_mt_model.attribute(
        texts,
        reference_texts,
        method=attribution_method,
        show_progress=False,
        attribute_target=attribute_target,
        return_convergence_delta=return_convergence_delta,
        step_scores=step_scores,
        internal_batch_size=50,
        n_steps=100,
        device=get_default_device(),
    )
    assert isinstance(out, FeatureAttributionOutput)
    assert isinstance(out.sequence_attributions[0], FeatureAttributionSequenceOutput)
    assert out.info["model_name"] == "Helsinki-NLP/opus-mt-en-it"
    assert out.info["constrained_decoding"] == use_reference
    assert out.info["attribution_method"] == attribution_method
    assert out.info["attribute_target"] == attribute_target
    assert out.info["step_scores"] == step_scores
    if "return_convergence_delta" in out.info:
        assert out.info["return_convergence_delta"] == return_convergence_delta
    if "internal_batch_size" in out.info:
        assert out.info["internal_batch_size"] == 50
    if "n_steps" in out.info:
        assert out.info["n_steps"] == 100


@mark.slow
@mark.parametrize(("texts", "reference_texts"), EXAMPLES["long_text"])
@mark.parametrize("attribution_method", ATTRIBUTION_METHODS)
@mark.parametrize("use_reference", USE_REFERENCE_TEXT)
def test_attribute_long_text_seq2seq(texts, reference_texts, attribution_method, use_reference, saliency_mt_model):
    if attribution_method == "discretized_integrated_gradients":
        pytest.skip("discretized_integrated_gradients currently unsupported")
    if not use_reference:
        reference_texts = None
    out = saliency_mt_model.attribute(
        texts,
        reference_texts,
        method=attribution_method,
        show_progress=False,
        internal_batch_size=10,
        n_steps=100,
        device=get_default_device(),
    )
    assert isinstance(out, FeatureAttributionOutput)
    assert isinstance(out.sequence_attributions[0], FeatureAttributionSequenceOutput)


def test_attribute_slice_seq2seq(saliency_mt_model):
    texts, reference_texts = EXAMPLES["texts"][2][0], EXAMPLES["texts"][2][1]
    out = saliency_mt_model.attribute(
        texts,
        reference_texts,
        show_progress=False,
        device=get_default_device(),
        attr_pos_start=13,
        attr_pos_end=17,
        attribute_target=True,
    )
    assert isinstance(out, FeatureAttributionOutput)
    assert len(out.sequence_attributions) == 3
    assert isinstance(out.sequence_attributions[0], FeatureAttributionSequenceOutput)
    ex1, ex2, ex3 = out.sequence_attributions[0], out.sequence_attributions[1], out.sequence_attributions[2]
    assert ex1.attr_pos_start == 13
    assert ex1.attr_pos_end == 17
    assert ex1.source_attributions.shape[1] == ex1.attr_pos_end - ex1.attr_pos_start
    assert ex1.target_attributions.shape[1] == ex1.attr_pos_end - ex1.attr_pos_start
    assert ex1.target_attributions.shape[0] == ex1.attr_pos_end
    # Empty attributions outputs have start and end set to seq length
    assert ex2.attr_pos_start == len(ex2.target)
    assert ex2.attr_pos_end == len(ex2.target)
    assert ex2.source_attributions.shape[1] == 0 and ex2.target_attributions.shape[1] == 0
    assert ex3.attr_pos_start == 13
    assert ex3.attr_pos_end == 15
    assert ex1.source_attributions.shape[1] == ex1.attr_pos_end - ex1.attr_pos_start
    assert ex1.target_attributions.shape[1] == ex1.attr_pos_end - ex1.attr_pos_start
    assert ex1.target_attributions.shape[0] == ex1.attr_pos_end
    assert out.info["attr_pos_start"] == 13
    assert out.info["attr_pos_end"] == 17
    aggregated = [attr.aggregate(attr._aggregator) for attr in out.sequence_attributions]
    assert all(isinstance(aggr_attr, FeatureAttributionSequenceOutput) for aggr_attr in aggregated)


def test_attribute_decoder(saliency_gpt2_model):
    texts = EXAMPLES["texts"][2][0]
    out = saliency_gpt2_model.attribute(
        texts,
        show_progress=False,
        device=get_default_device(),
        generation_args={"max_new_tokens": 10},
    )
    assert isinstance(out, FeatureAttributionOutput)
    assert len(out.sequence_attributions) == 3
    assert isinstance(out.sequence_attributions[0], FeatureAttributionSequenceOutput)
    ex1, ex2, ex3 = out.sequence_attributions[0], out.sequence_attributions[1], out.sequence_attributions[2]
    assert ex1.attr_pos_start == 17
    assert ex1.attr_pos_end == 27
    assert ex1.target_attributions.shape[1] == ex1.attr_pos_end - ex1.attr_pos_start
    assert ex1.target_attributions.shape[0] == ex1.attr_pos_end
    # Empty attributions outputs have start and end set to seq length
    assert ex2.attr_pos_start == 6
    assert ex2.attr_pos_end == 16
    assert ex2.target_attributions.shape[1] == ex2.attr_pos_end - ex2.attr_pos_start
    assert ex2.target_attributions.shape[0] == ex2.attr_pos_end
    assert ex3.attr_pos_start == 12
    assert ex3.attr_pos_end == 22
    assert ex3.target_attributions.shape[1] == ex3.attr_pos_end - ex3.attr_pos_start
    assert ex3.target_attributions.shape[0] == ex3.attr_pos_end
    assert out.info["attr_pos_start"] == 17
    assert out.info["attr_pos_end"] == 27
    aggregated = [attr.aggregate(attr._aggregator) for attr in out.sequence_attributions]
    assert all(isinstance(aggr_attr, FeatureAttributionSequenceOutput) for aggr_attr in aggregated)


def test_attribute_decoder_forced(saliency_gpt2_model):
    texts = [
        "Colorless green ideas sleep",
        "The scientist told the director that",
    ]
    forced_generations = [
        "Colorless green ideas sleep furiously.",
        "The scientist told the director that the experiment was a success.",
    ]
    out = saliency_gpt2_model.attribute(
        texts,
        forced_generations,
        show_progress=False,
        device=get_default_device(),
    )
    assert isinstance(out, FeatureAttributionOutput)
    assert len(out.sequence_attributions) == 2
    assert isinstance(out.sequence_attributions[0], FeatureAttributionSequenceOutput)
    ex1, ex2 = out.sequence_attributions[0], out.sequence_attributions[1]
    assert ex1.attr_pos_start == 5
    assert ex1.attr_pos_end == 7
    assert ex1.target_attributions.shape[1] == ex1.attr_pos_end - ex1.attr_pos_start
    assert ex1.target_attributions.shape[0] == ex1.attr_pos_end
    # Empty attributions outputs have start and end set to seq length
    assert ex2.attr_pos_start == 6
    assert ex2.attr_pos_end == 12
    assert ex2.target_attributions.shape[1] == ex2.attr_pos_end - ex2.attr_pos_start
    assert ex2.target_attributions.shape[0] == ex2.attr_pos_end
    assert out.info["attr_pos_start"] == 5
    assert out.info["attr_pos_end"] == 12
    aggregated = [attr.aggregate(attr._aggregator) for attr in out.sequence_attributions]
    assert all(isinstance(aggr_attr, FeatureAttributionSequenceOutput) for aggr_attr in aggregated)


def test_attribute_decoder_forced_sliced(saliency_gpt2_model):
    texts = [
        "Colorless green ideas sleep",
        "The scientist told the director that",
    ]
    forced_generations = [
        "Colorless green ideas sleep furiously.",
        "The scientist told the director that the experiment was a success.",
    ]
    out = saliency_gpt2_model.attribute(
        texts,
        forced_generations,
        show_progress=False,
        device=inseq.utils.get_default_device(),
        attr_pos_start=6,
        attr_pos_end=10,
    )
    assert isinstance(out, FeatureAttributionOutput)
    assert len(out.sequence_attributions) == 2
    assert isinstance(out.sequence_attributions[0], FeatureAttributionSequenceOutput)
    ex1, ex2 = out.sequence_attributions[0], out.sequence_attributions[1]
    assert ex1.attr_pos_start == 6
    assert ex1.attr_pos_end == 7
    assert ex1.target_attributions.shape[1] == ex1.attr_pos_end - ex1.attr_pos_start
    assert ex1.target_attributions.shape[0] == ex1.attr_pos_end
    assert ex2.attr_pos_start == 6
    assert ex2.attr_pos_end == 10
    assert ex2.target_attributions.shape[1] == ex2.attr_pos_end - ex2.attr_pos_start
    assert ex2.target_attributions.shape[0] == ex2.attr_pos_end
    assert out.info["attr_pos_start"] == 6
    assert out.info["attr_pos_end"] == 10
    aggregated = [attr.aggregate(attr._aggregator) for attr in out.sequence_attributions]
    assert all(isinstance(aggr_attr, FeatureAttributionSequenceOutput) for aggr_attr in aggregated)


@mark.slow
@mark.parametrize(("texts", "reference_texts"), EXAMPLES["texts"])
@mark.parametrize("layers", ATTENTION_IDX)
@mark.parametrize("heads", ATTENTION_IDX)
@mark.parametrize("aggregate_heads_fn", ATTENTION_AGGREGATE_FN)
@mark.parametrize("aggregate_layers_fn", ATTENTION_AGGREGATE_FN)
def test_attention_attribution_seq2seq(
    texts,
    reference_texts,
    layers,
    heads,
    aggregate_heads_fn,
    aggregate_layers_fn,
    saliency_mt_model,
):
    if isinstance(layers, int):
        aggregate_layers_fn = "single"
    if isinstance(heads, int):
        aggregate_heads_fn = "single"
    out = saliency_mt_model.attribute(
        texts,
        method="attention",
        show_progress=False,
        attribute_target=True,
        device=get_default_device(),
        layers=layers,
        heads=heads,
        aggregate_heads_fn=aggregate_heads_fn,
        aggregate_layers_fn=aggregate_layers_fn,
    )
    assert isinstance(out, FeatureAttributionOutput)
    assert isinstance(out.sequence_attributions[0], FeatureAttributionSequenceOutput)
    assert out.info["model_name"] == "Helsinki-NLP/opus-mt-en-it"
    assert out.info["constrained_decoding"] is False
    assert out.info["attribution_method"] == "attention"
    assert out.info["attribute_target"] is True
    assert len(out.sequence_attributions[0].source_attributions.shape) == 2


@mark.slow
@mark.parametrize(("texts", "reference_texts"), EXAMPLES["texts"])
@mark.parametrize("layers", ATTENTION_IDX)
@mark.parametrize("heads", ATTENTION_IDX)
@mark.parametrize("aggregate_heads_fn", ATTENTION_AGGREGATE_FN)
@mark.parametrize("aggregate_layers_fn", ATTENTION_AGGREGATE_FN)
def test_attention_attribution_decoder(
    texts,
    reference_texts,
    layers,
    heads,
    aggregate_heads_fn,
    aggregate_layers_fn,
    saliency_gpt2_model,
):
    if isinstance(layers, int):
        aggregate_layers_fn = "single"
    if isinstance(heads, int):
        aggregate_heads_fn = "single"
    out = saliency_gpt2_model.attribute(
        texts,
        method="attention",
        show_progress=False,
        device=get_default_device(),
        layers=layers,
        heads=heads,
        aggregate_heads_fn=aggregate_heads_fn,
        aggregate_layers_fn=aggregate_layers_fn,
    )
    assert isinstance(out, FeatureAttributionOutput)
    assert isinstance(out.sequence_attributions[0], FeatureAttributionSequenceOutput)
    assert out.info["model_name"] == "gpt2"
    assert out.info["constrained_decoding"] is False
    assert out.info["attribution_method"] == "attention"
    assert len(out.sequence_attributions[0].target_attributions.shape) == 2
