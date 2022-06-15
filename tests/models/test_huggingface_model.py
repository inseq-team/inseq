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


EXAMPLES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../fixtures/huggingface_model.json")
EXAMPLES = json.load(open(EXAMPLES_FILE))

USE_REFERENCE_TEXT = [True, False]
ATTRIBUTE_TARGET = [True, False]
RETURN_CONVERGENCE_DELTA = [True, False]
STEP_SCORES = [[], ["probability"]]
ATTRIBUTION_METHODS = list_feature_attribution_methods()


@fixture(scope="session")
def saliency_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency")


@mark.slow
@mark.require_gpu
@mark.parametrize(("texts", "reference_texts"), EXAMPLES["short_texts"])
@mark.parametrize("attribute_target", ATTRIBUTE_TARGET)
def test_cuda_attribution_consistency(texts, reference_texts, attribute_target, saliency_mt_model):
    out = {}
    for device in ["cpu", "cuda:0"]:
        out[device] = saliency_mt_model.attribute(
            texts, reference_texts, show_progress=False, attribute_target=attribute_target, device=device
        )
        assert isinstance(out[device], FeatureAttributionOutput)
        assert isinstance(out[device].sequence_attributions[0], FeatureAttributionSequenceOutput)
    for out_cpu, out_gpu in zip(out["cpu"].sequence_attributions, out["cuda:0"].sequence_attributions):
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
def test_batched_attribution_consistency(attribution_method, use_reference, attribute_target, saliency_mt_model):
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
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        method=attribution_method,
    )
    out_batch = saliency_mt_model.attribute(
        texts_batch,
        reference_batch,
        show_progress=False,
        attribute_target=attribute_target,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
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
def test_attribute(
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
        device="cuda:0" if torch.cuda.is_available() else "cpu",
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
def test_attribute_long_text(texts, reference_texts, attribution_method, use_reference, saliency_mt_model):
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
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    assert isinstance(out, FeatureAttributionOutput)
    assert isinstance(out.sequence_attributions[0], FeatureAttributionSequenceOutput)
