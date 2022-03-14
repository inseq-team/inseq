"""
TODO: Skipping layer attribution when attributing target and the DIG method
since it is bugged is not very elegant, this will need to be refactored.
"""

from textwrap import dedent

import numpy as np
import pytest
import torch
from pytest import fixture, mark

import inseq
from inseq import list_feature_attribution_methods
from inseq.data import FeatureAttributionOutput, FeatureAttributionSequenceOutput


EX_TEXTS = [
    ("Hello world!", "Buongiorno mondo!"),
    (
        [
            "Hello world!",
            "Colorless green ideas sleep furiously.",
        ],
        [
            "Buongiorno mondo!",
            "Le idee verdi senza colore dormono furiosamente",
        ],
    ),
    (
        [
            "The manager told the hairdresser that the haircut he made her was terrible.",
            "Colorless green ideas sleep furiously",
            "The scientist told the director that she made a new discovery.",
        ],
        [
            "La direttrice ha detto al parrucchiere che il taglio di capelli che le ha fatto è terribile.",
            "Le idee verdi senza colore dormono furiosamente",
            "La ricercatrice ha detto al direttore che ha fatto una nuova scoperta.",
        ],
    ),
]

EX_LONG_TEXT = [
    (
        dedent(
            """
        Integrated gradients is a simple, yet powerful axiomatic attribution
        method that requires almost no modification of the original network.
        It can be used for augmenting accuracy metrics, model debugging and
        feature or rule extraction."""
        ).replace("\n", " "),
        dedent(
            """
        Integrated gradients è un metodo di attribuzione assiomatico semplice ma
        potente che non richiede quasi nessuna modifica alla rete originale. Può
        essere usato per arricchire metriche di accuratezza, per effettuare il
        debug di modelli e per l'estrazione di feature o regole."""
        ).replace("\n", " "),
    )
]

EX_TEXTS_SHORT = EX_TEXTS[:2]

USE_REFERENCE_TEXT = [True, False]
ATTRIBUTE_TARGET = [True, False]
RETURN_CONVERGENCE_DELTA = [True, False]
RETURN_STEP_PROBABILITIES = [True, False]
ATTRIBUTION_METHODS = list_feature_attribution_methods()


@fixture(scope="session")
def saliency_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency")


@mark.slow
@mark.require_gpu
@mark.parametrize(("texts", "reference_texts"), EX_TEXTS_SHORT)
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
        assert all([tok_cpu == tok_gpu for tok_cpu, tok_gpu in zip(out_cpu.target_tokens, out_gpu.target_tokens)])
        attr_score_matches = [
            abs(el_cpu - el_gpu) < 1e-3
            for cpu_attr, gpu_attr in zip(out_cpu.source_attributions, out_gpu.source_attributions)
            for el_cpu, el_gpu in zip(cpu_attr, gpu_attr)
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
    texts_single, reference_single = EX_TEXTS[0]
    texts_batch, reference_batch = EX_TEXTS[1]
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
    assert np.allclose(
        out_single.sequence_attributions[0].source_scores, out_batch.sequence_attributions[0].source_scores, atol=8e-2
    )
    if attribute_target:
        assert np.allclose(
            out_single.sequence_attributions[0].target_scores,
            out_batch.sequence_attributions[0].target_scores,
            atol=8e-2,
            equal_nan=True,
        )


@mark.slow
@mark.parametrize(("texts", "reference_texts"), EX_TEXTS)
@mark.parametrize("attribution_method", ATTRIBUTION_METHODS)
@mark.parametrize("use_reference", USE_REFERENCE_TEXT)
@mark.parametrize("attribute_target", ATTRIBUTE_TARGET)
@mark.parametrize("return_convergence_delta", RETURN_CONVERGENCE_DELTA)
@mark.parametrize("return_step_probabilities", RETURN_STEP_PROBABILITIES)
def test_attribute(
    texts,
    reference_texts,
    attribution_method,
    use_reference,
    attribute_target,
    return_convergence_delta,
    return_step_probabilities,
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
        output_step_probabilities=return_step_probabilities,
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
    assert out.info["output_step_probabilities"] == return_step_probabilities
    if "return_convergence_delta" in out.info:
        assert out.info["return_convergence_delta"] == return_convergence_delta
    if "internal_batch_size" in out.info:
        assert out.info["internal_batch_size"] == 50
    if "n_steps" in out.info:
        assert out.info["n_steps"] == 100


@mark.slow
@mark.parametrize(("texts", "reference_texts"), EX_LONG_TEXT)
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
