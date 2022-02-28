import pytest

import inseq
from inseq.data import FeatureAttributionSequenceOutput


FULL_EXAMPLES = [
    # Single sentence, referenceless, source-side attribution
    ("Hello world!", None, False),
    # Single sentence with reference, source and target-side attribution
    ("Hello world!", "Buongiorno mondo!", True),
    # Sentence pair, referenceless, source-side attribution
    (
        [
            "The manager told the hairdresser that the haircut he made her was terrible.",
            "Colorless green ideas sleep furiously",
        ],
        None,
        False,
    ),
    # Three sentences with shorter middle sentence (edge case for filtered attribution step)
    # with references and target-side attribution
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
        True,
    ),
]

SHORT_EXAMPLES = FULL_EXAMPLES[:2]


@pytest.fixture(scope="session")
def saliency_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency")


@pytest.fixture(scope="session")
def ig_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "integrated_gradients")


@pytest.fixture(scope="session")
def deeplift_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "deeplift")


@pytest.fixture(scope="session")
def ixg_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "input_x_gradient")


@pytest.fixture(scope="session")
def gshap_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "gradient_shap")


@pytest.mark.slow
@pytest.mark.require_gpu
@pytest.mark.parametrize(
    ("texts", "reference_texts", "attribute_target"),
    SHORT_EXAMPLES,
)
def test_cuda_attribution_consistency(texts, reference_texts, attribute_target, saliency_mt_model):
    cpu_out = saliency_mt_model.attribute(
        texts, reference_texts, show_progress=False, attribute_target=attribute_target, device="cpu"
    )
    gpu_out = saliency_mt_model.attribute(
        texts, reference_texts, show_progress=False, attribute_target=attribute_target, device="cuda:0"
    )
    assert isinstance(cpu_out, FeatureAttributionSequenceOutput)
    assert isinstance(gpu_out, FeatureAttributionSequenceOutput)
    assert all([tok_cpu == tok_gpu for tok_cpu, tok_gpu in zip(cpu_out.target_tokens, gpu_out.target_tokens)])
    attr_score_matches = [
        abs(el_cpu - el_gpu) < 5e-3
        for cpu_attr, gpu_attr in zip(cpu_out.source_attributions, gpu_out.source_attributions)
        for el_cpu, el_gpu in zip(cpu_attr, gpu_attr)
    ]
    assert all(attr_score_matches)


@pytest.mark.slow
@pytest.mark.require_gpu
@pytest.mark.parametrize(
    ("texts", "reference_texts", "attribute_target"),
    FULL_EXAMPLES,
)
def test_attribute_with_and_without_reference(texts, reference_texts, attribute_target, saliency_mt_model):
    out = saliency_mt_model.attribute(
        texts, reference_texts, show_progress=False, attribute_target=attribute_target, device="cuda:0"
    )
    assert isinstance(out, FeatureAttributionSequenceOutput) or (
        isinstance(out, list) and isinstance(out[0], FeatureAttributionSequenceOutput)
    )


# Test attribution types


@pytest.mark.slow
@pytest.mark.require_gpu
@pytest.mark.parametrize(
    ("texts", "reference_texts", "attribute_target"),
    SHORT_EXAMPLES,
)
def test_saliency_attribution(texts, reference_texts, attribute_target, saliency_mt_model):
    out = saliency_mt_model.attribute(
        texts, reference_texts, show_progress=False, attribute_target=attribute_target, device="cuda:0"
    )
    assert isinstance(out, FeatureAttributionSequenceOutput) or (
        isinstance(out, list) and isinstance(out[0], FeatureAttributionSequenceOutput)
    )


@pytest.mark.slow
@pytest.mark.require_gpu
@pytest.mark.parametrize(
    ("texts", "reference_texts", "attribute_target", "return_convergence_delta"),
    [x + (True,) for x in SHORT_EXAMPLES],
)
def test_ig_attribution(texts, reference_texts, attribute_target, return_convergence_delta, ig_mt_model):
    out = ig_mt_model.attribute(
        texts,
        reference_texts,
        show_progress=False,
        return_convergence_delta=return_convergence_delta,
        attribute_target=attribute_target,
        n_steps=50,
        device="cuda:0",
    )
    assert isinstance(out, FeatureAttributionSequenceOutput) or (
        isinstance(out, list) and isinstance(out[0], FeatureAttributionSequenceOutput)
    )


@pytest.mark.slow
@pytest.mark.require_gpu
@pytest.mark.parametrize(
    ("texts", "reference_texts", "attribute_target"),
    SHORT_EXAMPLES,
)
def test_deeplift_attribution(texts, reference_texts, attribute_target, deeplift_mt_model):
    out = deeplift_mt_model.attribute(
        texts, reference_texts, show_progress=False, attribute_target=attribute_target, device="cuda:0"
    )
    assert isinstance(out, FeatureAttributionSequenceOutput) or (
        isinstance(out, list) and isinstance(out[0], FeatureAttributionSequenceOutput)
    )


@pytest.mark.slow
@pytest.mark.require_gpu
@pytest.mark.parametrize(
    ("texts", "reference_texts", "attribute_target"),
    SHORT_EXAMPLES,
)
def test_ixg_attribution(texts, reference_texts, attribute_target, ixg_mt_model):
    out = ixg_mt_model.attribute(
        texts, reference_texts, show_progress=False, attribute_target=attribute_target, device="cuda:0"
    )
    assert isinstance(out, FeatureAttributionSequenceOutput) or (
        isinstance(out, list) and isinstance(out[0], FeatureAttributionSequenceOutput)
    )


@pytest.mark.slow
@pytest.mark.require_gpu
@pytest.mark.parametrize(
    ("texts", "reference_texts", "attribute_target"),
    SHORT_EXAMPLES,
)
def test_gshap_attribution(texts, reference_texts, attribute_target, gshap_mt_model):
    out = gshap_mt_model.attribute(
        texts, reference_texts, show_progress=False, attribute_target=attribute_target, device="cuda:0"
    )
    assert isinstance(out, FeatureAttributionSequenceOutput) or (
        isinstance(out, list) and isinstance(out[0], FeatureAttributionSequenceOutput)
    )
