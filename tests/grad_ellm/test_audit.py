"""Regression checks for integration paths found in the 0.1.4 audit."""

from copy import deepcopy

import pytest
import torch

import inseq
from inseq.attr.feat.ops.grad_ellm import layer_scores
from inseq.data.aggregation_functions import DEFAULT_ATTRIBUTION_AGGREGATE_DICT
from inseq.data.aggregator import AggregatorPipeline, SequenceAttributionAggregator

from ._api import GradELLMCore, attribute_token_ids


def attribute(wrapped, **kwargs):
    return wrapped.attribute(
        "the movie is",
        generated_texts="the movie is good positive",
        attributed_fn="logit",
        show_progress=False,
        step_scores=["probability"],
        **kwargs,
    )


@pytest.mark.parametrize("style", ["list", "pipeline", "class", "subwords_pipeline"])
def test_all_scores_dispatch_forms(model, tokenizer, style):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    out = attribute(wrapped)
    aggregators = {
        "list": ["scores"],
        "pipeline": AggregatorPipeline(["scores"]),
        "class": SequenceAttributionAggregator,
        "subwords_pipeline": ["subwords", "scores"],
    }
    result = out.aggregate(aggregators[style])
    torch.testing.assert_close(result[0].target_attributions, out.aggregate()[0].target_attributions, equal_nan=True)
    assert result.show(display=False, return_html=True)


def test_aggregation_config_isolation_and_weighting(model, tokenizer, tmp_path):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    defaults = deepcopy(DEFAULT_ATTRIBUTION_AGGREGATE_DICT)
    out = attribute(wrapped)
    seq = out[0]
    assert seq._dict_aggregate_fn is not DEFAULT_ATTRIBUTION_AGGREGATE_DICT
    _ = out.aggregate(["scores", "scores"])
    _ = seq.aggregate("spans", source_spans=[(0, 2)], target_spans=[(0, 2)]).aggregate(["scores"])
    expected = seq.aggregate().target_attributions * seq.step_scores["probability"][None, :]
    weighted = seq.weight_attributions("probability")
    torch.testing.assert_close(
        weighted.target_attributions,
        expected,
        equal_nan=True,
    )
    _ = seq.aggregate("pair", paired_attr=seq, do_post_aggregation_checks=False)
    path = str(tmp_path / "out.json")
    out.save(path)
    restored = inseq.FeatureAttributionOutput.load(path)
    assert restored.show(display=False, return_html=True)
    assert DEFAULT_ATTRIBUTION_AGGREGATE_DICT == defaults
    assert seq._dict_aggregate_fn["target_attributions"]["scores"] == "grad_ellm_identity"


@pytest.mark.parametrize(
    "selection,indices",
    [
        (slice(None, 2), [0, 1]),
        (slice(1, None), [1, 2]),
        (slice(-2, None), [1, 2]),
        (slice(None), [0, 1, 2]),
        (0, [0]),
        (-1, [2]),
    ],
)
def test_prompt_slices_keep_generated_tokens(model, tokenizer, selection, indices):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    seq = attribute(wrapped)[0]
    sliced = seq[selection]
    expected_indices = indices + list(range(seq.attr_pos_start, len(seq.target)))
    assert sliced.target == [seq.target[index] for index in expected_indices]
    assert sliced.source == [seq.source[index] for index in indices]
    assert sliced.attr_pos_start == len(indices)
    torch.testing.assert_close(sliced.target_attributions, seq.target_attributions[expected_indices], equal_nan=True)
    assert sliced.show(display=False, return_html=True)
    with pytest.raises(ValueError, match="non-empty contiguous"):
        _ = seq[::2]
    with pytest.raises(ValueError, match="non-empty contiguous"):
        _ = seq[:0]


def test_explicit_method_keeps_settings_and_records_them(model, tokenizer):
    options = {"n_layers": 2, "normalize": False, "kv_expansion": "grouped", "gradient_source": "pre_o"}
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu", **options)
    implicit = attribute(wrapped)
    explicit = attribute(wrapped, method="grad_ellm")
    _ = attribute(wrapped, method="saliency")
    after_switch = attribute(wrapped, method="grad_ellm")
    for out in [implicit, explicit, after_switch]:
        assert out.info["grad_ellm_options"] == options
        torch.testing.assert_close(out[0].target_attributions, implicit[0].target_attributions, equal_nan=True)
    # A separately loaded wrapper still gets the ordinary defaults.
    other = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    assert other.attribution_method.core.n_layers == 1
    assert other.attribution_method.core.normalize is True


@pytest.mark.parametrize("keep_steps", [False, True])
@pytest.mark.parametrize("same_pad_eos", [False, True])
def test_exact_eos_replay_with_missing_unk(model, tokenizer, keep_steps, same_pad_eos):
    if same_pad_eos:
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, n_layers=2, device="cpu")
    tokenizer.unk_token = None
    ids = torch.tensor([[1, 4, 5, 10, 2]])
    out = attribute_token_ids(wrapped, ids, 3, show_progress=False, output_step_attributions=keep_steps)
    assert [token.id for token in out[0].target] == ids[0].tolist()
    assert out[0].target_attributions.shape == (5, 2)
    assert out.info["output_step_attributions"] is keep_steps
    assert (out.step_attributions is not None) is keep_steps
    for column, position in enumerate([3, 4]):
        expected = GradELLMCore(model, 2).attribute(ids[:, :position], ids[:, position])
        torch.testing.assert_close(out[0].target_attributions[:position, column], expected[0])
    assert out.show(display=False, return_html=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalize", [True, False])
def test_low_precision_scores_avoid_underflow_and_overflow(dtype, normalize):
    q = torch.tensor([[1.0, 0.0]], dtype=dtype)
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=dtype)
    v = torch.full_like(k, 10000.0)
    grad = torch.full_like(q, 10000.0)
    mask = torch.ones(1, 2, dtype=torch.bool)
    scores = layer_scores(q, k, v, grad, mask, normalize)
    expected = layer_scores(q.float(), k.float(), v.float(), grad.float(), mask, normalize)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(scores, expected)
    zero = layer_scores(torch.zeros_like(q), torch.zeros_like(k), v, grad, mask, True)
    torch.testing.assert_close(zero, torch.zeros_like(scores))


def test_invalid_options_masks_and_replay_inputs(model, tokenizer):
    with pytest.raises(TypeError, match="normalize must be a bool"):
        GradELLMCore(model, normalize="False")
    core = GradELLMCore(model)
    for mask in [torch.empty(1, 0), torch.empty(0, 2), torch.ones(2), torch.full((1, 2), 2)]:
        with pytest.raises(ValueError, match="attention mask|Attention mask"):
            core.explain(lambda: pytest.fail("Invalid mask must be rejected before forwarding."), mask)
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    with pytest.raises(TypeError, match="prompt_length must be an integer"):
        attribute_token_ids(wrapped, torch.tensor([[4, 5, 7]]), 1.5)
    with pytest.raises(ValueError, match="integer tensors"):
        attribute_token_ids(wrapped, torch.tensor([[4.0, 5.0, 7.0]]), 1)
    with pytest.raises(ValueError, match="embedding vocabulary"):
        attribute_token_ids(wrapped, torch.tensor([[4, 5, 999]]), 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_hf_forward_and_backward(model, tokenizer, dtype, tmp_path):
    model.to(dtype=dtype)
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu", n_layers=2)
    out = attribute(wrapped, output_step_attributions=True)
    raw = out[0].target_attributions
    assert raw.dtype == torch.float32
    for column in range(raw.shape[1]):
        valid = raw[: out[0].attr_pos_start + column, column]
        assert torch.isfinite(valid).all()
    assert out.show(display=False, return_html=True)
    path = str(tmp_path / "low_precision.json")
    out.save(path)
    restored = inseq.FeatureAttributionOutput.load(path)
    assert restored.show(display=False, return_html=True)
    for saved, original in zip(restored.step_attributions, out.step_attributions, strict=False):
        torch.testing.assert_close(saved.target_attributions, original.target_attributions)
        # Inseq's default serialization stores/reconstructs float32 scores.
        torch.testing.assert_close(
            saved.step_scores["probability"].float(), original.step_scores["probability"].float()
        )


@pytest.mark.parametrize("family", ["llama", "mistral"])
def test_cpu_sdpa_matches_eager(family):
    from .conftest import tiny_model

    model = tiny_model(family)
    core = GradELLMCore(model, n_layers=2)
    ids, target = torch.tensor([[1, 4, 5]]), torch.tensor([7])
    eager = core.attribute(ids, target)
    model.config._attn_implementation = "sdpa"
    sdpa = core.attribute(ids, target)
    torch.testing.assert_close(sdpa, eager, rtol=2e-5, atol=2e-7)


def test_int32_ids_and_core_shape_validation(model, tokenizer):
    ids = torch.tensor([[1, 4, 5, 7]], dtype=torch.int32)
    core = GradELLMCore(model)
    torch.testing.assert_close(
        core.attribute(ids[:, :3], ids[:, 3]), core.attribute(ids[:, :3].long(), ids[:, 3].long())
    )
    with pytest.raises(ValueError, match="same shape"):
        core.attribute(ids[:, :3], ids[:, 3], torch.ones(1, 2))
    with pytest.raises(ValueError, match="non-empty input_ids"):
        core.attribute(ids[0], ids[:, 3])
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    out = attribute_token_ids(wrapped, ids, 3, show_progress=False)
    assert [token.id for token in out[0].target] == ids[0].tolist()


def test_multi_forward_objective_rejected_and_hooks_cleaned(model):
    from .test_core import hook_count

    core = GradELLMCore(model, 2)
    ids = torch.tensor([[1, 4, 5]])

    def objective():
        first = model(ids, use_cache=False).logits[:, -1, 7]
        second = model(ids.flip(-1), use_cache=False).logits[:, -1, 8]
        return first - second

    with pytest.raises(NotImplementedError, match="one model forward"):
        core.explain(objective, torch.ones_like(ids))
    assert hook_count(model) == 0
    assert torch.isfinite(core.attribute(ids, torch.tensor([7]))).all()
