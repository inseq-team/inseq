"""Device-boundary and lifecycle regressions from the 0.1.5 audit."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

import inseq
from inseq.attr.feat.ops.grad_ellm_device import model_input_device
from inseq.attr.feat.ops.grad_ellm_token_ids import _attribute_token_ids

from ._api import GradELLMCore, attribute_token_ids, attribution_device


def test_actual_formatter_receives_ids_mask_and_baseline_on_embedding_device(model, tokenizer, monkeypatch):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    # Meta is a separate placement domain available in CPU CI. Stop before
    # numerical attribution; this does NOT emulate CUDA computation.
    model.to("meta")
    ids = torch.tensor([[1, 4, 5, 7]])
    original = ids.clone()
    seen = []

    class Prepared(Exception):
        pass

    def inspect_batch(batch, **kwargs):
        for name in ("input_ids", "attention_mask", "baseline_ids", "input_embeds", "baseline_embeds"):
            assert getattr(batch, name).device.type == "meta"
        seen.append(True)
        raise Prepared

    monkeypatch.setattr(wrapped.attribution_method, "attribute", inspect_batch)
    for mask in (None, torch.ones_like(ids)):
        with pytest.raises(Prepared):
            _attribute_token_ids(wrapped, ids, 2, attention_mask=mask, show_progress=False)
    assert len(seen) == 2 and ids.device.type == "cpu"
    torch.testing.assert_close(ids, original)


@pytest.mark.parametrize(
    "method",
    ["grad_ellm", "saliency", "input_x_gradient", "deeplift", "integrated_gradients", "attention", "value_zeroing"],
)
def test_exact_replay_other_methods_and_caller_ids_unchanged(model, tokenizer, method):
    ids = torch.tensor([[1, 4, 5, 7, 2]])
    original = ids.clone()
    wrapped = inseq.load_model(model, method, tokenizer=tokenizer, device="cpu")
    objective = "probability" if method in {"attention", "value_zeroing"} else "logit"
    options = {"n_steps": 4} if method == "integrated_gradients" else {}
    try:
        out = attribute_token_ids(
            wrapped, ids, 3, attributed_fn=objective, attribution_args=options, show_progress=False
        )
        assert [token.id for token in out[0].target] == original[0].tolist()
        scores = out.aggregate()[0].target_attributions
        assert scores.ndim == 2 and torch.isfinite(scores[:3]).all()
        torch.testing.assert_close(ids, original)
    finally:
        wrapped.attribution_method.unhook()


def test_indexed_cuda_scope_restores_after_nested_failure():
    state = {"current": 0}

    @contextmanager
    def fake_device(index):
        previous = state["current"]
        state["current"] = index
        try:
            yield
        finally:
            state["current"] = previous

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "device_count", return_value=2),
        patch.object(torch.cuda, "current_device", side_effect=lambda: state["current"]),
        patch.object(torch.cuda, "device", fake_device),
    ):
        with attribution_device("cuda:1") as backend:
            assert backend == "cuda" and state["current"] == 1
            with pytest.raises(RuntimeError, match="intentional"), attribution_device("cuda:0"):
                assert state["current"] == 0
                raise RuntimeError("intentional")
            assert state["current"] == 1
        assert state["current"] == 0
        with pytest.raises(ValueError, match="not visible"), attribution_device("cuda:2"):
            pytest.fail("Invalid index must be rejected.")


def test_no_cuda_failure_is_explicit():
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        pytest.raises(RuntimeError, match="CUDA is unavailable"),
        attribution_device("cuda:1"),
    ):
        pytest.fail("Unavailable CUDA must be rejected.")


def test_reject_offloaded_and_meta_models(model):
    assert model_input_device(model) == torch.device("cpu")
    model.hf_device_map = {"": "disk"}
    with pytest.raises(NotImplementedError, match="offloaded"):
        GradELLMCore(model)
    model.hf_device_map = {}
    model.to("meta")
    with pytest.raises(NotImplementedError, match="Meta/offloaded"):
        GradELLMCore(model)


def test_no_parameter_grad_accumulation_and_failure_cleanup(model):
    from .test_core import hook_count

    core = GradELLMCore(model, 2)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    old = [p.grad.clone() for p in model.parameters()]
    core.attribute(torch.tensor([[1, 4, 5]]), torch.tensor([7]))
    for p, previous in zip(model.parameters(), old, strict=False):
        torch.testing.assert_close(p.grad, previous)
    with pytest.raises(ValueError, match="differentiable"):
        core.explain(lambda: torch.tensor(1.0), torch.ones(1, 3))
    with pytest.raises(ValueError, match="selected attention"):
        core.explain(lambda: torch.tensor(1.0, requires_grad=True), torch.ones(1, 3))
    assert hook_count(model) == 0 and not core._active


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA hardware unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_cuda_cpu_ids_to_selected_gpu(model, tokenizer, dtype):
    index = 1 if torch.cuda.device_count() >= 2 else 0
    device = f"cuda:{index}"
    with attribution_device(device) as backend:
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("Selected GPU does not support bfloat16")
        model.to(device=device, dtype=dtype)
        wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device=backend, n_layers=2)
    # Call outside that context: exact-ID replay must infer and enter the actual
    # embedding device, even if the caller's current CUDA device differs.
    ids = torch.tensor([[1, 4, 5, 7, 2]])
    try:
        out = attribute_token_ids(wrapped, ids, 3, show_progress=False, step_scores=["probability"])
        expected = GradELLMCore(model, 2).attribute(ids[:, :3], ids[:, 3]).cpu()
        torch.testing.assert_close(out[0].target_attributions[:3, 0], expected[0])
        assert out.show(display=False, return_html=True)
        assert ids.device.type == "cpu"
    finally:
        wrapped.attribution_method.unhook()
