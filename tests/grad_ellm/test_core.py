"""Independent reference implements the supplied batch-one equations literally."""

import math

import pytest
import torch
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

from inseq.attr.feat.ops.grad_ellm import expand_kv, layer_scores

from ._api import GradELLMCore
from .conftest import tiny_model


def legacy_reference(model, ids, target, n_layers, normalize):
    captures = [{} for _ in range(n_layers)]
    handles = []
    try:
        for idx, layer in enumerate(model.model.layers[-n_layers:]):

            def proj_hook(name, i):
                def hook(module, args, output):
                    captures[i][name] = output.detach()

                return hook

            def attn_hook(i):
                def hook(module, args, output):
                    captures[i]["out"] = output[0]

                return hook

            for name in ["q", "k", "v"]:
                handles.append(getattr(layer.self_attn, name + "_proj").register_forward_hook(proj_hook(name, idx)))
            handles.append(layer.self_attn.register_forward_hook(attn_hook(idx)))
        logits = model(ids, use_cache=False).logits[:, -1, :]
        grads = torch.autograd.grad(logits[:, target].sum(), [c["out"] for c in captures])
        maps = []
        for c, grad in zip(captures, grads, strict=False):
            q, k, v = c["q"][:, -1, :], c["k"], c["v"]
            k = k.repeat(1, 1, 4) if k.shape[-1] < q.shape[-1] else k
            v = v.repeat(1, 1, 4) if v.shape[-1] < q.shape[-1] else v
            if normalize:
                similarity = (F.normalize(q, dim=-1) * F.normalize(k, dim=-1)).sum(-1)
                similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
            else:
                similarity = torch.softmax((q * k).sum(-1) / math.sqrt(k.shape[-1]), dim=-1)
            maps.append(F.relu((grad[:, -1, :].detach().unsqueeze(0) * v * similarity.unsqueeze(-1)).sum(-1)))
        return torch.stack(maps).sum(0)
    finally:
        for h in handles:
            h.remove()


def hook_count(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


@pytest.mark.parametrize("family", ["llama", "mistral"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("kv_heads", [2, 8])
def test_legacy_matches_exact_tokens(family, normalize, kv_heads):
    model = tiny_model(family, kv_heads)
    core = GradELLMCore(model, n_layers=2, normalize=normalize)
    ids = torch.tensor([[1, 4, 5, 6, 7]])
    for target in [10, 11, 2]:
        expected = legacy_reference(model, ids, target, 2, normalize)
        actual = core.attribute(ids, torch.tensor([target]))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
        ids = torch.cat([ids, torch.tensor([[target]])], dim=1)
    assert hook_count(model) == 0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("kv_expansion", ["legacy_repeat", "grouped"])
@pytest.mark.parametrize("gradient_source", ["attention_output", "pre_o"])
def test_batch_left_padding_matches_single(model, normalize, kv_expansion, gradient_source):
    core = GradELLMCore(model, 2, normalize, kv_expansion, gradient_source)
    ids = torch.tensor([[1, 4, 5, 6, 7], [0, 0, 1, 8, 6]])
    mask = ids.ne(0)
    batched = core.attribute(ids, torch.tensor([10, 11]), mask)
    for row, target in enumerate([10, 11]):
        single = core.attribute(ids[row : row + 1, mask[row]], torch.tensor([target]))
        torch.testing.assert_close(batched[row, mask[row]], single[0], rtol=2e-5, atol=2e-7)
    assert (batched[~mask] == 0).all()
    assert hook_count(model) == 0


def test_head_repeat_order():
    kv = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    assert expand_kv(kv, 8, 2, "legacy_repeat").tolist() == [[[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]]]
    assert expand_kv(kv, 8, 2, "grouped").tolist() == [[[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]]]


def test_constant_similarity_is_zero():
    q, kv = torch.ones(2, 4), torch.ones(2, 3, 4)
    score = layer_scores(q, kv, kv, q, torch.ones(2, 3, dtype=torch.bool), True)
    assert torch.isfinite(score).all() and (score == 0).all()


def test_cleanup_after_failure(model):
    core = GradELLMCore(model, 2)

    def fail():
        model(torch.tensor([[1, 4, 5]]), use_cache=False)
        raise RuntimeError("intentional forward failure")

    with pytest.raises(RuntimeError, match="intentional"):
        core.explain(fail, torch.ones(1, 3))
    assert hook_count(model) == 0 and not core._active and not core._captures
    assert torch.isfinite(core.attribute(torch.tensor([[1, 4, 5]]), torch.tensor([7]))).all()


@pytest.mark.parametrize("gradient_source", ["attention_output", "pre_o"])
def test_frozen_model_and_no_grad(model, gradient_source):
    model.requires_grad_(False)
    with torch.no_grad():
        scores = GradELLMCore(model, 2, gradient_source=gradient_source).attribute(
            torch.tensor([[1, 4, 5]]), torch.tensor([7])
        )
    assert torch.isfinite(scores).all()
    assert not any(p.requires_grad for p in model.parameters())
    assert hook_count(model) == 0


def test_reject_invalid_inputs(model):
    for layers in [0, 4, True, 1.5]:
        with pytest.raises(ValueError):
            GradELLMCore(model, layers)
    core = GradELLMCore(model)
    with pytest.raises(ValueError, match="left padding"):
        core.attribute(torch.tensor([[1, 4, 0]]), torch.tensor([7]), torch.tensor([[1, 1, 0]]))
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        core.attribute(torch.tensor([[1, 4]]), torch.tensor([7]))
    t5 = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=16, num_layers=1, num_heads=2))
    with pytest.raises(NotImplementedError, match="decoder-only"):
        GradELLMCore(t5)


def test_unequal_gqa_ratio_supported():
    model = tiny_model(kv_heads=4)  # Ratio 2: original hard-coded factor 4 would fail.
    for mode in ["legacy_repeat", "grouped"]:
        score = GradELLMCore(model, kv_expansion=mode).attribute(torch.tensor([[1, 4, 5]]), torch.tensor([7]))
        assert score.shape == (1, 3) and torch.isfinite(score).all()
