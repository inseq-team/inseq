"""Regression checks for tensor and tuple/list Transformer block outputs."""

from types import SimpleNamespace

import pytest
import torch

from inseq.attr.feat.ops.value_zeroing import ValueZeroing


@pytest.mark.parametrize("container", ["tensor", "tuple", "list"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_value_zeroing_patches_full_batch_and_preserves_output(container, dtype):
    corrupted = torch.arange(24).reshape(2, 3, 4).to(dtype)
    clean = torch.full((2, 3, 4), -1.0, dtype=dtype)
    owner = SimpleNamespace(clean_block_output_states={0: clean}, corrupted_block_output_states={})
    hook = ValueZeroing.get_states_extract_and_patch_hook(owner, 0)
    tail = object()
    output = corrupted if container == "tensor" else (corrupted, tail)
    if container == "list":
        output = list(output)
    patched = hook(None, (), output)
    torch.testing.assert_close(owner.corrupted_block_output_states[0], corrupted.float())
    assert owner.corrupted_block_output_states[0].shape == (2, 3, 4)
    assert not owner.corrupted_block_output_states[0].requires_grad
    if container == "tensor":
        assert isinstance(patched, torch.Tensor)
        torch.testing.assert_close(patched, clean)
    else:
        assert type(patched) is type(output)
        assert patched[1] is tail and output[0] is corrupted
        torch.testing.assert_close(patched[0], clean)


def test_value_zeroing_nonzero_tuple_index():
    corrupted = torch.randn(2, 3, 4)
    clean = torch.zeros_like(corrupted)
    owner = SimpleNamespace(clean_block_output_states={0: clean}, corrupted_block_output_states={})
    hook = ValueZeroing.get_states_extract_and_patch_hook(owner, 0, hidden_state_idx=1)
    head, tail = object(), object()
    patched = hook(None, (), (head, corrupted, tail))
    assert patched[0] is head and patched[2] is tail
    torch.testing.assert_close(patched[1], clean)
    torch.testing.assert_close(owner.corrupted_block_output_states[0], corrupted)


def test_value_zeroing_tensor_rejects_nonzero_index():
    owner = SimpleNamespace(clean_block_output_states={}, corrupted_block_output_states={})
    hook = ValueZeroing.get_states_extract_and_patch_hook(owner, 0, hidden_state_idx=1)
    with pytest.raises(ValueError, match="hidden_state_idx=0"):
        hook(None, (), torch.zeros(2, 3, 4))
