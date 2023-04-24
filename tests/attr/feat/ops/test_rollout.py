import torch

from inseq.attr.feat.ops.rollout import rollout_fn


def test_rollout_consistency_custom_dim():
    original = torch.randn(1, 2, 2, 6, 7, 3, 3)
    for dim in range(len(original.shape)):
        if dim != 1:
            dim_changed = original.transpose(1, dim)
            rolled_out_original = rollout_fn(original)
            rolled_out_dim_changed = rollout_fn(dim_changed, dim=dim)
            rolled_out_original = rolled_out_original.unsqueeze(1).transpose(1, dim).squeeze(dim)
            assert rolled_out_original.shape == rolled_out_dim_changed.shape
            assert torch.allclose(rolled_out_original, rolled_out_dim_changed)
