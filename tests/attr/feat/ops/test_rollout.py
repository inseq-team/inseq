import torch

from inseq.attr.feat.ops.rollout import rollout_fn


def test_rollout_consistency_custom_dim():
    original = torch.randint(low=0, high=10, size=(1, 2, 3, 4, 5, 6, 2, 2))
    rolled_out_original = rollout_fn(original)
    dim_changed_2 = original.permute(0, 2, 1, 3, 4, 5, 6, 7)
    rolled_out_dim_changed_2 = rollout_fn(dim_changed_2, dim=2)
    assert rolled_out_original.shape == rolled_out_dim_changed_2.shape
    assert torch.allclose(rolled_out_original, rolled_out_dim_changed_2)
    dim_changed_3 = original.permute(0, 2, 3, 1, 4, 5, 6, 7)
    rolled_out_dim_changed_3 = rollout_fn(dim_changed_3, dim=3)
    assert rolled_out_original.shape == rolled_out_dim_changed_3.shape
    assert torch.allclose(rolled_out_original, rolled_out_dim_changed_3)
    dim_changed_4 = original.permute(0, 2, 3, 4, 1, 5, 6, 7)
    rolled_out_dim_changed_4 = rollout_fn(dim_changed_4, dim=4)
    assert rolled_out_original.shape == rolled_out_dim_changed_4.shape
    assert torch.allclose(rolled_out_original, rolled_out_dim_changed_4)
    dim_changed_5 = original.permute(0, 2, 3, 4, 5, 1, 6, 7)
    rolled_out_dim_changed_5 = rollout_fn(dim_changed_5, dim=5)
    assert rolled_out_original.shape == rolled_out_dim_changed_5.shape
    assert torch.allclose(rolled_out_original, rolled_out_dim_changed_5)
