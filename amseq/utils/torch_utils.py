from typing import Any, Optional

import torch
from torch import Tensor, long
from torchtyping import TensorType


def remap_from_filtered(
    source: TensorType[..., Any],
    mask: TensorType["batch_size", 1, long],
    filtered: TensorType["filtered_batch_size", "seq_len", Any],
) -> Tensor:
    if len(filtered.shape) > 1:
        index = mask.squeeze().nonzero().expand_as(filtered)
    else:
        index = mask.squeeze().nonzero().squeeze()
    new_source = torch.ones_like(source, dtype=filtered.dtype) * float("nan")
    return new_source.scatter(0, index, filtered)


def sum_normalize(
    attributions: TensorType["batch_size", "seq_len", "hidden_size", float],
    dim_sum: Optional[int] = -1,
) -> TensorType["batch_size", "seq_len", float]:
    """
    Sum and normalize tensor across dim_sum.
    """
    attributions = attributions.sum(dim=dim_sum).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    if len(attributions.shape) == 1:
        return attributions.unsqueeze(0)
    return attributions


def pretty_tensor(t: Optional[torch.Tensor]) -> str:
    if t is None:
        return "None"
    if len(t.shape) > 3 or any([x > 20 for x in t.shape]):
        return f"tensor of shape {list(t.shape)}"
    else:
        return f"tensor of shape {list(t.shape)}: {t.tolist()}"
