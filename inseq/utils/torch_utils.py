from typing import Any, Optional

import torch
from torchtyping import TensorType

from .misc import _pretty_list
from .typing import AttributionOutputTensor, EmbeddingsTensor


def remap_from_filtered(
    source: TensorType[..., Any],
    mask: TensorType["batch_size", 1, int],
    filtered: TensorType["filtered_batch_size", "seq_len", Any],
) -> TensorType["batch_size", "seq_len", Any]:
    if len(filtered.shape) > 1:
        index = mask.squeeze().nonzero().expand_as(filtered)
    else:
        index = mask.squeeze().nonzero().squeeze()
    new_source = torch.ones_like(source, dtype=filtered.dtype) * float("nan")
    return new_source.scatter(0, index, filtered)


def sum_normalize(
    attributions: EmbeddingsTensor,
    dim_sum: Optional[int] = -1,
) -> AttributionOutputTensor:
    """
    Sum and normalize tensor across dim_sum.
    """
    attributions = attributions.sum(dim=dim_sum).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    if len(attributions.shape) == 1:
        return attributions.unsqueeze(0)
    return attributions


def pretty_tensor(t: Optional[torch.Tensor] = None) -> str:
    if t is None:
        return "None"
    if len(t.shape) > 3 or any([x > 20 for x in t.shape]):
        return f"tensor of shape {list(t.shape)}"
    else:
        out_list = t.tolist()
        return f"tensor of shape {list(t.shape)}:{_pretty_list(out_list) if isinstance(out_list, list) else out_list}"


def euclidean_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute the Euclidean distance between two points."""
    return (vec_a - vec_b).pow(2).sum(-1).sqrt()
