from typing import Any, Optional

import torch
from torchtyping import TensorType

from .typing import (
    AttributionOutputTensor,
    EmbeddingsTensor,
    FullLogitsTensor,
    TargetIdsTensor,
    TopProbabilitiesTensor,
)


@torch.no_grad()
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
    The outcome is a matrix of unit row vectors.
    """
    attributions = attributions.sum(dim=dim_sum).squeeze(0)
    attributions = attributions.T.div(torch.norm(attributions, dim=dim_sum)).T
    if len(attributions.shape) == 1:
        return attributions.unsqueeze(0)
    return attributions


def euclidean_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute the Euclidean distance between two points."""
    return (vec_a - vec_b).pow(2).sum(-1).sqrt()


@torch.no_grad()
def logits2probs(logits: FullLogitsTensor, target_ids: TargetIdsTensor) -> TopProbabilitiesTensor:
    """
    Compute the scores of the target_ids from the logits.
    The scores are computed as the probabilities of the target_ids after softmax.
    """
    softmax_out = torch.nn.functional.softmax(logits, dim=-1)
    target_ids = target_ids.reshape(logits.shape[0], 1)
    # Extracts the ith score from the softmax output over the vocabulary (dim -1 of the logits)
    # where i is the value of the corresponding index in target_ids.
    return softmax_out.gather(-1, target_ids).squeeze(-1)
