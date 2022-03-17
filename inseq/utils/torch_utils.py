from typing import Any, List, Optional, Sequence, Tuple, Union

import logging

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from .typing import (
    FullLogitsTensor,
    GranularSequenceAttributionTensor,
    SingleScorePerStepTensor,
    TargetIdsTensor,
    TokenSequenceAttributionTensor,
)


logger = logging.getLogger(__name__)


@torch.no_grad()
def remap_from_filtered(
    original_shape: Tuple[int, ...],
    mask: TensorType["batch_size", 1, int],
    filtered: TensorType["filtered_batch_size", Any],
) -> TensorType["batch_size", Any]:
    index = mask.squeeze(-1).nonzero().reshape(-1, 1)
    while len(index.shape) < len(filtered.shape):
        index = index.unsqueeze(-1)
    index = index.expand_as(filtered)
    new_source = torch.ones(original_shape, dtype=filtered.dtype, device=filtered.device) * float("nan")
    return new_source.scatter(0, index, filtered)


def sum_normalize_attributions(
    attributions: Union[
        GranularSequenceAttributionTensor, Tuple[GranularSequenceAttributionTensor, GranularSequenceAttributionTensor]
    ],
    cat_dim: int = 0,
    norm_dim: int = 0,
) -> TokenSequenceAttributionTensor:
    """
    Sum and normalize tensors across dim_sum.
    The outcome is a matrix of unit row vectors.
    """
    concat = False
    if isinstance(attributions, tuple):
        concat = True
        orig_sizes = [a.shape[cat_dim] for a in attributions]
        attributions = torch.cat(attributions, dim=cat_dim)
    # nansum is used to handle the target side sequence attribution case
    attributions = torch.nansum(attributions, dim=-1).squeeze(0)
    attributions = F.normalize(attributions, p=2, dim=norm_dim)
    if len(attributions.shape) == 1:
        attributions = attributions.unsqueeze(0)
    if concat:
        attributions = attributions.split(orig_sizes, dim=cat_dim)
        return attributions[0], attributions[1] + torch.tril(torch.ones_like(attributions[1]) * float("nan"))
    return attributions


def euclidean_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute the Euclidean distance between two points."""
    return (vec_a - vec_b).pow(2).sum(-1).sqrt()


@torch.no_grad()
def probits2probs(probits: FullLogitsTensor, target_ids: TargetIdsTensor) -> SingleScorePerStepTensor:
    """
    Compute the scores of the target_ids from the probits.
    The scores are computed as the probabilities of the target_ids after softmax.
    """
    target_ids = target_ids.reshape(probits.shape[0], 1)
    # Extracts the ith score from the softmax output over the vocabulary (dim -1 of the probits)
    # where i is the value of the corresponding index in target_ids.
    return probits.gather(-1, target_ids).squeeze(-1)


def get_sequences_from_batched_steps(
    bsteps: List[torch.Tensor], pad_dims: Optional[Sequence[int]] = None
) -> List[torch.Tensor]:
    """
    Given a sequence of batched step tensors of shape (batch_size, ...) builds a sequence
    of tensors of shape (len(sequence), ...) where each resulting tensor is the aggregation
    across batch steps for every batch element.

    If pad_dims is not None, the input tensors will be padded with nans up to max length in
    the specified dimensions to allow for stacking.
    """
    if pad_dims:
        for dim in pad_dims:
            max_dim = max(bstep.shape[dim] for bstep in bsteps)
            bsteps = [
                torch.cat(
                    [
                        bstep,
                        torch.ones(
                            *bstep.shape[:dim],
                            max_dim - bstep.shape[dim],
                            *bstep.shape[dim + 1 :],  # noqa
                            dtype=bstep.dtype,
                            device=bstep.device,
                        )
                        * float("nan"),
                    ],
                    dim=dim,
                )
                for bstep in bsteps
            ]
    if len(bsteps[0].shape) > 1:
        return [t.squeeze() for t in torch.stack(bsteps, dim=2).split(1, dim=0)]
    else:
        return [t.squeeze() for t in torch.stack(bsteps, dim=1).split(1, dim=0)]
