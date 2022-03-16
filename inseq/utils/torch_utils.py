from typing import Any, Tuple, Union

import logging

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from .misc import pretty_tensor
from .typing import (
    AttributionOutputTensor,
    EmbeddingsTensor,
    FullLogitsTensor,
    TargetIdsTensor,
    TopProbabilitiesTensor,
)


logger = logging.getLogger(__name__)


@torch.no_grad()
def remap_from_filtered(
    original: TensorType["batch_size", Any],
    mask: TensorType["batch_size", 1, int],
    filtered: TensorType["filtered_batch_size", Any],
) -> TensorType["batch_size", Any]:
    index = mask.squeeze().nonzero().squeeze()
    while len(index.shape) < len(filtered.shape):
        index = index.unsqueeze(-1)
    index = index.expand_as(filtered)
    new_source = torch.ones_like(original, dtype=filtered.dtype) * float("nan")
    return new_source.scatter(0, index, filtered)


def sum_normalize_attributions(
    attributions: Union[EmbeddingsTensor, Tuple[EmbeddingsTensor, EmbeddingsTensor]],
) -> AttributionOutputTensor:
    """
    Sum and normalize tensors across dim_sum.
    The outcome is a matrix of unit row vectors.
    """
    concat = False
    if isinstance(attributions, tuple):
        concat = True
        orig_sizes = [a.shape[1] for a in attributions]
        attributions = torch.cat(attributions, dim=1)
    attributions = attributions.sum(dim=-1).squeeze(0)
    logger.debug(f"pre-norm attributions: {pretty_tensor(attributions)}")
    attributions = F.normalize(attributions, p=2, dim=-1)
    if len(attributions.shape) == 1:
        attributions = attributions.unsqueeze(0)
    if concat:
        return attributions.split(orig_sizes, dim=1)
    return attributions


def euclidean_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute the Euclidean distance between two points."""
    return (vec_a - vec_b).pow(2).sum(-1).sqrt()


@torch.no_grad()
def probits2probs(probits: FullLogitsTensor, target_ids: TargetIdsTensor) -> TopProbabilitiesTensor:
    """
    Compute the scores of the target_ids from the probits.
    The scores are computed as the probabilities of the target_ids after softmax.
    """
    target_ids = target_ids.reshape(probits.shape[0], 1)
    # Extracts the ith score from the softmax output over the vocabulary (dim -1 of the probits)
    # where i is the value of the corresponding index in target_ids.
    return probits.gather(-1, target_ids).squeeze(-1)
