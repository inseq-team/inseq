from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

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


def logits2prob(logits: FullLogitsTensor, target_ids: TargetIdsTensor) -> SingleScorePerStepTensor:
    """
    Compute the probabilty of target_ids from the logits.
    """
    target_ids = target_ids.reshape(logits.shape[0], 1)
    logits = torch.softmax(logits, dim=-1)
    # Extracts the ith score from the softmax output over the vocabulary (dim -1 of the logits)
    # where i is the value of the corresponding index in target_ids.
    return logits.gather(-1, target_ids).squeeze(-1)


def logits2ent(logits: FullLogitsTensor, target_ids: TargetIdsTensor) -> SingleScorePerStepTensor:
    """
    Compute the entropy of the outputs from the logits.
    Target id is not used in the computation, but kept for consistency with the other functions.
    """
    out = torch.distributions.Categorical(logits=logits).entropy()
    if len(out.shape) > 1:
        out = out.squeeze(-1)
    return out


def logits2ce(logits: FullLogitsTensor, target_ids: TargetIdsTensor) -> SingleScorePerStepTensor:
    """
    Compute the cross entropy between the target_ids and the logits.
    See: https://github.com/ZurichNLP/nmtscore/blob/master/src/nmtscore/models/m2m100.py#L99
    """
    return -torch.log2(logits2prob(logits, target_ids))


def logits2ppl(logits: FullLogitsTensor, target_ids: TargetIdsTensor) -> SingleScorePerStepTensor:
    """
    Compute perplexity of the target_ids from the logits.
    Perplexity is the weighted branching factor. If we have a perplexity of 100,
    it means that whenever the model is trying to guess the next word it is as
    confused as if it had to pick between 100 words.
    Reference: https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/
    """
    return 2 ** logits2ce(logits, target_ids)


def aggregate_contiguous(
    t: torch.Tensor,
    spans: Sequence[Tuple[int, int]],
    aggregate_fn: Optional[Callable] = None,
    aggregate_dim: int = 1,
):
    if not spans:
        return t
    if aggregate_fn is None:
        aggregate_fn = torch.mean
    while len(t.shape) < 2:
        t = t.unsqueeze(-1)
    t = t.transpose(aggregate_dim, 1)
    slices = []
    base_val = 0
    for start, end in spans:
        slices.append(t[:, base_val:start])
        slices.append(aggregate_fn(t[:, start:end]))
        base_val = end
    slices.append(t[:, base_val:])
    out_cat = torch.cat(slices, dim=1).transpose(1, aggregate_dim)
    if 1 in out_cat.shape:
        out_cat = out_cat.transpose(1, 0).squeeze(0)
    return out_cat


def abs_max(t: torch.Tensor) -> torch.Tensor:
    return t.gather(1, t.abs().argmax(dim=1).unsqueeze(1))


def prod_fn(t: torch.Tensor) -> torch.Tensor:
    return t.prod(dim=1, keepdim=True)


def sum_fn(t: torch.Tensor) -> torch.Tensor:
    return t.sum(dim=1, keepdim=True)


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
