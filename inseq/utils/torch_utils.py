import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch.backends.cuda import is_built as is_cuda_built
from torch.backends.mps import is_available as is_mps_available
from torch.backends.mps import is_built as is_mps_built
from torch.cuda import is_available as is_cuda_available
from torch.linalg import vector_norm
from torchtyping import TensorType

from .typing import (
    GranularSequenceAttributionTensor,
    TokenSequenceAttributionTensor,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

TORCH_BACKEND_DEVICE_MAP = {
    "cuda": (is_cuda_built, is_cuda_available),
    "mps": (is_mps_built, is_mps_available),
}


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
    norm_dim: Optional[int] = 0,
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
    else:
        orig_sizes = [attributions.shape[cat_dim]]
    attributions = vector_norm(attributions, ord=2, dim=-1)
    if norm_dim is not None:
        # nansum is used to handle the target side sequence attribution case
        attributions = attributions / attributions.nansum(dim=norm_dim, keepdim=True)
    if len(attributions.shape) == 1:
        attributions = attributions.unsqueeze(0)
    if concat:
        attributions = attributions.split(orig_sizes, dim=cat_dim)
        return attributions[0], attributions[1]
    return attributions


def normalize_attributions(
    attributions: Union[
        TokenSequenceAttributionTensor, Tuple[TokenSequenceAttributionTensor, TokenSequenceAttributionTensor]
    ],
    cat_dim: int = 0,
    norm_dim: int = 0,
) -> TokenSequenceAttributionTensor:
    concat = False
    if isinstance(attributions, tuple):
        concat = True
        orig_sizes = [a.shape[cat_dim] for a in attributions]
        attributions = torch.cat(attributions, dim=cat_dim)
    else:
        orig_sizes = [attributions.shape[cat_dim]]
    # nansum is used to handle the target side sequence attribution case
    attributions = attributions / attributions.nansum(dim=norm_dim, keepdim=True)
    if len(attributions.shape) == 1:
        attributions = attributions.unsqueeze(0)
    if concat:
        attributions = attributions.split(orig_sizes, dim=cat_dim)
        return attributions[0], attributions[1]
    return attributions


def euclidean_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    """Compute the Euclidean distance between two points."""
    return (vec_a - vec_b).pow(2).sum(-1).sqrt()


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


def get_front_padding(t: torch.Tensor, pad: int = 0, dim: int = 1) -> List[int]:
    """Given a tensor of shape (batch, seq_len) of ids, return a list of length batch containing
    the number of padding tokens at the beginning of each sequence."""
    return (t != pad).int().argmax(dim).tolist()


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
            expanded_bsteps = []
            for bstep in bsteps:
                padded_bstep = torch.ones(
                    *bstep.shape[:dim],
                    max_dim - bstep.shape[dim],
                    *bstep.shape[dim + 1 :],  # noqa
                    dtype=bstep.dtype,
                    device=bstep.device,
                )
                padded_bstep = torch.cat([bstep, padded_bstep * float("nan")], dim=dim)
                expanded_bsteps.append(padded_bstep)
    else:
        expanded_bsteps = bsteps
    dim = 2 if len(bsteps[0].shape) > 1 else 1
    sequences = torch.stack(expanded_bsteps, dim=dim)
    sequences = sequences.split(1, dim=0)
    squeezed_sequences = [seq.squeeze(0) for seq in sequences]
    return squeezed_sequences


def check_device(device_name: str) -> bool:
    if device_name == "cpu":
        return True
    if device_name not in TORCH_BACKEND_DEVICE_MAP:
        raise ValueError(f"Unknown device {device_name}")
    available_fn, built_fn = TORCH_BACKEND_DEVICE_MAP[device_name]
    if not available_fn():
        raise ValueError(f"Cannot use {device_name} device, {device_name} is not available.")
    if not built_fn():
        raise ValueError(f"Current Pytorch distribution does not support {device_name} execution")
    return True


def get_default_device() -> str:
    if is_cuda_available() and is_cuda_built():
        return "cuda"
    elif is_mps_available() and is_mps_built():
        # temporarily fix mps-enabled devices on cpu until mps is able to support all operations this package needs
        # change this value on your own risk as it might break things depending on the attribution functions used
        return "cpu"
    else:
        return "cpu"
