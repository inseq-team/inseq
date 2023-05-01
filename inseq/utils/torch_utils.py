import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.backends.cuda import is_built as is_cuda_built
from torch.backends.mps import is_available as is_mps_available
from torch.backends.mps import is_built as is_mps_built
from torch.cuda import is_available as is_cuda_available
from torchtyping import TensorType

from .typing import TokenSequenceAttributionTensor

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
    while index.ndim < filtered.ndim:
        index = index.unsqueeze(-1)
    index = index.expand_as(filtered)
    new_source = torch.ones(original_shape, dtype=filtered.dtype, device=filtered.device) * float("nan")
    return new_source.scatter(0, index, filtered)


def normalize_attributions(
    attributions: Union[
        TokenSequenceAttributionTensor, Tuple[TokenSequenceAttributionTensor, TokenSequenceAttributionTensor]
    ],
    cat_dim: int = 0,
) -> TokenSequenceAttributionTensor:
    concat = False
    if isinstance(attributions, tuple):
        concat = True
        orig_sizes = [a.shape[cat_dim] for a in attributions]
        attributions = torch.cat(attributions, dim=cat_dim)
    else:
        orig_sizes = [attributions.shape[cat_dim]]
    # nansum is used to handle the target side sequence attribution case
    attributions = attributions / attributions.nansum(dim=cat_dim, keepdim=True)
    if attributions.ndim == 1:
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
    while t.ndim < 2:
        t = t.unsqueeze(-1)
    t = t.transpose(aggregate_dim, 1)
    slices = []
    base_val = 0
    for start, end in spans:
        slices.append(t[:, base_val:start])
        slices.append(aggregate_fn(t[:, start:end], dim=1).unsqueeze(1))
        base_val = end
    slices.append(t[:, base_val:])
    out_cat = torch.cat(slices, dim=1).transpose(1, aggregate_dim)
    if 1 in out_cat.shape:
        out_cat = out_cat.transpose(1, 0).squeeze(0)
    return out_cat


def get_front_padding(t: torch.Tensor, pad: int = 0, dim: int = 1) -> List[int]:
    """Given a tensor of shape (batch, seq_len) of ids, return a list of length batch containing
    the number of padding tokens at the beginning of each sequence.
    """
    return (t != pad).int().argmax(dim).tolist()


def get_sequences_from_batched_steps(bsteps: List[torch.Tensor]) -> List[torch.Tensor]:
    """Given a sequence of batched step tensors of shape (batch_size, ...) builds a sequence
    of tensors of shape (len(sequence), ...) where each resulting tensor is the aggregation
    across batch steps for every batch element.

    Input tensors will be padded with nans up to max length in non-uniform dimensions to allow for stacking.
    """
    dim_ranges = {dim: [bstep.shape[dim] for bstep in bsteps] for dim in range(bsteps[0].ndim)}
    for dim, dim_range in dim_ranges.items():
        # If dimension grows across batch steps, it will be padded
        if max(dim_range) > min(dim_range):
            for bstep_idx, bstep in enumerate(bsteps):
                padded_bstep = torch.ones(
                    *bstep.shape[:dim],
                    max(dim_range) - bstep.shape[dim],
                    *bstep.shape[dim + 1 :],  # noqa
                    dtype=bstep.dtype,
                    device=bstep.device,
                )
                padded_bstep = torch.cat([bstep, padded_bstep * float("nan")], dim=dim)
                bsteps[bstep_idx] = padded_bstep
    dim = 2 if bsteps[0].ndim > 1 else 1
    sequences = torch.stack(bsteps, dim=dim)
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


def find_block_stack(module):
    """Recursively searches for the first instance of a `nn.ModuleList` submodule within a given `torch.nn.Module`.

    Args:
        module (:obj:`torch.nn.Module`): A Pytorch :obj:`nn.Module` object.

    Returns:
        :obj:`torch.nn.ModuleList`: The first instance of a :obj:`nn.Module` submodule found within the given object.
        None: If no `nn.ModuleList` submodule is found within the given `nn.Module` object.
    """
    # Check if the current module is an instance of nn.ModuleList
    if isinstance(module, nn.ModuleList):
        return module

    # Recursively search for nn.ModuleList in the submodules of the current module
    for submodule in module.children():
        module_list = find_block_stack(submodule)
        if module_list is not None:
            return module_list

    # If nn.ModuleList is not found in any submodules, return None
    return None
