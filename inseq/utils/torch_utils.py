import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Literal, Optional, Union

import torch
import torch.nn.functional as F
from jaxtyping import Int, Num
from torch import nn
from torch.backends.cuda import is_built as is_cuda_built
from torch.backends.mps import is_available as is_mps_available
from torch.backends.mps import is_built as is_mps_built
from torch.cuda import is_available as is_cuda_available

from .typing import OneOrMoreIndices

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

TORCH_BACKEND_DEVICE_MAP = {
    "cuda": (is_cuda_built, is_cuda_available),
    "mps": (is_mps_built, is_mps_available),
}


@torch.no_grad()
def remap_from_filtered(
    original_shape: tuple[int, ...],
    mask: Int[torch.Tensor, "batch_size 1"],
    filtered: Num[torch.Tensor, "filtered_batch_size"],
) -> Num[torch.Tensor, "batch_size"]:
    index = mask.squeeze(-1).nonzero().reshape(-1, 1)
    while index.ndim < filtered.ndim:
        index = index.unsqueeze(-1)
    index = index.expand_as(filtered)
    new_source = torch.ones(original_shape, dtype=filtered.dtype, device=filtered.device) * float("nan")
    return new_source.scatter(0, index, filtered)


def normalize(
    attributions: Union[torch.Tensor, tuple[torch.Tensor, ...]],
    norm_dim: int = 0,
    norm_ord: int = 1,
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    multi_input = False
    if isinstance(attributions, tuple):
        orig_sizes = [a.shape[norm_dim] for a in attributions]
        attributions = torch.cat(attributions, dim=norm_dim)
        multi_input = True
    nan_mask = attributions.isnan()
    attributions[nan_mask] = 0.0
    attributions = F.normalize(attributions, p=norm_ord, dim=norm_dim)
    attributions[nan_mask] = float("nan")
    if multi_input:
        return tuple(attributions.split(orig_sizes, dim=norm_dim))
    return attributions


def top_p_logits_mask(logits: torch.Tensor, top_p: float, min_tokens_to_keep: int) -> torch.Tensor:
    """Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py"""
    # Compute cumulative probabilities of sorted tokens
    if top_p < 0 or top_p > 1.0:
        raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)

    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return indices_to_remove


def top_k_logits_mask(logits: torch.Tensor, top_k: int, min_tokens_to_keep: int) -> torch.Tensor:
    """Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py"""
    top_k = max(top_k, min_tokens_to_keep)
    return logits < logits.topk(top_k).values[..., -1, None]


def get_logits_from_filter_strategy(
    filter_strategy: Union[Literal["original"], Literal["contrast"], Literal["merged"]],
    original_logits: torch.Tensor,
    contrast_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if filter_strategy == "original":
        return original_logits
    elif filter_strategy == "contrast":
        return contrast_logits
    elif filter_strategy == "merged":
        return original_logits + contrast_logits


def filter_logits(
    original_logits: torch.Tensor,
    contrast_logits: Optional[torch.Tensor] = None,
    top_p: float = 1.0,
    top_k: int = 0,
    min_tokens_to_keep: int = 1,
    filter_strategy: Union[Literal["original"], Literal["contrast"], Literal["merged"], None] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Applies top-k and top-p filtering to logits, and optionally to an additional set of contrastive logits."""
    if top_k > original_logits.size(-1) or top_k < 0:
        raise ValueError(f"`top_k` has to be a positive integer < {original_logits.size(-1)}, but is {top_k}")
    if filter_strategy and filter_strategy != "original" and contrast_logits is None:
        raise ValueError(f"`filter_strategy` {filter_strategy} can only be used if `contrast_logits` is provided")
    if not filter_strategy:
        if contrast_logits is None:
            filter_strategy = "original"
        else:
            filter_strategy = "merged"
    if top_p < 1.0:
        indices_to_remove = top_p_logits_mask(
            get_logits_from_filter_strategy(filter_strategy, original_logits, contrast_logits),
            top_p,
            min_tokens_to_keep,
        )
        original_logits = original_logits.masked_fill(indices_to_remove, float("-inf"))
        if contrast_logits is not None:
            contrast_logits = contrast_logits.masked_fill(indices_to_remove, float("-inf"))
    if top_k > 0:
        indices_to_remove = top_k_logits_mask(
            get_logits_from_filter_strategy(filter_strategy, original_logits, contrast_logits),
            top_k,
            min_tokens_to_keep,
        )
        original_logits = original_logits.masked_fill(indices_to_remove, float("-inf"))
        if contrast_logits is not None:
            contrast_logits = contrast_logits.masked_fill(indices_to_remove, float("-inf"))
    if contrast_logits is not None:
        return original_logits, contrast_logits
    return original_logits


def euclidean_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    """Compute the Euclidean distance between two points."""
    return (vec_a - vec_b).pow(2).sum(-1).sqrt()


def aggregate_contiguous(
    t: torch.Tensor,
    spans: Sequence[tuple[int, int]],
    aggregate_fn: Optional[Callable] = None,
    aggregate_dim: int = 0,
):
    """Given a tensor, aggregate contiguous spans of the tensor along a given dimension using the provided
    aggregation function. If no aggregation function is provided, the mean is used.

    Args:
        t: Tensor to aggregate
        spans: Sequence of (start, end) tuples indicating contiguous spans to aggregate
        aggregate_fn: Aggregation function to use. If None, torch.mean is used.
        aggregate_dim: Dimension to aggregate along. Default is 0.
    """
    if not spans:
        return t
    if aggregate_fn is None:
        aggregate_fn = torch.mean
    if aggregate_dim > t.ndim:
        raise ValueError(f"aggregate_dim {aggregate_dim} is greater than tensor dimension {t.ndim}")
    if aggregate_dim != 0:
        t = t.transpose(aggregate_dim, 0)
    slices = []
    base_val = 0
    for start, end in spans:
        if start > base_val:
            slices.append(t[base_val:start, ...])
        slices.append(aggregate_fn(t[start:end, ...], dim=0).unsqueeze(0))
        base_val = end
    if base_val < t.shape[0]:
        slices.append(t[base_val:, ...])
    out_cat = torch.cat(slices, dim=0)
    if aggregate_dim != 0:
        out_cat = out_cat.transpose(aggregate_dim, 0)
    return out_cat


def get_front_padding(t: torch.Tensor, pad: int = 0, dim: int = 1) -> list[int]:
    """Given a tensor of shape (batch, seq_len) of ids, return a list of length batch containing
    the number of padding tokens at the beginning of each sequence.
    """
    return (t != pad).int().argmax(dim).tolist()


def get_sequences_from_batched_steps(
    bsteps: list[torch.Tensor], padding_dims: Sequence[int] = [], stack_dim: int = 2
) -> list[torch.Tensor]:
    """Given a sequence of batched step tensors of shape (batch_size, seq_len, ...) builds a sequence
    of tensors of shape (seq_len, ...) where each resulting tensor is the aggregation
    across batch steps for every batch element.

    Source attribution shape: (batch_size, source_seq_len, ...)
    Target attribution shape: (batch_size, target_seq_len, ...)
    Step scores shape: (batch_size)
    Sequence scores shape: (batch_size, source/target_seq_len, ...)

    Input tensors will be padded with nans up to max length in non-uniform dimensions to allow for stacking.
    """
    bsteps_num_dims = bsteps[0].ndim
    if stack_dim > bsteps_num_dims:
        raise ValueError(f"Stack dimension {stack_dim} is greater than tensor dimension {bsteps_num_dims}")
    if not padding_dims:
        sequences = torch.stack(bsteps, dim=stack_dim).split(1, dim=0)
        return [seq.squeeze(0) for seq in sequences]
    for dim in padding_dims:
        if dim >= bsteps_num_dims:
            raise ValueError(f"Padding dimension {dim} is greater than tensor dimension {bsteps_num_dims}")
    padding_dims = set(padding_dims)
    max_dims = tuple(max([bstep.shape[dim] for bstep in bsteps]) for dim in padding_dims)
    for bstep_idx, bstep in enumerate(bsteps):
        for curr_dim, max_dim in zip(padding_dims, max_dims):
            bstep_dim = bstep.shape[curr_dim]
            if bstep_dim < max_dim:
                # Pad the end of curr_dim with nans
                pad_shape = (0,) * ((bsteps_num_dims - curr_dim) * 2 - 1) + (max_dim - bstep_dim,)
                padded_bstep = F.pad(bstep, pad=pad_shape, mode="constant", value=float("nan"))
                bsteps[bstep_idx] = padded_bstep
    sequences = torch.stack(bsteps, dim=stack_dim).split(1, dim=0)
    return [seq.squeeze(0) for seq in sequences]


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


def validate_indices(
    scores: torch.Tensor,
    dim: int = -1,
    indices: Optional[OneOrMoreIndices] = None,
) -> OneOrMoreIndices:
    """Validates a set of indices for a given dimension of a tensor of scores. Supports single indices, spans and lists
    of indices, including negative indices to specify positions relative to the end of the tensor.

    Args:
        scores (torch.Tensor): The tensor of scores.
        dim (int, optional): The dimension of the tensor that will be indexed. Defaults to -1.
        indices (Union[int, tuple[int, int], list[int], None], optional):
            - If an integer, it is interpreted as a single index for the dimension.
            - If a tuple of two integers, it is interpreted as a span of indices for the dimension.
            - If a list of integers, it is interpreted as a list of individual indices for the dimension.

    Returns:
        ``Union[int, tuple[int, int], list[int]]``: The validated list of positive indices for indexing the dimension.
    """
    if dim >= scores.ndim:
        raise IndexError(f"Dimension {dim} is greater than tensor dimension {scores.ndim}")
    n_units = scores.shape[dim]
    if not isinstance(indices, (int, tuple, list)) and indices is not None:
        raise TypeError(
            "Indices must be an integer, a (start, end) tuple of indices representing a span, a list of individual"
            " indices or a single index."
        )
    if hasattr(indices, "__iter__"):
        if len(indices) == 0:
            raise RuntimeError("An empty sequence of indices is not allowed.")
        if len(indices) == 1:
            indices = indices[0]

    if isinstance(indices, int):
        if indices not in range(-n_units, n_units):
            raise IndexError(f"Index out of range. Scores only have {n_units} units.")
        indices = indices if indices >= 0 else n_units + indices
        return torch.tensor(indices)
    else:
        if indices is None:
            indices = (0, n_units)
            logger.info("No indices specified. Using all indices by default.")

        # Convert negative indices to positive indices
        if hasattr(indices, "__iter__"):
            indices = type(indices)([h_idx if h_idx >= 0 else n_units + h_idx for h_idx in indices])
        if not hasattr(indices, "__iter__") or (
            len(indices) == 2 and isinstance(indices, tuple) and indices[0] >= indices[1]
        ):
            raise RuntimeError(
                "A (start, end) tuple of indices representing a span, a list of individual indices"
                " or a single index must be specified."
            )
        max_idx_val = n_units if isinstance(indices, list) else n_units + 1
        if not all(h in range(-n_units, max_idx_val) for h in indices):
            raise IndexError(f"One or more index out of range. Scores only have {n_units} units.")
        if len(set(indices)) != len(indices):
            raise IndexError("Duplicate indices are not allowed.")
        if isinstance(indices, tuple):
            return torch.arange(indices[0], indices[1])
        else:
            return torch.tensor(indices)


def pad_with_nan(t: torch.Tensor, dim: int, pad_size: int, front: bool = False) -> torch.Tensor:
    """Utility to pad a tensor with nan values along a given dimension."""
    nan_tensor = torch.ones(
        *t.shape[:dim],
        pad_size,
        *t.shape[dim + 1 :],
        device=t.device,
    ) * float("nan")
    if front:
        return torch.cat([nan_tensor, t], dim=dim)
    return torch.cat([t, nan_tensor], dim=dim)
