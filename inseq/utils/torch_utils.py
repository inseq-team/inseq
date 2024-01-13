import logging
from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from jaxtyping import Int, Num
from torch.backends.cuda import is_built as is_cuda_built
from torch.backends.mps import is_available as is_mps_available
from torch.backends.mps import is_built as is_mps_built
from torch.cuda import is_available as is_cuda_available

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
    attributions: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    norm_dim: int = 0,
    norm_ord: int = 1,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
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
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
    spans: Sequence[Tuple[int, int]],
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


def get_front_padding(t: torch.Tensor, pad: int = 0, dim: int = 1) -> List[int]:
    """Given a tensor of shape (batch, seq_len) of ids, return a list of length batch containing
    the number of padding tokens at the beginning of each sequence.
    """
    return (t != pad).int().argmax(dim).tolist()


def get_sequences_from_batched_steps(
    bsteps: List[torch.Tensor], padding_dims: Sequence[int] = [], stack_dim: int = 2
) -> List[torch.Tensor]:
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
