from typing import Union

import torch
import torch.nn.functional as F

from ....utils import normalize as normalize_fn
from ....utils.typing import (
    MultiUnitScoreTensor,
    ScoreTensor,
)


def _check_matrix_shape(
    scores: Union[MultiUnitScoreTensor, tuple[MultiUnitScoreTensor, MultiUnitScoreTensor, MultiUnitScoreTensor]],
) -> None:
    """Checks that the shape of the provided scores is compatible with the rollout aggregation method."""

    def fix_target_scores(target_scores: MultiUnitScoreTensor) -> MultiUnitScoreTensor:
        has_prefix_target = False
        if target_scores.size(-2) - target_scores.size(-1) == 1:
            target_scores = torch.cat([torch.zeros_like(target_scores[..., -1])[..., None], target_scores], dim=-1)
            target_scores[..., 0, 0] = 1.0
            has_prefix_target = True
        if target_scores.size(-1) != target_scores.size(-2):
            raise ValueError(
                "Expected scores to be a tensor of shape (T, T) but got shape "
                f"{target_scores.size(-2), target_scores.size(-1)}. {msg}"
            )
        target_scores[target_scores.isnan()] = 0.0
        return target_scores, has_prefix_target

    msg = (
        "This can be due to a non-zero starting index used in generation, which is not supported by the rollout "
        "aggregation method. Use attribute_full_target=True in model.attribute to attribute the full target sequence."
    )
    if isinstance(scores, tuple):
        source_scores, cross_scores, target_scores = scores
        dim0, dim1 = -2, -1
        source_dim = source_scores.size(dim0)
        target_dim = target_scores.size(dim0)
        try:
            assert source_scores.size(dim1) == source_dim  # source scores S x S
            assert cross_scores.size(dim0) == source_dim and cross_scores.size(dim1) == target_dim  # x-scores S x T
            assert target_scores.size(dim1) == target_dim  # target scores T x T
        except AssertionError as e:
            raise ValueError(
                "Expected scores to be a tuple of tensors of shape (S, S), (S, T), (T, T) but got shapes "
                f"{source_dim, source_scores.size(dim1)}, {cross_scores.size(dim0), cross_scores.size(dim1)}, "
                f"{target_dim, target_scores.size(dim1)}. {msg}"
            ) from e
        target_scores, has_prefix_target = fix_target_scores(target_scores)
        return (source_scores, cross_scores, target_scores), has_prefix_target
    else:
        return fix_target_scores(scores)


def _rollout_single(
    scores: MultiUnitScoreTensor,
    normalize: bool = False,
) -> MultiUnitScoreTensor:
    """Performs rollout aggregation by `Abnar and Zuidema (2020) <https://aclanthology.org/2020.acl-main.385/>`__
    This is a helper function used in :func:`~inseq.attr.feat.ops.rollout` to rollout a single layer stack.
    """
    rollout_scores = torch.zeros_like(scores)
    rollout_scores[:, 0, ...] = scores[:, 0, ...]
    for i in range(1, scores.size(1)):
        # Rollout scores at layer i by matmul them with the scores at layer i-1
        layer_rollout_scores = scores[:, i, ...] @ rollout_scores[:, i - 1, ...]
        if normalize:
            rollout_scores[:, i, ...] = F.normalize(layer_rollout_scores, p=1, dim=-1)
        else:
            rollout_scores[:, i, ...] = layer_rollout_scores
    return rollout_scores


def _rollout_joint(
    final_source_scores: ScoreTensor,
    cross_scores: MultiUnitScoreTensor,
    target_scores: MultiUnitScoreTensor,
) -> tuple[ScoreTensor, ScoreTensor]:
    """Performs the rollout aggregation adapted for an encoder-decoder architecture with cross-importance scores."""
    target_scores = (target_scores.mT * cross_scores[..., -1, :]).mT
    joint_source_cross_scores = torch.einsum("bl...ij, b...jk -> bl...ik", cross_scores, final_source_scores)
    source_rollout_scores = torch.zeros_like(joint_source_cross_scores)
    source_rollout_scores[:, 0, ...] = joint_source_cross_scores[:, 0, ...]
    target_rollout_scores = torch.zeros_like(target_scores)
    target_rollout_scores[:, 0, ...] = target_scores[:, 0, ...]
    for i in range(1, target_scores.size(1)):
        # Target scores x previous cross rollout scores
        source_rollout_scores[:, i, ...] = (
            target_scores[:, i, ...] @ source_rollout_scores[:, i - 1, ...]
        ) + joint_source_cross_scores[:, i, ...]
        # Target scores x previous target rollout scores
        target_rollout_scores[:, i, ...] = target_scores[:, i, ...] @ target_rollout_scores[:, i - 1, ...]
    # Normalize scores across source and target
    source_rollout_scores, target_rollout_scores = normalize_fn(
        (source_rollout_scores, target_rollout_scores), cat_dim=-1
    )
    return source_rollout_scores, target_rollout_scores


def rollout_fn(
    scores: Union[MultiUnitScoreTensor, tuple[MultiUnitScoreTensor, MultiUnitScoreTensor, MultiUnitScoreTensor]],
    dim: int = 1,
) -> Union[ScoreTensor, tuple[ScoreTensor, ScoreTensor]]:
    """Reference implementations:
    * `samiraabnar/attention-flow
        <https://github.com/samiraabnar/attention_flow/blob/master/attention_graph_util.py#L104>`__
    * `mt-upc/transformer-contributions-nmt
        <https://github.com/mt-upc/transformer-contributions-nmt/blob/main/wrappers/transformer_wrapper.py#L506>`__.

    Args:
        scores (:obj:`torch.Tensor` or :obj:`tuple(torch.Tensor, torch.Tensor, torch.Tensor)`):
            Tensor of shape `(num_layers, ...)`, or a tuple of tensors of the same shape containing the
            scores computed for different layers. If a tuple is passed, rollout will be performed assuming tensors are
            (source_scores, cross_scores, target_scores) produced by an Transformer-like encoder-decoder architecture
            (i.e. rolled-out importance of the source in the encoder is modulated by cross_scores at every layer of the
            decoder). For an encoder-decoder architecture, the rollout procedure follows the procedure described by
            `Ferrando et al. (2022) <https://aclanthology.org/2022.emnlp-main.599/>`__.
        dim (:obj:`int`, `optional`, defaults to 1): The dimension along which to perform the rollout aggregation.

    Returns:
        :obj:`torch.Tensor` or :obj:`tuple(torch.Tensor, torch.Tensor)`:
            An aggregated score tensor of shape `(batch_size, ...)`, or a tuple of tensors of the same shape containing
            the scores aggregated using rollout until the topmost provided layer (e.g. for ``layers=[1,2,4]`` the
            rollout is done skipping layer 3, and only rolled out scores at layer 4 are returned). If encoder-decoder
            rollout is performed, a tuple of tensors ``(source_scores, target_scores)``.
    """
    squeeze_batch_dim = False
    if isinstance(scores, tuple):
        if dim < 0:
            dim = scores[0].ndim + dim
        if scores[0].ndim < 4:
            scores = tuple(t.unsqueeze(0) for t in scores)
            squeeze_batch_dim = True
            dim += 1
        if dim != 1:
            swap_dim = dim if dim < 2 else dim + 1
            scores = tuple(s[:, None, ...].transpose(swap_dim, 1).squeeze(swap_dim) for s in scores)
        scores, has_target_prefix = _check_matrix_shape(scores)
        source_scores, cross_scores, target_scores = scores

        # Get rolled out scores of encoder last layer with respect to source input
        final_source_scores = _rollout_single(source_scores.mT)[:, -1, ...].mT

        source_rollout_scores, target_rollout_scores = _rollout_joint(final_source_scores, cross_scores, target_scores)
        if has_target_prefix:
            target_rollout_scores = target_rollout_scores[..., 1:]
        source_rollout_scores = source_rollout_scores[:, -1, ...].unsqueeze(1)
        target_rollout_scores = target_rollout_scores[:, -1, ...].unsqueeze(1)
        if dim != 1:
            source_rollout_scores = source_rollout_scores.transpose(1, dim)
            target_rollout_scores = target_rollout_scores.transpose(1, dim)
        source_rollout_scores = source_rollout_scores.squeeze(dim)
        target_rollout_scores = target_rollout_scores.squeeze(dim)
        if squeeze_batch_dim:
            source_rollout_scores = source_rollout_scores.squeeze(0)
            target_rollout_scores = target_rollout_scores.squeeze(0)
        return source_rollout_scores, target_rollout_scores
    else:
        # Convert rollout dim to positive index to account for new dim insertions.
        if dim < 0:
            dim = scores.ndim + dim
        # Add batch dimension if not present. Assumed shape (batch_size, ...) with num_layers at position dim and at
        # least two dimensions representing scores that will be rolled out.
        if scores.ndim < 4:
            scores = scores[None, ...]
            squeeze_batch_dim = True
            dim += 1
        if dim != 1:
            scores = scores[:, None, ...]
            swap_dim = dim if dim < 2 else dim + 1
            scores = scores.transpose(swap_dim, 1).squeeze(swap_dim)
        scores, has_target_prefix = _check_matrix_shape(scores)
        target_rollout_scores = _rollout_single(scores.mT)[:, -1, ...].mT
        if has_target_prefix:
            target_rollout_scores = target_rollout_scores[..., 1:]
        if squeeze_batch_dim:
            target_rollout_scores = target_rollout_scores.squeeze(0)
        return target_rollout_scores
