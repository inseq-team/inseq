from typing import Tuple, Union

import torch
import torch.nn.functional as F

from ....utils import normalize_attributions
from ....utils.typing import (
    MultiUnitScoreTensor,
    ScoreTensor,
)


def _rollout_single(
    scores: MultiUnitScoreTensor,
) -> MultiUnitScoreTensor:
    """Performs rollout aggregation by `Abnar and Zuidema (2020) <https://aclanthology.org/2020.acl-main.385/>`__
    This is a helper function used in :func:`~inseq.attr.feat.ops.rollout` to rollout a single layer stack.
    """
    rollout_scores = torch.zeros_like(scores)
    rollout_scores[:, 0, ...] = scores[:, 0, ...]
    for i in range(1, scores.size(1)):
        # Rollout scores at layer i by matmul them with the scores at layer i-1
        layer_rollout_scores = scores[:, i, ...] @ rollout_scores[:, i - 1, ...]
        rollout_scores[:, i, ...] = F.normalize(layer_rollout_scores, p=1, dim=-1)
    return rollout_scores


def _rollout_joint(
    final_source_scores: ScoreTensor,
    cross_scores: MultiUnitScoreTensor,
    target_scores: MultiUnitScoreTensor,
) -> Tuple[ScoreTensor, ScoreTensor]:
    """Performs the rollout aggregation adapted for an encoder-decoder architecture with cross-importance scores."""
    target_scores = (target_scores.mT * cross_scores[..., None, :, -1]).mT
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
    source_rollout_scores, target_rollout_scores = normalize_attributions(
        (source_rollout_scores, target_rollout_scores), cat_dim=-1
    )
    return source_rollout_scores, target_rollout_scores


def rollout_fn(
    scores: Union[MultiUnitScoreTensor, Tuple[MultiUnitScoreTensor, MultiUnitScoreTensor, MultiUnitScoreTensor]],
    dim: int = 1,
) -> Union[ScoreTensor, Tuple[ScoreTensor, ScoreTensor]]:
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
    remove_padding = False
    if isinstance(scores, tuple):
        if dim < 0:
            dim = scores[0].ndim + dim
        if scores[0].ndim < 4:
            scores = tuple(t.unsqueeze(0) for t in scores)
            squeeze_batch_dim = True
        if dim != 1:
            source_scores, cross_scores, target_scores = tuple(t.transpose(dim, 1) for t in scores)
        else:
            source_scores, cross_scores, target_scores = scores

        # Get rolled out scores of encoder last layer with respect to source input
        source_scores = _rollout_single(source_scores)

        final_source_scores = source_scores[:, -1, ...]
        source_rollout_scores, target_rollout_scores = _rollout_joint(final_source_scores, cross_scores, target_scores)
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
        scores[scores.isnan()] = 0.0
        # If the matrix is not square (e.g. generating from a prefix), prepend zeros to make it square.
        if scores.size(-1) < scores.size(-2):
            pad_size = scores.size(-2) - scores.size(-1)
            scores = torch.cat([torch.zeros(*tuple(scores.shape[:-1]), pad_size), scores], dim=-1)
            remove_padding = True
        target_rollout_scores = _rollout_single(scores.mT)[:, -1, ...].mT
        if squeeze_batch_dim:
            target_rollout_scores = target_rollout_scores.squeeze(0)
        if remove_padding:
            target_rollout_scores = target_rollout_scores[..., pad_size:]
        return target_rollout_scores
