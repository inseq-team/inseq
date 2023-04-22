from typing import Tuple, Union

import torch

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
        rollout_scores[:, i, ...] = scores[:, i, ...] @ rollout_scores[:, i - 1, ...]
    return rollout_scores


def _rollout_joint(
    final_source_scores: ScoreTensor,
    cross_scores: MultiUnitScoreTensor,
    target_scores: MultiUnitScoreTensor,
) -> Tuple[MultiUnitScoreTensor, MultiUnitScoreTensor]:
    """Performs the rollout aggregation adapted for an encoder-decoder architecture with cross-importance scores."""
    target_scores = (target_scores.mT * cross_scores[..., None, :, -1]).mT
    joint_source_cross_scores = torch.einsum("blij, bjk -> blik", cross_scores, final_source_scores)
    source_rollout_scores = torch.zeros_like(joint_source_cross_scores)
    source_rollout_scores[:, 0, ...] = joint_source_cross_scores[:, 0, ...]
    target_rollout_scores = torch.zeros_like(target_scores)
    target_rollout_scores[:, 0, ...] = target_scores[:, 0, ...]
    for i in range(1, target_scores.size(1)):
        # Target scores x previous cross rollout scores
        source_rollout_scores[:, i, ...] = target_scores[:, i, ...] @ source_rollout_scores[:, i - 1, ...]
        # Target scores x previous target rollout scores
        target_rollout_scores[:, i, ...] = target_scores[:, i, ...] @ target_rollout_scores[:, i - 1, ...]
    # Normalize scores across source and target
    source_rollout_scores, target_rollout_scores = normalize_attributions(
        (source_rollout_scores, target_rollout_scores), cat_dim=-1, norm_dim=-1
    )
    return source_rollout_scores, target_rollout_scores


def rollout(
    scores: Union[MultiUnitScoreTensor, Tuple[MultiUnitScoreTensor, MultiUnitScoreTensor, MultiUnitScoreTensor]],
    add_residual: bool = False,
) -> Union[ScoreTensor, Tuple[ScoreTensor, ScoreTensor]]:
    """
    Reference implementations:
    * `samiraabnar/attention-flow
        <https://github.com/samiraabnar/attention_flow/blob/master/attention_graph_util.py#L104>`__
    * `mt-upc/transformer-contributions-nmt
        <https://github.com/mt-upc/transformer-contributions-nmt/blob/main/wrappers/transformer_wrapper.py#L506>`__

    Args:
        scores (:obj:`torch.Tensor` or :obj:`tuple(torch.Tensor, torch.Tensor, torch.Tensor)`):
            Tensor of shape `(batch_size, num_layers, ...)`, or a tuple of tensors of the same shape containing the
            scores computed for different layers. If a tuple is passed, rollout will be performed assuming tensors are
            (source_scores, cross_scores, target_scores) produced by an Transformer-like encoder-decoder architecture
            (i.e. rolled-out importance of the source in the encoder is modulated by cross_scores at every layer of the
            decoder). For an encoder-decoder architecture, the rollout procedure follows the procedure described by
            `Ferrando et al. (2022) <https://aclanthology.org/2022.emnlp-main.599/>`__.
        add_residual (:obj:`bool`):
            Whether to incorporate residual connection between the layers by adding an identity matrix and normalizing
            weights, as proposed by `Abnar and Zuidema (2020) <https://aclanthology.org/2020.acl-main.385/>`__.
            Defaults to False.

    Returns:
        :obj:`torch.Tensor` or :obj:`tuple(torch.Tensor, torch.Tensor)`:
            An aggregated score tensor of shape `(batch_size, ...)`, or a tuple of tensors of the same shape containing
            the scores aggregated using rollout until the topmost provided layer (e.g. for ``layers=[1,2,4]`` the
            rollout is done skipping layer 3, and only rolled out scores at layer 4 are returned). If encoder-decoder
            rollout is performed, a tuple of tensors ``(source_scores, target_scores)``.
    """
    if isinstance(scores, tuple):
        source_scores, cross_scores, target_scores = scores

        # Get rolled out scores of encoder last layer with respect to source input
        source_scores = _rollout_single(source_scores)

        final_source_scores = source_scores[:, -1, ...]
        source_rollout_scores, target_rollout_scores = _rollout_joint(final_source_scores, cross_scores, target_scores)
        return source_rollout_scores[:, -1, ...], target_rollout_scores[:, -1, ...]
    return _rollout_single(scores)[:, -1, ...]
