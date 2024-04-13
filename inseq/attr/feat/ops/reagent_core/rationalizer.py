import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
from jaxtyping import Int64
from typing_extensions import override

from .....utils.typing import IdsTensor, TargetIdsTensor
from .importance_score_evaluator import BaseImportanceScoreEvaluator


class BaseRationalizer(ABC):
    def __init__(self, importance_score_evaluator: BaseImportanceScoreEvaluator) -> None:
        super().__init__()
        self.importance_score_evaluator = importance_score_evaluator
        self.mean_importance_score = None

    @abstractmethod
    def __call__(
        self,
        input_ids: IdsTensor,
        target_id: TargetIdsTensor,
        decoder_input_ids: Optional[IdsTensor] = None,
        attribute_target: bool = False,
    ) -> Int64[torch.Tensor, "batch_size other_dims"]:
        """Compute rational of a sequence on a target

        Args:
            input_ids: The sequence [batch, sequence] (first dimension need to be 1)
            target_id: The target [batch]
            decoder_input_ids (optional): decoder input sequence for AutoModelForSeq2SeqLM [batch, sequence]
            attribute_target: whether attribute target for encoder-decoder models

        Return:
            pos_top_n: rational position in the sequence [batch, rational_size]

        """
        raise NotImplementedError()


class AggregateRationalizer(BaseRationalizer):
    """AggregateRationalizer"""

    @override
    def __init__(
        self,
        importance_score_evaluator: BaseImportanceScoreEvaluator,
        batch_size: int,
        overlap_threshold: int,
        overlap_strict_pos: bool = True,
        keep_top_n: int = 0,
        keep_ratio: float = 0,
    ) -> None:
        """Constructor

        Args:
            importance_score_evaluator: A ImportanceScoreEvaluator
            batch_size: Batch size for aggregate
            overlap_threshold: Overlap threshold of rational tokens within a batch
            overlap_strict_pos: Whether overlap strict to position ot not
            keep_top_n: If set to a value greater than 0, the top n tokens based on their importance score will be
                kept, and the rest will be flagged for replacement. If set to 0, the top n will be determined by
                ``keep_ratio``.
            keep_ratio: If ``keep_top_n`` is set to 0, this specifies the proportion of tokens to keep.
        """
        super().__init__(importance_score_evaluator)
        self.batch_size = batch_size
        self.overlap_threshold = overlap_threshold
        self.overlap_strict_pos = overlap_strict_pos
        self.keep_top_n = keep_top_n
        self.keep_ratio = keep_ratio
        assert overlap_strict_pos, "overlap_strict_pos = False is not supported yet"

    @override
    @torch.no_grad()
    def __call__(
        self,
        input_ids: IdsTensor,
        target_id: TargetIdsTensor,
        decoder_input_ids: Optional[IdsTensor] = None,
        attribute_target: bool = False,
    ) -> Int64[torch.Tensor, "batch_size other_dims"]:
        """Compute rational of a sequence on a target

        Args:
            input_ids: A tensor of ids of shape [batch, sequence_len]
            target_id: A tensor of predicted targets of size [batch]
            decoder_input_ids (optional): A tensor of ids representing the decoder input sequence for
                ``AutoModelForSeq2SeqLM``, with shape [batch, sequence_len]
            attribute_target: whether attribute target for encoder-decoder models

        Return:
            pos_top_n: rational position in the sequence [batch, rational_size]

        """
        assert input_ids.shape[0] == 1, "the first dimension of input (batch_size) need to be 1"
        batch_input_ids = input_ids.repeat(self.batch_size, 1)
        batch_decoder_input_ids = (
            decoder_input_ids.repeat(self.batch_size, 1) if decoder_input_ids is not None else None
        )
        batch_importance_score = self.importance_score_evaluator(
            batch_input_ids, target_id, batch_decoder_input_ids, attribute_target
        )
        importance_score_masked = batch_importance_score * torch.unsqueeze(
            self.importance_score_evaluator.stop_mask, -1
        )
        self.mean_importance_score = torch.sum(importance_score_masked, dim=0) / torch.sum(
            self.importance_score_evaluator.stop_mask
        )
        pos_sorted = torch.argsort(batch_importance_score, dim=-1, descending=True)
        top_n = int(math.ceil(self.keep_ratio * input_ids.shape[-1])) if not self.keep_top_n else self.keep_top_n
        pos_top_n = pos_sorted[:, :top_n]
        self.pos_top_n = pos_top_n
        if self.overlap_strict_pos:
            count_overlap = torch.bincount(pos_top_n.flatten(), minlength=input_ids.shape[1])
            pos_top_n_overlap = torch.unsqueeze(
                torch.nonzero(count_overlap >= self.overlap_threshold, as_tuple=True)[0], 0
            )
            return pos_top_n_overlap
        else:
            raise NotImplementedError("overlap_strict_pos = False not been supported yet")
            # TODO: Convert back to pos
            # token_id_top_n = input_ids[0, pos_top_n]
            # count_overlap = torch.bincount(token_id_top_n.flatten(), minlength=input_ids.shape[1])
            # _token_id_top_n_overlap = torch.unsqueeze(
            #     torch.nonzero(count_overlap >= self.overlap_threshold, as_tuple=True)[0], 0
            # )
