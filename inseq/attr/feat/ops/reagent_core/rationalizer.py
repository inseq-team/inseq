import math
from abc import ABC, abstractmethod

import torch
from typing_extensions import override

from .importance_score_evaluator import BaseImportanceScoreEvaluator


class BaseRationalizer(ABC):
    def __init__(self, importance_score_evaluator: BaseImportanceScoreEvaluator) -> None:
        super().__init__()

        self.importance_score_evaluator = importance_score_evaluator
        self.mean_important_score = None

    @abstractmethod
    def __call__(
        self, input_ids: torch.Tensor, target_id: torch.Tensor, decoder_input_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute rational of a sequence on a target

        Args:
            input_ids: The sequence [batch, sequence] (first dimension need to be 1)
            target_id: The target [batch]
            decoder_input_ids (optional): decoder input sequence for AutoModelForSeq2SeqLM [batch, sequence]

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
        top_n: float = 0,
        top_n_ratio: float = 0,
    ) -> None:
        """Constructor

        Args:
            importance_score_evaluator: A ImportanceScoreEvaluator
            batch_size: Batch size for aggregate
            overlap_threshold: Overlap threshold of rational tokens within a batch
            overlap_strict_pos: Whether overlap strict to position ot not
            top_n: Rational size
            top_n_ratio: Use ratio of sequence to define rational size

        """
        super().__init__(importance_score_evaluator)

        self.batch_size = batch_size
        self.overlap_threshold = overlap_threshold
        self.overlap_strict_pos = overlap_strict_pos
        self.top_n = top_n
        self.top_n_ratio = top_n_ratio

        assert overlap_strict_pos, "overlap_strict_pos = False not been supported yet"

    @override
    @torch.no_grad()
    def __call__(
        self, input_ids: torch.Tensor, target_id: torch.Tensor, decoder_input_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute rational of a sequence on a target

        Args:
            input_ids: The sequence [batch, sequence] (first dimension need to be 1)
            target_id: The target [batch]
            decoder_input_ids (optional): decoder input sequence for AutoModelForSeq2SeqLM [batch, sequence]

        Return:
            pos_top_n: rational position in the sequence [batch, rational_size]

        """
        assert input_ids.shape[0] == 1, "the first dimension of input (batch_size) need to be 1"

        batch_input_ids = input_ids.repeat(self.batch_size, 1)

        batch_importance_score = self.importance_score_evaluator(batch_input_ids, target_id, decoder_input_ids)

        important_score_masked = batch_importance_score * torch.unsqueeze(
            self.importance_score_evaluator.stop_mask, -1
        )
        self.mean_important_score = torch.sum(important_score_masked, dim=0) / torch.sum(
            self.importance_score_evaluator.stop_mask
        )

        pos_sorted = torch.argsort(batch_importance_score, dim=-1, descending=True)

        top_n = self.top_n

        if top_n == 0:
            top_n = int(math.ceil(self.top_n_ratio * input_ids.shape[-1]))

        pos_top_n = pos_sorted[:, :top_n]
        self.pos_top_n = pos_top_n

        if self.overlap_strict_pos:
            count_overlap = torch.bincount(pos_top_n.flatten(), minlength=input_ids.shape[1])
            pos_top_n_overlap = torch.unsqueeze(
                torch.nonzero(count_overlap >= self.overlap_threshold, as_tuple=True)[0], 0
            )
            return pos_top_n_overlap
        else:
            token_id_top_n = input_ids[0, pos_top_n]
            count_overlap = torch.bincount(token_id_top_n.flatten(), minlength=input_ids.shape[1])
            _token_id_top_n_overlap = torch.unsqueeze(
                torch.nonzero(count_overlap >= self.overlap_threshold, as_tuple=True)[0], 0
            )
            # TODO: Convert back to pos
            raise NotImplementedError("TODO")
