import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

from .....utils.typing import IdsTensor, MultipleScoresPerStepTensor, TargetIdsTensor
from .token_replacer import RankingTokenReplacer
from .token_sampler import TokenSampler


class StoppingConditionEvaluator(ABC):
    """Base class for Stopping Condition Evaluators"""

    @abstractmethod
    def __call__(
        self,
        input_ids: IdsTensor,
        target_id: TargetIdsTensor,
        importance_score: MultipleScoresPerStepTensor,
        decoder_input_ids: Optional[IdsTensor] = None,
        attribute_target: bool = False,
    ) -> TargetIdsTensor:
        """Evaluate stop condition according to the specified strategy.

        Args:
            input_ids: Input sequence [batch, sequence]
            target_id: Target token [batch]
            importance_score: Importance score of the input [batch, sequence]
            decoder_input_ids (optional): decoder input sequence for AutoModelForSeq2SeqLM [batch, sequence]
            attribute_target: whether attribute target for encoder-decoder models

        Return:
            Boolean flag per sequence signaling whether the stop condition was reached [batch]

        """
        raise NotImplementedError()


class TopKStoppingConditionEvaluator(StoppingConditionEvaluator):
    """
    Evaluator stopping when target exist among the top k predictions,
    while top n tokens based on importance_score are not been replaced.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        sampler: TokenSampler,
        top_k: int,
        keep_top_n: int = 0,
        keep_ratio: float = 0,
        invert_keep: bool = False,
    ) -> None:
        """Constructor for the TopKStoppingConditionEvaluator class.

        Args:
            model: A Huggingface AutoModelForCausalLM.
            sampler: A :class:`~inseq.attr.feat.ops.reagent_core.TokenSampler` object to sample replacement tokens.
            top_k: Top K predictions in which the target must be included in order to achieve the stopping condition.
            keep_top_n: If set to a value greater than 0, the top n tokens based on their importance score will be
                kept, and the rest will be flagged for replacement. If set to 0, the top n will be determined by
                ``keep_ratio``.
            keep_ratio: If ``keep_top_n`` is set to 0, this specifies the proportion of tokens to keep.
            invert_keep: If specified, the top tokens selected either via ``keep_top_n`` or ``keep_ratio`` will be
                replaced instead of being kept.
        """
        self.model = model
        self.top_k = top_k
        self.replacer = RankingTokenReplacer(sampler, keep_top_n, keep_ratio, invert_keep)

    def __call__(
        self,
        input_ids: IdsTensor,
        target_id: TargetIdsTensor,
        importance_score: MultipleScoresPerStepTensor,
        decoder_input_ids: Optional[IdsTensor] = None,
        attribute_target: bool = False,
    ) -> TargetIdsTensor:
        """Evaluate stop condition

        Args:
            input_ids: Input sequence [batch, sequence]
            target_id: Target token [batch]
            importance_score: Importance score of the input [batch, sequence]
            decoder_input_ids (optional): decoder input sequence for AutoModelForSeq2SeqLM [batch, sequence]
            attribute_target: whether attribute target for encoder-decoder models

        Return:
            Boolean flag per sequence signaling whether the stop condition was reached [batch]
        """
        # Replace tokens with low importance score and then inference \hat{y^{(e)}_{t+1}}
        self.replacer.set_score(importance_score)
        if not attribute_target:
            input_ids_replaced, mask_replacing = self.replacer(input_ids)
        else:
            ids_replaced, mask_replacing = self.replacer(torch.cat((input_ids, decoder_input_ids), 1))
            input_ids_replaced = ids_replaced[:, : input_ids.shape[1]]
            decoder_input_ids_replaced = ids_replaced[:, input_ids.shape[1] :]

        logging.debug(f"Replacing mask based on importance score -> { mask_replacing }")

        # Whether the result \hat{y^{(e)}_{t+1}} consistent with y_{t+1}
        assert not input_ids_replaced.requires_grad, "Error: auto-diff engine not disabled"
        with torch.no_grad():
            if decoder_input_ids is None:
                logits_replaced = self.model(input_ids_replaced)["logits"]
            elif not attribute_target:
                logits_replaced = self.model(input_ids=input_ids_replaced, decoder_input_ids=decoder_input_ids)[
                    "logits"
                ]
            else:
                logits_replaced = self.model(
                    input_ids=input_ids_replaced, decoder_input_ids=decoder_input_ids_replaced
                )["logits"]

        ids_prediction_sorted = torch.argsort(logits_replaced[:, -1, :], descending=True)
        ids_prediction_top_k = ids_prediction_sorted[:, : self.top_k]

        match_mask = ids_prediction_top_k == target_id
        match_hit = torch.sum(match_mask, dim=-1, dtype=torch.bool)

        # Stop flags for each sample in the batch
        return match_hit


class DummyStoppingConditionEvaluator(StoppingConditionEvaluator):
    """
    Stopping Condition Evaluator which stop when target exist in top k predictions,
    while top n tokens based on importance_score are not been replaced.
    """

    def __call__(self, input_ids: IdsTensor, **kwargs) -> TargetIdsTensor:
        """Evaluate stop condition

        Args:
            input_ids: Input sequence [batch, sequence]
            target_id: Target token [batch]
            importance_score: Importance score of the input [batch, sequence]
            attribute_target: whether attribute target for encoder-decoder models

        Return:
            Boolean flag per sequence signaling whether the stop condition was reached [batch]
        """
        return torch.ones([input_ids.shape[0]], dtype=torch.bool, device=input_ids.device)
