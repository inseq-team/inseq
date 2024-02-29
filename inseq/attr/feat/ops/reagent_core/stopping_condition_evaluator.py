import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from jaxtyping import Int64
from transformers import AutoModelWithLMHead, AutoTokenizer
from typing_extensions import override

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
    ) -> Int64[torch.Tensor, "batch_size"]:
        """Evaluate stop condition according to the specified strategy.

        Args:
            input_ids: Input sequence [batch, sequence]
            target_id: Target token [batch]
            importance_score: Importance score of the input [batch, sequence]
            decoder_input_ids (optional): decoder input sequence for AutoModelForSeq2SeqLM [batch, sequence]

        Return:
            Whether the stop condition achieved [batch]

        """
        raise NotImplementedError()


class TopKStoppingConditionEvaluator(StoppingConditionEvaluator):
    """
    Stopping Condition Evaluator which stop when target exist in top k predictions,
    while top n tokens based on importance_score are not been replaced.
    """

    @override
    def __init__(
        self,
        model: AutoModelWithLMHead,
        token_sampler: TokenSampler,
        top_k: int,
        top_n: int = 0,
        top_n_ratio: float = 0,
        tokenizer: AutoTokenizer = None,
    ) -> None:
        """Constructor

        Args:
            model: A Huggingface AutoModelWithLMHead.
            token_sampler: A TokenSampler to sample replacement tokens
            top_k: Stop condition achieved when target exist in top k predictions
            top_n: Top n tokens based on importance_score are not been replaced during the prediction inference.
                top_n_ratio will be used if top_n has been set to 0
            top_n_ratio: Use ratio of input length to control the top n
            tokenizer: (Optional) Used for logging top_k_words at each step

        """
        self.model = model
        self.token_sampler = token_sampler
        self.top_k = top_k
        self.token_replacer = RankingTokenReplacer(self.token_sampler, top_n, top_n_ratio)
        self.tokenizer = tokenizer

    @override
    def __call__(
        self,
        input_ids: IdsTensor,
        target_id: TargetIdsTensor,
        importance_score: MultipleScoresPerStepTensor,
        decoder_input_ids: Optional[IdsTensor] = None,
    ) -> Int64[torch.Tensor, "batch_size"]:
        """Evaluate stop condition

        Args:
            input_ids: Input sequence [batch, sequence]
            target_id: Target token [batch]
            importance_score: Importance score of the input [batch, sequence]
            decoder_input_ids (optional): decoder input sequence for AutoModelForSeq2SeqLM [batch, sequence]

        Return:
            Whether the stop condition achieved [batch]

        """
        # Replace tokens with low importance score and then inference \hat{y^{(e)}_{t+1}}

        self.token_replacer.set_score(importance_score)
        input_ids_replaced, mask_replacing = self.token_replacer(input_ids)

        logging.debug(f"Replacing mask based on importance score -> { mask_replacing }")

        # Whether the result \hat{y^{(e)}_{t+1}} consistent with y_{t+1}

        assert not input_ids_replaced.requires_grad, "Error: auto-diff engine not disabled"
        with torch.no_grad():
            if decoder_input_ids is None:
                logits_replaced = self.model(input_ids_replaced)["logits"]
            else:
                logits_replaced = self.model(input_ids=input_ids_replaced, decoder_input_ids=decoder_input_ids)[
                    "logits"
                ]

        ids_prediction_sorted = torch.argsort(logits_replaced[:, -1, :], descending=True)
        ids_prediction_top_k = ids_prediction_sorted[:, : self.top_k]

        if self.tokenizer:
            top_k_words = [[self.tokenizer.decode([token_id]) for token_id in seq] for seq in ids_prediction_top_k]
            logging.debug(f"Top K words -> {top_k_words}")

        match_mask = ids_prediction_top_k == target_id
        match_hit = torch.sum(match_mask, dim=-1, dtype=torch.bool)

        # Stop flags for each sample in the batch
        return match_hit


class DummyStoppingConditionEvaluator(StoppingConditionEvaluator):
    """
    Stopping Condition Evaluator which stop when target exist in top k predictions,
    while top n tokens based on importance_score are not been replaced.
    """

    @override
    def __init__(self) -> None:
        """Constructor"""

    @override
    def __call__(
        self,
        input_ids: IdsTensor,
        target_id: TargetIdsTensor,
        importance_score: MultipleScoresPerStepTensor,
        decoder_input_ids: Optional[IdsTensor] = None,
    ) -> Int64[torch.Tensor, "batch_size"]:
        """Evaluate stop condition

        Args:
            input_ids: Input sequence [batch, sequence]
            target_id: Target token [batch]
            importance_score: Importance score of the input [batch, sequence]

        Return:
            Whether the stop condition achieved [batch]

        """
        match_hit = torch.ones([input_ids.shape[0]], dtype=torch.bool, device=input_ids.device)

        # Stop flags for each sample in the batch
        return match_hit
