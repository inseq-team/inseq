import logging
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from typing_extensions import override

from .stopping_condition_evaluator import StoppingConditionEvaluator
from .token_replacer import TokenReplacer


class BaseImportanceScoreEvaluator(ABC):
    """Importance Score Evaluator"""

    def __init__(self, model: AutoModelForCausalLM | AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer) -> None:
        """Base Constructor

        Args:
            model: A Huggingface AutoModelForCausalLM or AutoModelForSeq2SeqLM model
            tokenizer: A Huggingface AutoTokenizer

        """

        self.model = model
        self.tokenizer = tokenizer

        self.important_score = None

    @abstractmethod
    def __call__(
        self, input_ids: torch.Tensor, target_id: torch.Tensor, decoder_input_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """Evaluate importance score of input sequence

        Args:
            input_ids: input sequence [batch, sequence]
            target_id: target token [batch]
            decoder_input_ids (optional): decoder input sequence for AutoModelForSeq2SeqLM [batch, sequence]

        Return:
            importance_score: evaluated importance score for each token in the input [batch, sequence]

        """
        raise NotImplementedError()


class DeltaProbImportanceScoreEvaluator(BaseImportanceScoreEvaluator):
    """Importance Score Evaluator"""

    @override
    def __init__(
        self,
        model: AutoModelForCausalLM | AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        token_replacer: TokenReplacer,
        stopping_condition_evaluator: StoppingConditionEvaluator,
        max_steps: float,
    ) -> None:
        """Constructor

        Args:
            model: A Huggingface AutoModelForCausalLM or AutoModelForSeq2SeqLM model
            tokenizer: A Huggingface AutoTokenizer
            token_replacer: A TokenReplacer
            stopping_condition_evaluator: A StoppingConditionEvaluator

        """

        super().__init__(model, tokenizer)

        self.token_replacer = token_replacer
        self.stopping_condition_evaluator = stopping_condition_evaluator
        self.max_steps = max_steps

        self.important_score = None
        self.num_steps = 0

    def update_importance_score(
        self,
        logit_importance_score: torch.Tensor,
        input_ids: torch.Tensor,
        target_id: torch.Tensor,
        prob_original_target: torch.Tensor,
        decoder_input_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Update importance score by one step

        Args:
            logit_importance_score: Current importance score in logistic scale [batch]
            input_ids: input tensor [batch, sequence]
            target_id: target tensor [batch]
            prob_original_target: predictive probability of the target on the original sequence [batch]
            decoder_input_ids (optional): decoder input sequence for AutoModelForSeq2SeqLM [batch, sequence]

        Return:
            logit_importance_score: updated importance score in logistic scale [batch]

        """
        # Randomly replace a set of tokens R to form a new sequence \hat{y_{1...t}}

        input_ids_replaced, mask_replacing = self.token_replacer(input_ids)

        logging.debug(f"Replacing mask:     { mask_replacing }")
        logging.debug(
            f"Replaced sequence:  { [[ self.tokenizer.decode(seq[i]) for i in range(input_ids_replaced.shape[1]) ] for seq in input_ids_replaced ] }"
        )

        # Inference \hat{p^{(y)}} = p(y_{t+1}|\hat{y_{1...t}})

        if decoder_input_ids is None:
            logits_replaced = self.model(input_ids_replaced)["logits"]
        else:
            logits_replaced = self.model(input_ids=input_ids_replaced, decoder_input_ids=decoder_input_ids)["logits"]

        prob_replaced_target = torch.softmax(logits_replaced[:, -1, :], -1)[:, target_id]

        # Compute changes delta = p^{(y)} - \hat{p^{(y)}}

        delta_prob_target = prob_original_target - prob_replaced_target
        logging.debug(f"likelihood delta: { delta_prob_target }")

        # Update importance scores based on delta (magnitude) and replacement (direction)

        delta_score = mask_replacing * delta_prob_target + ~mask_replacing * -delta_prob_target
        # TODO: better solution?
        # Rescaling from [-1, 1] to [0, 1] before logit function
        logit_delta_score = torch.logit(delta_score * 0.5 + 0.5)
        logit_importance_score = logit_importance_score + logit_delta_score
        logging.debug(f"Updated importance score: { torch.softmax(logit_importance_score, -1) }")

        return logit_importance_score

    @override
    def __call__(
        self, input_ids: torch.Tensor, target_id: torch.Tensor, decoder_input_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """Evaluate importance score of input sequence

        Args:
            input_ids: input sequence [batch, sequence]
            target_id: target token [batch]
            decoder_input_ids (optional): decoder input sequence for AutoModelForSeq2SeqLM [batch, sequence]

        Return:
            importance_score: evaluated importance score for each token in the input [batch, sequence]

        """

        self.stop_mask = torch.zeros([input_ids.shape[0]], dtype=torch.bool, device=input_ids.device)

        # Inference p^{(y)} = p(y_{t+1}|y_{1...t})
        if decoder_input_ids is None:
            logits_original = self.model(input_ids)["logits"]
        else:
            logits_original = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)["logits"]

        prob_original_target = torch.softmax(logits_original[:, -1, :], -1)[:, target_id]

        # Initialize importance score s for each token in the sequence y_{1...t}

        logit_importance_score = torch.rand(input_ids.shape, device=input_ids.device)
        logging.debug(f"Initialize importance score -> { torch.softmax(logit_importance_score, -1) }")

        # TODO: limit max steps
        self.num_steps = 0
        while self.num_steps < self.max_steps:
            self.num_steps += 1

            # Update importance score
            logit_importance_score_update = self.update_importance_score(
                logit_importance_score, input_ids, target_id, prob_original_target, decoder_input_ids
            )
            logit_importance_score = (
                ~torch.unsqueeze(self.stop_mask, 1) * logit_importance_score_update
                + torch.unsqueeze(self.stop_mask, 1) * logit_importance_score
            )

            self.important_score = torch.softmax(logit_importance_score, -1)

            # Evaluate stop condition
            self.stop_mask = self.stop_mask | self.stopping_condition_evaluator(
                input_ids, target_id, self.important_score, decoder_input_ids
            )
            if torch.prod(self.stop_mask) > 0:
                break

        logging.info(f"Importance score evaluated in {self.num_steps} steps.")

        return torch.softmax(logit_importance_score, -1)
