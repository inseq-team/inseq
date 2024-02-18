import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from typing_extensions import override

from ..stopping_condition_evaluator.base import StoppingConditionEvaluator
from ..token_replacement.token_replacer.base import TokenReplacer
from ..utils.traceable import Traceable


class BaseImportanceScoreEvaluator(Traceable):
    """Importance Score Evaluator
    
    """

    def __init__(self, model: AutoModelWithLMHead, tokenizer: AutoTokenizer) -> None:
        """Base Constructor

        Args:
            model: A Huggingface AutoModelWithLMHead model
            tokenizer: A Huggingface AutoTokenizer

        """

        self.model = model
        self.tokenizer = tokenizer
        
        self.important_score = None

        self.trace_importance_score = None
        self.trace_target_likelihood_original = None

    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Evaluate importance score of input sequence

        Args:
            input_ids: input sequence [batch, sequence]
            target_id: target token [batch]

        Return:
            importance_score: evaluated importance score for each token in the input [batch, sequence]

        """

        raise NotImplementedError()
    
    @override
    def trace_start(self):
        """Start tracing
        
        """
        super().trace_start()

        self.trace_importance_score = []
        self.trace_target_likelihood_original = -1
        self.stopping_condition_evaluator.trace_start()

    @override
    def trace_stop(self):
        """Stop tracing
        
        """
        super().trace_stop()

        self.trace_importance_score = None
        self.trace_target_likelihood_original = None
        self.stopping_condition_evaluator.trace_stop()
