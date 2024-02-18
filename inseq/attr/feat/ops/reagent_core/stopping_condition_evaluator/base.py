import torch

from ..utils.traceable import Traceable


class StoppingConditionEvaluator(Traceable):
    """Base class for Stopping Condition Evaluators
    
    """

    def __init__(self):
        """Base Constructor
        
        """
        self.trace_target_likelihood = []

    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor, importance_score: torch.Tensor) -> torch.Tensor:
        """Base evaluate
        
        """
