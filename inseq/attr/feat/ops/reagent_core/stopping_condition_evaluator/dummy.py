import torch
from typing_extensions import override

from .base import StoppingConditionEvaluator


class DummyStoppingConditionEvaluator(StoppingConditionEvaluator):
    """
    Stopping Condition Evaluator which stop when target exist in top k predictions, 
    while top n tokens based on importance_score are not been replaced.
    """

    @override
    def __init__(self) -> None:
        """Constructor

        """
        super().__init__()

    @override
    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor, importance_score: torch.Tensor) -> torch.Tensor:
        """Evaluate stop condition

        Args:
            input_ids: Input sequence [batch, sequence]
            target_id: Target token [batch]
            importance_score: Importance score of the input [batch, sequence]

        Return:
            Whether the stop condition achieved [batch]

        """
        super().evaluate(input_ids, target_id, importance_score)

        match_hit = torch.ones([input_ids.shape[0]], dtype=torch.bool, device=input_ids.device)

        # Stop flags for each sample in the batch
        return match_hit
