from .importance_score_evaluator.base import BaseImportanceScoreEvaluator
from .utils.traceable import Traceable


class BaseRationalizer(Traceable):

    def __init__(self, importance_score_evaluator: BaseImportanceScoreEvaluator) -> None:
        super().__init__()

        self.importance_score_evaluator = importance_score_evaluator
        self.mean_important_score = None
