
from typing import Any, Callable, Union, cast
from captum.attr._utils.attribution import PerturbationAttribution
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
import torch
from ReAGent.src.rationalization.rationalizer.aggregate_rationalizer import AggregateRationalizer
from ReAGent.src.rationalization.rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from ReAGent.src.rationalization.rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from ReAGent.src.rationalization.rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer

from ReAGent.src.rationalization.rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler
from ReAGent.src.rationalization.rationalizer.token_replacement.token_sampler.uniform import UniformTokenSampler

class ReAGent(PerturbationAttribution):

    def __init__(
        self,
        attribution_model: Callable,
    ) -> None:
        PerturbationAttribution.__init__(self, forward_func=attribution_model)

        # TODO: Handle parameters via args
        model = self.forward_func
        tokenizer = self.forward_func.tokenizer
        stopping_top_k = 1
        rational_size = 1
        rational_size_ratio = 1
        replacing = 0
        max_steps = 1
        batch = 1
        device = 'cpu'

        token_sampler = POSTagTokenSampler(tokenizer=tokenizer, device=device)
        # token_sampler = UniformTokenSampler(tokenizer=tokenizer)

        stopping_condition_evaluator = TopKStoppingConditionEvaluator(
            model=model, 
            token_sampler=token_sampler, 
            top_k=stopping_top_k, 
            top_n=rational_size, 
            top_n_ratio=rational_size_ratio, 
            tokenizer=tokenizer
        )

        importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
            model=model, 
            tokenizer=tokenizer, 
            token_replacer=UniformTokenReplacer(
                token_sampler=token_sampler, 
                ratio=replacing
            ),
            stopping_condition_evaluator=stopping_condition_evaluator,
            max_steps=max_steps
        )


        self.rationalizer = AggregateRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            batch_size=batch,
            overlap_threshold=2,
            overlap_strict_pos=True,
            top_n=rational_size, 
            top_n_ratio=rational_size_ratio
        )

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric,
        tuple[TensorOrTupleOfTensorsGeneric, Tensor],
    ]:
        # TODO: Actual ReAgent implementation
        res = torch.rand(inputs[0].shape)
        return (res,)
