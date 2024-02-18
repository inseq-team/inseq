
from typing import Any, Callable, Union, cast
from captum.attr._utils.attribution import PerturbationAttribution
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
import torch
from ReAGent.src.rationalization.rationalizer.aggregate_rationalizer import AggregateRationalizer
from ReAGent.src.rationalization.rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from ReAGent.src.rationalization.rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from ReAGent.src.rationalization.rationalizer.stopping_condition_evaluator.dummy import DummyStoppingConditionEvaluator
from ReAGent.src.rationalization.rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer

from ReAGent.src.rationalization.rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler
from ReAGent.src.rationalization.rationalizer.token_replacement.token_sampler.uniform import UniformTokenSampler

class ReAGent(PerturbationAttribution):
    r"""
    ReAGent

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it
        rational_size (int): Top n tokens based on importance_score are not been replaced during the prediction inference.
            top_n_ratio will be used if top_n has been set to 0
        rational_size_ratio (float): TUse ratio of input length to control the top n
        stopping_condition_top_k (int): Stop condition achieved when target exist in top k predictions
        replacing_ratio (float): replacing ratio of tokens for probing
        max_probe_steps (int): max_probe_steps
        num_probes (int): number of probes in parallel

    References:
        `ReAGent: A Model-agnostic Feature Attribution Method for Generative Language Models
        <https://arxiv.org/abs/2402.00794>`_

    Examples:
        ```
        import inseq

        model = inseq.load_model("gpt2-medium", "ReAGent", 
                                rational_size=5, 
                                rational_size_ratio=None,
                                stopping_condition_top_k=3,
                                replacing_ratio=0.3,
                                max_probe_steps=3000,
                                num_probes=8)
        out = model.attribute(
        "Super Mario Land is a game that developed by",
        )
        out.show()
        ```
    """

    def __init__(
        self,
        attribution_model: Callable,
        rational_size: int=5,
        rational_size_ratio: float=None,
        stopping_condition_top_k: int=3,
        replacing_ratio: float=0.3,
        max_probe_steps: int=3000,
        num_probes: int=8,
    ) -> None:
        PerturbationAttribution.__init__(self, forward_func=attribution_model)

        model = attribution_model.model
        tokenizer = attribution_model.tokenizer

        token_sampler = POSTagTokenSampler(tokenizer=tokenizer, device=model.device)

        stopping_condition_evaluator = TopKStoppingConditionEvaluator(
            model=model,
            token_sampler=token_sampler,
            top_k=stopping_condition_top_k,
            top_n=rational_size,
            top_n_ratio=rational_size_ratio,
            tokenizer=tokenizer
        )
        # stopping_condition_evaluator = DummyStoppingConditionEvaluator()

        importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            token_replacer=UniformTokenReplacer(
                token_sampler=token_sampler,
                ratio=replacing_ratio
            ),
            stopping_condition_evaluator=stopping_condition_evaluator,
            max_steps=max_probe_steps
        )

        self.rationalizer = AggregateRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            batch_size=num_probes,
            overlap_threshold=0,
            overlap_strict_pos=True,
            top_n=rational_size,
            top_n_ratio=rational_size_ratio
        )

    @override
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        _target: TargetType = None,
        additional_forward_args: Any = None,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric,
        tuple[TensorOrTupleOfTensorsGeneric, Tensor],
    ]:
        """Implement attribute
        """
        self.rationalizer.rationalize(additional_forward_args[0], additional_forward_args[1])
        mean_important_score = torch.unsqueeze(self.rationalizer.mean_important_score, 0)
        res = torch.unsqueeze(mean_important_score, 2).repeat(1, 1, inputs[0].shape[2])
        return (res,)
