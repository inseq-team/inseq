from typing import TYPE_CHECKING, Any, Union

import torch
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import PerturbationAttribution
from torch import Tensor
from typing_extensions import override

from .reagent_core.aggregate_rationalizer import AggregateRationalizer
from .reagent_core.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from .reagent_core.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from .reagent_core.token_replacement.token_replacer.uniform import UniformTokenReplacer
from .reagent_core.token_sampler import POSTagTokenSampler

if TYPE_CHECKING:
    from ....models import HuggingfaceModel


class Reagent(PerturbationAttribution):
    r"""Recursive attribution generator (ReAGent) method.

    Measures importance as the drop in prediction probability produced by replacing a token with a plausible
    alternative predicted by a LM.

    Reference implementation:
    `ReAGent: A Model-agnostic Feature Attribution Method for Generative Language Models
        <https://arxiv.org/abs/2402.00794>`__

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
        attribution_model: "HuggingfaceModel",
        rational_size: int = 5,
        rational_size_ratio: float = None,
        stopping_condition_top_k: int = 3,
        replacing_ratio: float = 0.3,
        max_probe_steps: int = 3000,
        num_probes: int = 8,
    ) -> None:
        PerturbationAttribution.__init__(self, forward_func=attribution_model)

        model = attribution_model.model
        tokenizer = attribution_model.tokenizer

        token_sampler = POSTagTokenSampler(
            tokenizer=tokenizer, identifier=attribution_model.model_name, device=attribution_model.device
        )

        stopping_condition_evaluator = TopKStoppingConditionEvaluator(
            model=model,
            token_sampler=token_sampler,
            top_k=stopping_condition_top_k,
            top_n=rational_size,
            top_n_ratio=rational_size_ratio,
            tokenizer=tokenizer,
        )

        importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            token_replacer=UniformTokenReplacer(token_sampler=token_sampler, ratio=replacing_ratio),
            stopping_condition_evaluator=stopping_condition_evaluator,
            max_steps=max_probe_steps,
        )

        self.rationalizer = AggregateRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            batch_size=num_probes,
            overlap_threshold=0,
            overlap_strict_pos=True,
            top_n=rational_size,
            top_n_ratio=rational_size_ratio,
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
        """Implement attribute"""
        self.rationalizer.rationalize(additional_forward_args[0], additional_forward_args[1])
        mean_important_score = torch.unsqueeze(self.rationalizer.mean_important_score, 0)
        res = torch.unsqueeze(mean_important_score, 2).repeat(1, 1, inputs[0].shape[2])
        return (res,)
