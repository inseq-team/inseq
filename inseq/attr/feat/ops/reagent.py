from typing import TYPE_CHECKING, Any, Union

import torch
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
from typing_extensions import override

from ....utils.typing import InseqAttribution
from .reagent_core import (
    AggregateRationalizer,
    DeltaProbImportanceScoreEvaluator,
    POSTagTokenSampler,
    TopKStoppingConditionEvaluator,
    UniformTokenReplacer,
)

if TYPE_CHECKING:
    from ....models import HuggingfaceModel


class Reagent(InseqAttribution):
    r"""Recursive attribution generator (ReAGent) method.

    Measures importance as the drop in prediction probability produced by replacing a token with a plausible
    alternative predicted by a LM.

    Reference implementation:
    `ReAGent: A Model-agnostic Feature Attribution Method for Generative Language Models
        <https://arxiv.org/abs/2402.00794>`__

    Args:
        forward_func (callable): The forward function of the model or any modification of it
        keep_top_n (int): If set to a value greater than 0, the top n tokens based on their importance score will be
            kept during the prediction inference. If set to 0, the top n will be determined by ``keep_ratio``.
        keep_ratio (float): If ``keep_top_n`` is set to 0, this specifies the proportion of tokens to keep.
        invert_keep: If specified, the top tokens selected either via ``keep_top_n`` or ``keep_ratio`` will be
            replaced instead of being kept.
        stopping_condition_top_k (int): Threshold indicating that the stop condition achieved when the predicted target
            exist in top k predictions
        replacing_ratio (float): replacing ratio of tokens for probing
        max_probe_steps (int): max_probe_steps
        num_probes (int): number of probes in parallel

    Example:
        ```
        import inseq

        model = inseq.load_model("gpt2-medium", "reagent",
            keep_top_n=5,
            stopping_condition_top_k=3,
            replacing_ratio=0.3,
            max_probe_steps=3000,
            num_probes=8
        )
        out = model.attribute("Super Mario Land is a game that developed by")
        out.show()
        ```
    """

    def __init__(
        self,
        attribution_model: "HuggingfaceModel",
        keep_top_n: int = 5,
        keep_ratio: float = None,
        invert_keep: bool = False,
        stopping_condition_top_k: int = 3,
        replacing_ratio: float = 0.3,
        max_probe_steps: int = 3000,
        num_probes: int = 16,
    ) -> None:
        super().__init__(attribution_model)

        model = attribution_model.model
        tokenizer = attribution_model.tokenizer
        model_name = attribution_model.model_name

        sampler = POSTagTokenSampler(tokenizer=tokenizer, identifier=model_name, device=attribution_model.device)
        stopping_condition_evaluator = TopKStoppingConditionEvaluator(
            model=model,
            sampler=sampler,
            top_k=stopping_condition_top_k,
            keep_top_n=keep_top_n,
            keep_ratio=keep_ratio,
            invert_keep=invert_keep,
        )
        importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            token_replacer=UniformTokenReplacer(sampler=sampler, ratio=replacing_ratio),
            stopping_condition_evaluator=stopping_condition_evaluator,
            max_steps=max_probe_steps,
        )

        self.rationalizer = AggregateRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            batch_size=num_probes,
            overlap_threshold=0,
            overlap_strict_pos=True,
            keep_top_n=keep_top_n,
            keep_ratio=keep_ratio,
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
        if len(additional_forward_args) == 8:
            # encoder-decoder with target
            self.rationalizer(additional_forward_args[0], additional_forward_args[2], additional_forward_args[1], True)

            mean_important_score = torch.unsqueeze(self.rationalizer.mean_important_score, 0)
            res = torch.unsqueeze(mean_important_score, 2).repeat(1, 1, inputs[0].shape[2])
            return (res[:, : additional_forward_args[0].shape[1], :], res[:, additional_forward_args[0].shape[1] :, :])
        elif len(additional_forward_args) == 9:
            # encoder-decoder
            self.rationalizer(additional_forward_args[1], additional_forward_args[3], additional_forward_args[2])
        elif len(additional_forward_args) == 6:
            # decoder only
            self.rationalizer(additional_forward_args[0], additional_forward_args[1])

        mean_important_score = torch.unsqueeze(self.rationalizer.mean_important_score, 0)
        res = torch.unsqueeze(mean_important_score, 2).repeat(1, 1, inputs[0].shape[2])
        return (res,)
