import logging
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from ...data.attribution import DEFAULT_ATTRIBUTION_AGGREGATE_DICT
from ...data.batch import DecoderOnlyBatch, EncoderDecoderBatch
from ...utils import extract_signature_args, output2ce, output2ent, output2ppl, output2prob
from ...utils.typing import (
    EmbeddingsTensor,
    IdsTensor,
    OneOrMoreAttributionSequences,
    OneOrMoreIdSequences,
    OneOrMoreTokenSequences,
    SingleScorePerStepTensor,
    TargetIdsTensor,
    TextInput,
    TokenWithId,
)

if TYPE_CHECKING:
    from ...models import AttributionModel
    from .feature_attribution import FeatureAttribution


StepScoreInput = Callable[
    [
        "AttributionModel",
        Union[EncoderDecoderBatch, DecoderOnlyBatch],
        IdsTensor,
        IdsTensor,
        EmbeddingsTensor,
        EmbeddingsTensor,
        IdsTensor,
        IdsTensor,
        TargetIdsTensor,
    ],
    SingleScorePerStepTensor,
]

STEP_SCORES_MAP = {
    "probability": output2prob,
    "entropy": output2ent,
    "crossentropy": output2ce,
    "perplexity": output2ppl,
}

logger = logging.getLogger(__name__)


def tok2string(
    attribution_model: "AttributionModel",
    token_lists: OneOrMoreTokenSequences,
    start: Optional[int] = None,
    end: Optional[int] = None,
    as_targets: bool = True,
) -> TextInput:
    """Enables bounded tokenization of a list of lists of tokens with start and end positions."""
    start = [0 if start is None else start for _ in token_lists]
    end = [len(tokens) if end is None else end for tokens in token_lists]
    return attribution_model.convert_tokens_to_string(
        [tokens[start[i] : end[i]] for i, tokens in enumerate(token_lists)],  # noqa: E203
        as_targets=as_targets,
    )


def rescale_attributions_to_tokens(
    attributions: OneOrMoreAttributionSequences, tokens: OneOrMoreTokenSequences
) -> OneOrMoreAttributionSequences:
    return [
        attr[: len(tokens)] if not all([math.isnan(x) for x in attr]) else []
        for attr, tokens in zip(attributions, tokens)
    ]


def check_attribute_positions(
    max_length: int,
    attr_pos_start: Optional[int] = None,
    attr_pos_end: Optional[int] = None,
) -> Tuple[int, int]:
    r"""
    Checks whether the combination of start/end positions for attribution is valid.

    Args:
        max_length (:obj:`int`): The maximum length of sequences in the batch.
        attr_pos_start (:obj:`int`, `optional`): The initial position for performing
            sequence attribution. Defaults to 1 (0 is the default BOS token).
        attr_pos_end (:obj:`int`, `optional`): The final position for performing sequence
            attribution. Defaults to None (full string).

    Raises:
        ValueError: If the start position is greater or equal than the end position or < 0.

    Returns:
        `tuple[int, int]`: The start and end positions for attribution.
    """
    if attr_pos_start is None:
        attr_pos_start = 0
    if attr_pos_end is None or attr_pos_end > max_length:
        attr_pos_end = max_length
    if attr_pos_start > attr_pos_end:
        raise ValueError("Invalid starting position for attribution")
    if attr_pos_start == attr_pos_end:
        raise ValueError("Start and end attribution positions cannot be the same.")
    return attr_pos_start, attr_pos_end


def get_step_scores(
    attribution_model: "AttributionModel",
    batch: Union[EncoderDecoderBatch, DecoderOnlyBatch],
    target_ids: TargetIdsTensor,
    score_identifier: str = "probability",
    step_scores_args: Dict[str, Any] = {},
) -> SingleScorePerStepTensor:
    """
    Returns step scores for the target tokens.
    """
    if attribution_model is None:
        raise ValueError("Attribution model is not set.")
    with torch.no_grad():
        output = attribution_model.get_forward_output(
            **attribution_model.format_forward_args(
                batch, use_embeddings=attribution_model.attribution_method.forward_batch_embeds
            )
        )
        step_scores_args = attribution_model.format_step_function_args(
            forward_output=output,
            encoder_input_ids=batch.source_ids,
            decoder_input_ids=batch.target_ids,
            encoder_input_embeds=batch.source_embeds,
            decoder_input_embeds=batch.target_embeds,
            target_ids=target_ids,
            encoder_attention_mask=batch.source_mask,
            decoder_attention_mask=batch.target_mask,
            **step_scores_args,
        )
        return STEP_SCORES_MAP[score_identifier](**step_scores_args)


def join_token_ids(tokens: OneOrMoreTokenSequences, ids: OneOrMoreIdSequences) -> List[TokenWithId]:
    return [[TokenWithId(token, id) for token, id in zip(tok_seq, idx_seq)] for tok_seq, idx_seq in zip(tokens, ids)]


def extract_args(
    attribution_method: "FeatureAttribution",
    attributed_fn: Callable[..., SingleScorePerStepTensor],
    step_scores: List[str],
    default_args: List[str],
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    attribution_args = kwargs.pop("attribution_args", {})
    attributed_fn_args = kwargs.pop("attributed_fn_args", {})
    step_scores_args = kwargs.pop("step_scores_args", {})
    extra_attribution_args, attribution_unused_args = attribution_method.get_attribution_args(**kwargs)
    extra_attributed_fn_args, attributed_fn_unused_args = extract_signature_args(
        kwargs, attributed_fn, exclude_args=default_args, return_remaining=True
    )
    extra_step_scores_args = {}
    for step_score in step_scores:
        if step_score not in STEP_SCORES_MAP:
            raise AttributeError(
                f"Step score {step_score} not found. Available step scores are: "
                f"{', '.join(list(STEP_SCORES_MAP.keys()))}. Use the inseq.register_step_score"
                "function to register a custom step score."
            )
        extra_step_scores_args.update(
            **extract_signature_args(
                kwargs,
                STEP_SCORES_MAP[step_score],
                exclude_args=default_args,
                return_remaining=False,
            )
        )
    step_scores_unused_args = {k: v for k, v in kwargs.items() if k not in extra_step_scores_args}
    unused_args = {
        k: v
        for k, v in kwargs.items()
        if k in attribution_unused_args.keys() & attributed_fn_unused_args.keys() & step_scores_unused_args.keys()
    }
    if unused_args:
        logger.warning(f"Unused arguments during attribution: {unused_args}")
    attribution_args.update(extra_attribution_args)
    attributed_fn_args.update(extra_attributed_fn_args)
    step_scores_args.update(extra_step_scores_args)
    return attribution_args, attributed_fn_args, step_scores_args


def list_step_scores() -> List[str]:
    """
    Lists identifiers for all available step scores. One or more step scores identifiers can be passed to the
    :meth:`~inseq.models.AttributionModel.attribute` method either to compute scores while attributing (`step_scores`
    parameter), or as target function for the attribution, if supported by the attribution method (`attributed_fn`
    parameter).
    """
    return list(STEP_SCORES_MAP.keys())


def register_step_score(
    fn: StepScoreInput,
    identifier: str,
    aggregate_map: Optional[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
) -> None:
    """
    Registers a function to be used to compute step scores and store them in the
    :class:`~inseq.data.attribution.FeatureAttributionOutput` object. Registered step functions can also be used as
    attribution targets by gradient-based feature attribution methods.

    Args:
        fn (:obj:`callable`): The function to be used to compute step scores. Default parameters (use kwargs to capture
        unused ones when defining your function):

            - :obj:`attribution_model`: an :class:`~inseq.models.AttributionModel` instance, corresponding to the model
                used for computing the score.

            - :obj:`forward_output`: the output of the forward pass from the attribution model.

            - :obj:`encoder_input_ids`, :obj:`decoder_input_ids`, :obj:`encoder_input_embeds`,
                :obj:`decoder_input_embeds`, :obj:`encoder_attention_mask`, :obj:`decoder_attention_mask`: all the
                elements composing the :class:`~inseq.data.Batch` used as context of the model.

            - :obj:`target_ids`: :obj:`torch.Tensor` of target token ids of size `(batch_size,)` and type long,
                corresponding to the target predicted tokens for the next generation step.

            The function can also define an arbitrary number of custom parameters that can later be provided directly
            to the `model.attribute` function call, and it must return a :obj:`torch.Tensor` of size `(batch_size,)` of
            float or long. If parameter names conflict with `model.attribute` ones, pass them as key-value pairs in the
            :obj:`step_scores_args` dict parameter.

        identifier (:obj:`str`): The identifier that will be used for the registered step score.
        aggregate_map (:obj:`dict`, `optional`): An optional dictionary mapping from :class:`~inseq.data.Aggregator`
            name identifiers to functions taking in input a tensor of shape `(batch_size, seq_len)` and producing
            tensors of shape `(batch_size, aggregated_seq_len)` in output that will be used to aggregate the
            registered step score when used in conjunction with the corresponding aggregator. E.g. the `probability`
            step score uses the aggregate_map `{"span_aggregate": lambda x: t.prod(dim=1, keepdim=True)}` to aggregate
            probabilities with a product when aggregating scores over spans.
    """
    STEP_SCORES_MAP[identifier] = fn
    if isinstance(aggregate_map, dict):
        for agg_name, agg_fn in aggregate_map.items():
            if agg_name not in DEFAULT_ATTRIBUTION_AGGREGATE_DICT["step_scores"]:
                DEFAULT_ATTRIBUTION_AGGREGATE_DICT["step_scores"][agg_name] = {}
            DEFAULT_ATTRIBUTION_AGGREGATE_DICT["step_scores"][agg_name][identifier] = agg_fn


def get_source_target_attributions(
    attr: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    is_encoder_decoder: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if is_encoder_decoder:
        if isinstance(attr, tuple) and len(attr) > 1:
            return attr[0], attr[1]
        elif isinstance(attr, tuple) and len(attr) == 1:
            return attr[0], None
        else:
            return attr, None
    else:
        if isinstance(attr, tuple):
            return None, attr[0]
        else:
            return None, attr
