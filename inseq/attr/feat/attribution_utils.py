import logging
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from ...data.batch import DecoderOnlyBatch, EncoderDecoderBatch
from ...utils import extract_signature_args
from ...utils.typing import (
    OneOrMoreAttributionSequences,
    OneOrMoreIdSequences,
    OneOrMoreTokenSequences,
    SingleScorePerStepTensor,
    TargetIdsTensor,
    TextInput,
    TokenWithId,
)
from ..step_functions import STEP_SCORES_MAP

if TYPE_CHECKING:
    from ...models import AttributionModel
    from .feature_attribution import FeatureAttribution


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
    Returns step scores for the target tokens in the batch.
    """
    if attribution_model is None:
        raise ValueError("Attribution model is not set.")
    if score_identifier not in STEP_SCORES_MAP:
        raise AttributeError(
            f"Step score {score_identifier} not found. Available step scores are: "
            f"{', '.join(list(STEP_SCORES_MAP.keys()))}. Use the inseq.register_step_function"
            "function to register a custom step score."
        )
    with torch.no_grad():
        output = attribution_model.get_forward_output(
            **attribution_model.format_forward_args(
                batch, use_embeddings=attribution_model.attribution_method.forward_batch_embeds
            ),
            use_embeddings=attribution_model.attribution_method.forward_batch_embeds,
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
    """Builds a list of TokenWithId objects from a list of token sequences and a list of id sequences."""
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
                f"{', '.join(list(STEP_SCORES_MAP.keys()))}. Use the inseq.register_step_function"
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
