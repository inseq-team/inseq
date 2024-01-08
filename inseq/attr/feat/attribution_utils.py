import logging
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from ...utils import extract_signature_args, get_aligned_idx
from ...utils.typing import (
    OneOrMoreAttributionSequences,
    OneOrMoreIdSequences,
    OneOrMoreTokenSequences,
    SingleScorePerStepTensor,
    StepAttributionTensor,
    TextInput,
    TokenWithId,
)
from ..step_functions import get_step_scores_args

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
        attr[: len(tokens)] if not all(math.isnan(x) for x in attr) else []
        for attr, tokens in zip(attributions, tokens)
    ]


def check_attribute_positions(
    max_length: int,
    attr_pos_start: Optional[int] = None,
    attr_pos_end: Optional[int] = None,
) -> Tuple[int, int]:
    r"""Checks whether the combination of start/end positions for attribution is valid.

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
        attr_pos_start = 1
    if attr_pos_end is None or attr_pos_end > max_length:
        attr_pos_end = max_length
    if attr_pos_start < -max_length:
        raise ValueError(f"Invalid starting position for attribution: {attr_pos_start}")
    if attr_pos_start < 0:
        attr_pos_start = max_length + attr_pos_start
    if attr_pos_end < -max_length:
        raise ValueError(f"Invalid ending position for attribution: {attr_pos_end}")
    if attr_pos_end < 0:
        attr_pos_end = max_length + attr_pos_end
    if attr_pos_start > attr_pos_end:
        raise ValueError(f"Invalid starting position for attribution: {attr_pos_start} > {attr_pos_end}")
    if attr_pos_start == attr_pos_end:
        raise ValueError("Start and end attribution positions cannot be the same.")
    return attr_pos_start, attr_pos_end


def join_token_ids(
    tokens: OneOrMoreTokenSequences,
    ids: OneOrMoreIdSequences,
    contrast_tokens: Optional[OneOrMoreTokenSequences] = None,
    contrast_targets_alignments: Optional[List[List[Tuple[int, int]]]] = None,
) -> List[TokenWithId]:
    """Joins tokens and ids into a list of TokenWithId objects."""
    if contrast_tokens is None:
        contrast_tokens = tokens
    # 1:1 alignment between target and contrast tokens
    if contrast_targets_alignments is None:
        contrast_targets_alignments = [[(idx, idx) for idx, _ in enumerate(seq)] for seq in tokens]
    sequences = []
    for target_tokens_seq, contrast_target_tokens_seq, input_ids_seq, alignments_seq in zip(
        tokens, contrast_tokens, ids, contrast_targets_alignments
    ):
        curr_seq = []
        for pos_idx, (token, token_idx) in enumerate(zip(target_tokens_seq, input_ids_seq)):
            contrast_pos_idx = get_aligned_idx(pos_idx, alignments_seq)
            if contrast_pos_idx != -1 and token != contrast_target_tokens_seq[contrast_pos_idx]:
                curr_seq.append(TokenWithId(f"{contrast_target_tokens_seq[contrast_pos_idx]} → {token}", -1))
            else:
                curr_seq.append(TokenWithId(token, token_idx))
        sequences.append(curr_seq)
    return sequences


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
    extra_step_scores_args = get_step_scores_args(step_scores, kwargs, default_args)
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
    attr: Union[StepAttributionTensor, Tuple[StepAttributionTensor, StepAttributionTensor]],
    is_encoder_decoder: bool,
) -> Tuple[Optional[StepAttributionTensor], Optional[StepAttributionTensor]]:
    if isinstance(attr, tuple):
        if is_encoder_decoder:
            return (attr[0], attr[1]) if len(attr) > 1 else (attr[0], None)
        else:
            return (None, attr[0])
    else:
        return (attr, None) if is_encoder_decoder else (None, attr)
