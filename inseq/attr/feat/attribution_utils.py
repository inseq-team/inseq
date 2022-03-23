from typing import List, Optional, Tuple

import math

from ...data.attribution import FeatureAttributionStepOutput
from ...data.batch import EncoderDecoderBatch
from ...utils import probits2probs
from ...utils.typing import (
    OneOrMoreAttributionSequences,
    OneOrMoreIdSequences,
    OneOrMoreTokenSequences,
    TargetIdsTensor,
    TextInput,
    TokenWithId,
)


def tok2string(
    attribution_model: "AttributionModel",
    token_lists: OneOrMoreTokenSequences,
    start: Optional[int] = None,
    end: Optional[int] = None,
    as_targets: bool = True,
) -> TextInput:
    start = [0 if start is None else start for _ in token_lists]
    end = [len(tokens) if end is None else end for tokens in token_lists]
    return attribution_model.convert_tokens_to_string(
        [tokens[start[i] : end[i]] for i, tokens in enumerate(token_lists)],  # noqa: E203
        as_targets=as_targets,
    )


def get_attribution_sentences(
    attribution_model: "AttributionModel",
    batch: EncoderDecoderBatch,
    start: int,
    end: int,
) -> Tuple[List[str], List[str], List[int]]:
    source_sentences = tok2string(attribution_model, batch.sources.input_tokens, as_targets=False)
    target_sentences = tok2string(attribution_model, batch.targets.input_tokens)
    if isinstance(source_sentences, str):
        source_sentences = [source_sentences]
        target_sentences = [target_sentences]
    tokenized_target_sentences = [
        attribution_model.convert_string_to_tokens(sent, as_targets=True) for sent in target_sentences
    ]
    lengths = [min(end, len(tts) + 1) - start for tts in tokenized_target_sentences]
    return source_sentences, target_sentences, lengths


def get_split_targets(
    attribution_model: "AttributionModel",
    targets: OneOrMoreTokenSequences,
    start: int,
    end: int,
    step: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    skipped_prefixes = tok2string(attribution_model, targets, end=start)
    attributed_sentences = tok2string(attribution_model, targets, start, step + 1)
    unattributed_suffixes = tok2string(attribution_model, targets, step + 1, end)
    skipped_suffixes = tok2string(attribution_model, targets, start=end)
    return skipped_prefixes, attributed_sentences, unattributed_suffixes, skipped_suffixes


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
        attr_pos_start = 1
    if attr_pos_end is None or attr_pos_end > max_length:
        attr_pos_end = max_length
    if attr_pos_start > attr_pos_end or attr_pos_start < 1:
        raise ValueError("Invalid starting position for attribution")
    if attr_pos_start == attr_pos_end:
        raise ValueError("Start and end attribution positions cannot be the same.")
    return attr_pos_start, attr_pos_end


def get_step_prediction_probabilities(
    attribution_model: "AttributionModel", batch: EncoderDecoderBatch, target_ids: TargetIdsTensor
) -> float:
    """
    Returns the probabilities of the target tokens.
    """
    if attribution_model is None:
        raise ValueError("Attribution model is not set.")
    probits = attribution_model(
        encoder_tensors=batch.sources.input_embeds,
        decoder_embeds=batch.targets.input_embeds,
        encoder_attention_mask=batch.sources.attention_mask,
        decoder_attention_mask=batch.targets.attention_mask,
        use_embeddings=True,
    )
    return probits2probs(probits, target_ids)


def join_token_ids(tokens: OneOrMoreTokenSequences, ids: OneOrMoreIdSequences) -> List[TokenWithId]:
    return [[TokenWithId(token, id) for token, id in zip(tok_seq, idx_seq)] for tok_seq, idx_seq in zip(tokens, ids)]


def enrich_step_output(
    step_output: FeatureAttributionStepOutput,
    batch: EncoderDecoderBatch,
    target_tokens: OneOrMoreTokenSequences,
    target_ids: TargetIdsTensor,
) -> FeatureAttributionStepOutput:
    r"""
    Enriches the attribution output with token information, producing the finished
    :class:`~inseq.data.FeatureAttributionStepOutput` object.

    Args:
        step_output (:class:`~inseq.data.FeatureAttributionStepOutput`): The output produced
            by the attribution step, with missing batch information.
        batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch on which attribution was performed.
        target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size, 1)` corresponding to tokens
            for which the attribution step was performed.

    Returns:
        :class:`~inseq.data.FeatureAttributionStepOutput`: The enriched attribution output.
    """
    if len(target_ids.shape) == 0:
        target_ids = target_ids.unsqueeze(0)
    step_output.source = join_token_ids(batch.sources.input_tokens, batch.sources.input_ids.tolist())
    step_output.target = [[TokenWithId(token[0], id)] for token, id in zip(target_tokens, target_ids.tolist())]
    step_output.prefix = join_token_ids(batch.targets.input_tokens, batch.targets.input_ids.tolist())
    return step_output
