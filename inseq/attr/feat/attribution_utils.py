from typing import List, Optional, Tuple

import math

from inseq.data.batch import EncoderDecoderBatch

from ...utils.typing import OneOrMoreAttributionSequences, OneOrMoreTokenSequences, TextInput


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
