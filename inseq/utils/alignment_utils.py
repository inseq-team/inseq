import logging
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import chain
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .errors import MissingAlignmentsError

logger = logging.getLogger(__name__)

ALIGN_MODEL_ID = "sentence-transformers/LaBSE"


@dataclass
class AlignedSequences:
    source_tokens: List[str]
    target_tokens: List[str]
    alignments: List[Tuple[int, int]]

    @property
    def aligned_tokens(self) -> List[Tuple[str, str]]:
        return [(self.source_tokens[a_idx], self.target_tokens[b_idx]) for a_idx, b_idx in self.alignments]

    def reverse(self) -> "AlignedSequences":
        return AlignedSequences(
            source_tokens=self.target_tokens,
            target_tokens=self.source_tokens,
            alignments=[(b_idx, a_idx) for a_idx, b_idx in self.alignments],
        )

    def __str__(self) -> str:
        return f"{', '.join([f'{a}→{b} ({self.source_tokens[a]}→{self.target_tokens[b]})'for a,b in self.alignments])}"


class AlignmentMethod(Enum):
    AUTO = "auto"


@lru_cache
def get_aligner_model() -> PreTrainedModel:
    return AutoModel.from_pretrained(ALIGN_MODEL_ID)


@lru_cache
def get_aligner_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(ALIGN_MODEL_ID)


def _preprocess_sequence_for_alignment(tokenized_seq: List[str]) -> Tuple[torch.Tensor, List[List[int]]]:
    aligner_tokenizer = get_aligner_tokenizer()
    idxs = [aligner_tokenizer.convert_tokens_to_ids(x) for x in tokenized_seq]
    idxs = aligner_tokenizer.prepare_for_model(
        list(chain(*idxs)),
        return_tensors="pt",
        truncation=True,
        model_max_length=aligner_tokenizer.model_max_length,
    )["input_ids"]
    sub2word_map = []
    for i, word_list in enumerate(tokenized_seq):
        sub2word_map += [i for x in word_list]
    return idxs, sub2word_map


def _get_aligner_subword_aligns(
    src: List[str],
    tgt: List[str],
    align_layer: int,
    score_threshold: float,
) -> torch.Tensor:
    aligner = get_aligner_model()
    tokenizer = get_aligner_tokenizer()
    tok_aenized = [tokenizer.tokenize(word) for word in src]
    tok_benized = [tokenizer.tokenize(word) for word in tgt]
    ids_src, sub2word_map_src = _preprocess_sequence_for_alignment(tok_aenized)
    ids_tgt, sub2word_map_tgt = _preprocess_sequence_for_alignment(tok_benized)
    aligner.eval()
    with torch.no_grad():
        out_src = aligner(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = aligner(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)
        softmax_inter = (softmax_srctgt > score_threshold) * (softmax_tgtsrc > score_threshold)
    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    return align_subwords, sub2word_map_src, sub2word_map_tgt


def compute_word_aligns(
    src: Union[str, List[str]],
    tgt: Union[str, List[str]],
    split_pattern: str = r"\s+|\b",
    align_layer: int = 8,
    score_threshold: float = 1e-3,
) -> AlignedSequences:
    if isinstance(src, str):
        src = [word for word in re.split(split_pattern, src) if word]
    if isinstance(tgt, str):
        tgt = [word for word in re.split(split_pattern, tgt) if word]
    align_subwords, sub2word_map_src, sub2word_map_tgt = _get_aligner_subword_aligns(
        src, tgt, align_layer, score_threshold
    )
    align_words = set()
    for i, j in align_subwords:
        align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))
    word_alignments = [(a_idx, b_idx) for a_idx, b_idx in sorted(align_words, key=lambda x: (x[0], x[1]))]
    return AlignedSequences(
        source_tokens=src.copy(),
        target_tokens=tgt.copy(),
        alignments=word_alignments.copy(),
    )


def align_tokenizations(
    tok_a: List[str],
    tok_b: List[str],
) -> AlignedSequences:
    """Align tokens from a sentence tokenized by different tokenizers.

    Args:
        tok_a (:obj:`str` or :obj:`list` of :obj:`str`):
            Sequence of tokens produced by the first tokenizer.
        tok_b (:obj:`str` or :obj:`list` of :obj:`str`):
            Sequence of tokens produced by the second tokenizer.

    Raises:
        `ValueError`: Raised if the provided sequences do not have the same contents when concatenated.
    """
    if "".join(tok_a) != "".join(tok_b):
        raise ValueError(
            "The provided sequences must have the same contents when concatenated.\n"
            f"Sequence A: {tok_a}\nSequence B: {tok_b}\n"
        )
    aligns = []
    orig_tok_a = tok_a.copy()
    orig_tok_b = tok_b.copy()
    a_idx, b_idx = 0, 0
    while a_idx < len(tok_a):
        curr_tok_a = tok_a[a_idx]
        curr_tok_b = tok_b[b_idx]
        if curr_tok_a == curr_tok_b:
            aligns.append((a_idx, b_idx))
            a_idx += 1
            b_idx += 1
        elif curr_tok_a in curr_tok_b:
            aligns.append((a_idx, b_idx))
            tok_b[b_idx] = tok_b[b_idx].replace(curr_tok_a, "", 1)
            a_idx += 1
        elif curr_tok_b in curr_tok_a:
            aligns.append((a_idx, b_idx))
            tok_a[a_idx] = tok_a[a_idx].replace(curr_tok_b, "", 1)
            b_idx += 1
        else:
            raise ValueError(
                f"Found mismatching tokens '{curr_tok_a}' and '{curr_tok_b}' when aligning tokens. "
                "Please provide tokenizations that can be aligned."
            )
    return AlignedSequences(
        source_tokens=orig_tok_a,
        target_tokens=orig_tok_b,
        alignments=aligns.copy(),
    )


def propagate_alignments(aligns_a_b: AlignedSequences, aligns_b_c: AlignedSequences) -> AlignedSequences:
    """Given two set of alignments corresponding to the aligned tokens of strings A and B
    and those of strings B and C respectively, returns the alignment of tokens between
    string A and C.

    Args:
        aligns_a_b (:obj:`list` of :obj:`tuple` of :obj:`int`): List of alignment index pairs
            between sequences A and B.
        aligns_b_c (:obj:`list` of :obj:`tuple` of :obj:`int`): List of alignment index pairs
            between sequences B and C.

    Returns:
        :class:`AlignedSequences`: Alignment pairs between sequences A and C.
    """
    aligns_a_c = []
    for idx_a, idx_b_in_ab in aligns_a_b.alignments:
        for idx_b_in_bc, idx_c in aligns_b_c.alignments:
            if idx_b_in_ab == idx_b_in_bc:
                aligns_a_c.append((idx_a, idx_c))
    return AlignedSequences(
        source_tokens=aligns_a_b.source_tokens.copy(),
        target_tokens=aligns_b_c.target_tokens.copy(),
        alignments=aligns_a_c.copy(),
    )


def auto_align_sequences(
    a_sequence: Optional[str] = None,
    a_tokens: Optional[List[str]] = None,
    b_sequence: Optional[str] = None,
    b_tokens: Optional[List[str]] = None,
    filter_special_tokens: List[str] = [],
    split_pattern: str = r"\s+|\b",
) -> AlignedSequences:
    if not a_sequence or not b_sequence or not a_tokens or not b_tokens:
        raise ValueError(
            "Missing required arguments to compute alignments. Please provide target and contrast sequence and tokens."
        )
    try:
        for token in filter_special_tokens:
            b_sequence = b_sequence.replace(token, "")
        # 1. Use aligner to get alignments at word level
        # Alignments are target to contrast word-level alignment pairs
        a_words = [word for word in re.split(split_pattern, a_sequence) if word]
        b_words = [word for word in re.split(split_pattern, b_sequence) if word]
        a_to_b_word_align = compute_word_aligns(a_words, b_words)
        # 2. Align word-level alignments to token-level alignments from the generative model tokenizer.
        # Requires cleaning up the model tokens from special tokens (special characters already removed)
        clean_a_tokens = []
        removed_a_token_idxs = []
        for idx_a, tok_a in enumerate(a_tokens):
            if tok_a not in filter_special_tokens:
                clean_a_tokens += [tok_a.strip()]
            else:
                removed_a_token_idxs += [idx_a]
        clean_b_tokens = []
        removed_b_token_idxs = []
        for idx_b, tok_b in enumerate(b_tokens):
            if tok_b not in filter_special_tokens:
                clean_b_tokens += [tok_b.strip()]
            else:
                removed_b_token_idxs += [idx_b]
        if len(removed_a_token_idxs) != len(removed_b_token_idxs):
            raise ValueError(
                "The number of special tokens in the target and contrast sequences do not match. "
                "Please provide sequences with the same number of special tokens."
            )
        aligned_special_tokens = [(rm_a, rm_b) for rm_a, rm_b in zip(removed_a_token_idxs, removed_b_token_idxs)]
        a_word_to_token_align = align_tokenizations(a_words, clean_a_tokens)
        b_word_to_token_align = align_tokenizations(b_words, clean_b_tokens)
        # 3. Propagate word-level alignments to token-level alignments.
        # target token-level -> target word-level -> contrast word-level -> contrast token-level
        # First step: get target token-level -> contrast word-level
        a_token_to_word_align = a_word_to_token_align.reverse()
        a_token_to_b_word_align = propagate_alignments(a_token_to_word_align, a_to_b_word_align)
        # Second step: get target token-level -> contrast token-level using previous step outputs
        a_to_b_token_align = propagate_alignments(a_token_to_b_word_align, b_word_to_token_align)
        # 4. Add special tokens alignments
        for s_idx_a, s_idx_b in aligned_special_tokens:
            for pos, (idx_a, idx_b) in enumerate(a_to_b_token_align.alignments):
                a_val, b_val = idx_a, idx_b
                if idx_a >= s_idx_a:
                    a_val += 1
                if idx_b >= s_idx_b:
                    b_val += 1
                a_to_b_token_align.alignments[pos] = (a_val, b_val)
        return AlignedSequences(
            source_tokens=a_tokens,
            target_tokens=b_tokens,
            alignments=a_to_b_token_align.alignments + aligned_special_tokens,
        )
    except Exception as e:
        logger.warning(
            "Failed to compute alignments using the aligner. "
            f"Please check the following error and provide custom alignments if needed.\n{e}"
        )
        raise e


def get_adjusted_alignments(
    alignments: Union[List[Tuple[int, int]], str],
    target_sequence: Optional[str] = None,
    target_tokens: Optional[List[str]] = None,
    contrast_sequence: Optional[str] = None,
    contrast_tokens: Optional[List[str]] = None,
    do_sort: bool = True,
    fill_missing: bool = False,
    special_tokens: List[str] = [],
) -> List[Tuple[int, int]]:
    is_auto_aligned = False
    if fill_missing and not target_tokens:
        raise ValueError("Missing target tokens. Please provide target tokens to fill missing alignments.")
    if alignments is None and fill_missing:
        alignments = [(idx, idx) for idx in range(len(target_tokens))]
    elif isinstance(alignments, str):
        if alignments == AlignmentMethod.AUTO.value:
            auto_aligned = auto_align_sequences(
                a_sequence=target_sequence,
                a_tokens=target_tokens,
                b_sequence=contrast_sequence,
                b_tokens=contrast_tokens,
                filter_special_tokens=special_tokens,
            )
            is_auto_aligned = True
            alignments = auto_aligned.alignments
        else:
            raise ValueError(
                f"Unknown alignment method: {alignments}. "
                f"Available methods: {','.join([m.value for m in AlignmentMethod])}"
            )
    if do_sort:
        # Sort alignments
        alignments = sorted(set(alignments), key=lambda x: (x[0], x[1]))

    # Filling alignments with missing tokens
    if fill_missing:
        filled_alignments = []
        for pair_idx in range(len(target_tokens)):
            match_pairs = [pair for pair in alignments if pair[0] == pair_idx]

            if not match_pairs:
                # Assuming 1:1 mapping to cover all tokens from the original sequence
                filled_alignments.append((pair_idx, pair_idx))
            else:
                match_pairs_unaligned = [p for p in match_pairs if p[1] not in [f[1] for f in filled_alignments]]
                # If found, use the first match that containing an unaligned target token, first match otherwise
                valid_match = match_pairs_unaligned[0] if match_pairs_unaligned else match_pairs[0]
                filled_alignments.append(valid_match)
        if alignments != filled_alignments:
            logger.warning(
                f"Provided alignments do not cover all {len(target_tokens)} tokens from the original sequence.\n"
                "Filling missing position with 1:1 position alignments."
            )
        if is_auto_aligned:
            logger.warning(
                f"Using {ALIGN_MODEL_ID} for automatic alignments. Provide custom alignments for non-linguistic "
                f"sequences, or for languages not covered by the aligner.\nGenerated alignments: {filled_alignments}"
            )
        alignments = filled_alignments
    return alignments


def get_aligned_idx(a_idx: int, alignments: List[Tuple[int, int]]) -> int:
    if alignments:
        # Find all alignment pairs for the current original target
        aligned_idxs = [t_idx for s_idx, t_idx in alignments if s_idx == a_idx]
        if not aligned_idxs:
            raise MissingAlignmentsError(
                f"No alignment found for token at index {a_idx}. "
                "Please provide alignment pairs that cover all original target tokens."
            )
        # Select the minimum index to identify the next target token
        return min(aligned_idxs)
    return a_idx
