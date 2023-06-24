import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .errors import MissingAlignmentsError


@dataclass
class AlignedSequences:
    source_tokens: List[str]
    target_tokens: List[str]
    alignments: List[Tuple[int, int]]


class AlignmentMethod(Enum):
    AUTO = "auto"


@lru_cache
def get_aligner_model() -> PreTrainedModel:
    return AutoModel.from_pretrained("sentence-transformers/LaBSE")


@lru_cache
def get_aligner_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")


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
    src_tokenized = [tokenizer.tokenize(word) for word in src]
    tgt_tokenized = [tokenizer.tokenize(word) for word in tgt]
    ids_src, sub2word_map_src = _preprocess_sequence_for_alignment(src_tokenized)
    ids_tgt, sub2word_map_tgt = _preprocess_sequence_for_alignment(tgt_tokenized)
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


def get_word_aligns(
    src: Union[str, List[str]],
    tgt: Union[str, List[str]],
    split_pattern: str = r"\s+|\b",
    align_layer: int = 8,
    score_threshold: float = 1e-3,
) -> Dict[str, Any]:
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
    word_alignments = [(src_idx, tgt_idx) for src_idx, tgt_idx in sorted(align_words, key=lambda x: (x[0], x[1]))]
    return AlignedSequences(
        source_tokens=src,
        target_tokens=tgt,
        alignments=word_alignments,
    )


def get_adjusted_alignments(
    alignments: Union[List[Tuple[int, int]], str],
    do_sort: bool = True,
    fill_missing_len: Optional[int] = None,
) -> List[Tuple[int, int]]:
    if alignments is None and isinstance(fill_missing_len, int):
        alignments = [(idx, idx) for idx in range(fill_missing_len)]
    elif isinstance(alignments, str):
        if alignments == AlignmentMethod.AUTO:
            raise NotImplementedError
            # TODO: Implement alignment method. Wrap it in a try-except block that raises a Runtime error in case any
            # of the steps fail.
            # 1. Use LaBSE to get alignments at word level
            # 2. Align word-level alignments to token-level alignments from the generative model tokenizer.
            # 2.1 Requires cleaning up the model tokens from special tokens and characters, check if something native
            # exists in the tokenizer.
            # 3. Propagate word-level alignments to token-level alignments.
        else:
            raise ValueError(f"Unknown alignment method: {alignments}")
    if do_sort:
        # Sort alignments
        alignments = sorted(set(alignments), key=lambda x: (x[0], x[1]))

    # Filling alignments with missing tokens
    if isinstance(fill_missing_len, int):
        filled_alignments = []
        for pair_idx in range(fill_missing_len):
            match_pairs = [x for x in alignments if x[0] == pair_idx]
            if not match_pairs:
                # Assuming 1:1 mapping to cover all tokens from the original sequence
                filled_alignments.append((pair_idx, pair_idx))
            else:
                # Use only first match for the source sequence
                filled_alignments.append(match_pairs[0])
        alignments = filled_alignments
    return alignments


def get_aligned_idx(src_idx: int, alignments: List[Tuple[int, int]]) -> int:
    if alignments:
        # Find all alignment pairs for the current original target
        aligned_idxs = [t_idx for s_idx, t_idx in alignments if s_idx == src_idx]
        if not aligned_idxs:
            raise MissingAlignmentsError(
                f"No alignment found for token at index {src_idx}. "
                "Please provide alignment pairs that cover all original target tokens."
            )
        # Select the minimum index to identify the next target token
        return min(aligned_idxs)
    return src_idx
