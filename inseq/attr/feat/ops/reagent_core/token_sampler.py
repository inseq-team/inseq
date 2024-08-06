import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing_extensions import override

from .....utils import INSEQ_ARTIFACTS_CACHE, cache_results, is_nltk_available
from .....utils.typing import IdsTensor

logger = logging.getLogger(__name__)


class TokenSampler(ABC):
    """Base class for token samplers"""

    @abstractmethod
    def __call__(self, input: IdsTensor, **kwargs) -> IdsTensor:
        """Sample tokens according to the specified strategy.

        Args:
            input: input tensor [batch, sequence]

        Returns:
            token_uniform: A sampled tensor where its shape is the same with the input
        """
        raise NotImplementedError()


class POSTagTokenSampler(TokenSampler):
    """Sample tokens from Uniform distribution on a set of words with the same POS tag."""

    def __init__(
        self,
        tokenizer: str | PreTrainedTokenizerBase,
        identifier: str = "pos_tag_sampler",
        save_cache: bool = True,
        overwrite_cache: bool = False,
        cache_dir: Path = INSEQ_ARTIFACTS_CACHE / "pos_tag_sampler_cache",
        device: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = {},
    ) -> None:
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_kwargs)
        cache_filename = cache_dir / f"{identifier.split('/')[-1]}.pkl"
        self.pos2ids = self.build_pos_mapping_from_vocab(
            cache_dir,
            cache_filename,
            save_cache,
            overwrite_cache,
            tokenizer=self.tokenizer,
        )
        num_postags = len(self.pos2ids)
        self.id2pos = torch.zeros([self.tokenizer.vocab_size], dtype=torch.long, device=device)
        for pos_idx, ids in enumerate(self.pos2ids.values()):
            self.id2pos[ids] = pos_idx
        self.num_ids_per_pos = torch.tensor(
            [len(ids) for ids in self.pos2ids.values()], dtype=torch.long, device=device
        )
        self.offsets = torch.sum(
            torch.tril(torch.ones([num_postags, num_postags], device=device), diagonal=-1) * self.num_ids_per_pos,
            dim=-1,
        )
        self.compact_idx = torch.cat(
            tuple(torch.tensor(v, dtype=torch.long, device=device) for v in self.pos2ids.values())
        )

    @staticmethod
    @cache_results
    def build_pos_mapping_from_vocab(
        tokenizer: PreTrainedTokenizerBase,
        log_every: int = 5000,
    ) -> dict[str, list[int]]:
        """Build mapping from POS tags to list of token ids from tokenizer's vocabulary."""
        if not is_nltk_available():
            raise ImportError("nltk is required to build POS tag mapping. Please install nltk.")
        import nltk

        nltk.download("averaged_perceptron_tagger")
        pos2ids = defaultdict(list)
        for i in range(tokenizer.vocab_size):
            word = tokenizer.decode([i])
            _, tag = nltk.pos_tag([word.strip()])[0]
            pos2ids[tag].append(i)
            if i % log_every == 0:
                logger.info(f"Loading vocab from tokenizer - {i / tokenizer.vocab_size * 100:.2f}%")
        return pos2ids

    @override
    def __call__(self, input_ids: IdsTensor) -> IdsTensor:
        """Sample a tensor

        Args:
            input: input tensor [batch, sequence]

        Returns:
            token_uniform: A sampled tensor where its shape is the same with the input
        """
        input_ids_pos = self.id2pos[input_ids]
        sample_uniform = torch.rand(input_ids.shape, device=input_ids.device)
        compact_group_idx = (sample_uniform * self.num_ids_per_pos[input_ids_pos] + self.offsets[input_ids_pos]).long()
        return self.compact_idx[compact_group_idx]
