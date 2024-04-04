import math
from abc import ABC, abstractmethod

import torch
from typing_extensions import override

from .....utils.typing import IdsTensor
from .token_sampler import TokenSampler


class TokenReplacer(ABC):
    """
    Base class for token replacers

    """

    def __init__(self, sampler: TokenSampler) -> None:
        self.sampler = sampler

    @abstractmethod
    def __call__(self, input: IdsTensor) -> tuple[IdsTensor, IdsTensor]:
        """Replace tokens according to the specified strategy.

        Args:
            input: input sequence [batch, sequence]

        Returns:
            input_replaced: A replaced sequence [batch, sequence]
            replacement_mask: Boolean mask identifying which token has been replaced [batch, sequence]

        """
        raise NotImplementedError()


class RankingTokenReplacer(TokenReplacer):
    """Replace tokens in a sequence based on top-N ranking"""

    @override
    def __init__(
        self, sampler: TokenSampler, keep_top_n: int = 0, keep_ratio: float = 0, invert_keep: bool = False
    ) -> None:
        """Constructor for the RankingTokenReplacer class.

        Args:
            sampler: A :class:`~inseq.attr.feat.ops.reagent_core.TokenSampler` object for sampling replacement tokens.
            keep_top_n: If set to a value greater than 0, the top n tokens based on their importance score will be
                kept, and the rest will be flagged for replacement. If set to 0, the top n will be determined by
                ``keep_ratio``.
            keep_ratio: If ``keep_top_n`` is set to 0, this specifies the proportion of tokens to keep.
            invert_keep: If specified, the top tokens selected either via ``keep_top_n`` or ``keep_ratio`` will be
                replaced instead of being kept.
        """
        super().__init__(sampler)
        self.keep_top_n = keep_top_n
        self.keep_ratio = keep_ratio
        self.invert_keep = invert_keep

    def set_score(self, value: torch.Tensor) -> None:
        pos_sorted = torch.argsort(value, descending=True)
        top_n = int(math.ceil(self.keep_ratio * value.shape[-1])) if not self.keep_top_n else self.keep_top_n
        pos_top_n = pos_sorted[..., :top_n]
        self.replacement_mask = torch.ones_like(value, device=value.device, dtype=torch.bool).scatter(
            -1, pos_top_n, self.invert_keep
        )

    @override
    def __call__(self, input: IdsTensor) -> tuple[IdsTensor, IdsTensor]:
        """Sample a sequence

        Args:
            input: Input sequence of ids of shape [batch, sequence]

        Returns:
            input_replaced: A replaced sequence [batch, sequence]
            replacement_mask: Boolean mask identifying which token has been replaced [batch, sequence]
        """
        token_sampled = self.sampler(input)
        input_replaced = input * ~self.replacement_mask + token_sampled * self.replacement_mask
        return input_replaced, self.replacement_mask


class UniformTokenReplacer(TokenReplacer):
    """Replace tokens in a sequence where selecting is base on uniform distribution"""

    @override
    def __init__(self, sampler: TokenSampler, ratio: float) -> None:
        """Constructor

        Args:
            sampler: A :class:`~inseq.attr.feat.ops.reagent_core.TokenSampler` object for sampling replacement tokens.
            ratio: Ratio of tokens to replace in the sequence.
        """
        super().__init__(sampler)
        self.ratio = ratio

    @override
    def __call__(self, input: IdsTensor) -> tuple[IdsTensor, IdsTensor]:
        """Sample a sequence

        Args:
            input: Input sequence of ids of shape [batch, sequence]

        Returns:
            input_replaced: A replaced sequence [batch, sequence]
            replacement_mask: Boolean mask identifying which token has been replaced [batch, sequence]
        """
        sample_uniform = torch.rand(input.shape, device=input.device)
        replacement_mask = sample_uniform < self.ratio
        token_sampled = self.sampler(input)
        input_replaced = input * ~replacement_mask + token_sampled * replacement_mask
        return input_replaced, replacement_mask
