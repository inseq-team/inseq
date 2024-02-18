import math
from typing import Union

import torch
from typing_extensions import override

from ..token_sampler.base import TokenSampler
from .base import TokenReplacer


class RankingTokenReplacer(TokenReplacer):
    """Replace tokens in a sequence based on top-N ranking

    """

    @override
    def __init__(self, token_sampler: TokenSampler, top_n: int = 0, top_n_ratio: float = 0, replace_greater: bool = False) -> None:
        """Constructor

        Args:
            token_sampler: A TokenSampler for sampling replace token.
            top_n: Top N as the threshold. If top_n is 0, use top_n_ratio instead.
            top_n_ratio: Use ratio of input to control to top_n
            replace_greater: Whether replace top-n. Otherwise, replace the rests.

        """
        super().__init__(token_sampler)

        self.top_n = top_n
        self.top_n_ratio = top_n_ratio
        self.replace_greater = replace_greater

    def set_score(self, value: torch.Tensor) -> None:
        
        pos_sorted = torch.argsort(value, descending=True)

        top_n = self.top_n
        

        if top_n == 0:
            top_n = int(math.ceil(self.top_n_ratio * value.shape[-1]))

        pos_top_n = pos_sorted[..., :top_n]

        if not self.replace_greater:
            self.mask_replacing = torch.ones(value.shape, device=value.device, dtype=torch.bool).scatter(-1, pos_top_n, 0)
        else:
            self.mask_replacing = torch.zeros(value.shape, device=value.device, dtype=torch.bool).scatter(-1, pos_top_n, 1)

    @override
    def sample(self, input: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        """Sample a sequence

        Args:
            input: input sequence [batch, sequence]
        
        Returns:
            input_replaced: A replaced sequence [batch, sequence]
            mask_replacing: Identify which token has been replaced [batch, sequence]

        """
        super().sample(input)

        token_sampled = self.token_sampler.sample(input)

        input_replaced = input * ~self.mask_replacing + token_sampled * self.mask_replacing

        return input_replaced, self.mask_replacing
