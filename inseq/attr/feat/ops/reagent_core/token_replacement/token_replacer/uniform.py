
from typing import Union

import torch
from typing_extensions import override

from ..token_sampler.base import TokenSampler
from .base import TokenReplacer


class UniformTokenReplacer(TokenReplacer):
    """Replace tokens in a sequence where selecting is base on uniform distribution

    """

    @override
    def __init__(self, token_sampler: TokenSampler, ratio: float) -> None:
        """Constructor

        Args:
            token_sampler: A TokenSampler for sampling replace token.
            ratio: replacing ratio

        """
        super().__init__(token_sampler)

        self.ratio = ratio

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

        sample_uniform = torch.rand(input.shape, device=input.device)
        mask_replacing = sample_uniform < self.ratio

        token_sampled = self.token_sampler.sample(input)

        input_replaced = input * ~mask_replacing + token_sampled * mask_replacing

        return input_replaced, mask_replacing
    