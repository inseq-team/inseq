
from typing import Union

import torch
from typing_extensions import override

from ..token_sampler.base import TokenSampler
from .base import TokenReplacer


class ThresholdTokenReplacer(TokenReplacer):
    """Replace tokens in a sequence based on a threshold

    """

    @override
    def __init__(self, token_sampler: TokenSampler, threshold: float, replace_greater: bool = False) -> None:
        """Constructor

        Args:
            token_sampler: A TokenSampler for sampling replace token.
            threshold: replacing threshold
            replace_greater: Whether replace top-n. Otherwise, replace the rests.

        """
        super().__init__(token_sampler)

        self.threshold = threshold
        self.replace_greater = replace_greater

    def set_value(self, value: torch.Tensor) -> None:
        """Set the value for threshold control
        
        Args:
            value: value [batch, sequence]

        """
        if not self.replace_greater:
            self.mask_replacing = value < self.threshold
        else:
            self.mask_replacing = value > self.threshold

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

    