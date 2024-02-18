import torch
from typing_extensions import override

from ...utils.traceable import Traceable


class TokenSampler(Traceable):
    """Base class for token samplers

    """
    
    @override
    def __init__(self) -> None:
        """Base Constructor
        
        """
        super().__init__()

    def sample(self, input: torch.Tensor) -> torch.Tensor:
        """Base sample

        """
