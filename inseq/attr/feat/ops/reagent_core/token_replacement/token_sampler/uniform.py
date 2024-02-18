import torch
from transformers import AutoTokenizer
from typing_extensions import override

from .base import TokenSampler


class UniformTokenSampler(TokenSampler):
    """Sample tokens from Uniform distribution

    """

    @override
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        """Constructor

        Args:
            tokenizer: A Huggingface AutoTokenizer.

        """
        super().__init__()

        self.tokenizer = tokenizer

        # masking tokens
        avail_mask = torch.ones(tokenizer.vocab_size)

        # mask out special tokens
        avail_mask[tokenizer.bos_token_id] = 0
        avail_mask[tokenizer.eos_token_id] = 0
        avail_mask[tokenizer.unk_token_id] = 0

        # collect available tokens
        self.avail_tokens = torch.arange(tokenizer.vocab_size)[avail_mask != 0]
    
    @override
    def sample(self, input: torch.Tensor) -> torch.Tensor:
        """Sample a tensor

        Args:
            input: input tensor [batch, sequence]
        
        Returns:
            token_uniform: A sampled tensor where its shape is the same with the input

        """
        super().sample(input)

        # sample idx form uniform distribution
        sample_uniform = torch.rand(input.shape, device=input.device)
        sample_uniform_idx = (sample_uniform * self.avail_tokens.shape[0]).type(torch.int32)
        # map idx to tokens
        token_uniform = self.avail_tokens.to(sample_uniform_idx)[sample_uniform_idx]

        return token_uniform
