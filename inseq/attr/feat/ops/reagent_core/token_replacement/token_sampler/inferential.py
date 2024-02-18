import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from typing_extensions import override

from .base import TokenSampler


class InferentialTokenSampler(TokenSampler):
    """Sample tokens from a seq-2-seq model

    """

    @override
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelWithLMHead) -> None:
        """Constructor

        Args:
            tokenizer: A Huggingface AutoTokenizer.
            model: A Huggingface AutoModelWithLMHead for inference the output.

        """
        super().__init__()

        self.tokenizer = tokenizer
        self.model = model

    @override
    def sample(self, input: torch.Tensor) -> torch.Tensor:
        """Sample a tensor

        Args:
            input: input tensor [batch, sequence]
        
        Returns:
            token_inferences: sampled (placement) tokens by inference

        """
        super().sample(input)

        logits_replacing = self.model(input)['logits']
        ids_infer = torch.argmax(logits_replacing, dim=-1)

        token_inferences = torch.cat([ input[:, 0:1], ids_infer[:, :-1] ], dim=1)

        return token_inferences
