""" Outputs produced by models and attribution methods """

from typing import List, Optional

from dataclasses import dataclass

import torch


@dataclass
class TokenizedOutput:
    """
    Output produced by the tokenization process.

    Attributes:
        input_ids (torch.Tensor): Batch of token ids with shape
            `(batch_size, max_seq_length)`
        attention_mask (torch.Tensor): Batch of attention masks with shape
            `(batch_size, max_seq_length)`
        ref_input_ids (torch.Tensor, optional): Batch of reference token ids
            with shape `(batch_size, max_ref_seq_length)`. Useful for attribution
            methods requiring a reference input (e.g. integrated gradients).
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    ref_input_ids: Optional[torch.Tensor]


@dataclass
class GradientAttributionOutput:
    """
    Output produced by a standard attribution method.

    Attributes:
        source_tokens (list[str]): Tokenized source sequence.
        target_tokens (list[str]): Tokenized target sequence.
        attributions (list[list[str]]): List of length len(target_tokens) containing
            lists of attributions of length len(source_tokens) for each 
            source-target token pair.
        deltas (list[float], optional): List of length len(target_tokens) containing
            the deltas for the approximate integration of the gradients for each
            target token.
        
    Example:
        >> model = AttributionModel('Helsinki-NLP/opus-mt-en-it')
        >> attr_output = model.attribute( \
                method='integrated_gradients', \
                source_text='I like to eat cake.', \
                n_steps=300, \
                internal_batch_size=50 \
            )
        >> attr_output
        # 0.42 is the attribution for the first target token '▁Mi' 
        # to the second source token '▁like'.
        # 0.01 is the convergence delta for the first target token.
        IntegratedGradientAttributionOutput(
            source_tokens=['▁I', '▁like', '▁to', '▁eat', '▁cake', '.', '</s>'],
            target_tokens=['▁Mi', '▁piace', '▁mangiare', '▁la', '▁tor', 'ta', '.' '</s>'],
            attributions=[ [ 0.85, ... ], [ 0.42, ... ], ... ],
            deltas=[ 0.01, ... ]
        )
    """

    source_tokens: List[str]
    target_tokens: List[str]
    attributions: List[List[float]]
    deltas: Optional[List[float]]

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  source_tokens={self.source_tokens},\n"
            f"  target_tokens={self.target_tokens},\n"
            f"  attributions={[[round(v, 2) for v in src_attr] for src_attr in self.attributions]},\n"
            f"  deltas={self.deltas}\n"
            ")"
        )
