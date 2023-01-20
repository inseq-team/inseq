from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from ..utils.typing import EmbeddingsTensor, ExpandedTargetIdsTensor, IdsTensor, OneOrMoreTokenSequences
from .data_utils import TensorWrapper


@dataclass(eq=False, repr=False)
class BatchEncoding(TensorWrapper):
    """
    Output produced by the tokenization process.

    Attributes:
        input_ids (torch.Tensor): Batch of token ids with shape
            (batch_size, longest_seq_length). Extra tokens for each sentence
            are padded, and truncation to max_seq_length is performed.
        input_tokens (:obj:`list(list(str))`): List of lists containing tokens
            for each sentence in the batch.
        attention_mask (torch.Tensor): Batch of attention masks with shape
            (batch_size, longest_seq_length). 1 for positions that are valid,
            0 for padded positions.
        baseline_ids (torch.Tensor, optional): Batch of reference token ids
            with shape `(batch_size, longest_seq_length)`. Useful for attribution
            methods requiring a reference input (e.g. integrated gradients).
    """

    input_ids: IdsTensor
    input_tokens: OneOrMoreTokenSequences
    attention_mask: IdsTensor
    baseline_ids: Optional[IdsTensor]

    def __len__(self) -> int:
        return len(self.input_tokens)


@dataclass(eq=False, repr=False)
class BatchEmbedding(TensorWrapper):
    input_embeds: Optional[EmbeddingsTensor] = None
    baseline_embeds: Optional[EmbeddingsTensor] = None

    def __len__(self) -> Optional[int]:
        if self.input_embeds is not None:
            return self.input_embeds.shape[0]
        return None


@dataclass(eq=False, repr=False)
class Batch(TensorWrapper):
    encoding: BatchEncoding
    embedding: BatchEmbedding

    @property
    def input_ids(self) -> IdsTensor:
        return self.encoding.input_ids

    @property
    def input_tokens(self) -> OneOrMoreTokenSequences:
        return self.encoding.input_tokens

    @property
    def attention_mask(self) -> IdsTensor:
        return self.encoding.attention_mask

    @property
    def baseline_ids(self) -> Optional[IdsTensor]:
        return self.encoding.baseline_ids

    @property
    def input_embeds(self) -> Optional[EmbeddingsTensor]:
        return self.embedding.input_embeds

    @property
    def baseline_embeds(self) -> Optional[EmbeddingsTensor]:
        return self.embedding.baseline_embeds

    @input_ids.setter
    def input_ids(self, value: IdsTensor):
        self.encoding.input_ids = value

    @input_tokens.setter
    def input_tokens(self, value: List[List[str]]):
        self.encoding.input_tokens = value

    @attention_mask.setter
    def attention_mask(self, value: IdsTensor):
        self.encoding.attention_mask = value

    @baseline_ids.setter
    def baseline_ids(self, value: Optional[IdsTensor]):
        self.encoding.baseline_ids = value

    @input_embeds.setter
    def input_embeds(self, value: Optional[EmbeddingsTensor]):
        self.embedding.input_embeds = value

    @baseline_embeds.setter
    def baseline_embeds(self, value: Optional[EmbeddingsTensor]):
        self.embedding.baseline_embeds = value


@dataclass(eq=False, repr=False)
class EncoderDecoderBatch(TensorWrapper):
    sources: Batch
    targets: Batch

    def __getitem__(self, subscript: Union[slice, int]) -> "EncoderDecoderBatch":
        return EncoderDecoderBatch(sources=self.sources, targets=self.targets[subscript])

    @property
    def max_generation_length(self) -> int:
        return self.targets.input_ids.shape[1]

    @property
    def source_tokens(self) -> OneOrMoreTokenSequences:
        return self.sources.input_tokens

    @property
    def target_tokens(self) -> OneOrMoreTokenSequences:
        return self.targets.input_tokens

    @property
    def source_ids(self) -> IdsTensor:
        return self.sources.input_ids

    @property
    def target_ids(self) -> IdsTensor:
        return self.targets.input_ids

    @property
    def source_embeds(self) -> EmbeddingsTensor:
        return self.sources.input_embeds

    @property
    def target_embeds(self) -> EmbeddingsTensor:
        return self.targets.input_embeds

    @property
    def source_mask(self) -> IdsTensor:
        return self.sources.attention_mask

    @property
    def target_mask(self) -> IdsTensor:
        return self.targets.attention_mask

    def get_step_target(
        self, step: int, with_attention: bool = False
    ) -> Union[ExpandedTargetIdsTensor, Tuple[ExpandedTargetIdsTensor, ExpandedTargetIdsTensor]]:
        tgt = self.targets.input_ids[:, step]
        if with_attention:
            return tgt, self.targets.attention_mask[:, step]
        return tgt


@dataclass(eq=False, repr=False)
class DecoderOnlyBatch(Batch):
    @property
    def max_generation_length(self) -> int:
        return self.input_ids.shape[1]

    @property
    def source_tokens(self) -> OneOrMoreTokenSequences:
        return None

    @property
    def target_tokens(self) -> OneOrMoreTokenSequences:
        return self.input_tokens

    @property
    def source_ids(self) -> IdsTensor:
        return None

    @property
    def target_ids(self) -> IdsTensor:
        return self.input_ids

    @property
    def source_embeds(self) -> EmbeddingsTensor:
        return None

    @property
    def target_embeds(self) -> EmbeddingsTensor:
        return self.input_embeds

    @property
    def source_mask(self) -> IdsTensor:
        return None

    @property
    def target_mask(self) -> IdsTensor:
        return self.attention_mask

    def get_step_target(
        self, step: int, with_attention: bool = False
    ) -> Union[ExpandedTargetIdsTensor, Tuple[ExpandedTargetIdsTensor, ExpandedTargetIdsTensor]]:
        tgt = self.input_ids[:, step]
        if with_attention:
            return tgt, self.attention_mask[:, step]
        return tgt
