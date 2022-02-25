from typing import List, NoReturn, Optional, Union

from copy import deepcopy
from dataclasses import dataclass

from torchtyping import TensorType

from ..utils import pretty_dict, pretty_list, pretty_tensor
from ..utils.typing import EmbeddingsTensor, IdsTensor, OneOrMoreTokenSequences


@dataclass
class BatchEncoding:
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

    def __getitem__(self, subscript: Union[slice, int]) -> "BatchEncoding":
        return BatchEncoding(
            self.input_ids[:, subscript],
            [seq[subscript] for seq in self.input_tokens],
            self.attention_mask[:, subscript],
            self.baseline_ids[:, subscript] if self.baseline_ids is not None else None,
        )

    def to(self, device: str) -> "BatchEncoding":
        self.input_ids.to(device)
        self.attention_mask.to(device)
        self.baseline_ids.to(device) if self.baseline_ids is not None else None
        return self

    def clone(self) -> "BatchEncoding":
        cloned_baseline_ids = None
        if self.baseline_ids is not None:
            cloned_baseline_ids = self.baseline_ids.clone()
        return BatchEncoding(
            self.input_ids.clone(),
            deepcopy(self.input_tokens),
            self.attention_mask.clone(),
            cloned_baseline_ids,
        )

    def select_active(self, mask: TensorType["batch_size", 1, int]) -> "BatchEncoding":
        ids_shape = self.input_ids.shape[1:]
        active_mask = mask.bool()
        active_input_ids = self.input_ids.masked_select(active_mask)
        active_input_tokens = [seq for i, seq in enumerate(self.input_tokens) if active_mask.tolist()[i]]
        active_attention_mask = self.attention_mask.masked_select(active_mask)
        active_baseline_ids = None
        if self.baseline_ids is not None:
            active_baseline_ids = self.baseline_ids.masked_select(active_mask)
        return BatchEncoding(
            active_input_ids.reshape(-1, *ids_shape),
            active_input_tokens,
            active_attention_mask.reshape(-1, *ids_shape),
            active_baseline_ids.reshape(-1, *ids_shape),
        )

    def __str__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.__dict__)})"


@dataclass
class BatchEmbedding:
    input_embeds: Optional[EmbeddingsTensor] = None
    baseline_embeds: Optional[EmbeddingsTensor] = None

    def __getitem__(self, subscript: Union[slice, int]) -> "BatchEmbedding":
        return BatchEmbedding(
            self.input_embeds[:, subscript, :] if self.input_embeds is not None else None,
            self.baseline_embeds[:, subscript, :] if self.baseline_embeds is not None else None,
        )

    def to(self, device: str) -> "BatchEmbedding":
        self.input_embeds.to(device) if self.input_embeds is not None else None
        self.baseline_embeds.to(device) if self.baseline_embeds is not None else None
        return self

    def clone(self) -> "BatchEmbedding":
        cloned_input_embeds = None
        cloned_baseline_embeds = None
        if self.input_embeds is not None:
            cloned_input_embeds = self.input_embeds.clone()
        if self.baseline_embeds is not None:
            cloned_baseline_embeds = self.baseline_embeds.clone()
        return BatchEmbedding(cloned_input_embeds, cloned_baseline_embeds)

    def select_active(self, mask: TensorType["batch_size", 1, int]) -> "BatchEmbedding":
        active_input_embeds = None
        active_baseline_embeds = None
        active_mask_embeds = mask.unsqueeze(-1).bool()
        if self.input_embeds is not None:
            embeds_shape = self.input_embeds.shape[1:]
            active_input_embeds = self.input_embeds.masked_select(active_mask_embeds).reshape(-1, *embeds_shape)
        if self.baseline_embeds is not None:
            embeds_shape = self.baseline_embeds.shape[1:]
            active_baseline_embeds = self.baseline_embeds.masked_select(active_mask_embeds).reshape(-1, *embeds_shape)
        return BatchEmbedding(active_input_embeds, active_baseline_embeds)

    def __str__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.__dict__)})"


@dataclass
class Batch:
    encoding: BatchEncoding
    embedding: BatchEmbedding

    def __getitem__(self, subscript: Union[slice, int]) -> "Batch":
        return Batch(encoding=self.encoding[subscript], embedding=self.embedding[subscript])

    def to(self, device: str) -> "Batch":
        self.encoding.to(device)
        self.embedding.to(device)
        return self

    def clone(self) -> "Batch":
        return Batch(
            encoding=self.encoding.clone(),
            embedding=self.embedding.clone(),
        )

    def select_active(
        self, mask: TensorType["batch_size", 1, int], inplace: Optional[bool] = False
    ) -> Union[NoReturn, "Batch"]:
        if inplace:
            self.encoding.select_active(mask),
            self.embedding.select_active(mask)
        else:
            return Batch(
                encoding=self.encoding.select_active(mask),
                embedding=self.embedding.select_active(mask),
            )

    @property
    def input_ids(self) -> IdsTensor:
        return self.encoding.input_ids

    @property
    def input_tokens(self) -> List[List[str]]:
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

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    input_ids={pretty_tensor(self.input_ids)},\n"
            f"    input_tokens={pretty_list(self.input_tokens)},\n"
            f"    attention_mask={pretty_tensor(self.attention_mask)},\n"
            f"    baseline_ids={pretty_tensor(self.baseline_ids)},\n"
            f"    input_embeds={pretty_tensor(self.input_embeds)},\n"
            f"    baseline_embeds={pretty_tensor(self.baseline_embeds)},\n"
            ")"
        )


@dataclass
class EncoderDecoderBatch:
    sources: Batch
    targets: Batch

    def __getitem__(self, subscript: Union[slice, int]) -> "EncoderDecoderBatch":
        return EncoderDecoderBatch(sources=self.sources, targets=self.targets[subscript])

    def to(self, device: str) -> "EncoderDecoderBatch":
        self.sources.to(device)
        self.targets.to(device)
        return self

    def select_active(
        self, mask: TensorType["batch_size", 1, int], inplace: Optional[bool] = False
    ) -> Union[NoReturn, "EncoderDecoderBatch"]:
        if inplace:
            self.sources.select_active(mask, inplace)
            self.targets.select_active(mask, inplace)
        else:
            return EncoderDecoderBatch(
                sources=self.sources.select_active(mask),
                targets=self.targets.select_active(mask),
            )

    def clone(self) -> "EncoderDecoderBatch":
        return EncoderDecoderBatch(sources=self.sources.clone(), targets=self.targets.clone())

    def __str__(self):
        source_str = str(self.sources).replace("\n", "\n    ")
        target_str = str(self.targets).replace("\n", "\n    ")
        return f"{self.__class__.__name__}(\n" f"    sources={source_str},\n" f"    targets={target_str}\n" ")"
