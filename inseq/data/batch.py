from typing import List, Optional, Union

from copy import deepcopy
from dataclasses import dataclass, fields

import torch
from torchtyping import TensorType

from ..utils import pretty_dict
from ..utils.typing import EmbeddingsTensor, IdsTensor, OneOrMoreTokenSequences


@dataclass
class TensorWrapper:
    @staticmethod
    def _getitem(attr, subscript):
        if isinstance(attr, torch.Tensor):
            if len(attr.shape) == 1:
                return attr[subscript]
            if len(attr.shape) >= 2:
                return attr[:, subscript, ...]
        elif isinstance(attr, TensorWrapper):
            return attr[subscript]
        elif isinstance(attr, list) and isinstance(attr[0], list):
            return [seq[subscript] for seq in attr]
        elif isinstance(attr, dict):
            return {key: TensorWrapper._getitem(val, subscript) for key, val in attr.items()}
        else:
            return attr

    @staticmethod
    def _select_active(attr, mask):
        if isinstance(attr, torch.Tensor):
            if len(attr.shape) <= 1:
                return attr
            else:
                curr_mask = mask.clone()
                if curr_mask.dtype != torch.bool:
                    curr_mask = curr_mask.bool()
                while len(curr_mask.shape) < len(attr.shape):
                    curr_mask = curr_mask.unsqueeze(-1)
                orig_shape = attr.shape[1:]
                return attr.masked_select(curr_mask).reshape(-1, *orig_shape)
        elif isinstance(attr, TensorWrapper):
            return attr.select_active(mask)
        elif isinstance(attr, list):
            return [val for i, val in enumerate(attr) if mask.tolist()[i]]
        elif isinstance(attr, dict):
            return {key: TensorWrapper._select_active(val, mask) for key, val in attr.items()}
        else:
            return attr

    @staticmethod
    def _to(attr, device: str):
        if isinstance(attr, torch.Tensor) or isinstance(attr, TensorWrapper):
            return attr.to(device)
        elif isinstance(attr, dict):
            return {key: TensorWrapper._to(val, device) for key, val in attr.items()}
        else:
            return attr

    @staticmethod
    def _detach(attr):
        if isinstance(attr, torch.Tensor) or isinstance(attr, TensorWrapper):
            return attr.detach()
        elif isinstance(attr, dict):
            return {key: TensorWrapper._detach(val) for key, val in attr.items()}
        else:
            return attr

    def __getitem__(self, subscript):
        return self.__class__(
            **{field.name: self._getitem(getattr(self, field.name), subscript) for field in fields(self.__class__)}
        )

    def select_active(self, mask: TensorType["batch_size", 1, int]):
        return self.__class__(
            **{field.name: self._select_active(getattr(self, field.name), mask) for field in fields(self.__class__)}
        )

    def to(self, device: str):
        for field in fields(self.__class__):
            attr = getattr(self, field.name)
            setattr(self, field.name, self._to(attr, device))
        return self

    def detach(self):
        for field in fields(self.__class__):
            attr = getattr(self, field.name)
            setattr(self, field.name, self._detach(attr))
        return self

    def clone(self):
        out_params = {}
        for field in fields(self.__class__):
            attr = getattr(self, field.name)
            if isinstance(attr, torch.Tensor) or isinstance(attr, TensorWrapper):
                out_params[field.name] = attr.clone()
            elif attr is not None:
                out_params[field.name] = deepcopy(attr)
            else:
                out_params[field.name] = None
        return self.__class__(**out_params)

    def to_dict(self):
        return self.__dict__

    def __str__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.to_dict())})"


@dataclass
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


@dataclass
class BatchEmbedding(TensorWrapper):
    input_embeds: Optional[EmbeddingsTensor] = None
    baseline_embeds: Optional[EmbeddingsTensor] = None


@dataclass
class Batch(TensorWrapper):
    encoding: BatchEncoding
    embedding: BatchEmbedding

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


@dataclass
class EncoderDecoderBatch(TensorWrapper):
    sources: Batch
    targets: Batch

    def __getitem__(self, subscript: Union[slice, int]) -> "EncoderDecoderBatch":
        return EncoderDecoderBatch(sources=self.sources, targets=self.targets[subscript])
