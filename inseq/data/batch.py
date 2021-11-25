from typing import NoReturn, Optional, Union

from copy import deepcopy
from dataclasses import dataclass

from torchtyping import TensorType

from inseq.utils.misc import pretty_list

from ..utils import pretty_tensor
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

    def to(
        self, device: str, inplace: Optional[bool] = False
    ) -> Union[NoReturn, "BatchEncoding"]:
        if inplace:
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.baseline_ids.to(device) if self.baseline_ids is not None else None
        else:
            return BatchEncoding(
                self.input_ids.to(device),
                self.input_tokens,
                self.attention_mask.to(device),
                self.baseline_ids.to(device) if self.baseline_ids is not None else None,
            )

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    input_ids={pretty_tensor(self.input_ids)},\n"
            f"    input_tokens={pretty_list(self.input_tokens)},\n"
            f"    attention_mask={pretty_tensor(self.attention_mask)},\n"
            f"    baseline_ids={pretty_tensor(self.baseline_ids)},\n"
            ")"
        )


@dataclass
class BatchEmbedding:
    input_embeds: Optional[EmbeddingsTensor]
    baseline_embeds: Optional[EmbeddingsTensor]

    def __getitem__(self, subscript: Union[slice, int]) -> "BatchEmbedding":
        return BatchEmbedding(
            self.input_embeds[:, subscript, :]
            if self.input_embeds is not None
            else None,
            self.baseline_embeds[:, subscript, :]
            if self.baseline_embeds is not None
            else None,
        )

    def to(
        self, device: str, inplace: Optional[bool] = False
    ) -> Union[NoReturn, "BatchEmbedding"]:
        if inplace:
            self.input_embeds.to(device) if self.input_embeds is not None else None,
            self.baseline_embeds.to(
                device
            ) if self.baseline_embeds is not None else None,
        else:
            return BatchEmbedding(
                self.input_embeds.to(device) if self.input_embeds is not None else None,
                self.baseline_embeds.to(device)
                if self.baseline_embeds is not None
                else None,
            )

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    input_embeds={pretty_tensor(self.input_embeds)},\n"
            f"    baseline_embeds={pretty_tensor(self.baseline_embeds)},\n"
            ")"
        )


@dataclass
class Batch(BatchEncoding, BatchEmbedding):
    @classmethod
    def from_encoding_embeds(
        cls, encoding: BatchEncoding, embedding: BatchEmbedding
    ) -> "Batch":
        return cls(
            input_ids=encoding.input_ids,
            input_tokens=encoding.input_tokens,
            attention_mask=encoding.attention_mask,
            baseline_ids=encoding.baseline_ids,
            input_embeds=embedding.input_embeds,
            baseline_embeds=embedding.baseline_embeds,
        )

    def __getitem__(self, subscript: Union[slice, int]) -> "Batch":
        return Batch(
            input_ids=self.input_ids[:, subscript],
            input_tokens=[seq[subscript] for seq in self.input_tokens],
            attention_mask=self.attention_mask[:, subscript],
            baseline_ids=self.baseline_ids[:, subscript]
            if self.baseline_ids is not None
            else None,
            input_embeds=self.input_embeds[:, subscript, :]
            if self.input_embeds is not None
            else None,
            baseline_embeds=self.baseline_embeds[:, subscript, :]
            if self.baseline_embeds is not None
            else None,
        )

    def to(
        self, device: str, inplace: Optional[bool] = False
    ) -> Union[NoReturn, "Batch"]:
        if inplace:
            self.input_ids.to(device),
            self.input_tokens,
            self.attention_mask.to(device),
            self.baseline_ids.to(device) if self.baseline_ids is not None else None,
            self.input_embeds.to(device) if self.input_embeds is not None else None,
            self.baseline_embeds.to(
                device
            ) if self.baseline_embeds is not None else None,
        else:
            return Batch(
                input_ids=self.input_ids.to(device),
                input_tokens=self.input_tokens,
                attention_mask=self.attention_mask.to(device),
                baseline_ids=self.baseline_ids.to(device)
                if self.baseline_ids is not None
                else None,
                input_embeds=self.input_embeds.to(device)
                if self.input_embeds is not None
                else None,
                baseline_embeds=self.baseline_embeds.to(device)
                if self.baseline_embeds is not None
                else None,
            )

    def clone(self) -> "Batch":
        return Batch(
            input_ids=self.input_ids.clone(),
            input_tokens=deepcopy(self.input_tokens),
            attention_mask=self.attention_mask.clone(),
            baseline_ids=self.baseline_ids.clone()
            if self.baseline_ids is not None
            else None,
            input_embeds=self.input_embeds.clone()
            if self.input_embeds is not None
            else None,
            baseline_embeds=self.baseline_embeds.clone()
            if self.baseline_embeds is not None
            else None,
        )

    def select_active(
        self, mask: TensorType["batch_size", 1, int], inplace: Optional[bool] = False
    ) -> Union[NoReturn, "Batch"]:
        # Masked select of ids
        ids_shape = self.input_ids.shape[1:]
        active_mask = mask.bool()
        active_input_ids = self.input_ids.masked_select(active_mask)
        active_attention_mask = self.attention_mask.masked_select(active_mask)
        active_baseline_ids = None
        if self.baseline_ids is not None:
            active_baseline_ids = self.baseline_ids.masked_select(active_mask).reshape(
                -1, *ids_shape
            )
        # Masked select of embeddings
        active_input_embeds = None
        active_baseline_embeds = None
        if self.input_embeds is not None:
            embeds_shape = self.input_embeds.shape[1:]
            active_mask_embeds = mask.unsqueeze(-1).bool()
            active_input_embeds = self.input_embeds.masked_select(
                active_mask_embeds
            ).reshape(-1, *embeds_shape)
            if self.baseline_embeds is not None:
                active_baseline_embeds = self.baseline_embeds.masked_select(
                    active_mask_embeds
                ).reshape(-1, *embeds_shape)
        if inplace:
            self.input_ids = active_input_ids.reshape(-1, *ids_shape)
            self.input_tokens = [
                sent
                for i, sent in enumerate(self.input_tokens)
                if active_mask.tolist()[i]
            ]
            self.attention_mask = active_attention_mask.reshape(-1, *ids_shape)
            self.baseline_ids = active_baseline_ids
            self.input_embeds = active_input_embeds
            self.baseline_embeds = active_baseline_embeds
        else:
            return Batch(
                input_ids=active_input_ids.reshape(-1, *ids_shape),
                input_tokens=[
                    sent
                    for i, sent in enumerate(self.input_tokens)
                    if active_mask.tolist()[i]
                ],
                attention_mask=active_attention_mask.reshape(-1, *ids_shape),
                baseline_ids=active_baseline_ids,
                input_embeds=active_input_embeds,
                baseline_embeds=active_baseline_embeds,
            )

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
        return EncoderDecoderBatch(
            sources=self.sources, targets=self.targets[subscript]
        )

    def to(
        self, device: str, inplace: Optional[bool] = False
    ) -> Union[NoReturn, "EncoderDecoderBatch"]:
        if inplace:
            self.sources.to(device),
            self.targets.to(device)
        else:
            return EncoderDecoderBatch(
                sources=self.sources.to(device), targets=self.targets.to(device)
            )

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
        return EncoderDecoderBatch(
            sources=self.sources.clone(), targets=self.targets.clone()
        )

    def __str__(self):
        source_str = str(self.sources).replace("\n", "\n    ")
        target_str = str(self.targets).replace("\n", "\n    ")
        return (
            f"{self.__class__.__name__}(\n"
            f"    sources={source_str},\n"
            f"    targets={target_str}\n"
            ")"
        )
