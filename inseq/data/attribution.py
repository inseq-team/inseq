from typing import Any, Dict, List, Optional, Tuple, Union

import json
from dataclasses import dataclass

import numpy as np
from torchtyping import TensorType

from ..utils import pad, pretty_dict, remap_from_filtered
from ..utils.typing import (
    AttributionOutputTensor,
    DeltaOutputTensor,
    IdsTensor,
    OneOrMoreAttributionSequences,
    OneOrMoreIdSequences,
    OneOrMoreTokenSequences,
    TextInput,
    TopProbabilitiesTensor,
)
from .batch import Batch, BatchEncoding


FeatureAttributionInput = Union[TextInput, BatchEncoding, Batch]


@dataclass
class FeatureAttributionStepOutput:
    """
    Raw output of a single step of feature attribution
    """

    source_attributions: AttributionOutputTensor
    target_attributions: Optional[AttributionOutputTensor] = None
    deltas: Optional[DeltaOutputTensor] = None
    probabilities: Optional[TopProbabilitiesTensor] = None

    def detach(self) -> "FeatureAttributionStepOutput":
        self.source_attributions.detach()
        if self.target_attributions is not None:
            self.target_attributions.detach()
        if self.deltas is not None:
            self.deltas.detach()
        if self.probabilities is not None:
            self.probabilities.detach()
        return self

    def to(self, device: str) -> "FeatureAttributionStepOutput":
        self.source_attributions.to(device)
        if self.target_attributions is not None:
            self.target_attributions.to(device)
        if self.deltas is not None:
            self.deltas.to(device)
        if self.probabilities is not None:
            self.probabilities.to(device)
        return self

    def remap_from_filtered(
        self,
        source_token_indexes: IdsTensor,
        target_token_indexes: IdsTensor,
        target_attention_mask: TensorType["batch_size", 1, int],
    ):
        self.source_attributions = remap_from_filtered(
            source=source_token_indexes,  # orig_batch.sources.input_ids,
            mask=target_attention_mask,
            filtered=self.source_attributions,
        )
        if self.target_attributions is not None:
            self.target_attributions = remap_from_filtered(
                source=target_token_indexes,  # orig_batch.targets.input_ids,
                mask=target_attention_mask,
                filtered=self.target_attributions,
            )
        if self.deltas is not None:
            self.deltas = remap_from_filtered(
                source=target_attention_mask.squeeze(),
                mask=target_attention_mask,
                filtered=self.deltas,
            )
        if self.probabilities is not None:
            self.probabilities = remap_from_filtered(
                source=target_attention_mask.squeeze(),
                mask=target_attention_mask,
                filtered=self.probabilities,
            )


@dataclass
class FeatureAttributionOutput:
    source_attributions: Optional[OneOrMoreAttributionSequences] = None
    target_attributions: Optional[OneOrMoreAttributionSequences] = None
    delta: Optional[List[float]] = None
    probabilities: Optional[List[float]] = None
    source_ids: Optional[OneOrMoreIdSequences] = None
    prefix_ids: Optional[OneOrMoreIdSequences] = None
    target_ids: Optional[OneOrMoreIdSequences] = None
    source_tokens: Optional[OneOrMoreTokenSequences] = None
    prefix_tokens: Optional[OneOrMoreTokenSequences] = None
    target_tokens: Optional[OneOrMoreTokenSequences] = None

    def __str__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.__dict__)}"

    def __getitem__(self, index: Union[int, slice]) -> "FeatureAttributionOutput":
        return FeatureAttributionOutput(
            source_ids=self.source_ids[index] if self.source_ids is not None else None,
            prefix_ids=self.prefix_ids[index] if self.prefix_ids is not None else None,
            target_ids=self.target_ids[index] if self.target_ids is not None else None,
            source_tokens=self.source_tokens[index] if self.source_tokens is not None else None,
            prefix_tokens=self.prefix_tokens[index] if self.prefix_tokens is not None else None,
            target_tokens=self.target_tokens[index] if self.target_tokens is not None else None,
            source_attributions=self.source_attributions[index],
            target_attributions=self.target_attributions[index],
            delta=self.delta[index] if self.delta is not None else None,
            probabilities=self.probabilities[index] if self.probabilities is not None else None,
        )


@dataclass
class FeatureAttributionSequenceOutput:
    """
    Output produced by a standard attribution method.

    Attributes:
        source_tokens (list[str]): Tokenized source sequence.
        target_tokens (list[str]): Tokenized target sequence.
        source_attributions (list[list[str]]): List of length len(target_tokens) containing
            lists of attributions of length len(source_tokens) for each
            source-target token pair (full matrix).
        target_attributions (list[list[str]]): List of length len(source_tokens) containing
            lists of attributions of length len(target_tokens) for each
            target-target token pair (triangular matrix).
        deltas (list[float], optional): List of length len(target_tokens) containing
            the deltas for the approximate integration of the gradients for each
            target token.
        probabilities (list[float], optional): List of length len(target_tokens) containing
            the probabilities of each target token.

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
            source_attributions=[ [ 0.85, ... ], [ 0.42, ... ], ... ],
            target_attributions=[ [ 0.85, ... ], [ 0.42, ... ], ... ],
            deltas=[ 0.01, ... ],
            probabilities=[ 0.42, ... ]
        )
    """

    source_ids: List[int]
    source_tokens: List[str]
    target_ids: List[int]
    target_tokens: List[str]
    source_attributions: OneOrMoreAttributionSequences
    target_attributions: Optional[OneOrMoreAttributionSequences] = None
    deltas: Optional[List[float]] = None
    probabilities: Optional[List[float]] = None

    def __str__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.__dict__)})"

    @classmethod
    def from_attributions(
        cls, attributions: List[FeatureAttributionOutput]
    ) -> "OneOrMoreFeatureAttributionSequenceOutputs":
        num_sequences = len(attributions[0].source_attributions)
        if not all([len(curr.source_attributions) == num_sequences for curr in attributions]):
            raise ValueError("All the attributions must include the same number of sequences.")
        feat_attr_seq = []
        for seq_id in range(num_sequences):
            feat_attr_seq_args = {
                "source_ids": attributions[0].source_ids[seq_id],
                "source_tokens": attributions[0].source_tokens[seq_id],
                "target_ids": [
                    attr.target_ids[seq_id][0] for attr in attributions if attr.source_attributions[seq_id]
                ],
                "target_tokens": [
                    attr.target_tokens[seq_id][0] for attr in attributions if attr.source_attributions[seq_id]
                ],
                "source_attributions": [
                    attr.source_attributions[seq_id] for attr in attributions if attr.source_attributions[seq_id]
                ],
            }
            if all(a.delta is not None for a in attributions):
                feat_attr_seq_args["deltas"] = [
                    attr.delta[seq_id] for attr in attributions if attr.source_attributions[seq_id]
                ]
            if all(a.probabilities is not None for a in attributions):
                feat_attr_seq_args["probabilities"] = [
                    attr.probabilities[seq_id] for attr in attributions if attr.source_attributions[seq_id]
                ]
            if all(a.target_attributions is not None for a in attributions):
                feat_attr_seq_args["target_attributions"] = [
                    attr.target_attributions[seq_id] for attr in attributions if attr.source_attributions[seq_id]
                ]
            feat_attr_seq.append(cls(**feat_attr_seq_args))
        if len(feat_attr_seq) == 1:
            return feat_attr_seq[0]
        return feat_attr_seq

    def show(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        display: bool = True,
        return_html: Optional[bool] = False,
    ) -> Optional[str]:
        from inseq import show_attributions

        return show_attributions(self, min_val, max_val, display, return_html)

    @property
    def minimum(self) -> float:
        values = [np.amin(self.source_scores)]
        if self.target_attributions is not None:
            values.append(np.amin(self.target_scores))
        return min(values)

    @property
    def maximum(self) -> float:
        values = [np.amax(self.source_scores)]
        if self.target_attributions is not None:
            values.append(np.amax(self.target_scores))
        return min(values)

    @property
    def source_scores(self) -> np.ndarray:
        return np.array(self.source_attributions).T

    @property
    def target_scores(self) -> Optional[np.ndarray]:
        if self.target_attributions is None:
            return None
        # Add an empty row to the target_attributions to make it a square matrix.
        return np.vstack(
            (np.array(pad(self.target_attributions, np.nan)).T, np.ones(len(self.target_attributions)) * np.nan)
        )

    def as_dict(self) -> Dict[str, List[Any]]:
        return self.__dict__


OneOrMoreFeatureAttributionSequenceOutputs = Union[
    FeatureAttributionSequenceOutput, List[FeatureAttributionSequenceOutput]
]
# One or more FeatureAttributionSequenceOutput, with an optional added list
# of single FeatureAttributionOutputs for each step.
OneOrMoreFeatureAttributionSequenceOutputsWithStepOutputs = Union[
    OneOrMoreFeatureAttributionSequenceOutputs,
    Tuple[OneOrMoreFeatureAttributionSequenceOutputs, List[FeatureAttributionOutput]],
]


def save_attributions(
    attributions: OneOrMoreFeatureAttributionSequenceOutputs,
    path: str,
    overwrite: bool = False,
) -> None:
    """
    Save attributions to a file.

    Args:
        attributions: Attributions to save.
        path: Path to save the attributions to.
        overwrite: If True, overwrite the file if it exists.
    """
    if isinstance(attributions, FeatureAttributionSequenceOutput):
        attributions = [attributions]
    with open(path, "w" if overwrite else "a") as f:
        for attribution in attributions:
            f.write(f"{json.dumps(attribution.as_dict())}\n")


def load_attributions(
    path: str,
) -> OneOrMoreFeatureAttributionSequenceOutputs:
    """
    Load attributions from a file.

    Args:
        path: Path to a JSONlines file containing one serialized FeatureAttributionSequenceOutput
            object per line.

    Returns:
        A list of FeatureAttributionSequenceOutput loaded from the file.
    """
    with open(path) as f:
        attributions = [json.loads(line) for line in f]
    return [FeatureAttributionSequenceOutput(**attribution) for attribution in attributions]
