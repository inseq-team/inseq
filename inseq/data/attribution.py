from typing import List, NoReturn, Optional, Sequence, Tuple, Type, Union

import math
from dataclasses import dataclass
from itertools import dropwhile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..utils import pretty_list
from ..utils.typing import (
    AttributionOutputTensor,
    DeltaOutputTensor,
    OneOrMoreAttributionSequences,
    OneOrMoreIdSequences,
    OneOrMoreTokenSequences,
    TextInput,
)
from .batch import Batch, BatchEncoding

FeatureAttributionInput = Union[TextInput, BatchEncoding, Batch]
FeatureAttributionStepOutput = Union[
    Tuple[
        AttributionOutputTensor,
    ],
    Tuple[AttributionOutputTensor, DeltaOutputTensor],
]


@dataclass
class FeatureAttributionOutput:
    attributions: Optional[OneOrMoreAttributionSequences] = None
    delta: Optional[List[float]] = None
    source_ids: Optional[OneOrMoreIdSequences] = None
    prefix_ids: Optional[OneOrMoreIdSequences] = None
    target_ids: Optional[OneOrMoreIdSequences] = None
    source_tokens: Optional[OneOrMoreTokenSequences] = None
    prefix_tokens: Optional[OneOrMoreTokenSequences] = None
    target_tokens: Optional[OneOrMoreTokenSequences] = None

    def __str__(self):
        pretty_attrs = (
            [[round(v, 2) for v in a] for a in self.attributions]
            if self.attributions is not None
            else None
        )
        return (
            f"{self.__class__.__name__}(\n"
            f"   source_tokens={pretty_list(self.source_tokens)},\n"
            f"   prefix_tokens={pretty_list(self.prefix_tokens)},\n"
            f"   target_tokens={pretty_list(self.target_tokens)},\n"
            f"   source_ids={pretty_list(self.source_ids)},\n"
            f"   prefix_ids={pretty_list(self.prefix_ids)},\n"
            f"   target_ids={pretty_list(self.target_ids)},\n"
            f"   attributions={pretty_list(pretty_attrs)},\n"
            f"   delta={self.delta},\n"
            ")"
        )

    def set_attributions(
        self,
        attributions: AttributionOutputTensor,
        delta: Optional[DeltaOutputTensor] = None,
    ) -> None:
        attributions = attributions.detach().cpu().tolist()
        if delta is not None:
            self.delta = delta.detach().cpu().squeeze().tolist()
            if not isinstance(self.delta, list):
                self.delta = [self.delta]
        self.attributions = [
            list(reversed(list(dropwhile(lambda x: x == 0, reversed(sequence)))))
            if not all([math.isnan(x) for x in sequence])
            else []
            for sequence in attributions
        ]

    def __getitem__(self, index: Union[int, slice]) -> "FeatureAttributionOutput":
        return FeatureAttributionOutput(
            source_ids=self.source_ids[index] if self.source_ids is not None else None,
            prefix_ids=self.prefix_ids[index] if self.prefix_ids is not None else None,
            target_ids=self.target_ids[index] if self.target_ids is not None else None,
            source_tokens=self.source_tokens[index]
            if self.source_tokens is not None
            else None,
            prefix_tokens=self.prefix_tokens[index]
            if self.prefix_tokens is not None
            else None,
            target_tokens=self.target_tokens[index]
            if self.target_tokens is not None
            else None,
            attributions=self.attributions[index],
            delta=self.delta[index] if self.delta is not None else None,
        )


@dataclass
class FeatureAttributionSequenceOutput:
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

    source_ids: List[int]
    source_tokens: List[str]
    target_ids: List[int]
    target_tokens: List[str]
    attributions: OneOrMoreAttributionSequences
    deltas: Optional[List[float]] = None

    def __str__(self):
        pretty_attrs = [
            " " * 6
            + "[ "
            + " ".join([f" {v:.2f}" if v > 0 else f"{v:.2f}" for v in src_attr])
            + " ],\n"
            for src_attr in self.attributions
        ]
        pretty_deltas = (
            [f"{d:.2f}" if d > 0 else f"{d:.2f}" for d in self.deltas]
            if self.deltas
            else []
        )
        return (
            f"{self.__class__.__name__}(\n"
            f"   source_tokens={pretty_list(self.source_tokens)},\n"
            f"   target_tokens={pretty_list(self.target_tokens)},\n"
            f"   source_ids={pretty_list(self.source_ids)},\n"
            f"   target_ids={pretty_list(self.target_ids)},\n"
            f"   attributions=[\n"
            f"{''.join(pretty_attrs)}],\n"
            f"   deltas={pretty_deltas}\n"
            ")"
        )

    @classmethod
    def from_attributions(
        cls, attributions: List[FeatureAttributionOutput]
    ) -> "OneOrMoreFeatureAttributionSequenceOutputs":
        num_sequences = len(attributions[0].attributions)
        if not all([len(curr.attributions) == num_sequences for curr in attributions]):
            raise ValueError(
                "All the attributions must include the same number of sequences."
            )
        feat_attr_seq = []
        for seq_id in range(num_sequences):
            feat_attr_seq_args = {
                "source_ids": attributions[0].source_ids[seq_id],
                "source_tokens": attributions[0].source_tokens[seq_id],
                "target_ids": [
                    attr.target_ids[seq_id][0]
                    for attr in attributions
                    if attr.attributions[seq_id]
                ],
                "target_tokens": [
                    attr.target_tokens[seq_id][0]
                    for attr in attributions
                    if attr.attributions[seq_id]
                ],
                "attributions": [
                    attr.attributions[seq_id]
                    for attr in attributions
                    if attr.attributions[seq_id]
                ],
            }
            if all(a.delta is not None for a in attributions):
                feat_attr_seq_args["deltas"] = [
                    attr.delta[seq_id]
                    for attr in attributions
                    if attr.attributions[seq_id]
                ]
            feat_attr_seq.append(cls(**feat_attr_seq_args))
        if len(feat_attr_seq) == 1:
            return feat_attr_seq[0]
        return feat_attr_seq

    def heatmap(self, cmap=None, figsize=None) -> NoReturn:
        if not cmap:
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
        if not figsize:
            figsize = (1.5 * len(self.source_tokens), 0.7 * len(self.target_tokens))
        plt.subplots(figsize=figsize)
        sns.heatmap(
            np.array(self.attributions).T,
            xticklabels=self.target_tokens,
            yticklabels=self.source_tokens,
            cmap=cmap,
        )

    @property
    def minimum(self) -> float:
        return min(min(attr) for attr in self.attributions)

    @property
    def maximum(self) -> float:
        return max(max(attr) for attr in self.attributions)

    @property
    def scores(self) -> np.ndarray:
        return np.array(self.attributions).T


OneOrMoreFeatureAttributionSequenceOutputs = Union[
    FeatureAttributionSequenceOutput, List[FeatureAttributionSequenceOutput]
]
