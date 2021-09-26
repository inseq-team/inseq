from typing import List, NoReturn, Optional, Sequence, Tuple, Union

import math
from dataclasses import dataclass
from itertools import dropwhile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from inseq import attr

from ..utils import pretty_list
from .batch import Batch, BatchEncoding

TextInput = Union[str, Sequence[str]]

OneOrMoreIdSequences = Sequence[Sequence[int]]
OneOrMoreTokenSequences = Sequence[Sequence[str]]
OneOrMoreAttributionSequences = Sequence[Sequence[float]]

FeatureAttributionInput = Union[TextInput, BatchEncoding, Batch]

# For Huggingface it's a string identifier e.g. "t5-base", "Helsinki-NLP/opus-mt-en-it"
# For Fairseq it's a tuple of strings containing repo and model name e.g. ("pytorch/fairseq", "transformer.wmt14.en-fr")
ModelIdentifier = Union[str, Tuple[str, str]]


@dataclass
class FeatureAttributionOutput:
    attributions: OneOrMoreAttributionSequences
    delta: Optional[List[float]] = None
    source_ids: Optional[OneOrMoreIdSequences] = None
    prefix_ids: Optional[OneOrMoreIdSequences] = None
    target_ids: Optional[OneOrMoreIdSequences] = None
    source_tokens: Optional[OneOrMoreTokenSequences] = None
    prefix_tokens: Optional[OneOrMoreTokenSequences] = None
    target_tokens: Optional[OneOrMoreTokenSequences] = None

    def __str__(self):
        pretty_attrs = [[round(v, 2) for v in a] for a in self.attributions]
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

    def fix_attributions(self):
        attributions = self.attributions.detach().cpu().tolist()
        if self.delta is not None:
            self.delta = self.delta.detach().cpu().squeeze().tolist()
        if isinstance(self.delta, float):
            self.delta = [self.delta]
        self.attributions = [
            list(reversed(list(dropwhile(lambda x: x == 0, reversed(sequence)))))
            if not all([math.isnan(x) for x in sequence])
            else []
            for sequence in attributions
        ]

    def check_consistency(self):
        # No batch is missing
        assert (
            len(self.source_tokens)
            == len(self.prefix_tokens)
            == len(self.target_tokens)
            == len(self.attributions)
        )
        # Number of in-batch elements is consistent
        assert all(
            len(t) == len(id) for t, id in zip(self.source_tokens, self.source_ids)
        )
        assert all(
            len(t) == len(id) for t, id in zip(self.prefix_tokens, self.prefix_ids)
        )
        assert all(
            len(t) == len(id) for t, id in zip(self.target_tokens, self.target_ids)
        )
        if self.delta is not None:
            assert isinstance(self.delta, list) and all(
                isinstance(x, float) for x in self.delta
            )
        # Attributions size match either the source or the prefix
        # assert all(len(a) == len(t) for a, t in zip(self.attributions, self.source_tokens)) or \
        #    all(len(a) == len(t) for a, t in zip(self.attributions, self.prefix_tokens)), (self.attributions, self.source_tokens)

    def __getitem__(self, index: Union[int, slice]):
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
        return (
            f"{self.__class__.__name__}(\n"
            f"   source_tokens={pretty_list(self.source_tokens)},\n"
            f"   target_tokens={pretty_list(self.target_tokens)},\n"
            f"   source_ids={pretty_list(self.source_ids)},\n"
            f"   target_ids={pretty_list(self.target_ids)},\n"
            f"   attributions={[[round(v, 2) for v in src_attr] for src_attr in self.attributions]},\n"
            f"   deltas={self.deltas}\n"
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


OneOrMoreFeatureAttributionSequenceOutputs = Union[
    FeatureAttributionSequenceOutput, List[FeatureAttributionSequenceOutput]
]
