from typing import Any, Dict, List, Optional, Union

import json
import os
from dataclasses import dataclass, field

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
class FeatureAttributionRawStepOutput:
    """
    Raw output of a single step of feature attribution
    """

    source_attributions: AttributionOutputTensor
    target_attributions: Optional[AttributionOutputTensor] = None
    deltas: Optional[DeltaOutputTensor] = None
    probabilities: Optional[TopProbabilitiesTensor] = None

    def detach(self) -> "FeatureAttributionRawStepOutput":
        self.source_attributions.detach()
        if self.target_attributions is not None:
            self.target_attributions.detach()
        if self.deltas is not None:
            self.deltas.detach()
        if self.probabilities is not None:
            self.probabilities.detach()
        return self

    def to(self, device: str) -> "FeatureAttributionRawStepOutput":
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
class FeatureAttributionStepOutput:
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

    def __getitem__(self, index: Union[int, slice]) -> "FeatureAttributionStepOutput":
        return FeatureAttributionStepOutput(
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
    def from_step_attributions(
        cls, attributions: List[FeatureAttributionStepOutput]
    ) -> List["FeatureAttributionSequenceOutput"]:
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


@dataclass
class FeatureAttributionOutput:
    """
    Output produced by the `AttributionModel.attribute` method.

    Attributes:
        sequence_attributions (list of :class:`~inseq.data.FeatureAttributionSequenceOutput`): List
                        containing all attributions performed on input sentences (one per input sentence, including
                        source and optionally target-side attribution).
                step_attributions (list of :class:`~inseq.data.FeatureAttributionStepOutput`, optional): List
                        containing all step attributions (one per generation step performed on the batch), returned if
                        `output_step_attributions=True`.
                info (dict with str keys and any values): Dictionary including all available parameters used to
                        perform the attribution.
    """

    sequence_attributions: List[FeatureAttributionSequenceOutput]
    step_attributions: Optional[List[FeatureAttributionStepOutput]] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.__dict__)})"

    def save(self, path: str, overwrite: bool = False) -> None:
        """
        Save class contents to a JSON file.

        Args:
            path (str): Path to save the attributions to.
            overwrite (bool): If True, overwrite the file if it exists, raise error otherwise.
        """
        if not overwrite and os.path.exists(path):
            raise ValueError(f"File {path} already exists.")
        out = json.dumps(
            {
                "sequence_attributions": [seq.__dict__ for seq in self.sequence_attributions],
                "step_attributions": [step.__dict__ for step in self.step_attributions]
                if self.step_attributions
                else None,
                "info": self.info,
            }
        )
        with open(path, "w") as f:
            f.write(out)

    @classmethod
    def load(cls, path: str) -> "FeatureAttributionOutput":
        """Loads a saved attribution output into an object

        Args:
            path (str): Path to the saved attribution output

        Returns:
            :class:`~inseq.data.FeatureAttributionOutput`: Loaded attribution output
        """
        with open(path) as f:
            output = json.loads(f.read())
        return cls(
            sequence_attributions=[FeatureAttributionSequenceOutput(**seq) for seq in output["sequence_attributions"]],
            step_attributions=[FeatureAttributionStepOutput(**step) for step in output["step_attributions"]]
            if output["step_attributions"]
            else None,
            info=output["info"],
        )

    def show(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        display: bool = True,
        return_html: Optional[bool] = False,
    ) -> Optional[str]:
        """Visualize the sequence attributions.

        Args:
            min_val (int, optional): Minimum value for color scale.
            max_val (int, optional): Maximum value for color scale.
            display (bool, optional): If True, display the attribution visualization.
            return_html (bool, optional): If True, return the attribution visualization as HTML.

        Returns:
            str: Attribution visualization as HTML if `return_html=True`, None otherwise.
        """
        from inseq import show_attributions

        return show_attributions(self.sequence_attributions, min_val, max_val, display, return_html)
