from typing import Any, Dict, List, Optional, Type, Union

import json
import os
from dataclasses import dataclass, field, fields

import torch

from ..utils import drop_padding, get_sequences_from_batched_steps, pretty_dict, remap_from_filtered
from ..utils.typing import (
    MultipleScoresPerSequenceTensor,
    MultipleScoresPerStepTensor,
    OneOrMoreTokenWithIdSequences,
    SequenceAttributionTensor,
    SingleScorePerStepTensor,
    SingleScoresPerSequenceTensor,
    StepAttributionTensor,
    TargetIdsTensor,
    TextInput,
    TokenWithId,
)
from .aggregator import Aggregator, SumNormAggregator
from .batch import Batch, BatchEncoding, TensorWrapper


FeatureAttributionInput = Union[TextInput, BatchEncoding, Batch]


@dataclass
class FeatureAttributionStepOutput(TensorWrapper):
    """
    Output of a single step of feature attribution, plus
    extra information related to what was attributed.
    """

    source_attributions: StepAttributionTensor
    target_attributions: Optional[StepAttributionTensor] = None
    step_scores: Optional[Dict[str, SingleScorePerStepTensor]] = None
    sequence_scores: Optional[Dict[str, MultipleScoresPerStepTensor]] = None
    source: Optional[OneOrMoreTokenWithIdSequences] = None
    prefix: Optional[OneOrMoreTokenWithIdSequences] = None
    target: Optional[OneOrMoreTokenWithIdSequences] = None

    def remap_from_filtered(
        self,
        target_attention_mask: TargetIdsTensor,
    ) -> None:
        self.source_attributions = remap_from_filtered(
            original_shape=(len(self.source), *self.source_attributions.shape[1:]),
            mask=target_attention_mask,
            filtered=self.source_attributions,
        )
        if self.target_attributions is not None:
            self.target_attributions = remap_from_filtered(
                original_shape=(len(self.prefix), *self.target_attributions.shape[1:]),
                mask=target_attention_mask,
                filtered=self.target_attributions,
            )
        if self.step_scores is not None:
            for score_name, score_tensor in self.step_scores.items():
                self.step_scores[score_name] = remap_from_filtered(
                    original_shape=(len(self.prefix), 1),
                    mask=target_attention_mask,
                    filtered=score_tensor.unsqueeze(-1),
                ).squeeze(-1)
        if self.sequence_scores is not None:
            for score_name, score_tensor in self.sequence_scores.items():
                self.sequence_scores[score_name] = remap_from_filtered(
                    original_shape=(len(self.source), *self.source_attributions.shape[1:]),
                    mask=target_attention_mask,
                    filtered=score_tensor,
                )


@dataclass
class FeatureAttributionSequenceOutput(TensorWrapper):
    """
    Output produced by a standard attribution method.

    Attributes:
        source (list of :class:`~inseq.utils.typing.TokenWithId): Tokenized source sequence.
        target (list of :class:`~inseq.utils.typing.TokenWithId): Tokenized target sequence.
        source_attributions (:obj:`SequenceAttributionTensor`): Tensor of shape (`source_len`,
            `target_len`) plus an optional third dimension if the attribution is granular (e.g.
            gradient attribution) containing the attribution scores produced at each generation step of
            the target for every source token.
        target_attributions (:obj:`SequenceAttributionTensor`, optional): Tensor of shape
            (`target_len`, `target_len`), plus an optional third dimension if
            the attribution is granular containing the attribution scores produced at each generation
            step of the target for every token in the target prefix.
        step_scores (:obj:`dict[str, SingleScorePerStepTensor]`, optional): Dictionary of step scores
            produced alongside attributions (one per generation step).
        sequence_scores (:obj:`dict[str, MultipleScoresPerStepTensor]`, optional): Dictionary of sequence
            scores produced alongside attributions (n per generation step, as for attributions).
    """

    source: List[TokenWithId]
    target: List[TokenWithId]
    source_attributions: SequenceAttributionTensor
    target_attributions: Optional[SequenceAttributionTensor] = None
    step_scores: Optional[Dict[str, SingleScoresPerSequenceTensor]] = None
    sequence_scores: Optional[Dict[str, MultipleScoresPerSequenceTensor]] = None
    aggregator: Type[Aggregator] = SumNormAggregator

    @classmethod
    def from_step_attributions(
        cls,
        attributions: List[FeatureAttributionStepOutput],
        pad_id: Optional[Any] = None,
        has_bos_token: bool = True,
    ) -> List["FeatureAttributionSequenceOutput"]:
        attr = attributions[0]
        num_sequences = len(attr.source_attributions)
        if not all([len(att.source_attributions) == num_sequences for att in attributions]):
            raise ValueError("All the attributions must include the same number of sequences.")
        source_attributions = get_sequences_from_batched_steps([att.source_attributions for att in attributions])
        seq_attributions = []
        sources = [drop_padding(attr.source[seq_id], pad_id) for seq_id in range(num_sequences)]
        targets = [
            drop_padding([a.target[seq_id][0] for a in attributions], pad_id) for seq_id in range(num_sequences)
        ]
        for seq_id in range(num_sequences):
            # Remove padding from tensor
            filtered_source_attribution = source_attributions[seq_id][
                : len(sources[seq_id]), : len(targets[seq_id]), ...
            ]
            seq_attributions.append(
                cls(
                    source=sources[seq_id],
                    target=targets[seq_id],
                    source_attributions=filtered_source_attribution,
                )
            )
        if attr.target_attributions is not None:
            target_attributions = get_sequences_from_batched_steps(
                [att.target_attributions for att in attributions], pad_dims=(1,)
            )
            for seq_id in range(num_sequences):
                if has_bos_token:
                    target_attributions[seq_id] = target_attributions[seq_id][1:, ...]
                target_attributions[seq_id] = target_attributions[seq_id][
                    : len(targets[seq_id]), : len(targets[seq_id]), ...
                ]
                if target_attributions[seq_id].shape[0] != len(targets[seq_id]):
                    empty_final_row = torch.ones(1, *target_attributions[seq_id].shape[1:]) * float("nan")
                    target_attributions[seq_id] = torch.cat([target_attributions[seq_id], empty_final_row], dim=0)
                seq_attributions[seq_id].target_attributions = target_attributions[seq_id]
        if attr.step_scores is not None:
            step_scores = [{} for _ in range(num_sequences)]
            for step_score_name in attr.step_scores.keys():
                out_step_scores = get_sequences_from_batched_steps(
                    [att.step_scores[step_score_name] for att in attributions]
                )
                for seq_id in range(num_sequences):
                    step_scores[seq_id][step_score_name] = out_step_scores[seq_id][: len(targets[seq_id])]
            for seq_id in range(num_sequences):
                seq_attributions[seq_id].step_scores = step_scores[seq_id]
        if attr.sequence_scores is not None:
            seq_scores = [{} for _ in range(num_sequences)]
            for seq_score_name in attr.sequence_scores.keys():
                out_seq_scores = get_sequences_from_batched_steps(
                    [att.sequence_scores[seq_score_name] for att in attributions]
                )
                for seq_id in range(num_sequences):
                    seq_scores[seq_id][seq_score_name] = out_seq_scores[seq_id][
                        : len(sources[seq_id]), : len(targets[seq_id]), ...
                    ]
            for seq_id in range(num_sequences):
                seq_attributions[seq_id].sequence_scores = seq_scores[seq_id]
        return seq_attributions

    def show(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        display: bool = True,
        return_html: Optional[bool] = False,
        aggregator: Optional[Type[Aggregator]] = None,
    ) -> Optional[str]:
        from inseq import show_attributions

        if aggregator is None:
            aggregator = self.aggregator
        return show_attributions(aggregator.aggregate(self), min_val, max_val, display, return_html)

    @property
    def minimum(self) -> float:
        minimum = float(self.source_attributions.min())
        if self.target_attributions is not None:
            minimum = min(minimum, float(self.target_attributions.min()))
        return minimum

    @property
    def maximum(self) -> float:
        maxmimum = float(self.source_attributions.max())
        if self.target_attributions is not None:
            maxmimum = max(maxmimum, float(self.target_attributions.max()))
        return maxmimum

    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name) for field in fields(self.__class__) if field.name != "aggregator"
        }


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
                "sequence_attributions": [seq.to_dict() for seq in self.sequence_attributions],
                "step_attributions": [step.to_dict() for step in self.step_attributions]
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
        aggregator: Optional[Type[Aggregator]] = None,
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

        if aggregator is None:
            attributions = [attr.aggregator.aggregate(attr) for attr in self.sequence_attributions]
        else:
            attributions = [aggregator.aggregate(attr) for attr in self.sequence_attributions]
        return show_attributions(attributions, min_val, max_val, display, return_html)


# Gradient attribution classes


@dataclass
class GradientFeatureAttributionStepOutput(FeatureAttributionStepOutput):
    """Raw output of a single step of gradient feature attribution.
    Adds the convergence delta to the base class.
    """

    pass
