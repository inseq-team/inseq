from typing import Any, Dict, List, Optional, Type, Union

import os
from dataclasses import dataclass, field

import json_tricks
import torch

from ..utils import (
    abs_max,
    drop_padding,
    get_sequences_from_batched_steps,
    identity_fn,
    pretty_dict,
    prod_fn,
    remap_from_filtered,
    sum_fn,
    sum_normalize_attributions,
)
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
from .aggregator import AggregableMixin, Aggregator, AggregatorPipeline, SequenceAttributionAggregator
from .batch import Batch, BatchEncoding
from .data_utils import TensorWrapper


FeatureAttributionInput = Union[TextInput, BatchEncoding, Batch]

DEFAULT_ATTRIBUTION_AGGREGATE_DICT = {
    "source_attributions": {"sequence_aggregate": identity_fn, "span_aggregate": abs_max},
    "target_attributions": {"sequence_aggregate": identity_fn, "span_aggregate": abs_max},
    "step_scores": {
        "span_aggregate": {
            "probability": prod_fn,
            "crossentropy": sum_fn,
            "perplexity": prod_fn,
        }
    },
}


@dataclass(eq=False, repr=False)
class FeatureAttributionSequenceOutput(TensorWrapper, AggregableMixin):
    """
    Output produced by a standard attribution method.

    Attributes:
        source (list of :class:`~inseq.utils.typing.TokenWithId`): Tokenized source sequence.
        target (list of :class:`~inseq.utils.typing.TokenWithId`): Tokenized target sequence.
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
    attr_pos_start: int = 0
    attr_pos_end: Optional[int] = None
    _aggregator: Union[AggregatorPipeline, Type[Aggregator]] = None
    _dict_aggregate_fn: Dict[str, Any] = None

    def __post_init__(self):
        aggregate_dict = DEFAULT_ATTRIBUTION_AGGREGATE_DICT
        if self._dict_aggregate_fn is None or self._dict_aggregate_fn == {}:
            self._dict_aggregate_fn = aggregate_dict
        elif isinstance(self._dict_aggregate_fn, dict):
            aggregate_dict.update(self._dict_aggregate_fn)
            self._dict_aggregate_fn = aggregate_dict
        if self._aggregator is None:
            self._aggregator = SequenceAttributionAggregator
        if self.attr_pos_end is None or self.attr_pos_end > len(self.target) + 1:
            self.attr_pos_end = len(self.target) + 1

    @classmethod
    def from_step_attributions(
        cls,
        attributions: List["FeatureAttributionStepOutput"],
        tokenized_target_sentences: Optional[List[List[TokenWithId]]] = None,
        pad_id: Optional[Any] = None,
        has_bos_token: bool = True,
        attr_pos_start: int = 0,
        attr_pos_end: Optional[int] = None,
    ) -> List["FeatureAttributionSequenceOutput"]:
        attr = attributions[0]
        seq_attr_cls = attr._sequence_cls
        num_sequences = len(attr.source_attributions)
        if not all([len(att.source_attributions) == num_sequences for att in attributions]):
            raise ValueError("All the attributions must include the same number of sequences.")
        source_attributions = get_sequences_from_batched_steps([att.source_attributions for att in attributions])
        seq_attributions = []
        sources = [drop_padding(attr.source[seq_id], pad_id) for seq_id in range(num_sequences)]
        targets = [
            drop_padding([a.target[seq_id][0] for a in attributions], pad_id) for seq_id in range(num_sequences)
        ]
        if tokenized_target_sentences is None:
            tokenized_target_sentences = targets
        if attr_pos_end is None:
            attr_pos_end = max([len(t) for t in tokenized_target_sentences])
        for seq_id in range(num_sequences):
            # Remove padding from tensor
            filtered_source_attribution = source_attributions[seq_id][
                : len(sources[seq_id]), : len(targets[seq_id]), ...
            ]
            seq_attributions.append(
                seq_attr_cls(
                    source=sources[seq_id],
                    target=tokenized_target_sentences[seq_id],
                    source_attributions=filtered_source_attribution,
                    attr_pos_start=attr_pos_start,
                    attr_pos_end=attr_pos_end,
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
                    : len(tokenized_target_sentences[seq_id]), : len(targets[seq_id]), ...
                ]
                if target_attributions[seq_id].shape[0] != len(tokenized_target_sentences[seq_id]):
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
        aggregator: Union[AggregatorPipeline, Type[Aggregator]] = None,
        **kwargs,
    ) -> Optional[str]:
        from inseq import show_attributions

        aggregated = self.aggregate(aggregator, **kwargs)
        return show_attributions(aggregated, min_val, max_val, display, return_html)

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

    def weight_attributions(self, step_score_id: str):
        aggregated_attr = self.aggregate()
        step_scores = self.step_scores[step_score_id].T.unsqueeze(1)
        source_attr = aggregated_attr.source_attributions.float().T
        self.source_attributions = (step_scores * source_attr).T
        if self.target_attributions is not None:
            target_attr = aggregated_attr.target_attributions.float().T
            self.target_attributions = (step_scores * target_attr).T
        self._aggregator = AggregatorPipeline([])
        return self


@dataclass(eq=False, repr=False)
class FeatureAttributionStepOutput(TensorWrapper):
    """
    Output of a single step of feature attribution, plus
    extra information related to what was attributed.
    """

    source_attributions: StepAttributionTensor
    step_scores: Dict[str, SingleScorePerStepTensor]
    target_attributions: Optional[StepAttributionTensor] = None
    sequence_scores: Optional[Dict[str, MultipleScoresPerStepTensor]] = None
    source: Optional[OneOrMoreTokenWithIdSequences] = None
    prefix: Optional[OneOrMoreTokenWithIdSequences] = None
    target: Optional[OneOrMoreTokenWithIdSequences] = None
    _sequence_cls: Type["FeatureAttributionSequenceOutput"] = FeatureAttributionSequenceOutput

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

    # These fields of the info dictionary should be matching to allow merging
    _merge_match_info_fields = [
        "attribute_target",
        "attribution_method",
        "constrained_decoding",
        "include_eos_baseline",
        "model_class",
        "model_name",
        "step_scores",
        "prepend_bos_token",
        "tokenizer_class",
        "tokenizer_name",
    ]

    sequence_attributions: List[FeatureAttributionSequenceOutput]
    step_attributions: Optional[List[FeatureAttributionStepOutput]] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.__dict__)})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        for self_seq, other_seq in zip(self.sequence_attributions, other.sequence_attributions):
            if self_seq != other_seq:
                return False
        if self.step_attributions is not None and other.step_attributions is not None:
            for self_step, other_step in zip(self.step_attributions, other.step_attributions):
                if self_step != other_step:
                    return False
        if self.info != other.info:
            return False
        return True

    def save(self, path: str, overwrite: bool = False, compress: bool = False) -> None:
        """
        Save class contents to a JSON file.

        Args:
            path (str): Path to the folder where attributions and their configuration will be stored.
            overwrite (bool): If True, overwrite the file if it exists, raise error otherwise.
        """
        if not overwrite and os.path.exists(path):
            raise ValueError(f"{path} already exists. Override with overwrite=True.")
        with open(path, "w") as f:
            json_tricks.dump(
                self,
                f,
                allow_nan=True,
                indent=4,
                sort_keys=True,
                compression=compress,
                properties={"ndarray_compact": True},
            )

    @staticmethod
    def load(path: str) -> "FeatureAttributionOutput":
        """Load saved attribution outputs into a new FeatureAttributionOutput object.

        Args:
            path (str): Path to the folder containing the saved attribution outputs.

        Returns:
            :class:`~inseq.data.FeatureAttributionOutput`: Loaded attribution output
        """
        with open(path) as f:
            out = json_tricks.load(f)
        out.sequence_attributions = [seq.torch() for seq in out.sequence_attributions]
        if out.step_attributions is not None:
            out.step_attributions = [step.torch() for step in out.step_attributions]
        return out

    def show(
        self,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        display: bool = True,
        return_html: Optional[bool] = False,
        aggregator: Union[AggregatorPipeline, Type[Aggregator]] = None,
        **kwargs,
    ) -> Optional[str]:
        """Visualize the sequence attributions.

        Args:
            min_val (int, optional): Minimum value for color scale.
            max_val (int, optional): Maximum value for color scale.
            display (bool, optional): If True, display the attribution visualization.
            return_html (bool, optional): If True, return the attribution visualization as HTML.
            aggregator (:obj:`AggregatorPipeline` or :obj:`Type[Aggregator]`, optional): Aggregator
                or pipeline to use. If not provided, the default aggregator for every sequence attribution
                is used.

        Returns:
            str: Attribution visualization as HTML if `return_html=True`, None otherwise.
        """
        from inseq import show_attributions

        attributions = [attr.aggregate(aggregator, **kwargs) for attr in self.sequence_attributions]
        return show_attributions(attributions, min_val, max_val, display, return_html)

    @classmethod
    def merge_attributions(cls, attributions: List["FeatureAttributionOutput"]) -> "FeatureAttributionOutput":
        """Merges multiple FeatureAttributionOutput object into a single one.

        Merging is allowed only if the attribution process was the same (by checking info).

        Args:
            attributions (`list(FeatureAttributionOutput)`):
                The single FeatureAttributionOutput objects to be merged

        Returns:
            `FeatureAttributionOutput`: Merged object
        """
        assert all(
            isinstance(x, FeatureAttributionOutput) for x in attributions
        ), "Only FeatureAttributionOutput objects can be merged."
        first = attributions[0]
        for match_field in cls._merge_match_info_fields:
            assert all(
                attr.info[match_field] == first.info[match_field]
                if match_field in first.info
                else match_field not in attr.info
                for attr in attributions
            ), f"Cannot merge: incompatible values for field {match_field}"
        out_info = first.info.copy()
        if "attr_pos_end" in first.info:
            out_info.update({"attr_pos_end": max(attr.info["attr_pos_end"] for attr in attributions)})
        if "generated_texts" in first.info:
            out_info.update(
                {"generated_texts": [text for attr in attributions for text in attr.info["generated_texts"]]}
            )
        if "input_texts" in first.info:
            out_info.update({"input_texts": [text for attr in attributions for text in attr.info["input_texts"]]})
        return cls(
            sequence_attributions=[seqattr for attr in attributions for seqattr in attr.sequence_attributions],
            step_attributions=[stepattr for attr in attributions for stepattr in attr.step_attributions]
            if first.step_attributions is not None
            else None,
            info=out_info,
        )

    def weight_attributions(self, step_score_id: str):
        for i, attr in enumerate(self.sequence_attributions):
            self.sequence_attributions[i] = attr.weight_attributions(step_score_id)


# Gradient attribution classes


@dataclass(eq=False, repr=False)
class GradientFeatureAttributionSequenceOutput(FeatureAttributionSequenceOutput):
    """Raw output of a single sequence of gradient feature attribution.
    Adds the convergence delta to the base class.
    """

    def __post_init__(self):
        super().__post_init__()
        self._dict_aggregate_fn["source_attributions"]["sequence_aggregate"] = sum_normalize_attributions
        self._dict_aggregate_fn["target_attributions"]["sequence_aggregate"] = sum_normalize_attributions
        if "deltas" not in self._dict_aggregate_fn["step_scores"]["span_aggregate"]:
            self._dict_aggregate_fn["step_scores"]["span_aggregate"]["deltas"] = abs_max


@dataclass(eq=False, repr=False)
class GradientFeatureAttributionStepOutput(FeatureAttributionStepOutput):
    """Raw output of a single step of gradient feature attribution.
    Adds the convergence delta to the base class.
    """

    _sequence_cls: Type["FeatureAttributionSequenceOutput"] = GradientFeatureAttributionSequenceOutput
