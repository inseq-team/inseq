import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch

from ..utils import (
    abs_max,
    drop_padding,
    get_sequences_from_batched_steps,
    identity_fn,
    json_advanced_dump,
    json_advanced_load,
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

logger = logging.getLogger(__name__)


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
    source_attributions: Optional[SequenceAttributionTensor] = None
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
        if self.attr_pos_end is None or self.attr_pos_end > len(self.target):
            self.attr_pos_end = len(self.target)

    @classmethod
    def from_step_attributions(
        cls,
        attributions: List["FeatureAttributionStepOutput"],
        tokenized_target_sentences: Optional[List[List[TokenWithId]]] = None,
        pad_id: Optional[Any] = None,
        has_bos_token: bool = True,
        attr_pos_end: Optional[int] = None,
    ) -> List["FeatureAttributionSequenceOutput"]:
        attr = attributions[0]
        seq_attr_cls = attr._sequence_cls
        num_sequences = len(attr.prefix)
        if not all([len(attr.prefix) == num_sequences for attr in attributions]):
            raise ValueError("All the attributions must include the same number of sequences.")
        seq_attributions = []
        sources = None
        if attr.source_attributions is not None:
            sources = [drop_padding(attr.source[seq_id], pad_id) for seq_id in range(num_sequences)]
        targets = [
            drop_padding([a.target[seq_id][0] for a in attributions], pad_id) for seq_id in range(num_sequences)
        ]
        if tokenized_target_sentences is None:
            tokenized_target_sentences = targets
        if attr_pos_end is None:
            attr_pos_end = max([len(t) for t in tokenized_target_sentences])
        pos_start = [
            min(len(tokenized_target_sentences[seq_id]), attr_pos_end) - len(targets[seq_id])
            for seq_id in range(num_sequences)
        ]
        for seq_id in range(num_sequences):
            source = tokenized_target_sentences[seq_id][: pos_start[seq_id]] if sources is None else sources[seq_id]
            seq_attributions.append(
                seq_attr_cls(
                    source=source,
                    target=tokenized_target_sentences[seq_id],
                    attr_pos_start=pos_start[seq_id],
                    attr_pos_end=attr_pos_end,
                )
            )
        if attr.source_attributions is not None:
            source_attributions = get_sequences_from_batched_steps([att.source_attributions for att in attributions])
            for seq_id in range(num_sequences):
                # Remove padding from tensor
                filtered_source_attribution = source_attributions[seq_id][
                    : len(sources[seq_id]), : len(targets[seq_id]), ...
                ]
                seq_attributions[seq_id].source_attributions = filtered_source_attribution
        if attr.target_attributions is not None:
            target_attributions = get_sequences_from_batched_steps(
                [att.target_attributions for att in attributions], pad_dims=(1,)
            )
            for seq_id in range(num_sequences):
                if has_bos_token:
                    target_attributions[seq_id] = target_attributions[seq_id][1:, ...]
                start_idx = max(pos_start) - pos_start[seq_id]
                end_idx = start_idx + len(tokenized_target_sentences[seq_id])
                target_attributions[seq_id] = target_attributions[seq_id][
                    start_idx:end_idx, : len(targets[seq_id]), ...  # noqa: E203
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
        do_aggregation: bool = True,
        **kwargs,
    ) -> Optional[str]:
        from inseq import show_attributions

        # If no aggregator is specified, the default aggregator for the class is used
        aggregated = self.aggregate(aggregator, **kwargs) if do_aggregation else self
        if (aggregated.source_attributions is not None and aggregated.source_attributions.shape[1] == 0) or (
            aggregated.target_attributions is not None and aggregated.target_attributions.shape[1] == 0
        ):
            tokens = "".join(tid.token for tid in self.target)
            logger.warning(f"Found empty attributions, skipping attribution matching generation: {tokens}")
        else:
            return show_attributions(aggregated, min_val, max_val, display, return_html)

    @property
    def minimum(self) -> float:
        minimum = 0
        if self.source_attributions is not None:
            minimum = min(minimum, float(torch.nan_to_num(self.source_attributions).min()))
        if self.target_attributions is not None:
            minimum = min(minimum, float(torch.nan_to_num(self.target_attributions).min()))
        return minimum

    @property
    def maximum(self) -> float:
        maximum = 0
        if self.source_attributions is not None:
            maximum = max(maximum, float(torch.nan_to_num(self.source_attributions).max()))
        if self.target_attributions is not None:
            maximum = max(maximum, float(torch.nan_to_num(self.target_attributions).max()))
        return maximum

    def weight_attributions(self, step_score_id: str):
        aggregated_attr = self.aggregate()
        step_scores = self.step_scores[step_score_id].T.unsqueeze(1)
        if self.source_attributions is not None:
            source_attr = aggregated_attr.source_attributions.float().T
            self.source_attributions = (step_scores * source_attr).T
        if self.target_attributions is not None:
            target_attr = aggregated_attr.target_attributions.float().T
            self.target_attributions = (step_scores * target_attr).T
        self._aggregator = AggregatorPipeline([])
        return self

    def get_scores_dicts(
        self,
        aggregator: Union[AggregatorPipeline, Type[Aggregator]] = None,
        do_aggregation: bool = True,
        **kwargs,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        # If no aggregator is specified, the default aggregator for the class is used
        aggr = self.aggregate(aggregator, **kwargs) if do_aggregation else self
        return_dict = {"source_attributions": {}, "target_attributions": {}, "step_scores": {}}
        if aggr.source_attributions is not None:
            score_map_source = {}
            for tgt_idx, tgt_tok in enumerate(aggr.target):
                score_map_source[tgt_tok.token] = {}
                for src_idx, src_tok in enumerate(aggr.source):
                    score_map_source[tgt_tok.token][src_tok.token] = aggr.source_attributions[src_idx, tgt_idx].item()
            return_dict["source_attributions"] = score_map_source
        if aggr.target_attributions is not None:
            score_map_target = {}
            for tgt_idx_b, tgt_tok_b in enumerate(aggr.target):
                score_map_target[tgt_tok_b.token] = {}
                for tgt_idx_a, tgt_tok_a in enumerate(aggr.target):
                    score_map_target[tgt_tok_b.token][tgt_tok_a.token] = aggr.target_attributions[
                        tgt_idx_a, tgt_idx_b
                    ].item()
            return_dict["target_attributions"] = score_map_target
        if aggr.step_scores is not None:
            step_scores_map = {}
            for tgt_idx, tgt_tok in enumerate(aggr.target):
                step_scores_map[tgt_tok.token] = {}
                for step_score_id, step_score in aggr.step_scores.items():
                    step_scores_map[tgt_tok.token][step_score_id] = step_score[tgt_idx].item()
            return_dict["step_scores"] = step_scores_map
        return return_dict


@dataclass(eq=False, repr=False)
class FeatureAttributionStepOutput(TensorWrapper):
    """
    Output of a single step of feature attribution, plus
    extra information related to what was attributed.
    """

    source_attributions: Optional[StepAttributionTensor] = None
    step_scores: Optional[Dict[str, SingleScorePerStepTensor]] = None
    target_attributions: Optional[StepAttributionTensor] = None
    sequence_scores: Optional[Dict[str, MultipleScoresPerStepTensor]] = None
    source: Optional[OneOrMoreTokenWithIdSequences] = None
    prefix: Optional[OneOrMoreTokenWithIdSequences] = None
    target: Optional[OneOrMoreTokenWithIdSequences] = None
    _sequence_cls: Type["FeatureAttributionSequenceOutput"] = FeatureAttributionSequenceOutput

    def __post_init__(self):
        self.to(torch.float32)

    def remap_from_filtered(
        self,
        target_attention_mask: TargetIdsTensor,
    ) -> None:
        if self.source_attributions is not None:
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

    def save(
        self,
        path: str,
        overwrite: bool = False,
        compress: bool = False,
        ndarray_compact: bool = True,
        use_primitives: bool = False,
        split_sequences: bool = False,
    ) -> None:
        """
        Save class contents to a JSON file.

        Args:
            path (:obj:`str`): Path to the folder where the attribution output will be stored (e.g. ``./out.json``).
            overwrite (:obj:`bool`, *optional*, defaults to False):
                If True, overwrite the file if it exists, raise error otherwise.
            compress (:obj:`bool`, *optional*, defaults to False):
                If True, the output file is compressed using gzip. Especially useful for large sequences and granular
                attributions with umerged hidden dimensions.
            ndarray_compact (:obj:`bool`, *optional*, defaults to True):
                If True, the arrays for scores and attributions are stored in a compact b64 format. Otherwise, they are
                stored as plain lists of floats.
            use_primitives (:obj:`bool`, *optional*, defaults to False):
                If True, the output is stored as a list of dictionaries with primitive types (e.g. int, float, str).
                Note that an attribution saved with this option cannot be loaded with the `load` method.
            split_sequences (:obj:`bool`, *optional*, defaults to False):
                If True, the output is split into multiple files, one per sequence. The file names are generated by
                appending the sequence index to the given path (e.g. ``./out.json`` with two sequences ->
                ``./out_0.json``, ``./out_1.json``)
        """
        if not overwrite and Path(path).exists():
            raise ValueError(f"{path} already exists. Override with overwrite=True.")
        save_outs = []
        paths = []
        if split_sequences:
            for i, seq in enumerate(self.sequence_attributions):
                attr_out = deepcopy(self)
                attr_out.sequence_attributions = [seq]
                attr_out.step_attributions = None
                attr_out.info["input_texts"] = [attr_out.info["input_texts"][i]]
                attr_out.info["generated_texts"] = [attr_out.info["generated_texts"][i]]
                save_outs.append(attr_out)
                paths.append(f"{str(path).split('.json')[0]}_{i}.json{'.gz' if compress else ''}")
        else:
            save_outs.append(self)
            paths.append(path)
        for attr_out, path_out in zip(save_outs, paths):
            with open(path_out, f"w{'b' if compress else ''}") as f:
                json_advanced_dump(
                    attr_out,
                    f,
                    allow_nan=True,
                    indent=4,
                    sort_keys=True,
                    ndarray_compact=ndarray_compact,
                    compression=compress,
                    use_primitives=use_primitives,
                )

    @staticmethod
    def load(
        path: str,
        decompress: bool = False,
    ) -> "FeatureAttributionOutput":
        """Load saved attribution output into a new :class:`~inseq.data.FeatureAttributionOutput` object.

        Args:
            path (:obj:`str`): Path to the JSON file containing the saved attribution output.
                Note that the file must have been saved with the :meth:`~inseq.data.FeatureAttributionOutput.save`
                method with ``use_primitives=False`` in order to be loaded correctly.
            decompress (:obj:`bool`, *optional*, defaults to False):
                If True, the input file is decompressed using gzip.

        Returns:
            :class:`~inseq.data.FeatureAttributionOutput`: Loaded attribution output
        """
        out = json_advanced_load(path, decompression=decompress)
        out.sequence_attributions = [seq.torch() for seq in out.sequence_attributions]
        if out.step_attributions is not None:
            out.step_attributions = [step.torch() for step in out.step_attributions]
        return out

    def aggregate(
        self,
        aggregator: Union[AggregatorPipeline, Type[Aggregator]] = None,
        **kwargs,
    ) -> "FeatureAttributionOutput":
        """Aggregate the sequence attributions.

        Args:
            aggregator (:obj:`AggregatorPipeline` or :obj:`Type[Aggregator]`, optional): Aggregator
                or pipeline to use. If not provided, the default aggregator for every sequence attribution
                is used.

        Returns:
            :class:`~inseq.data.FeatureAttributionOutput`: Aggregated attribution output
        """
        aggregated = deepcopy(self)
        for idx, seq in enumerate(aggregated.sequence_attributions):
            aggregated.sequence_attributions[idx] = seq.aggregate(aggregator, **kwargs)
        return aggregated

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
        out_str = ""
        for attr in self.sequence_attributions:
            if return_html:
                out_str += attr.show(min_val, max_val, display, return_html, aggregator, **kwargs)
            else:
                attr.show(min_val, max_val, display, return_html, aggregator, **kwargs)
        if return_html:
            return out_str

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

    def get_scores_dicts(
        self, aggregator: Union[AggregatorPipeline, Type[Aggregator]] = None, do_aggregation: bool = True, **kwargs
    ) -> List[Dict[str, Dict[str, Dict[str, float]]]]:
        """Get all computed scores (attributions and step scores) for all sequences as a list of dictionaries.

        Returns:
            :obj:`list(dict)`: List containing one dictionary per sequence. Every dictionary contains the keys
            "source_attributions", "target_attributions" and "step_scores". For each of these keys, the value is a
            dictionary with generated tokens as keys, and for values a final dictionary. For  "step_scores", the keys
            of the final dictionary are the step score ids, and the values are the scores.
            For "source_attributions" and "target_attributions", the keys of the final dictionary are respectively
            source and target tokens, and the values are the attribution scores.

        This output is intended to be easily converted to a pandas DataFrame. The following example produces a list of
        DataFrames, one for each sequence, matching the source attributions that would be visualized by out.show().

        ```python
        dfs = [pd.DataFrame(x["source_attributions"]) for x in out.get_scores_dicts()]
        ```
        """
        return [attr.get_scores_dicts(aggregator, do_aggregation, **kwargs) for attr in self.sequence_attributions]


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
