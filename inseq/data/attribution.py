import base64
import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import treescope as ts

from ..utils import (
    convert_from_safetensor,
    convert_to_safetensor,
    drop_padding,
    get_sequences_from_batched_steps,
    json_advanced_dump,
    json_advanced_load,
    pad_with_nan,
    pretty_dict,
    remap_from_filtered,
)
from ..utils.typing import (
    MultipleScoresPerSequenceTensor,
    MultipleScoresPerStepTensor,
    OneOrMoreTokenWithIdSequences,
    ScorePrecision,
    SequenceAttributionTensor,
    SingleScorePerStepTensor,
    SingleScoresPerSequenceTensor,
    StepAttributionTensor,
    TargetIdsTensor,
    TextInput,
    TokenWithId,
)
from .aggregation_functions import DEFAULT_ATTRIBUTION_AGGREGATE_DICT
from .aggregator import AggregableMixin, Aggregator, AggregatorPipeline
from .batch import Batch, BatchEmbedding, BatchEncoding, DecoderOnlyBatch, EncoderDecoderBatch
from .data_utils import TensorWrapper
from .viz import get_saliency_heatmap_treescope, get_tokens_heatmap_treescope

if TYPE_CHECKING:
    from ..models import AttributionModel

FeatureAttributionInput = TextInput | BatchEncoding | Batch


logger = logging.getLogger(__name__)


DEFAULT_ATTRIBUTION_DIM_NAMES = {
    "source_attributions": {0: "Input Tokens", 1: "Generated Tokens"},
    "target_attributions": {0: "Input Tokens", 1: "Generated Tokens"},
}


def get_batch_from_inputs(
    attribution_model: "AttributionModel",
    inputs: FeatureAttributionInput,
    include_eos_baseline: bool = False,
    as_targets: bool = False,
    skip_special_tokens: bool = False,
) -> Batch:
    if isinstance(inputs, Batch):
        batch = inputs
    else:
        if isinstance(inputs, str | list):
            encodings: BatchEncoding = attribution_model.encode(
                inputs,
                as_targets=as_targets,
                return_baseline=True,
                include_eos_baseline=include_eos_baseline,
                add_special_tokens=not skip_special_tokens,
            )
        elif isinstance(inputs, BatchEncoding):
            encodings = inputs
        else:
            raise ValueError(
                f"Error: Found inputs of type {type(inputs)}. "
                "Inputs must be either a string, a list of strings, a BatchEncoding or a Batch."
            )
        embeddings = BatchEmbedding(
            input_embeds=attribution_model.embed(
                encodings.input_ids, as_targets=as_targets, add_special_tokens=not skip_special_tokens
            ),
            baseline_embeds=attribution_model.embed(
                encodings.baseline_ids, as_targets=as_targets, add_special_tokens=not skip_special_tokens
            ),
        )
        batch = Batch(encodings, embeddings)
    return batch


def merge_attributions(attributions: list["FeatureAttributionOutput"]) -> "FeatureAttributionOutput":
    """Merges multiple :class:`~inseq.data.FeatureAttributionOutput` objects into a single one.

    Merging is allowed only if the two outputs match on the fields specified in ``_merge_match_info_fields``.

    Args:
        attributions (:obj:`list` of :class:`~inseq.data.FeatureAttributionOutput`): The FeatureAttributionOutput
            objects to be merged.

    Returns:
        :class:`~inseq.data.FeatureAttributionOutput`: Merged object.
    """
    assert all(
        isinstance(x, FeatureAttributionOutput) for x in attributions
    ), "Only FeatureAttributionOutput objects can be merged."
    first = attributions[0]
    for match_field in FeatureAttributionOutput._merge_match_info_fields:
        assert all(
            (
                attr.info[match_field] == first.info[match_field]
                if match_field in first.info
                else match_field not in attr.info
            )
            for attr in attributions
        ), f"Cannot merge: incompatible values for field {match_field}"
    out_info = first.info.copy()
    if "attr_pos_end" in first.info:
        out_info.update({"attr_pos_end": max(attr.info["attr_pos_end"] for attr in attributions)})
    if "generated_texts" in first.info:
        out_info.update({"generated_texts": [text for attr in attributions for text in attr.info["generated_texts"]]})
    if "input_texts" in first.info:
        out_info.update({"input_texts": [text for attr in attributions for text in attr.info["input_texts"]]})
    return FeatureAttributionOutput(
        sequence_attributions=[seqattr for attr in attributions for seqattr in attr.sequence_attributions],
        step_attributions=(
            [stepattr for attr in attributions for stepattr in attr.step_attributions]
            if first.step_attributions is not None
            else None
        ),
        info=out_info,
    )


@dataclass(eq=False, repr=False)
class FeatureAttributionSequenceOutput(TensorWrapper, AggregableMixin):
    """Output produced by a standard attribution method.

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

    source: list[TokenWithId]
    target: list[TokenWithId]
    source_attributions: SequenceAttributionTensor | None = None
    target_attributions: SequenceAttributionTensor | None = None
    step_scores: dict[str, SingleScoresPerSequenceTensor] | None = None
    sequence_scores: dict[str, MultipleScoresPerSequenceTensor] | None = None
    attr_pos_start: int = 0
    attr_pos_end: int | None = None
    _aggregator: str | list[str] | None = None
    _dict_aggregate_fn: dict[str, str] | None = None
    _attribution_dim_names: dict[str, dict[int, str]] | None = None

    def __post_init__(self):
        if self._dict_aggregate_fn is None:
            self._dict_aggregate_fn = {}
        default_aggregate_fn = DEFAULT_ATTRIBUTION_AGGREGATE_DICT
        default_aggregate_fn.update(self._dict_aggregate_fn)
        self._dict_aggregate_fn = default_aggregate_fn
        if self._attribution_dim_names is None:
            self._attribution_dim_names = {}
        default_dim_names = DEFAULT_ATTRIBUTION_DIM_NAMES
        default_dim_names.update(self._attribution_dim_names)
        self._attribution_dim_names = default_dim_names
        if self._aggregator is None:
            self._aggregator = "scores"
        if self.attr_pos_end is None or self.attr_pos_end > len(self.target):
            self.attr_pos_end = len(self.target)

    def __getitem__(self, s: slice | int) -> "FeatureAttributionSequenceOutput":
        source_spans = None if self.source_attributions is None else (s.start, s.stop)
        target_spans = None if self.source_attributions is not None else (s.start, s.stop)
        return self.aggregate("slices", source_spans=source_spans, target_spans=target_spans)

    def __sub__(self, other: "FeatureAttributionSequenceOutput") -> "FeatureAttributionSequenceOutput":
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot compare {type(other)} with {type(self)}")
        return self.aggregate("pair", paired_attr=other, do_post_aggregation_checks=False)

    def __treescope_repr__(
        self,
        path: str,
        subtree_renderer: Callable[[Any, str | None], ts.rendering_parts.Rendering],
    ) -> ts.rendering_parts.Rendering:
        def granular_attribution_visualizer(
            value: Any,
            path: tuple[Any, ...] | None,
        ):
            if isinstance(value, torch.Tensor):
                tname = path.split(".")[-1]
                column_labels = [t.token for t in self.target[self.attr_pos_start : self.attr_pos_end]]
                if tname == "source_attributions":
                    row_labels = [t.token for t in self.source]
                elif tname == "target_attributions":
                    row_labels = [t.token for t in self.target]
                elif tname.startswith("sequence_scores"):
                    tname = tname[17:].split("_")[0]
                    if tname.startswith("encoder"):
                        row_labels = [t.token for t in self.source]
                        column_labels = [t.token for t in self.source]
                    elif tname.startswith("decoder"):
                        row_labels = [t.token for t in self.target]
                        column_labels = [t.token for t in self.target]
                adapter = ts.type_registries.lookup_ndarray_adapter(value)
                if value.ndim >= 2:
                    return ts.IPythonVisualization(
                        ts.figures.inline(
                            adapter.get_array_summary(value, fast=False),
                            get_saliency_heatmap_treescope(
                                scores=value.numpy(),
                                column_labels=column_labels,
                                row_labels=row_labels,
                                dim_names=self._attribution_dim_names.get(tname, None),
                            ),
                        ),
                        replace=True,
                    )
                else:
                    return ts.IPythonVisualization(
                        ts.figures.inline(
                            adapter.get_array_summary(value, fast=False) + "\n\n",
                            ts.figures.figure_from_treescope_rendering_part(
                                ts.rendering_parts.indented_children(
                                    [
                                        get_tokens_heatmap_treescope(
                                            tokens=column_labels,
                                            scores=value.numpy(),
                                            max_val=value.max().item(),
                                        )
                                    ]
                                )
                            ),
                        ),
                        replace=True,
                    )

        with ts.active_autovisualizer.set_scoped(granular_attribution_visualizer):
            return ts.repr_lib.render_object_constructor(
                object_type=type(self),
                attributes=self.__dict__,
                path=path,
                subtree_renderer=subtree_renderer,
                roundtrippable=True,
            )

    def _convert_to_safetensors(self, scores_precision: ScorePrecision = "float32"):
        """
        Converts tensor attributes within the class to the specified precision.
        The conversion is based on the specified `scores_precision`.
        If the input tensor is already of the desired precision, no conversion occurs.
        For float8, the function performs scaling and converts to uint8, which can be later converted back to float16 upon reloading.

        Args:
            scores_precision (str, optional): Desired output data type precision. Defaults to "float32".
        Returns:
            self: The function modifies the class attributes in-place.
        """

        if self.source_attributions is not None:
            self.source_attributions = convert_to_safetensor(
                self.source_attributions.contiguous(), scores_precision=scores_precision
            )
        if self.target_attributions is not None:
            self.target_attributions = convert_to_safetensor(
                self.target_attributions.contiguous(), scores_precision=scores_precision
            )
        if self.step_scores is not None:
            self.step_scores = {
                k: convert_to_safetensor(v.contiguous(), scores_precision=scores_precision)
                for k, v in self.step_scores.items()
            }
        if self.sequence_scores is not None:
            self.sequence_scores = {
                k: convert_to_safetensor(v.contiguous(), scores_precision=scores_precision)
                for k, v in self.sequence_scores.items()
            }
        return self

    def _recover_from_safetensors(self):
        """
        Converts tensor attributes within the class from b64-encoded safetensors to torch tensors.`.
        """
        if self.source_attributions is not None:
            self.source_attributions = convert_from_safetensor(base64.b64decode(self.source_attributions))
        if self.target_attributions is not None:
            self.target_attributions = convert_from_safetensor(base64.b64decode(self.target_attributions))
        if self.step_scores is not None:
            self.step_scores = {k: convert_from_safetensor(base64.b64decode(v)) for k, v in self.step_scores.items()}
        if self.sequence_scores is not None:
            self.sequence_scores = {
                k: convert_from_safetensor(base64.b64decode(v)) for k, v in self.sequence_scores.items()
            }
        return self

    @staticmethod
    def get_remove_pad_fn(attr: "FeatureAttributionStepOutput", name: str) -> Callable:
        if attr.source_attributions is None or name.startswith("decoder"):
            remove_pad_fn = lambda scores, _, targets, seq_id: scores[seq_id][
                : len(targets[seq_id]), : len(targets[seq_id]), ...
            ]
        elif name.startswith("encoder"):
            remove_pad_fn = lambda scores, sources, _, seq_id: scores[seq_id][
                : len(sources[seq_id]), : len(sources[seq_id]), ...
            ]
        else:  # default case: cross-attention
            remove_pad_fn = lambda scores, sources, targets, seq_id: scores[seq_id][
                : len(sources[seq_id]), : len(targets[seq_id]), ...
            ]
        return remove_pad_fn

    @classmethod
    def from_step_attributions(
        cls,
        attributions: list["FeatureAttributionStepOutput"],
        tokenized_target_sentences: list[list[TokenWithId]],
        pad_token: Any | None = None,
        attr_pos_end: int | None = None,
    ) -> list["FeatureAttributionSequenceOutput"]:
        """Converts a list of :class:`~inseq.data.attribution.FeatureAttributionStepOutput` objects containing multiple
        examples outputs per step into a list of :class:`~inseq.data.attribution.FeatureAttributionSequenceOutput` with
        every object containing all step outputs for an individual example.

        Raises:
            `ValueError`: If the number of sequences in the attributions is not the same for all input sequences.

        Returns:
            `List[FeatureAttributionSequenceOutput]`: List of
            :class:`~inseq.data.attribution.FeatureAttributionSequenceOutput` objects.
        """
        attr = attributions[0]
        num_sequences = len(attr.prefix)
        if not all(len(attr.prefix) == num_sequences for attr in attributions):
            raise ValueError("All the attributions must include the same number of sequences.")
        sources = []
        targets = []
        pos_start = []
        for seq_idx in range(num_sequences):
            if attr.source_attributions is not None:
                sources.append(drop_padding(attr.source[seq_idx], pad_token))
            curr_target = [a.target[seq_idx][0] for a in attributions]
            targets.append(drop_padding(curr_target, pad_token))
            if all(attr.prefix[seq_idx][0] == pad_token for seq_idx in range(num_sequences)):
                tokenized_target_sentences[seq_idx] = tokenized_target_sentences[seq_idx][:1] + drop_padding(
                    tokenized_target_sentences[seq_idx][1:], pad_token
                )
            else:
                tokenized_target_sentences[seq_idx] = drop_padding(tokenized_target_sentences[seq_idx], pad_token)
        if attr_pos_end is None:
            attr_pos_end = max(len(t) for t in tokenized_target_sentences)
        for seq_idx in range(num_sequences):
            # If the model is decoder-only, the source is the input prefix
            curr_pos_start = min(len(tokenized_target_sentences[seq_idx]), attr_pos_end) - len(targets[seq_idx])
            pos_start.append(curr_pos_start)
        if attr.source_attributions is not None:
            source_attributions = get_sequences_from_batched_steps([att.source_attributions for att in attributions])
            for seq_id in range(num_sequences):
                # Remove padding from tensor
                source_attributions[seq_id] = source_attributions[seq_id][
                    : len(sources[seq_id]), : len(targets[seq_id]), ...
                ]
        if attr.target_attributions is not None:
            target_attributions = get_sequences_from_batched_steps(
                [att.target_attributions for att in attributions], padding_dims=[1]
            )
            for seq_id in range(num_sequences):
                start_idx = max(pos_start) - pos_start[seq_id]
                end_idx = start_idx + len(tokenized_target_sentences[seq_id])
                target_attributions[seq_id] = target_attributions[seq_id][
                    start_idx:end_idx, : len(targets[seq_id]), ...  # noqa: E203
                ]
                if target_attributions[seq_id].shape[0] != len(tokenized_target_sentences[seq_id]):
                    target_attributions[seq_id] = pad_with_nan(target_attributions[seq_id], dim=0, pad_size=1)
        if attr.step_scores is not None:
            step_scores = [{} for _ in range(num_sequences)]
            for step_score_name in attr.step_scores.keys():
                out_step_scores = get_sequences_from_batched_steps(
                    [att.step_scores[step_score_name] for att in attributions], stack_dim=1
                )
                for seq_id in range(num_sequences):
                    step_scores[seq_id][step_score_name] = out_step_scores[seq_id][: len(targets[seq_id])]
        if attr.sequence_scores is not None:
            seq_scores = [{} for _ in range(num_sequences)]
            for seq_score_name in attr.sequence_scores.keys():
                # Since we need to know in advance the length of the sequence to remove padding from
                # batching sequences with different lengths, we rely on code names for sequence scores
                # that are not source-to-target (default for encoder-decoder) or target-to-target
                # (default for decoder only).
                remove_pad_fn = cls.get_remove_pad_fn(attr, seq_score_name)
                if seq_score_name.startswith("encoder") or seq_score_name.startswith("decoder"):
                    out_seq_scores = [attr.sequence_scores[seq_score_name][i, ...] for i in range(num_sequences)]
                else:
                    out_seq_scores = get_sequences_from_batched_steps(
                        [att.sequence_scores[seq_score_name] for att in attributions], padding_dims=[1]
                    )
                for seq_id in range(num_sequences):
                    seq_scores[seq_id][seq_score_name] = remove_pad_fn(out_seq_scores, sources, targets, seq_id)
        seq_attributions: list[FeatureAttributionSequenceOutput] = []
        for seq_idx in range(num_sequences):
            curr_seq_attribution: FeatureAttributionSequenceOutput = attr.get_sequence_cls(
                source=deepcopy(
                    tokenized_target_sentences[seq_idx][: pos_start[seq_idx]] if not sources else sources[seq_idx]
                ),
                target=deepcopy(tokenized_target_sentences[seq_idx]),
                source_attributions=source_attributions[seq_idx] if attr.source_attributions is not None else None,
                target_attributions=target_attributions[seq_idx] if attr.target_attributions is not None else None,
                step_scores=step_scores[seq_idx] if attr.step_scores is not None else None,
                sequence_scores=seq_scores[seq_idx] if attr.sequence_scores is not None else None,
                attr_pos_start=pos_start[seq_idx],
                attr_pos_end=attr_pos_end,
            )
            seq_attributions.append(curr_seq_attribution)
        return seq_attributions

    def show(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        max_show_size: int | None = None,
        show_dim: int | str | None = None,
        slice_dims: dict[int | str, tuple[int, int]] | None = None,
        display: bool = True,
        return_html: bool | None = False,
        return_figure: bool = False,
        aggregator: AggregatorPipeline | type[Aggregator] = None,
        do_aggregation: bool = True,
        **kwargs,
    ) -> str | None:
        """Visualize the attributions.

        Args:
            min_val (:obj:`int`, *optional*, defaults to None):
                Minimum value in the color range of the visualization. If None, the minimum value of the attributions
                across all visualized examples is used.
            max_val (:obj:`int`, *optional*, defaults to None):
                Maximum value in the color range of the visualization. If None, the maximum value of the attributions
                across all visualized examples is used.
            max_show_size (:obj:`int`, *optional*, defaults to None):
                For granular visualization, this parameter specifies the maximum dimension size for additional dimensions
                to be visualized. Default: 20.
            show_dim (:obj:`int` or :obj:`str`, *optional*, defaults to None):
                For granular visualization, this parameter specifies the dimension that should be visualized along with
                the source and target tokens. Can be either the dimension index or the dimension name. Works only if
                the dimension size is less than or equal to `max_show_size`.
            slice_dims (:obj:`dict[int or str, tuple[int, int]]`, *optional*, defaults to None):
                For granular visualization, this parameter specifies the dimensions that should be sliced and visualized
                along with the source and target tokens. The dictionary should contain the dimension index or name as the
                key and the slice range as the value.
            display (:obj:`bool`, *optional*, defaults to True):
                Whether to display the visualization. Can be set to False if the visualization is produced and stored
                for later use.
            return_html (:obj:`bool`, *optional*, defaults to False):
                Whether to return the HTML code of the visualization.
            return_figure (:obj:`bool`, *optional*, defaults to False):
                For granular visualization, whether to return the Treescope figure object for further manipulation.
            aggregator (:obj:`AggregatorPipeline`, *optional*, defaults to None):
                Aggregates attributions before visualizing them. If not specified, the default aggregator for the class
                is used.
            do_aggregation (:obj:`bool`, *optional*, defaults to True):
                Whether to aggregate the attributions before visualizing them. Allows to skip aggregation if the
                attributions are already aggregated.

        Returns:
            :obj:`str`: The HTML code of the visualization if :obj:`return_html` is set to True, otherwise None.
        """
        from inseq import show_attributions, show_granular_attributions

        # If no aggregator is specified, the default aggregator for the class is used
        aggregated = self.aggregate(aggregator, **kwargs) if do_aggregation else self
        if (aggregated.source_attributions is not None and aggregated.source_attributions.shape[1] == 0) or (
            aggregated.target_attributions is not None and aggregated.target_attributions.shape[1] == 0
        ):
            tokens = "".join(tid.token for tid in self.target)
            logger.warning(f"Found empty attributions, skipping attribution matching generation: {tokens}")
        if (
            (aggregated.source_attributions is not None and aggregated.source_attributions.ndim == 2)
            or (aggregated.target_attributions is not None and aggregated.target_attributions.ndim == 2)
            or (aggregated.source_attributions is None and aggregated.target_attributions is None)
        ):
            return show_attributions(
                attributions=aggregated, min_val=min_val, max_val=max_val, display=display, return_html=return_html
            )
        else:
            return show_granular_attributions(
                attributions=aggregated,
                max_show_size=max_show_size,
                min_val=min_val,
                max_val=max_val,
                show_dim=show_dim,
                display=display,
                return_html=return_html,
                return_figure=return_figure,
                slice_dims=slice_dims,
            )

    def show_granular(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        max_show_size: int | None = None,
        show_dim: int | str | None = None,
        slice_dims: dict[int | str, tuple[int, int]] | None = None,
        display: bool = True,
        return_html: bool | None = False,
        return_figure: bool = False,
    ) -> str | None:
        """Visualizes granular attribution heatmaps in HTML format.

        Args:
            min_val (:obj:`int`, *optional*, defaults to None):
                Lower attribution score threshold for color map.
            max_val (:obj:`int`, *optional*, defaults to None):
                Upper attribution score threshold for color map.
            max_show_size (:obj:`int`, *optional*, defaults to None):
                Maximum dimension size for additional dimensions to be visualized. Default: 20.
            show_dim (:obj:`int` or :obj:`str`, *optional*, defaults to None):
                Dimension to be visualized along with the source and target tokens. Can be either the dimension index or
                the dimension name. Works only if the dimension size is less than or equal to `max_show_size`.
            slice_dims (:obj:`dict[int or str, tuple[int, int]]`, *optional*, defaults to None):
                Dimensions to be sliced and visualized along with the source and target tokens. The dictionary should
                contain the dimension index or name as the key and the slice range as the value.
            display (:obj:`bool`, *optional*, defaults to True):
                Whether to show the output of the visualization function.
            return_html (:obj:`bool`, *optional*, defaults to False):
                If true, returns the HTML corresponding to the notebook visualization of the attributions in
                string format, for saving purposes.
            return_figure (:obj:`bool`, *optional*, defaults to False):
                If true, returns the Treescope figure object for further manipulation.

        Returns:
            `str`: Returns the HTML output if `return_html=True`
        """
        from inseq import show_granular_attributions

        return show_granular_attributions(
            attributions=self,
            max_show_size=max_show_size,
            min_val=min_val,
            max_val=max_val,
            show_dim=show_dim,
            slice_dims=slice_dims,
            display=display,
            return_html=return_html,
            return_figure=return_figure,
        )

    def show_tokens(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        display: bool = True,
        return_html: bool | None = False,
        return_figure: bool = False,
        replace_char: dict[str, str] | None = None,
        wrap_after: int | str | list[str] | tuple[str] | None = None,
        step_score_highlight: str | None = None,
        aggregator: AggregatorPipeline | type[Aggregator] = None,
        do_aggregation: bool = True,
        **kwargs,
    ) -> str | None:
        """Visualizes token-level attributions in HTML format.

        Args:
            attributions (:class:`~inseq.data.attribution.FeatureAttributionSequenceOutput`):
                Sequence attributions to be visualized.
            min_val (:obj:`int`, *optional*, defaults to None):
                Lower attribution score threshold for color map.
            max_val (:obj:`int`, *optional*, defaults to None):
                Upper attribution score threshold for color map.
            display (:obj:`bool`, *optional*, defaults to True):
                Whether to show the output of the visualization function.
            return_html (:obj:`bool`, *optional*, defaults to False):
                If true, returns the HTML corresponding to the notebook visualization of the attributions in string format,
                for saving purposes.
            return_figure (:obj:`bool`, *optional*, defaults to False):
                If true, returns the Treescope figure object for further manipulation.
            replace_char (:obj:`dict[str, str]`, *optional*, defaults to None):
                Dictionary mapping strings to be replaced to replacement options, used for cleaning special characters.
                Default: {}.
            wrap_after (:obj:`int` or :obj:`str` or :obj:`list[str]` :obj:`tuple[str]]`, *optional*, defaults to None):
                Token indices or tokens after which to wrap lines. E.g. 10 = wrap after every 10 tokens, "hi" = wrap after
                word hi occurs, ["." "!", "?"] or ".!?" = wrap after every sentence-ending punctuation.
            step_score_highlight (`str`, *optional*, defaults to None):
                Name of the step score to use to highlight generated tokens in the visualization. If None, no highlights are
                shown. Default: None.
        """
        from inseq import show_token_attributions

        aggregated = self.aggregate(aggregator, **kwargs) if do_aggregation else self
        return show_token_attributions(
            attributions=aggregated,
            min_val=min_val,
            max_val=max_val,
            display=display,
            return_html=return_html,
            return_figure=return_figure,
            replace_char=replace_char,
            wrap_after=wrap_after,
            step_score_highlight=step_score_highlight,
        )

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

    def weight_attributions(self, step_fn_id: str):
        """Weights attribution scores in place by the value of the selected step function for every generation step.

        Args:
            step_fn_id (`str`):
                The id of the step function to use for weighting the attributions (e.g. ``probability``)
        """
        aggregated_attr = self.aggregate()
        step_scores = self.step_scores[step_fn_id].T.unsqueeze(1)
        if self.source_attributions is not None:
            source_attr = aggregated_attr.source_attributions.float().T
            self.source_attributions = (step_scores * source_attr).T
        if self.target_attributions is not None:
            target_attr = aggregated_attr.target_attributions.float().T
            self.target_attributions = (step_scores * target_attr).T
        # Empty aggregator pipeline -> no aggregation
        self._aggregator = []
        return self

    def get_scores_dicts(
        self,
        aggregator: AggregatorPipeline | type[Aggregator] = None,
        do_aggregation: bool = True,
        **kwargs,
    ) -> dict[str, dict[str, dict[str, float]]]:
        # If no aggregator is specified, the default aggregator for the class is used
        aggr = self.aggregate(aggregator, **kwargs) if do_aggregation else self
        return_dict = {"source_attributions": {}, "target_attributions": {}, "step_scores": {}}
        for tgt_idx in range(aggr.attr_pos_start, aggr.attr_pos_end):
            tgt_tok = aggr.target[tgt_idx]
            if aggr.source_attributions is not None:
                return_dict["source_attributions"][(tgt_idx, tgt_tok.token)] = {}
                for src_idx, src_tok in enumerate(aggr.source):
                    return_dict["source_attributions"][(tgt_idx, tgt_tok.token)][
                        (src_idx, src_tok.token)
                    ] = aggr.source_attributions[src_idx, tgt_idx - aggr.attr_pos_start].item()
            if aggr.target_attributions is not None:
                return_dict["target_attributions"][(tgt_idx, tgt_tok.token)] = {}
                for tgt_idx_attr in range(aggr.attr_pos_end):
                    tgt_tok_attr = aggr.target[tgt_idx_attr]
                    return_dict["target_attributions"][(tgt_idx, tgt_tok.token)][
                        (tgt_idx_attr, tgt_tok_attr.token)
                    ] = aggr.target_attributions[tgt_idx_attr, tgt_idx - aggr.attr_pos_start].item()
            if aggr.step_scores is not None:
                return_dict["step_scores"][(tgt_idx, tgt_tok.token)] = {}
                for step_score_id, step_score in aggr.step_scores.items():
                    return_dict["step_scores"][(tgt_idx, tgt_tok.token)][step_score_id] = step_score[
                        tgt_idx - aggr.attr_pos_start
                    ].item()
        return return_dict


@dataclass(eq=False, repr=False)
class FeatureAttributionStepOutput(TensorWrapper):
    """Output of a single step of feature attribution, plus extra information related to what was attributed."""

    source_attributions: StepAttributionTensor | None = None
    step_scores: dict[str, SingleScorePerStepTensor] | None = None
    target_attributions: StepAttributionTensor | None = None
    sequence_scores: dict[str, MultipleScoresPerStepTensor] | None = None
    source: OneOrMoreTokenWithIdSequences | None = None
    prefix: OneOrMoreTokenWithIdSequences | None = None
    target: OneOrMoreTokenWithIdSequences | None = None
    _sequence_cls: type["FeatureAttributionSequenceOutput"] = FeatureAttributionSequenceOutput

    def __post_init__(self):
        self.to(torch.float32)
        if self.step_scores is None:
            self.step_scores = {}
        if self.sequence_scores is None:
            self.sequence_scores = {}

    def get_sequence_cls(self, **kwargs):
        return self._sequence_cls(**kwargs)

    def remap_from_filtered(
        self,
        target_attention_mask: TargetIdsTensor,
        batch: DecoderOnlyBatch | EncoderDecoderBatch,
        is_final_step_method: bool = False,
    ) -> None:
        """Remaps the attributions to the original shape of the input sequence."""
        batch_size = (
            len(batch.sources.input_tokens) if self.source_attributions is not None else len(batch.target_tokens)
        )
        source_len = len(batch.sources.input_tokens[0])
        target_len = len(batch.target_tokens[0])
        # Normal per-step attribution outputs have shape (batch_size, seq_len, ...)
        other_dims_start_idx = 2
        # Final step attribution outputs have shape (batch_size, seq_len, seq_len, ...)
        if is_final_step_method:
            other_dims_start_idx += 1
        other_dims = (
            self.source_attributions.shape[other_dims_start_idx:]
            if self.source_attributions is not None
            else self.target_attributions.shape[other_dims_start_idx:]
        )
        if self.source_attributions is not None:
            self.source_attributions = remap_from_filtered(
                original_shape=(batch_size, *self.source_attributions.shape[1:]),
                mask=target_attention_mask,
                filtered=self.source_attributions,
            )
        if self.target_attributions is not None:
            self.target_attributions = remap_from_filtered(
                original_shape=(batch_size, *self.target_attributions.shape[1:]),
                mask=target_attention_mask,
                filtered=self.target_attributions,
            )
        if self.step_scores is not None:
            for score_name, score_tensor in self.step_scores.items():
                self.step_scores[score_name] = remap_from_filtered(
                    original_shape=(batch_size, 1),
                    mask=target_attention_mask,
                    filtered=score_tensor.unsqueeze(-1),
                ).squeeze(-1)
        if self.sequence_scores is not None:
            for score_name, score_tensor in self.sequence_scores.items():
                if score_name.startswith("decoder"):
                    original_shape = (batch_size, target_len, target_len, *other_dims)
                elif score_name.startswith("encoder"):
                    original_shape = (batch_size, source_len, source_len, *other_dims)
                else:  # default case: cross-attention
                    original_shape = (batch_size, source_len, target_len, *other_dims)
                self.sequence_scores[score_name] = remap_from_filtered(
                    original_shape=original_shape,
                    mask=target_attention_mask,
                    filtered=score_tensor,
                )


@dataclass
class FeatureAttributionOutput:
    """Output produced by the `AttributionModel.attribute` method.

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

    sequence_attributions: list[FeatureAttributionSequenceOutput]
    step_attributions: list[FeatureAttributionStepOutput] | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.__dict__)})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        for self_seq, other_seq in zip(self.sequence_attributions, other.sequence_attributions, strict=False):
            if self_seq != other_seq:
                return False
        if self.step_attributions is not None and other.step_attributions is not None:
            for self_step, other_step in zip(self.step_attributions, other.step_attributions, strict=False):
                if self_step != other_step:
                    return False
        if self.info != other.info:
            return False
        return True

    def __getitem__(self, item) -> FeatureAttributionSequenceOutput:
        return self.sequence_attributions[item]

    def __len__(self) -> int:
        return len(self.sequence_attributions)

    def __iter__(self):
        return iter(self.sequence_attributions)

    def __add__(self, other) -> "FeatureAttributionOutput":
        return merge_attributions([self, other])

    def __radd__(self, other) -> "FeatureAttributionOutput":
        return self.__add__(other)

    def save(
        self,
        path: PathLike,
        overwrite: bool = False,
        compress: bool = False,
        ndarray_compact: bool = True,
        use_primitives: bool = False,
        split_sequences: bool = False,
        scores_precision: ScorePrecision = "float32",
    ) -> None:
        """Save class contents to a JSON file.

        Args:
            path (:obj:`os.PathLike`): Path to the folder where the attribution output will be stored
                (e.g. ``./out.json``).
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
            scores_precision (:obj:`str`, *optional*, defaults to "float32"):
                Rounding precision for saved scores. Can be used to reduce space on disk but introduces rounding
                errors. Can be combined with compress=True for further space reduction.
                Accepted values: "float32", "float16", or "float8". Default: "float32" (no rounding).
        """
        if not overwrite and Path(path).exists():
            raise ValueError(f"{path} already exists. Override with overwrite=True.")
        save_outs = []
        paths = []
        if split_sequences:
            for seq_id in range(len(self.sequence_attributions)):
                attr_out = deepcopy(self)
                attr_out.sequence_attributions = [
                    attr_out.sequence_attributions[seq_id]._convert_to_safetensors(scores_precision=scores_precision)
                ]  # this overwrites the original
                attr_out.step_attributions = None
                attr_out.info["input_texts"] = [attr_out.info["input_texts"][seq_id]]
                attr_out.info["generated_texts"] = [attr_out.info["generated_texts"][seq_id]]
                save_outs.append(attr_out)
                paths.append(f"{str(path).split('.json')[0]}_{seq_id}.json{'.gz' if compress else ''}")
        else:
            self_out = deepcopy(self)
            self_out.sequence_attributions = [
                seq._convert_to_safetensors(scores_precision=scores_precision)
                for seq in self_out.sequence_attributions
            ]
            save_outs.append(self_out)
            paths.append(path)
        for attr_out, path_out in zip(save_outs, paths, strict=False):
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
        path: PathLike,
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
        out.sequence_attributions = [seq._recover_from_safetensors() for seq in out.sequence_attributions]
        if out.step_attributions is not None:
            out.step_attributions = [step._recover_from_safetensors() for step in out.step_attributions]
        return out

    def aggregate(
        self,
        aggregator: AggregatorPipeline | type[Aggregator] = None,
        **kwargs,
    ) -> "FeatureAttributionOutput":
        """Aggregate the sequence attributions using one or more aggregators.

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
        min_val: int | None = None,
        max_val: int | None = None,
        max_show_size: int | None = None,
        show_dim: int | str | None = None,
        slice_dims: dict[int | str, tuple[int, int]] | None = None,
        display: bool = True,
        return_html: bool | None = False,
        return_figure: bool = False,
        aggregator: AggregatorPipeline | type[Aggregator] = None,
        do_aggregation: bool = True,
        **kwargs,
    ) -> str | list | None:
        """Visualize the sequence attributions.

        Args:
            min_val (int, optional): Minimum value for color scale.
            max_val (int, optional): Maximum value for color scale.
            max_show_size (int, optional): Maximum size of the dimension to show.
            show_dim (int or str, optional): Dimension to show.
            slice_dims (dict[int or str, tuple[int, int]], optional): Dimensions to slice.
            display (bool, optional): If True, display the attribution visualization.
            return_html (bool, optional): If True, return the attribution visualization as HTML.
            return_figure (bool, optional): If True, return the Treescope figure object for further manipulation.
            aggregator (:obj:`AggregatorPipeline` or :obj:`Type[Aggregator]`, optional): Aggregator
                or pipeline to use. If not provided, the default aggregator for every sequence attribution
                is used.
            do_aggregation (:obj:`bool`, *optional*, defaults to True):
                Whether to aggregate the attributions before visualizing them. Allows to skip aggregation if the
                attributions are already aggregated.

        Returns:
            str: Attribution visualization as HTML if `return_html=True`
            list: List of Treescope figure objects if `return_figure=True`
            None if `return_html=False` and `return_figure=False`

        """
        out_str = ""
        out_figs = []
        for attr in self.sequence_attributions:
            curr_out = attr.show(
                min_val=min_val,
                max_val=max_val,
                max_show_size=max_show_size,
                show_dim=show_dim,
                slice_dims=slice_dims,
                display=display,
                return_html=return_html,
                return_figure=return_figure,
                aggregator=aggregator,
                do_aggregation=do_aggregation,
                **kwargs,
            )
            if return_html:
                out_str += curr_out
            if return_figure:
                out_figs.append(curr_out)
        if return_html:
            return out_str
        if return_figure:
            return out_figs

    def show_granular(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        max_show_size: int | None = None,
        show_dim: int | str | None = None,
        slice_dims: dict[int | str, tuple[int, int]] | None = None,
        display: bool = True,
        return_html: bool = False,
        return_figure: bool = False,
    ) -> str | None:
        """Visualizes granular attribution heatmaps in HTML format.

        Args:
            min_val (:obj:`int`, *optional*, defaults to None):
                Lower attribution score threshold for color map.
            max_val (:obj:`int`, *optional*, defaults to None):
                Upper attribution score threshold for color map.
            max_show_size (:obj:`int`, *optional*, defaults to None):
                Maximum dimension size for additional dimensions to be visualized. Default: 20.
            show_dim (:obj:`int` or :obj:`str`, *optional*, defaults to None):
                Dimension to be visualized along with the source and target tokens. Can be either the dimension index or
                the dimension name. Works only if the dimension size is less than or equal to `max_show_size`.
            slice_dims (:obj:`dict[int or str, tuple[int, int]]`, *optional*, defaults to None):
                Dimensions to be sliced and visualized along with the source and target tokens. The dictionary should
                contain the dimension index or name as the key and the slice range as the value.
            display (:obj:`bool`, *optional*, defaults to True):
                Whether to show the output of the visualization function.
            return_html (:obj:`bool`, *optional*, defaults to False):
                If true, returns the HTML corresponding to the notebook visualization of the attributions in
                string format, for saving purposes.
            return_figure (:obj:`bool`, *optional*, defaults to False):
                If true, returns the Treescope figure object for further manipulation.

        Returns:
            `str`: Returns the HTML output if `return_html=True`
        """
        out_str = ""
        out_figs = []
        for attr in self.sequence_attributions:
            curr_out = attr.show_granular(
                min_val=min_val,
                max_val=max_val,
                max_show_size=max_show_size,
                show_dim=show_dim,
                slice_dims=slice_dims,
                display=display,
                return_html=return_html,
            )
            if return_html:
                out_str += curr_out
            if return_figure:
                out_figs.append(curr_out)
        if return_html:
            return out_str
        if return_figure:
            return out_figs

    def show_tokens(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        display: bool = True,
        return_html: bool = False,
        return_figure: bool = False,
        replace_char: dict[str, str] | None = None,
        wrap_after: int | str | list[str] | tuple[str] | None = None,
        step_score_highlight: str | None = None,
        aggregator: AggregatorPipeline | type[Aggregator] = None,
        do_aggregation: bool = True,
        **kwargs,
    ) -> str | None:
        """Visualizes token-level attributions in HTML format.

        Args:
            min_val (:obj:`int`, *optional*, defaults to None):
                Lower attribution score threshold for color map.
            max_val (:obj:`int`, *optional*, defaults to None):
                Upper attribution score threshold for color map.
            display (:obj:`bool`, *optional*, defaults to True):
                Whether to show the output of the visualization function.
            return_html (:obj:`bool`, *optional*, defaults to False):
                If true, returns the HTML corresponding to the notebook visualization of the attributions in string format,
                for saving purposes.
            return_figure (:obj:`bool`, *optional*, defaults to False):
                If true, returns the Treescope figure object for further manipulation.
            replace_char (:obj:`dict[str, str]`, *optional*, defaults to None):
                Dictionary mapping strings to be replaced to replacement options, used for cleaning special characters.
                Default: {}.
            wrap_after (:obj:`int` or :obj:`str` or :obj:`list[str]` :obj:`tuple[str]]`, *optional*, defaults to None):
                Token indices or tokens after which to wrap lines. E.g. 10 = wrap after every 10 tokens, "hi" = wrap after
                word hi occurs, ["." "!", "?"] or ".!?" = wrap after every sentence-ending punctuation.
            step_score_highlight (`str`, *optional*, defaults to None):
                Name of the step score to use to highlight generated tokens in the visualization. If None, no highlights are
                shown. Default: None.
        """
        out_str = ""
        out_figs = []
        for attr in self.sequence_attributions:
            curr_out = attr.show_tokens(
                min_val=min_val,
                max_val=max_val,
                display=display,
                return_html=return_html,
                return_figure=return_figure,
                replace_char=replace_char,
                wrap_after=wrap_after,
                step_score_highlight=step_score_highlight,
                aggregator=aggregator,
                do_aggregation=do_aggregation,
                **kwargs,
            )
            if return_html:
                out_str += curr_out
            if return_figure:
                out_figs.append(curr_out)
        if return_html:
            return out_str
        if return_figure:
            return out_figs

    def weight_attributions(self, step_score_id: str):
        for i, attr in enumerate(self.sequence_attributions):
            self.sequence_attributions[i] = attr.weight_attributions(step_score_id)

    def get_scores_dicts(
        self, aggregator: AggregatorPipeline | type[Aggregator] = None, do_aggregation: bool = True, **kwargs
    ) -> list[dict[str, dict[str, dict[str, float]]]]:
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


@dataclass(eq=False, repr=False)
class GranularFeatureAttributionSequenceOutput(FeatureAttributionSequenceOutput):
    """Raw output of a single sequence of granular feature attribution.

    An example of granular feature attribution methods are gradient-based attribution methods such as Integrated
    Gradients, returning one score per hidden dimension of the model for every generated token.

    Adds the convergence delta and default L2 + normalization merging of attributions to the base class.
    """

    def __post_init__(self):
        super().__post_init__()
        self._aggregator = "vnorm"
        self._dict_aggregate_fn["source_attributions"]["scores"] = "vnorm"
        self._dict_aggregate_fn["target_attributions"]["scores"] = "vnorm"
        if "deltas" not in self._dict_aggregate_fn["step_scores"]["spans"]:
            self._dict_aggregate_fn["step_scores"]["spans"]["deltas"] = "absmax"
        self._attribution_dim_names = {
            "source_attributions": {0: "Input Tokens", 1: "Generated Tokens", 2: "Embedding Dimension"},
            "target_attributions": {0: "Input Tokens", 1: "Generated Tokens", 2: "Embedding Dimension"},
        }


@dataclass(eq=False, repr=False)
class GranularFeatureAttributionStepOutput(FeatureAttributionStepOutput):
    """Raw output of a single step of gradient feature attribution."""

    _sequence_cls: type["FeatureAttributionSequenceOutput"] = GranularFeatureAttributionSequenceOutput


@dataclass(eq=False, repr=False)
class CoarseFeatureAttributionSequenceOutput(FeatureAttributionSequenceOutput):
    """Raw output of a single sequence of coarse-grained feature attribution.

    An example of coarse-grained feature attribution methods are occlusion methods in which a whole token is masked at
    once, producing a single output score per token.
    """

    def __post_init__(self):
        super().__post_init__()
        self._aggregator = []


@dataclass(eq=False, repr=False)
class CoarseFeatureAttributionStepOutput(FeatureAttributionStepOutput):
    """Raw output of a single step of coarse-grained feature attribution."""

    _sequence_cls: type["FeatureAttributionSequenceOutput"] = CoarseFeatureAttributionSequenceOutput


@dataclass(eq=False, repr=False)
class MultiDimensionalFeatureAttributionSequenceOutput(FeatureAttributionSequenceOutput):
    """Raw output of a single sequence of multi-dimensional feature attribution.

    Multi-dimensional feature attribution methods are a generalization of granular feature attribution methods
    allowing for an arbitrary number of extra dimensions. For example, the attention method returns one score per
    attention head and per layer for every source-target token pair in the source attributions (i.e. 2 dimensions).
    """

    _num_dimensions: int = 2

    def __post_init__(self):
        super().__post_init__()
        self._aggregator = ["mean"] * self._num_dimensions
        self._attribution_dim_names = {
            "source_attributions": {0: "Input Tokens", 1: "Generated Tokens", 2: "Model Layer"},
            "target_attributions": {0: "Input Tokens", 1: "Generated Tokens", 2: "Model Layer"},
            "encoder": {0: "Input Tokens", 1: "Input Tokens", 2: "Model Layer"},
            "decoder": {0: "Generated Tokens", 1: "Generated Tokens", 2: "Model Layer"},
        }
        if self._num_dimensions == 2:
            for key in self._attribution_dim_names.keys():
                self._attribution_dim_names[key][3] = "Attention Head"


@dataclass(eq=False, repr=False)
class MultiDimensionalFeatureAttributionStepOutput(FeatureAttributionStepOutput):
    """Raw output of a single step of multi-dimensional feature attribution."""

    _num_dimensions: int = 2
    _sequence_cls: type["FeatureAttributionSequenceOutput"] = MultiDimensionalFeatureAttributionSequenceOutput

    def get_sequence_cls(self, **kwargs):
        return MultiDimensionalFeatureAttributionSequenceOutput(_num_dimensions=self._num_dimensions, **kwargs)
