import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import torch

from ..utils import (
    Registry,
    aggregate_contiguous,
    aggregate_token_pair,
    aggregate_token_sequence,
    available_classes,
    extract_signature_args,
)
from ..utils import normalize as normalize_fn
from ..utils.typing import IndexSpan, TokenWithId
from .aggregation_functions import AggregationFunction
from .data_utils import TensorWrapper

if TYPE_CHECKING:
    from .attribution import FeatureAttributionSequenceOutput


logger = logging.getLogger(__name__)

AggregableMixinClass = TypeVar("AggregableMixinClass", bound="AggregableMixin")


class DictWithDefault(dict):
    """Used to pass specific values to field-specific calls of the aggregate function in Aggregator.

    DictWithDefault dictionary objects won't be passed as a whole to all field-specific functions called by
    Aggregator.aggregate, and instead only the values with the name of the corresponding field will be used.
    When these are missing, the default field of DictWithDefault will be used as fallback.
    """

    @staticmethod
    def _get_fn(name: str) -> Callable:
        if name not in available_classes(AggregationFunction):
            raise ValueError(
                f"Unknown aggregation function {name}. Choose from {','.join(available_classes(AggregationFunction))}."
            )
        return AggregationFunction.available_classes()[name]()

    def __init__(self, default: Union[str, Callable], **kwargs):
        super().__init__(**kwargs)
        self.default = self._get_fn(default) if isinstance(default, str) else default

    def __getitem__(self, key):
        try:
            value = super().__getitem__(key)
            if isinstance(value, str):
                return self._get_fn(value)
            elif isinstance(value, dict):
                return DictWithDefault(self.default, **value)
            return value
        except KeyError:
            return self.default


class Aggregator(Registry):
    registry_attr = "aggregator_name"

    @classmethod
    def start_aggregation_hook(cls, tensors: TensorWrapper, **kwargs):
        """Hook called at the start of the aggregation process.

        Use to ensure a prerequisite that is independent of previous
        aggregation steps and fundamental to the aggregation process
        (e.g. parameters are of the correct type). Will avoid performing aggregation steps before
        returning an error.
        """
        pass

    @classmethod
    def pre_aggregate_hook(cls, tensors: TensorWrapper, **kwargs):
        """Hook called right before the aggregation function is called.

        Use to ensure a prerequisite that is functional of previous
        aggregation steps and fundamental to the aggregation process
        (e.g. the aggregatable object produced by the previous step has correct shapes).
        """
        pass

    @classmethod
    @abstractmethod
    def _aggregate(cls, tensors: TensorWrapper, **kwargs):
        pass

    @classmethod
    def aggregate(
        cls,
        tensors: AggregableMixinClass,
        do_pre_aggregation_checks: bool = True,
        do_post_aggregation_checks: bool = True,
        **kwargs,
    ) -> AggregableMixinClass:
        if do_pre_aggregation_checks:
            cls.start_aggregation_hook(tensors, **kwargs)
        cls.pre_aggregate_hook(tensors, **kwargs)
        aggregated = cls._aggregate(tensors, **kwargs)
        cls.post_aggregate_hook(aggregated, **kwargs)
        if do_post_aggregation_checks:
            cls.end_aggregation_hook(aggregated, **kwargs)
        return aggregated

    @classmethod
    def post_aggregate_hook(cls, tensors: TensorWrapper, **kwargs):
        """Hook called right after the aggregation function is called.

        Verifies that the aggregated object has the correct properties.
        """
        pass

    @classmethod
    def end_aggregation_hook(cls, tensors: TensorWrapper, **kwargs):
        """Hook called at the end of the aggregation process.

        Use to ensure that the final product of aggregation is compliant with
        the requirements of individual aggregators.
        """
        pass


def _get_aggregators_from_id(
    aggregator: str,
    aggregate_fn: Optional[str] = None,
) -> Tuple[Type[Aggregator], Optional[AggregationFunction]]:
    if aggregator in available_classes(Aggregator):
        aggregator = Aggregator.available_classes()[aggregator]
    elif aggregator in available_classes(AggregationFunction):
        if aggregate_fn is not None:
            raise ValueError(
                "If aggregator is a string identifying an aggregation function, aggregate_fn should not be provided."
            )
        aggregate_fn = aggregator
        aggregator = SequenceAttributionAggregator
    else:
        raise ValueError(
            f"Unknown aggregator {aggregator}. Choose from {', '.join(available_classes(Aggregator))}.\n"
            f"Alternatively, choose from the aggregate_fn options {', '.join(available_classes(AggregationFunction))} "
            "for scores aggregation with the chosen function."
        )
    if aggregate_fn is None:
        return aggregator, aggregate_fn
    if aggregate_fn not in available_classes(AggregationFunction):
        raise ValueError(
            f"Unknown aggregation function {aggregate_fn}. "
            f"Choose from {', '.join(available_classes(AggregationFunction))}"
        )
    aggregate_fn = AggregationFunction.available_classes()[aggregate_fn]()
    return aggregator, aggregate_fn


class AggregatorPipeline:
    def __init__(
        self,
        aggregators: List[Union[str, Type[Aggregator]]],
        aggregate_fn: Optional[List[Union[str, Callable]]] = None,
    ):
        self.aggregators: List[Type[Aggregator]] = []
        self.aggregate_fn: List[Callable] = []
        if aggregate_fn is not None:
            if len(aggregate_fn) != len(aggregators):
                raise ValueError(
                    "If custom aggregate_fn are provided, their number should match the number of aggregators."
                )
        for idx in range(len(aggregators)):
            curr_aggregator = aggregators[idx]
            curr_aggregate_fn = aggregate_fn[idx] if aggregate_fn is not None else None
            if isinstance(curr_aggregator, str):
                curr_aggregator, curr_aggregate_fn = _get_aggregators_from_id(curr_aggregator, curr_aggregate_fn)
            self.aggregators.append(curr_aggregator)
            self.aggregate_fn.append(curr_aggregate_fn)

    def aggregate(
        self,
        tensors: AggregableMixinClass,
        do_pre_aggregation_checks: bool = True,
        do_post_aggregation_checks: bool = True,
        **kwargs,
    ) -> AggregableMixinClass:
        if do_pre_aggregation_checks:
            for aggregator in self.aggregators:
                aggregator.start_aggregation_hook(tensors, **kwargs)
        for aggregator, aggregate_fn in zip(self.aggregators, self.aggregate_fn):
            curr_aggregation_kwargs = kwargs.copy()
            if aggregate_fn is not None:
                curr_aggregation_kwargs["aggregate_fn"] = aggregate_fn
            tensors = aggregator.aggregate(
                tensors, do_pre_aggregation_checks=False, do_post_aggregation_checks=False, **curr_aggregation_kwargs
            )
        if do_post_aggregation_checks:
            for aggregator in self.aggregators:
                aggregator.end_aggregation_hook(tensors, **kwargs)
        return tensors


AggregatorInput = Union[AggregatorPipeline, Type[Aggregator], str, Sequence[Union[str, Type[Aggregator]]], None]


def list_aggregators() -> List[str]:
    """Lists identifiers for all available aggregators."""
    return available_classes(Aggregator)


class AggregableMixin(ABC):
    _aggregator: Union[AggregatorPipeline, Type[Aggregator]]

    def aggregate(
        self: AggregableMixinClass,
        aggregator: AggregatorInput = None,
        aggregate_fn: Union[str, Sequence[str], None] = None,
        do_pre_aggregation_checks: bool = True,
        do_post_aggregation_checks: bool = True,
        **kwargs,
    ) -> AggregableMixinClass:
        """Aggregate outputs using the default or provided aggregator.

        Args:
            aggregator (:obj:`AggregatorPipeline` or :obj:`Type[Aggregator]` or :obj:`str` or , optional): Aggregator
                pipeline to use. If not provided, the default aggregator pipeline is used.

        Returns:
            :obj:`AggregableMixin`: The aggregated output class.
        """
        if aggregator is None:
            aggregator = self._aggregator
        if isinstance(aggregator, str):
            if isinstance(aggregate_fn, (list, tuple)):
                raise ValueError(
                    "If a single aggregator is used, aggregate_fn should also be a string identifier for the "
                    "corresponding aggregation function if defined."
                )
            aggregator, aggregate_fn = _get_aggregators_from_id(aggregator, aggregate_fn)
            if aggregate_fn is not None:
                kwargs["aggregate_fn"] = aggregate_fn
        elif isinstance(aggregator, (list, tuple)):
            if all(isinstance(a, (str, type)) for a in aggregator):
                aggregator = AggregatorPipeline(aggregator, aggregate_fn)
            elif all(isinstance(agg, tuple) for agg in aggregator):
                if all(isinstance(idx, (str, type)) for agg in aggregator for idx in agg):
                    aggregator = AggregatorPipeline([a[0] for a in aggregator], [a[1] for a in aggregator])
            else:
                raise ValueError(
                    "If aggregator is a sequence, it should contain either strings/classes identifying aggregators"
                    "or tuples of pairs of strings/classes identifying aggregators and aggregate functions."
                )
        return aggregator.aggregate(
            self,
            do_pre_aggregation_checks=do_pre_aggregation_checks,
            do_post_aggregation_checks=do_post_aggregation_checks,
            **kwargs,
        )

    @abstractmethod
    def __post_init__(self):
        pass


class SequenceAttributionAggregator(Aggregator):
    """Aggregates sequence attributions using a custom function. By default, the mean function is used.

    Enables aggregation for the FeatureAttributionSequenceOutput class using an aggregation function of choice.

    Args:
        attr (:class:`~inseq.data.FeatureAttributionSequenceOutput`): The attribution object to aggregate.
        aggregate_fn (:obj:`Callable`, optional): Function used to aggregate sequence attributions.
            Defaults to summing over the last dimension and renormalizing by the norm of the
            source(+target) attributions for granular attributions, no aggregation for token-level
            attributions.
    """

    aggregator_name = "scores"
    aggregator_family = "scores"
    default_fn = "mean"

    @classmethod
    def _aggregate(
        cls, attr: "FeatureAttributionSequenceOutput", aggregate_fn: Union[str, Callable, None] = None, **kwargs
    ) -> "FeatureAttributionSequenceOutput":
        if aggregate_fn is None and isinstance(attr._dict_aggregate_fn, dict):
            aggregate_fn = DictWithDefault(default=cls.default_fn, **attr._dict_aggregate_fn)
        elif aggregate_fn is not None:
            aggregate_fn = DictWithDefault(default=aggregate_fn)

        # Dispatch kwargs to the corresponding field-specific functions.
        # E.g. aggregate_source_attributions will take care of the source_attributions field.
        aggregated_sequence_attribution_fields = {}
        for field in attr.to_dict().keys():
            if aggregate_fn is not None:
                kwargs["aggregate_fn"] = aggregate_fn[field]

                # If the subclass is a dict, then we assume its fields represent variants depending on the aggregator
                # family that is being used (see e.g. step_scores in DEFAULT_ATTRIBUTION_AGGREGATE_DICT)
                if isinstance(kwargs["aggregate_fn"], dict):
                    kwargs["aggregate_fn"] = kwargs["aggregate_fn"][cls.aggregator_family]
            field_func = getattr(cls, f"aggregate_{field}")
            aggregated_sequence_attribution_fields[field] = field_func(attr, **kwargs)
        return attr.__class__(**aggregated_sequence_attribution_fields)

    @classmethod
    def _process_attribution_scores(
        cls,
        attr: "FeatureAttributionSequenceOutput",
        aggregate_fn: AggregationFunction,
        select_idx: Union[int, Tuple[int, int], List[int], None] = None,
        normalize: bool = True,
        **kwargs,
    ):
        fn_kwargs = extract_signature_args(kwargs, aggregate_fn)
        # If select_idx is a single int, no aggregation is performed
        do_aggregate = not isinstance(select_idx, int)
        has_source = attr.source_attributions is not None
        has_target = attr.target_attributions is not None
        src_scores = None
        if has_source:
            src_scores = cls._filter_scores(attr.source_attributions, dim=-1, indices=select_idx)
        tgt_scores = None
        if has_target:
            tgt_scores = cls._filter_scores(attr.target_attributions, dim=-1, indices=select_idx)
        if has_source and has_target:
            scores = (src_scores, tgt_scores)
        else:
            scores = src_scores if src_scores is not None else tgt_scores
        if aggregate_fn.takes_sequence_scores:
            fn_kwargs["sequence_scores"] = attr.sequence_scores
        if do_aggregate:
            scores = cls._aggregate_scores(scores, aggregate_fn, dim=-1, **fn_kwargs)
        if normalize:
            scores = normalize_fn(scores)
        return scores

    @classmethod
    def post_aggregate_hook(cls, attr: "FeatureAttributionSequenceOutput", **kwargs):
        super().post_aggregate_hook(attr, **kwargs)
        cls.is_compatible(attr)

    @classmethod
    def end_aggregation_hook(cls, attr: "FeatureAttributionSequenceOutput", **kwargs):
        super().end_aggregation_hook(attr, **kwargs)
        # Needed to ensure the attribution can be visualized
        try:
            if attr.source_attributions is not None:
                assert attr.source_attributions.ndim == 2, attr.source_attributions.shape
            if attr.target_attributions is not None:
                assert attr.target_attributions.ndim == 2, attr.target_attributions.shape
        except AssertionError as e:
            raise RuntimeError(
                f"The aggregated attributions should be 2-dimensional to be visualized. Found dimensions: {e.args[0]}"
                "If you're performing intermediate aggregation and don't aim to visualize the output right away, use"
                "do_post_aggregation_checks=False in the aggregate method to bypass this check."
            ) from e

    @staticmethod
    def aggregate_source(attr: "FeatureAttributionSequenceOutput", **kwargs):
        return attr.source

    @staticmethod
    def aggregate_target(attr: "FeatureAttributionSequenceOutput", **kwargs):
        return attr.target

    @classmethod
    def aggregate_source_attributions(
        cls,
        attr: "FeatureAttributionSequenceOutput",
        aggregate_fn: AggregationFunction,
        select_idx: Union[int, Tuple[int, int], List[int], None] = None,
        normalize: bool = True,
        **kwargs,
    ):
        if attr.source_attributions is None:
            return attr.source_attributions
        scores = cls._process_attribution_scores(attr, aggregate_fn, select_idx, normalize, **kwargs)
        return scores[0] if attr.target_attributions is not None else scores

    @classmethod
    def aggregate_target_attributions(
        cls,
        attr: "FeatureAttributionSequenceOutput",
        aggregate_fn: AggregationFunction,
        select_idx: Union[int, Tuple[int, int], List[int], None] = None,
        normalize: bool = True,
        **kwargs,
    ):
        if attr.target_attributions is None:
            return attr.target_attributions
        scores = cls._process_attribution_scores(attr, aggregate_fn, select_idx, normalize, **kwargs)
        return scores[1] if attr.source_attributions is not None else scores

    @staticmethod
    def aggregate_step_scores(attr: "FeatureAttributionSequenceOutput", **kwargs):
        return attr.step_scores

    @classmethod
    def aggregate_sequence_scores(
        cls,
        attr: "FeatureAttributionSequenceOutput",
        aggregate_fn: AggregationFunction,
        select_idx: Union[int, Tuple[int, int], List[int], None] = None,
        **kwargs,
    ):
        if aggregate_fn.takes_sequence_scores:
            return attr.sequence_scores
        fn_kwargs = extract_signature_args(kwargs, aggregate_fn)
        new_sequence_scores = {}
        for scores_id, seq_scores in attr.sequence_scores.items():
            filtered_scores = cls._filter_scores(seq_scores, dim=-1, indices=select_idx)
            if not isinstance(select_idx, int):
                filtered_scores = cls._aggregate_scores(filtered_scores, aggregate_fn, dim=-1, **fn_kwargs)
            new_sequence_scores[scores_id] = filtered_scores
        return new_sequence_scores

    @staticmethod
    def aggregate_attr_pos_start(attr: "FeatureAttributionSequenceOutput", **kwargs):
        return attr.attr_pos_start

    @staticmethod
    def aggregate_attr_pos_end(attr: "FeatureAttributionSequenceOutput", **kwargs):
        return attr.attr_pos_end

    @staticmethod
    def is_compatible(attr: "FeatureAttributionSequenceOutput"):
        from .attribution import FeatureAttributionSequenceOutput

        assert isinstance(attr, FeatureAttributionSequenceOutput)
        if attr.source_attributions is not None:
            assert attr.source_attributions.shape[0] == len(attr.source)
            assert attr.source_attributions.shape[1] == attr.attr_pos_end - attr.attr_pos_start
        if attr.target_attributions is not None:
            assert attr.target_attributions.shape[0] == min(len(attr.target), attr.attr_pos_end)
            assert attr.target_attributions.shape[1] == attr.attr_pos_end - attr.attr_pos_start
        if attr.step_scores is not None:
            for step_score in attr.step_scores.values():
                assert len(step_score) == attr.attr_pos_end - attr.attr_pos_start

    @staticmethod
    def _filter_scores(
        scores: torch.Tensor,
        dim: int = -1,
        indices: Union[int, Tuple[int, int], List[int], None] = None,
    ) -> torch.Tensor:
        n_units = scores.shape[dim]

        if hasattr(indices, "__iter__"):
            if len(indices) == 0:
                raise RuntimeError("At least two indices must be specified for aggregation.")
            if len(indices) == 1:
                indices = indices[0]

        if isinstance(indices, int):
            if indices not in range(-n_units, n_units):
                raise IndexError(f"Index out of range. Scores only have {n_units} units.")
            indices = indices if indices >= 0 else n_units + indices
            return scores.select(dim, torch.tensor(indices, device=scores.device))
        else:
            if indices is None:
                indices = (0, n_units)
                logger.info("No indices specified for extraction. Using all units by default.")

            # Convert negative indices to positive indices
            if hasattr(indices, "__iter__"):
                indices = type(indices)([h_idx if h_idx >= 0 else n_units + h_idx for h_idx in indices])
            if not hasattr(indices, "__iter__") or (
                len(indices) == 2 and isinstance(indices, tuple) and indices[0] >= indices[1]
            ):
                raise RuntimeError(
                    "A (start, end) tuple of indices representing a span, a list of individual indices"
                    " or a single index must be specified for select_idx."
                )
            max_idx_val = n_units if isinstance(indices, list) else n_units + 1
            if not all(h in range(-n_units, max_idx_val) for h in indices):
                raise IndexError("One or more index out of range. Scores only have {n_units} units.")
            if len(set(indices)) != len(indices):
                raise IndexError("Duplicate indices are not allowed.")
            if isinstance(indices, tuple):
                scores = scores.index_select(dim, torch.arange(indices[0], indices[1]))
            else:
                scores = scores.index_select(dim, torch.tensor(indices, device=scores.device))
            return scores

    @staticmethod
    def _aggregate_scores(
        scores: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        aggregate_fn: AggregationFunction,
        dim: int = -1,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(scores, tuple) and aggregate_fn.takes_single_tensor:
            return tuple(aggregate_fn(score, dim=dim, **kwargs) for score in scores)
        return aggregate_fn(scores, dim=dim, **kwargs)


class ContiguousSpanAggregator(SequenceAttributionAggregator):
    """Reduces sequence attributions across one or more contiguous spans.

    Args:
        attr (:class:`~inseq.data.FeatureAttributionSequenceOutput`): The attribution object to aggregate.
        aggregate_fn (:obj:`Callable`, optional): Function used to aggregate sequence attributions.
            Defaults to the highest absolute value score across the aggregated span, with original sign
            preserved (e.g. [0.3, -0.7, 0.1] -> -0.7).
        source_spans (tuple of [int, int] or sequence of tuples of [int, int], optional): Spans to aggregate
            over for the source sequence. Defaults to no aggregation performed.
        target_spans (tuple of [int, int] or sequence of tuples of [int, int], optional): Spans to aggregate
            over for the target sequence. Defaults to no aggregation performed.

    """

    aggregator_name = "spans"
    aggregator_family = "spans"
    default_fn = "absmax"

    @classmethod
    def start_aggregation_hook(
        cls,
        attr: "FeatureAttributionSequenceOutput",
        source_spans: Optional[IndexSpan] = None,
        target_spans: Optional[IndexSpan] = None,
        **kwargs,
    ):
        super().start_aggregation_hook(attr, **kwargs)
        cls.validate_spans(attr.source, source_spans)
        cls.validate_spans(attr.target, target_spans)

    @classmethod
    def end_aggregation_hook(cls, attr: "FeatureAttributionSequenceOutput", **kwargs):
        pass

    @classmethod
    def aggregate(
        cls,
        attr: "FeatureAttributionSequenceOutput",
        source_spans: Optional[IndexSpan] = None,
        target_spans: Optional[IndexSpan] = None,
        **kwargs,
    ):
        """Spans can be:

        1. A list of the form [pos_start, pos_end] including the contiguous positions of tokens that
            are to be aggregated, if all values are integers and len(span) < len(original_seq)
        2. A list of the form [(pos_start_0, pos_end_0), (pos_start_1, pos_end_1)], same as above but
            for multiple contiguous spans.
        """
        source_spans = cls.format_spans(source_spans)
        target_spans = cls.format_spans(target_spans)
        return super().aggregate(attr, source_spans=source_spans, target_spans=target_spans, **kwargs)

    @staticmethod
    def format_spans(spans) -> List[Tuple[int, int]]:
        if not spans:
            return spans
        return [spans] if isinstance(spans[0], int) else spans

    @classmethod
    def validate_spans(cls, span_sequence: "FeatureAttributionSequenceOutput", spans: Optional[IndexSpan] = None):
        if not spans:
            return
        allmatch = lambda l, type: all(isinstance(x, type) for x in l)
        assert allmatch(spans, int) or allmatch(
            spans, tuple
        ), f"All items must be either indices (int) or spans (tuple), got {spans}"
        spans = cls.format_spans(spans)
        prev_span_max = -1
        for span in spans:
            assert len(span) == 2, f"Spans must contain at least two indexes, got {spans}"
            assert span[1] > span[0] + 1, f"Spans must be non-empty, got {spans}"
            assert (
                span[0] >= prev_span_max
            ), f"Spans must be postive-valued, non-overlapping and in ascending order, got {spans}"
            assert span[1] < len(span_sequence), f"Span values must be indexes of the original span, got {spans}"
            prev_span_max = span[1]

    @staticmethod
    def _aggregate_sequential_scores(scores, x_spans, y_spans, aggregate_fn):
        # First aggregate alongside the y-axis
        scores_aggregated_y = aggregate_contiguous(scores, y_spans, aggregate_fn, aggregate_dim=1)
        # Then aggregate alonside the x-axis
        scores_aggregated_x = aggregate_contiguous(scores_aggregated_y, x_spans, aggregate_fn, aggregate_dim=0)
        return scores_aggregated_x

    @staticmethod
    def aggregate_source(attr, source_spans, **kwargs):
        return aggregate_token_sequence(attr.source, source_spans)

    @staticmethod
    def aggregate_target(attr, target_spans, **kwargs):
        return aggregate_token_sequence(attr.target, target_spans)

    @staticmethod
    def aggregate_source_attributions(attr, source_spans, target_spans, aggregate_fn, **kwargs):
        if attr.source_attributions is None:
            return attr.source_attributions
        # First aggregate along generated target sequence, then along attributed source
        return ContiguousSpanAggregator._aggregate_sequential_scores(
            attr.source_attributions, source_spans, target_spans, aggregate_fn
        )

    @staticmethod
    def aggregate_target_attributions(attr, target_spans, aggregate_fn, **kwargs):
        if attr.target_attributions is None:
            return attr.target_attributions
        # First aggregate along generated target sequence, then along attributed prefix
        return ContiguousSpanAggregator._aggregate_sequential_scores(
            attr.target_attributions, target_spans, target_spans, aggregate_fn
        )

    @staticmethod
    def aggregate_step_scores(attr, target_spans, aggregate_fn, **kwargs):
        if not attr.step_scores:
            return attr.step_scores
        out_dict = {}
        for name, step_scores in attr.step_scores.items():
            agg_fn = aggregate_fn[name] if isinstance(aggregate_fn, dict) else aggregate_fn
            out_dict[name] = aggregate_contiguous(step_scores, target_spans, agg_fn, aggregate_dim=0)
        return out_dict

    @staticmethod
    def aggregate_sequence_scores(attr, source_spans, target_spans, aggregate_fn, **kwargs):
        # Assume sequence scores are shaped like source attributions
        if not attr.sequence_scores:
            return attr.sequence_scores
        out_dict = {}
        for name, step_scores in attr.sequence_scores.items():
            aggregate_fn = aggregate_fn[name] if isinstance(aggregate_fn, dict) else aggregate_fn
            if name.startswith("decoder"):
                out_dict[name] = ContiguousSpanAggregator._aggregate_sequential_scores(
                    step_scores, target_spans, target_spans, aggregate_fn
                )
            elif name.startswith("encoder"):
                out_dict[name] = ContiguousSpanAggregator._aggregate_sequential_scores(
                    step_scores, source_spans, source_spans, aggregate_fn
                )
            else:
                out_dict[name] = ContiguousSpanAggregator._aggregate_sequential_scores(
                    step_scores, source_spans, target_spans, aggregate_fn
                )
        return out_dict


class SubwordAggregator(ContiguousSpanAggregator):
    """Aggregates over subwords by automatic detecting contiguous subword spans.

    Args:
        attr (:class:`~inseq.data.FeatureAttributionSequenceOutput`): The attribution object to aggregate.
        aggregate_fn (:obj:`Callable`, optional): Function to aggregate over the subwords.
            Defaults to the highest absolute value score across the aggregated span, with original sign
            preserved (e.g. [0.3, -0.7, 0.1] -> -0.7).
        aggregate_source (bool, optional): Whether to aggregate over the source sequence. Defaults to True.
        aggregate_target (bool, optional): Whether to aggregate over the target sequence. Defaults to True.
        special_symbol (str, optional): Symbol used to identify subwords. Defaults to '▁', used by SentencePiece.
            If is_suffix_symbol=True, then this symbol is used to identify parts to be aggregated (e.g. # in WordPiece,
            ['phen', '##omen', '##al']). Otherwise, it identifies the roots that should be preserved (e.g. ▁ in
            SentencePiece, ['▁phen', 'omen', 'al']).
        is_suffix_symbol (bool, optional): Whether the special symbol is used to identify suffixes or prefixes.
            Defaults to False.
    """

    aggregator_name = "subwords"

    @classmethod
    def aggregate(
        cls,
        attr: "FeatureAttributionSequenceOutput",
        aggregate_source: bool = True,
        aggregate_target: bool = True,
        special_symbol: str = "▁",
        is_suffix_symbol: bool = False,
        **kwargs,
    ):
        source_spans = []
        target_spans = []
        if aggregate_source:
            source_spans = cls.get_spans(attr.source, special_symbol, is_suffix_symbol)
        if aggregate_target:
            target_spans = cls.get_spans(attr.target, special_symbol, is_suffix_symbol)
        return super().aggregate(attr, source_spans=source_spans, target_spans=target_spans, **kwargs)

    @staticmethod
    def get_spans(tokens: List[TokenWithId], special_symbol: str, is_suffix_symbol: bool):
        spans = []
        last_prefix_idx = 0
        for curr_idx, token in enumerate(tokens):
            # Suffix if token start with special suffix symbol, or if it doesn't have the special prefix symbol.
            is_suffix = token.token.startswith(special_symbol) == is_suffix_symbol
            if is_suffix:
                if curr_idx == len(tokens) - 1 and curr_idx - last_prefix_idx > 1:
                    spans.append((last_prefix_idx, curr_idx))
                continue
            if curr_idx - last_prefix_idx > 1:
                spans.append((last_prefix_idx, curr_idx))
            last_prefix_idx = curr_idx
        return spans


class PairAggregator(SequenceAttributionAggregator):
    """Aggregates two FeatureAttributionSequenceOutput object into a single one containing the diff.

    Args:
        attr (:class:`~inseq.data.FeatureAttributionSequenceOutput`): The starting attribution object.
        paired_attr (:class:`~inseq.data.FeatureAttributionSequenceOutput`): The attribution object with whom
            the diff is computed, representing a change from `attr_start` (e.g. minimal pair edit).
        aggregate_fn (:obj:`Callable`, optional): Function to aggregate elementwise values of the pair.
            Defaults to the difference between the two elements.
    """

    aggregator_name = "pair"
    aggregator_family = "pair"
    default_fn = lambda x, y: y - x

    @classmethod
    def pre_aggregate_hook(
        cls, attr: "FeatureAttributionSequenceOutput", paired_attr: "FeatureAttributionSequenceOutput", **kwargs
    ):
        super().pre_aggregate_hook(attr, **kwargs)
        cls.validate_pair(attr, paired_attr)

    @classmethod
    def validate_pair(cls, attr, paired_attr):
        assert len(attr.source) == len(paired_attr.source), "Source sequences must be the same length."
        assert len(attr.target) == len(paired_attr.target), "Target sequences must be the same length."
        if attr.source_attributions is not None:
            assert (
                attr.source_attributions.shape == paired_attr.source_attributions.shape
            ), "Source attributions must be the same shape."
        if attr.target_attributions is not None:
            assert (
                attr.target_attributions.shape == paired_attr.target_attributions.shape
            ), "Target attributions must be the same shape."
        if attr.step_scores is not None:
            assert paired_attr.step_scores is not None, "Paired attribution must have step scores."
            for key, value in attr.step_scores.items():
                assert key in paired_attr.step_scores, f"Step score {key} must be in paired attribution."
                assert value.shape == paired_attr.step_scores[key].shape, f"Step score {key} must be the same shape."
        if attr.sequence_scores is not None:
            assert paired_attr.sequence_scores is not None, "Paired attribution must have sequence scores."
            for key, value in attr.sequence_scores.items():
                assert key in paired_attr.sequence_scores, f"Sequence score {key} must be in paired attribution."
                assert (
                    value.shape == paired_attr.sequence_scores[key].shape
                ), f"Sequence score {key} must be the same shape."

    @staticmethod
    def aggregate_source(attr, paired_attr, **kwargs):
        return aggregate_token_pair(attr.source, paired_attr.source)

    @staticmethod
    def aggregate_target(attr, paired_attr, **kwargs):
        return aggregate_token_pair(attr.target, paired_attr.target)

    @staticmethod
    def aggregate_source_attributions(attr, paired_attr, aggregate_fn, **kwargs):
        if attr.source_attributions is None:
            return attr.source_attributions
        return aggregate_fn(attr.source_attributions, paired_attr.source_attributions)

    @staticmethod
    def aggregate_target_attributions(attr, paired_attr, aggregate_fn, **kwargs):
        if attr.target_attributions is None:
            return attr.target_attributions
        return aggregate_fn(attr.target_attributions, paired_attr.target_attributions)

    @staticmethod
    def aggregate_step_scores(attr, paired_attr, aggregate_fn, **kwargs):
        if not attr.step_scores:
            return attr.step_scores
        out_dict = {}
        for name, step_scores in attr.step_scores.items():
            agg_fn = aggregate_fn[name] if isinstance(aggregate_fn, dict) else aggregate_fn
            out_dict[name] = agg_fn(step_scores, paired_attr.step_scores[name])
        return out_dict

    @staticmethod
    def aggregate_sequence_scores(attr, paired_attr, aggregate_fn, **kwargs):
        if not attr.sequence_scores:
            return attr.sequence_scores
        out_dict = {}
        for name, sequence_scores in attr.sequence_scores.items():
            agg_fn = aggregate_fn[name] if isinstance(aggregate_fn, dict) else aggregate_fn
            out_dict[name] = agg_fn(sequence_scores, paired_attr.sequence_scores[name])
        return out_dict
