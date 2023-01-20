from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from ..utils import abs_max, aggregate_contiguous, aggregate_token_pair, aggregate_token_sequence, identity_fn
from ..utils.typing import IndexSpan, TokenWithId
from .data_utils import TensorWrapper


class DispatchableDict(dict):
    """Used to pass specific values to field-specific calls of the aggregate function in Aggregator.

    DispatchableDict dictionary objects won't be passed as a whole to all field-specific functions called by
    Aggregator.aggregate, and instead only the values with the name of the corresponding field will be used.
    When these are missing, the default field of DispatchableDict will be used as fallback.
    """

    def __init__(self, default: Optional[Any] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default = default


class Aggregator(ABC):
    @classmethod
    @abstractmethod
    def start_aggregation_hook(cls, tensors: TensorWrapper, **kwargs):
        """Hook called at the start of the aggregation process.

        Use to ensure a prerequisite that is independent of previous
        aggregation steps and fundamental to the aggregation process
        (e.g. parameters are of the correct type). Will avoid performing aggregation steps before
        returning an error.
        """
        pass

    @classmethod
    @abstractmethod
    def pre_aggregate_hook(cls, tensors: TensorWrapper, **kwargs):
        """Hook called right before the aggregation function is called.

        Use to ensure a prerequisite that is functional of previous
        aggregation steps and fundamental to the aggregation process
        (e.g. the aggregatable object produced by the previous step has correct shapes).
        """
        pass

    @classmethod
    def _aggregate(cls, tensors: TensorWrapper, **kwargs):
        aggregated_sequence_attribution_fields = {}
        for field in tensors.to_dict().keys():
            field_func = getattr(cls, f"aggregate_{field}")
            dispatched_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, DispatchableDict):
                    dispatched_kwargs[k] = v.get(field, v.default)
                    # If the subclass is a dict, then we assume its fields represent
                    # variants depending on the aggregator that is being used.
                    if isinstance(dispatched_kwargs[k], dict):
                        dispatched_kwargs[k] = dispatched_kwargs[k].get(cls.name, v.default)
                else:
                    dispatched_kwargs[k] = v
            aggregated_sequence_attribution_fields[field] = field_func(tensors, **dispatched_kwargs)
        return tensors.__class__(**aggregated_sequence_attribution_fields)

    @classmethod
    def aggregate(
        cls, tensors: TensorWrapper, do_start_aggregation: bool = True, do_end_aggregation: bool = True, **kwargs
    ):
        if do_start_aggregation:
            cls.start_aggregation_hook(tensors, **kwargs)
        cls.pre_aggregate_hook(tensors, **kwargs)
        aggregated = cls._aggregate(tensors, **kwargs)
        cls.post_aggregate_hook(aggregated, **kwargs)
        if do_end_aggregation:
            cls.end_aggregation_hook(aggregated, **kwargs)
        return aggregated

    @classmethod
    @abstractmethod
    def post_aggregate_hook(cls, tensors: TensorWrapper, **kwargs):
        """Hook called right after the aggregation function is called.

        Verifies that the aggregated object has the correct properties.
        """
        pass

    @classmethod
    @abstractmethod
    def end_aggregation_hook(cls, tensors: TensorWrapper, **kwargs):
        """Hook called at the end of the aggregation process.

        Use to ensure that the final product of aggregation is compliant with
        the requirements of individual aggregators.
        """
        pass


class AggregatorPipeline:
    def __init__(self, aggregators: List[Type[Aggregator]]):
        self.aggregators = aggregators

    def aggregate(self, tensors: TensorWrapper, **kwargs):
        for aggregator in self.aggregators:
            aggregator.start_aggregation_hook(tensors, **kwargs)
        for aggregator in self.aggregators:
            tensors = aggregator.aggregate(tensors, do_start_aggregation=False, do_end_aggregation=False, **kwargs)
        for aggregator in self.aggregators:
            aggregator.end_aggregation_hook(tensors, **kwargs)
        return tensors


class AggregableMixin(ABC):
    _aggregator: Union[AggregatorPipeline, Type[Aggregator]]

    def aggregate(
        self,
        aggregator: Union[AggregatorPipeline, Type[Aggregator]] = None,
        **kwargs,
    ) -> "AggregableMixin":
        """Aggregate attributions using the default or provided aggregator.

        Args:
            aggregator (:obj:`AggregatorPipeline` or :obj:`Type[Aggregator]`, optional): Aggregator
                pipeline to use. If not provided, the default aggregator pipeline is used.

        Returns:
            :obj:`AggregableMixin`: The aggregated output class.
        """
        if aggregator is None:
            aggregator = self._aggregator
        return aggregator.aggregate(self, **kwargs)

    @abstractmethod
    def __post_init__(self):
        pass


class SequenceAttributionAggregator(Aggregator):
    """Aggregates sequence attributions using a custom function. By default, the identity function is used.

    Represent the identity aggregator for the FeatureAttributionSequenceOutput class.

    Args:
        attr (:class:`~inseq.data.FeatureAttributionSequenceOutput`): The attribution object to aggregate.
        aggregate_fn (:obj:`Callable`, optional): Function used to aggregate sequence attributions.
            Defaults to summing over the last dimension and renormalizing by the norm of the
            source(+target) attributions for granular attributions, no aggregation for token-level
            attributions.
    """

    name = "sequence_aggregate"
    default_fn = identity_fn

    @classmethod
    def aggregate(cls, attr, aggregate_fn: Union[Callable, Dict[str, Any], None] = None, **kwargs):
        if aggregate_fn is None:
            aggregate_fn = attr._dict_aggregate_fn if isinstance(attr._dict_aggregate_fn, dict) else cls.default_fn
        # By default we treat dicts as key-value maps for aggregation functions that
        # will be applied to specific fields
        if isinstance(aggregate_fn, dict):
            aggregate_fn = DispatchableDict(default=cls.default_fn, **aggregate_fn)
        aggregated = super().aggregate(attr, aggregate_fn=aggregate_fn, **kwargs)
        return aggregated

    @classmethod
    def post_aggregate_hook(cls, attr, **kwargs):
        super().post_aggregate_hook(attr, **kwargs)
        cls.is_compatible(attr)

    @classmethod
    def end_aggregation_hook(cls, attr: TensorWrapper, **kwargs):
        super().end_aggregation_hook(attr, **kwargs)
        # Needed to ensure the attribution can be visualized
        if attr.source_attributions is not None:
            assert len(attr.source_attributions.shape) == 2
        if attr.target_attributions is not None:
            assert len(attr.target_attributions.shape) == 2

    @staticmethod
    def aggregate_source(attr, **kwargs):
        return attr.source

    @staticmethod
    def aggregate_target(attr, **kwargs):
        return attr.target

    @staticmethod
    def aggregate_source_attributions(attr, aggregate_fn: Union[Dict[str, Callable], Callable], **kwargs):
        if attr.source_attributions is None:
            return attr.source_attributions
        if attr.target_attributions is None:
            return aggregate_fn(attr.source_attributions)
        else:
            return aggregate_fn((attr.source_attributions, attr.target_attributions))[0]

    @staticmethod
    def aggregate_target_attributions(attr, aggregate_fn: Callable, **kwargs):
        if attr.target_attributions is None:
            return attr.target_attributions
        if attr.source_attributions is None:
            return aggregate_fn(attr.target_attributions)
        else:
            return aggregate_fn((attr.source_attributions, attr.target_attributions))[1]

    @staticmethod
    def aggregate_step_scores(attr, **kwargs):
        return attr.step_scores

    @staticmethod
    def aggregate_sequence_scores(attr, **kwargs):
        return attr.sequence_scores

    @staticmethod
    def aggregate_attr_pos_start(attr, **kwargs):
        return attr.attr_pos_start

    @staticmethod
    def aggregate_attr_pos_end(attr, **kwargs):
        return attr.attr_pos_end

    @staticmethod
    def is_compatible(attr):
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
        if attr.sequence_scores is not None:
            for sequence_score in attr.sequence_scores.values():
                assert sequence_score.shape == attr.source_attributions.shape


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

    name = "span_aggregate"
    default_fn = abs_max

    @classmethod
    def start_aggregation_hook(cls, attr, source_spans=None, target_spans=None, **kwargs):
        super().start_aggregation_hook(attr, **kwargs)
        cls.validate_spans(attr.source, source_spans)
        cls.validate_spans(attr.target, target_spans)

    @classmethod
    def end_aggregation_hook(cls, attr: TensorWrapper, **kwargs):
        pass

    @classmethod
    def aggregate(
        cls,
        attr,
        aggregate_fn: Union[Callable, Dict[str, Any], None] = None,
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
        return super().aggregate(
            attr, aggregate_fn=aggregate_fn, source_spans=source_spans, target_spans=target_spans, **kwargs
        )

    @staticmethod
    def format_spans(spans) -> List[Tuple[int, int]]:
        if not spans:
            return spans
        return [spans] if isinstance(spans[0], int) else spans

    @classmethod
    def validate_spans(cls, span_sequence, spans):
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

    @classmethod
    def aggregate(
        cls,
        attr,
        aggregate_fn: Union[Callable, Dict[str, Any], None] = None,
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
        return super().aggregate(
            attr, aggregate_fn=aggregate_fn, source_spans=source_spans, target_spans=target_spans, **kwargs
        )

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

    name = "pair_aggregate"
    default_fn = lambda x, y: y - x

    @classmethod
    def pre_aggregate_hook(cls, attr, paired_attr, **kwargs):
        super().pre_aggregate_hook(attr, **kwargs)
        cls.validate_pair(attr, paired_attr)

    @classmethod
    def aggregate(
        cls,
        attr,
        paired_attr,
        aggregate_fn: Union[Callable, Dict[str, Any], None] = None,
        **kwargs,
    ):
        return super().aggregate(attr, aggregate_fn=aggregate_fn, paired_attr=paired_attr, **kwargs)

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
