from typing import Any, Callable, Dict, List, Optional, Type, Union

from abc import ABC

from ..utils import abs_max, aggregate_contiguous, aggregate_token_sequence, identity_fn
from ..utils.typing import IndexSpan
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
    def pre_aggregate_hook(cls, tensors: TensorWrapper, **kwargs):
        pass

    @classmethod
    def aggregate(cls, tensors: TensorWrapper, **kwargs):
        aggregated_sequence_attribution_fields = {}
        for field in tensors.to_dict().keys():
            field_func = getattr(cls, f"aggregate_{field}")
            dispatched_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, DispatchableDict):
                    dispatched_kwargs[k] = v.get(field, v.default)
                    # If the subclass is a dict, then we assume its fields represent
                    # variants depending on the aggregator that is being used.
                    if isinstance(dispatched_kwargs[k], dict) and cls.name in dispatched_kwargs[k]:
                        dispatched_kwargs[k] = dispatched_kwargs[k].get(cls.name, v.default)
                else:
                    dispatched_kwargs[k] = v
            aggregated_sequence_attribution_fields[field] = field_func(tensors, **dispatched_kwargs)
        return tensors.__class__(**aggregated_sequence_attribution_fields)

    @classmethod
    def post_aggregate_hook(cls, tensors: TensorWrapper, **kwargs):
        pass


class AggregatorPipeline:
    def __init__(self, aggregators: List[Type[Aggregator]]):
        self.aggregators = aggregators

    def aggregate(self, tensors: TensorWrapper, **kwargs):
        for aggregator in self.aggregators:
            aggregator.pre_aggregate_hook(tensors, **kwargs)
        for aggregator in self.aggregators:
            tensors = aggregator.aggregate(tensors, **kwargs)
        for aggregator in self.aggregators:
            aggregator.post_aggregate_hook(tensors, **kwargs)
        return tensors


class AggregableMixin(ABC):
    _aggregator: Union[AggregatorPipeline, Type[Aggregator]]
    _dict_aggregate_fn: DispatchableDict

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

    def __post_init__(self):
        pass


class FeatureAttributionSequenceOutputAggregator(Aggregator):
    """Identity aggregator for FeatureAttributionSequenceOutput objects"""

    @classmethod
    def aggregate(cls, attr, **kwargs):
        return super().aggregate(attr, **kwargs)

    @classmethod
    def post_aggregate_hook(cls, attr: TensorWrapper, **kwargs):
        super().post_aggregate_hook(attr, **kwargs)
        cls.is_compatible(attr)

    @staticmethod
    def aggregate_source(attr, **kwargs):
        return attr.source

    @staticmethod
    def aggregate_target(attr, **kwargs):
        return attr.target

    @staticmethod
    def aggregate_source_attributions(attr, **kwargs):
        return attr.source_attributions

    @staticmethod
    def aggregate_target_attributions(attr, **kwargs):
        return attr.target_attributions

    @staticmethod
    def aggregate_step_scores(attr, **kwargs):
        return attr.step_scores

    @staticmethod
    def aggregate_sequence_scores(attr, **kwargs):
        return attr.sequence_scores

    @staticmethod
    def is_compatible(attr):
        from .attribution import FeatureAttributionSequenceOutput

        assert isinstance(attr, FeatureAttributionSequenceOutput)
        assert len(attr.source_attributions.shape) == 2
        assert attr.source_attributions.shape[0] == len(attr.source)
        assert attr.source_attributions.shape[1] == len(attr.target)
        if attr.target_attributions is not None:
            assert len(attr.target_attributions.shape) == 2
            assert attr.target_attributions.shape[0] == len(attr.target)
            assert attr.target_attributions.shape[1] == len(attr.target)
        if attr.step_scores is not None:
            for step_score in attr.step_scores.values():
                assert len(step_score) == len(attr.target)
        if attr.sequence_scores is not None:
            for sequence_score in attr.sequence_scores.values():
                assert sequence_score.shape == attr.source_attributions.shape


class SequenceAttributionAggregator(FeatureAttributionSequenceOutputAggregator):
    """Aggregates sequence attributions using a custom function. By default, sum over the last dimension and
    dividing by the norm of the sequence attributions.

    Args:
        aggregate_fn (callable, optional): Function used to aggregate sequence attributions.
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

    @staticmethod
    def aggregate_source_attributions(attr, aggregate_fn: Union[Dict[str, Callable], Callable], **kwargs):
        if attr.target_attributions is None:
            return aggregate_fn(attr.source_attributions)
        else:
            return aggregate_fn((attr.source_attributions, attr.target_attributions))[0]

    @staticmethod
    def aggregate_target_attributions(attr, aggregate_fn: Callable, **kwargs):
        if attr.target_attributions is None:
            return attr.target_attributions
        else:
            return aggregate_fn((attr.source_attributions, attr.target_attributions))[1]


class ContiguousSpanAggregator(FeatureAttributionSequenceOutputAggregator):
    """Reduces sequence attributions across one or more contiguous spans.

    Args:
        aggregate_fn (callable, optional): Function used to aggregate sequence attributions.
            Defaults to summing over the last dimension (for score-level attributions) and renormalizing
            by the norm of the source(+target) attributions.
        source_spans (tuple of [int, int] or sequence of tuples of [int, int], optional): Spans to aggregate
            over for the source sequence. Defaults to no aggregation performed.
        target_spans (tuple of [int, int] or sequence of tuples of [int, int], optional): Spans to aggregate
            over for the target sequence. Defaults to no aggregation performed.

    """

    name = "span_aggregate"
    default_fn = abs_max

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
        if aggregate_fn is None:
            aggregate_fn = attr._dict_aggregate_fn if isinstance(attr._dict_aggregate_fn, dict) else cls.default_fn
        if isinstance(aggregate_fn, dict):
            aggregate_fn = DispatchableDict(default=cls.default_fn, **aggregate_fn)
        source_spans = cls.validate_spans(attr.source, source_spans)
        target_spans = cls.validate_spans(attr.target, target_spans)
        return super().aggregate(attr, aggregate_fn=aggregate_fn, source_spans=source_spans, target_spans=target_spans)

    @staticmethod
    def validate_spans(span_sequence, spans):
        if not spans:
            return spans
        allmatch = lambda l, type: all(isinstance(x, type) for x in l)
        assert allmatch(spans, int) or allmatch(
            spans, tuple
        ), "All items must be either indices (int) or spans (tuple)"
        spans = [spans] if isinstance(spans[0], int) else spans
        prev_span_max = -1
        for span in spans:
            assert len(span) == 2, "Spans must contain at least two indexes"
            assert span[1] > span[0] + 1, "Spans must be non-empty"
            assert span[0] > prev_span_max, "Spans must be postive-valued, non-overlapping and in ascending order"
            assert span[1] < len(span_sequence), "Span values must be indexes of the original span"
            prev_span_max = span[1]
        return spans

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
        # First aggregate along generated target sequence, then along attributed source
        return ContiguousSpanAggregator._aggregate_sequential_scores(
            attr.source_attributions, source_spans, target_spans, aggregate_fn
        )

    @staticmethod
    def aggregate_target_attributions(attr, target_spans, aggregate_fn, **kwargs):
        if not attr.target_attributions:
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
