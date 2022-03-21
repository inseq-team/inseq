from typing import Any, Callable, List, Optional, Type, Union

from abc import ABC

from ..utils import abs_max, aggregate_contiguous, sum_normalize_attributions
from ..utils.typing import IndexSpan, TokenWithId
from .batch import TensorWrapper


IDENTITY = lambda x: x


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
    def aggregate(cls, tensors: TensorWrapper, **kwargs):
        aggregated_sequence_attribution_fields = {}
        for field in tensors.to_dict().keys():
            field_func = getattr(cls, f"aggregate_{field}")
            dispatched_kwargs = {
                k: v if not isinstance(v, DispatchableDict) else v.get(k, v.default) for k, v in kwargs.items()
            }
            aggregated_sequence_attribution_fields[field] = field_func(tensors, **dispatched_kwargs)
        return tensors.__class__(**aggregated_sequence_attribution_fields)


class AggregatorPipeline:
    def __init__(self, aggregators: List[Type[Aggregator]]):
        self.aggregators = aggregators

    def aggregate(self, tensors: TensorWrapper, **kwargs):
        for aggregator in self.aggregators:
            tensors = aggregator.aggregate(tensors, **kwargs)
        return tensors


class FeatureAttributionSequenceOutputAggregator(Aggregator):
    """Identity aggregator for FeatureAttributionSequenceOutput objects"""

    @classmethod
    def aggregate(cls, attr, **kwargs):
        return super().aggregate(attr, **kwargs)

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


class CustomAttributionAggregator(FeatureAttributionSequenceOutputAggregator):
    """Aggregates sequence attributions by summing over the last dimension and
    dividing by the norm of the sequence attributions.

    Args:
        aggregate_fn (callable, optional): Function used to aggregate sequence attributions.
            Defaults to summing over the last dimension and renormalizing by the norm of the
            source(+target) attributions for granular attributions, no aggregation for token-level
            attributions.
    """

    @classmethod
    def aggregate(cls, attr, aggregate_fn: Union[Callable, DispatchableDict, None] = None, **kwargs):
        if aggregate_fn is None:
            if len(attr.source_attributions.shape) == 2:
                aggregate_fn = IDENTITY
            else:
                aggregate_fn = sum_normalize_attributions
        aggregated = super().aggregate(attr, aggregate_fn=aggregate_fn, **kwargs)
        return aggregated

    @staticmethod
    def aggregate_source_attributions(attr, aggregate_fn: Callable, **kwargs):
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

    @classmethod
    def aggregate(
        cls,
        attr,
        aggregate_fn: Union[Callable, DispatchableDict, None] = None,
        source_spans: Optional[IndexSpan] = None,
        target_spans: Optional[IndexSpan] = None,
    ):
        """Spans can be:

        1. A list of the form [pos_start, pos_end] including the contiguous positions of tokens that
            are to be aggregated, if all values are integers and len(span) < len(original_seq)
        2. A list of the form [(pos_start_0, pos_end_0), (pos_start_1, pos_end_1)], same as above but
            for multiple contiguous spans.
        """
        if aggregate_fn is None:
            aggregate_fn = DispatchableDict(
                default=abs_max,
                step_scores={"deltas": abs_max, "probabilities": lambda t: t.prod(dim=0, keepdim=True)},
            )
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
    def _aggregate_token_sequence(token_sequence, spans):
        if not spans:
            return token_sequence
        out_sequence = []
        span_start_idxs = [span[0] for span in spans]
        curr_idx = 0
        for tok_idx, token in enumerate(token_sequence):
            if tok_idx < curr_idx:
                continue
            if curr_idx in span_start_idxs:
                end_idx = spans[span_start_idxs.index(curr_idx)][1]
                # We use -1 as token index to indicate the token is product of an aggregation
                # (i.e. not contained in the original vocabulary)
                out_sequence.append(TokenWithId("".join([t.token for t in token_sequence[curr_idx:end_idx]]), -1))
                curr_idx = end_idx
            else:
                out_sequence.append(token)
                curr_idx += 1
        return out_sequence

    @staticmethod
    def _aggregate_sequential_scores(scores, x_spans, y_spans, aggregate_fn):
        # First aggregate alongside the y-axis
        scores_aggregated_y = aggregate_contiguous(scores, y_spans, aggregate_fn, aggregate_dim=1)
        # Then aggregate alonside the x-axis
        scores_aggregated_x = aggregate_contiguous(scores_aggregated_y, x_spans, aggregate_fn, aggregate_dim=0)
        return scores_aggregated_x

    @staticmethod
    def aggregate_source(attr, source_spans, **kwargs):
        return ContiguousSpanAggregator._aggregate_token_sequence(attr.source, source_spans)

    @staticmethod
    def aggregate_target(attr, target_spans, **kwargs):
        return ContiguousSpanAggregator._aggregate_token_sequence(attr.target, target_spans)

    @staticmethod
    def aggregate_source_attributions(attr, source_spans, target_spans, aggregate_fn, **kwargs):
        # First aggregate along generated target sequence, then along attributed source
        return ContiguousSpanAggregator._aggregate_sequential_scores(
            attr.source_attributions, source_spans, target_spans, aggregate_fn
        )

    @staticmethod
    def aggregate_target_attributions(attr, target_spans, aggregate_fn, **kwargs):
        # First aggregate along generated target sequence, then along attributed prefix
        return ContiguousSpanAggregator._aggregate_sequential_scores(
            attr.target_attributions, target_spans, target_spans, aggregate_fn
        )

    @staticmethod
    def aggregate_step_scores(attr, target_spans, aggregate_fn, **kwargs):
        if not attr.step_scores:
            return attr.step_scores
        return {
            k: aggregate_contiguous(v, target_spans, aggregate_fn, aggregate_dim=0)
            for k, v in attr.step_scores.items()
        }

    @staticmethod
    def aggregate_sequence_scores(attr, source_spans, target_spans, aggregate_fn, **kwargs):
        # Assume sequence scores are shaped like source attributions
        if not attr.sequence_scores:
            return attr.sequence_scores
        return {
            k: ContiguousSpanAggregator._aggregate_sequential_scores(v, source_spans, target_spans, aggregate_fn)
            for k, v in attr.sequence_scores.items()
        }
