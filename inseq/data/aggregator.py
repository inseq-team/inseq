from abc import ABC

from ..utils import sum_normalize_attributions
from .batch import TensorWrapper


class Aggregator(ABC):
    @classmethod
    def aggregate(cls, tensors: TensorWrapper):
        aggregated_sequence_attribution_fields = {}
        for field in tensors.to_dict().keys():
            field_func = getattr(cls, f"aggregate_{field}")
            aggregated_sequence_attribution_fields[field] = field_func(tensors)
        return tensors.__class__(**aggregated_sequence_attribution_fields)


class FeatureAttributionSequenceOutputAggregator(Aggregator):
    """Identity aggregator for FeatureAttributionSequenceOutput objects"""

    @classmethod
    def aggregate(cls, attr):
        aggregated = super().aggregate(attr)
        cls.is_compatible(aggregated)
        return aggregated

    @staticmethod
    def aggregate_source(attr):
        return attr.source

    @staticmethod
    def aggregate_target(attr):
        return attr.target

    @staticmethod
    def aggregate_source_attributions(attr):
        return attr.source_attributions

    @staticmethod
    def aggregate_target_attributions(attr):
        return attr.target_attributions

    @staticmethod
    def aggregate_step_scores(attr):
        return attr.step_scores

    @staticmethod
    def aggregate_sequence_scores(attr):
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


class SumNormAggregator(FeatureAttributionSequenceOutputAggregator):
    """Aggregates sequence attributions by summing over the last dimension and
    dividing by the norm of the sequence attributions."""

    @staticmethod
    def aggregate_source_attributions(attr):
        if attr.target_attributions is None:
            return sum_normalize_attributions(attr.source_attributions)
        else:
            return sum_normalize_attributions((attr.source_attributions, attr.target_attributions))[0]

    @staticmethod
    def aggregate_target_attributions(attr):
        if attr.target_attributions is None:
            return attr.target_attributions
        else:
            return sum_normalize_attributions((attr.source_attributions, attr.target_attributions))[1]
