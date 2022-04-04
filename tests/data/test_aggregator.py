import torch
from pytest import fixture

import inseq
from inseq.data.aggregator import AggregatorPipeline, ContiguousSpanAggregator, SequenceAttributionAggregator


@fixture(scope="session")
def saliency_mt_model():
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency", device="cpu")


def test_sequence_attribution_aggregator(saliency_mt_model):
    out = saliency_mt_model.attribute(
        "This is a test.",
        output_step_probabilities=True,
        attribute_target=True,
        output_step_attributions=True,
        device="cpu",
    )
    seqattr = out.sequence_attributions[0]
    assert seqattr.source_attributions.shape == (6, 7, 512)
    assert seqattr.target_attributions.shape == (7, 7, 512)
    assert seqattr.step_scores["probabilities"].shape == (7,)
    for i, step in enumerate(out.step_attributions):
        assert step.source_attributions.shape == (1, 6, 512)
        assert step.target_attributions.shape == (1, i + 1, 512)
    out_agg = seqattr.aggregate()
    assert out_agg.source_attributions.shape == (6, 7)
    assert out_agg.target_attributions.shape == (7, 7)
    assert out_agg.step_scores["probabilities"].shape == (7,)


def test_continuous_span_aggregator(saliency_mt_model):
    out = saliency_mt_model.attribute(
        "This is a test.", attribute_target=True, output_step_probabilities=True, device="cpu"
    )
    seqattr = out.sequence_attributions[0]
    out_agg = seqattr.aggregate(ContiguousSpanAggregator, source_spans=(3, 5), target_spans=[(0, 3), (4, 6)])
    assert out_agg.source_attributions.shape == (5, 4, 512)
    assert out_agg.target_attributions.shape == (4, 4, 512)
    assert out_agg.step_scores["probabilities"].shape == (4,)


def test_aggregator_pipeline(saliency_mt_model):
    out = saliency_mt_model.attribute(
        "This is a test.", attribute_target=True, output_step_probabilities=True, device="cpu"
    )
    seqattr = out.sequence_attributions[0]
    squeezesum = AggregatorPipeline([ContiguousSpanAggregator, SequenceAttributionAggregator])
    out_agg_squeezesum = seqattr.aggregate(squeezesum, source_spans=(3, 5), target_spans=[(0, 3), (4, 6)])
    assert out_agg_squeezesum.source_attributions.shape == (5, 4)
    assert out_agg_squeezesum.target_attributions.shape == (4, 4)
    assert out_agg_squeezesum.step_scores["probabilities"].shape == (4,)
    sumsqueeze = AggregatorPipeline([SequenceAttributionAggregator, ContiguousSpanAggregator])
    out_agg_sumsqueeze = seqattr.aggregate(sumsqueeze, source_spans=(3, 5), target_spans=[(0, 3), (4, 6)])
    assert out_agg_sumsqueeze.source_attributions.shape == (5, 4)
    assert out_agg_sumsqueeze.target_attributions.shape == (4, 4)
    assert out_agg_sumsqueeze.step_scores["probabilities"].shape == (4,)
    assert not torch.allclose(out_agg_squeezesum.source_attributions, out_agg_sumsqueeze.source_attributions)
    assert not torch.allclose(out_agg_squeezesum.target_attributions, out_agg_sumsqueeze.target_attributions)
