import json
import os

import torch
from pytest import fixture

import inseq
from inseq.data.aggregator import (
    AggregatorPipeline,
    ContiguousSpanAggregator,
    PairAggregator,
    SequenceAttributionAggregator,
    SubwordAggregator,
)
from inseq.models import HuggingfaceDecoderOnlyModel, HuggingfaceEncoderDecoderModel

EXAMPLES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../fixtures/aggregator.json")
EXAMPLES = json.load(open(EXAMPLES_FILE))


@fixture(scope="session")
def saliency_mt_model() -> HuggingfaceEncoderDecoderModel:
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency", device="cpu")


@fixture(scope="session")
def saliency_gpt_model() -> HuggingfaceDecoderOnlyModel:
    return inseq.load_model("gpt2", "saliency", device="cpu")


def test_sequence_attribution_aggregator(saliency_mt_model: HuggingfaceEncoderDecoderModel):
    out = saliency_mt_model.attribute(
        "This is a test.",
        step_scores=["probability"],
        attribute_target=True,
        output_step_attributions=True,
        device="cpu",
        show_progress=False,
    )
    seqattr = out.sequence_attributions[0]
    assert seqattr.source_attributions.shape == (6, 7, 512)
    assert seqattr.target_attributions.shape == (7, 7, 512)
    assert seqattr.step_scores["probability"].shape == (7,)
    for i, step in enumerate(out.step_attributions):
        assert step.source_attributions.shape == (1, 6, 512)
        assert step.target_attributions.shape == (1, i + 1, 512)
    out_agg = seqattr.aggregate()
    assert out_agg.source_attributions.shape == (6, 7)
    assert out_agg.target_attributions.shape == (7, 7)
    assert out_agg.step_scores["probability"].shape == (7,)


def test_continuous_span_aggregator(saliency_mt_model: HuggingfaceEncoderDecoderModel):
    out = saliency_mt_model.attribute(
        "This is a test.", attribute_target=True, step_scores=["probability"], device="cpu", show_progress=False
    )
    seqattr = out.sequence_attributions[0]
    out_agg = seqattr.aggregate(ContiguousSpanAggregator, source_spans=(3, 5), target_spans=[(0, 3), (4, 6)])
    assert out_agg.source_attributions.shape == (5, 4, 512)
    assert out_agg.target_attributions.shape == (4, 4, 512)
    assert out_agg.step_scores["probability"].shape == (4,)


def test_span_aggregator_with_prefix(saliency_gpt_model: HuggingfaceDecoderOnlyModel):
    out = saliency_gpt_model.attribute("Hello, world! I am,:.", "Hello, world! I am,:.!,. Last")
    aggregated = out.aggregate("subwords", special_symbol=("Ġ", "Ċ")).aggregate()
    assert aggregated[0].target_attributions.shape == (5, 2)
    assert aggregated[0].attr_pos_start == 3
    assert aggregated[0].attr_pos_end == 5


def test_aggregator_pipeline(saliency_mt_model: HuggingfaceEncoderDecoderModel):
    out = saliency_mt_model.attribute(
        "This is a test.", attribute_target=True, step_scores=["probability"], device="cpu", show_progress=False
    )
    seqattr = out.sequence_attributions[0]
    squeezesum = AggregatorPipeline([ContiguousSpanAggregator, SequenceAttributionAggregator])
    out_agg_squeezesum = seqattr.aggregate(squeezesum, source_spans=(3, 5), target_spans=[(0, 3), (4, 6)])
    assert out_agg_squeezesum.source_attributions.shape == (5, 4)
    assert out_agg_squeezesum.target_attributions.shape == (4, 4)
    assert out_agg_squeezesum.step_scores["probability"].shape == (4,)
    sumsqueeze = AggregatorPipeline([SequenceAttributionAggregator, ContiguousSpanAggregator])
    out_agg_sumsqueeze = seqattr.aggregate(sumsqueeze, source_spans=(3, 5), target_spans=[(0, 3), (4, 6)])
    assert out_agg_sumsqueeze.source_attributions.shape == (5, 4)
    assert out_agg_sumsqueeze.target_attributions.shape == (4, 4)
    assert out_agg_sumsqueeze.step_scores["probability"].shape == (4,)
    assert not torch.allclose(out_agg_squeezesum.source_attributions, out_agg_sumsqueeze.source_attributions)
    assert not torch.allclose(out_agg_squeezesum.target_attributions, out_agg_sumsqueeze.target_attributions)
    # Named indexing version
    named_squeezesum = ["spans", "scores"]
    named_sumsqueeze = ["scores", "spans"]
    out_agg_squeezesum_named = seqattr.aggregate(named_squeezesum, source_spans=(3, 5), target_spans=[(0, 3), (4, 6)])
    out_agg_sumsqueeze_named = seqattr.aggregate(named_sumsqueeze, source_spans=(3, 5), target_spans=[(0, 3), (4, 6)])
    assert out_agg_squeezesum_named.source_attributions.shape == (5, 4)
    assert out_agg_squeezesum_named.target_attributions.shape == (4, 4)
    assert out_agg_squeezesum_named.step_scores["probability"].shape == (4,)
    assert out_agg_sumsqueeze_named.source_attributions.shape == (5, 4)
    assert out_agg_sumsqueeze_named.target_attributions.shape == (4, 4)
    assert out_agg_sumsqueeze_named.step_scores["probability"].shape == (4,)
    assert not torch.allclose(
        out_agg_squeezesum_named.source_attributions, out_agg_sumsqueeze_named.source_attributions
    )
    assert not torch.allclose(
        out_agg_squeezesum_named.target_attributions, out_agg_sumsqueeze_named.target_attributions
    )


def test_subword_aggregator(saliency_mt_model: HuggingfaceEncoderDecoderModel):
    out = saliency_mt_model.attribute(EXAMPLES["source"], show_progress=False)
    seqattr = out.sequence_attributions[0]
    for idx, token in enumerate(seqattr.source):
        assert token.token == EXAMPLES["source_subwords"][idx]
    for idx, token in enumerate(seqattr.target):
        assert token.token == EXAMPLES["target_subwords"][idx]
    # Full aggregation
    out_agg = seqattr.aggregate(SubwordAggregator)
    for idx, token in enumerate(out_agg.source):
        assert token.token == EXAMPLES["source_merged"][idx]
    for idx, token in enumerate(out_agg.target):
        assert token.token == EXAMPLES["target_merged"][idx]
    # Source-only aggregation
    out_agg = seqattr.aggregate(SubwordAggregator, aggregate_target=False)
    for idx, token in enumerate(out_agg.source):
        assert token.token == EXAMPLES["source_merged"][idx]
    for idx, token in enumerate(out_agg.target):
        assert token.token == EXAMPLES["target_subwords"][idx]
    # Target-only aggregation
    out_agg = seqattr.aggregate(SubwordAggregator, aggregate_source=False)
    for idx, token in enumerate(out_agg.source):
        assert token.token == EXAMPLES["source_subwords"][idx]
    for idx, token in enumerate(out_agg.target):
        assert token.token == EXAMPLES["target_merged"][idx]


def test_pair_aggregator(saliency_mt_model: HuggingfaceEncoderDecoderModel):
    out = saliency_mt_model.attribute([EXAMPLES["source"], EXAMPLES["alternative_source"]], show_progress=False)
    orig_seqattr = out.sequence_attributions[0].aggregate(["vnorm"])
    alt_seqattr = out.sequence_attributions[1].aggregate(["vnorm"])
    diff_seqattr = orig_seqattr.aggregate(PairAggregator, paired_attr=alt_seqattr)
    for idx, token in enumerate(diff_seqattr.source):
        assert token.token == EXAMPLES["diff_subwords"][idx]
    assert torch.allclose(
        alt_seqattr.source_attributions - orig_seqattr.source_attributions, diff_seqattr.source_attributions
    )
    # Default aggregation with SequenceAttributionAggregator
    orig_seqattr_other = out.sequence_attributions[0].aggregate()
    alt_seqattr_other = out.sequence_attributions[1].aggregate()
    # Aggregate with aggregator name
    diff_seqattr_other = orig_seqattr_other.aggregate("pair", paired_attr=alt_seqattr_other)
    assert torch.allclose(diff_seqattr_other.source_attributions, diff_seqattr.source_attributions)


def test_named_aggregate_fn_aggregation(saliency_mt_model: HuggingfaceEncoderDecoderModel):
    out = saliency_mt_model.attribute(
        [EXAMPLES["source"], EXAMPLES["alternative_source"]],
        show_progress=False,
        attribute_target=True,
        method="attention",
    )
    out_headmean = out.aggregate(aggregator=["mean", "mean"])
    assert out_headmean.sequence_attributions[0].source_attributions.ndim == 2
    assert out_headmean.sequence_attributions[0].target_attributions.ndim == 2
    assert out_headmean.sequence_attributions[1].source_attributions.ndim == 2
    assert out_headmean.sequence_attributions[1].target_attributions.ndim == 2
    out_allmean_subwords = out.aggregate(aggregator=["mean", "mean", "subwords"])

    # Check whether scores aggregation worked correctly
    assert out_allmean_subwords.sequence_attributions[0].source_attributions.ndim == 2
    assert out_allmean_subwords.sequence_attributions[0].target_attributions.ndim == 2
    assert out_allmean_subwords.sequence_attributions[1].source_attributions.ndim == 2
    assert out_allmean_subwords.sequence_attributions[1].target_attributions.ndim == 2

    # Check whether subword aggregation worked correctly
    assert (
        out_allmean_subwords.sequence_attributions[0].source_attributions.shape[0]
        < out.sequence_attributions[0].source_attributions.shape[0]
    )
    assert (
        out_allmean_subwords.sequence_attributions[0].target_attributions.shape[0]
        < out.sequence_attributions[0].target_attributions.shape[0]
    )
    assert (
        out_allmean_subwords.sequence_attributions[1].source_attributions.shape[0]
        < out.sequence_attributions[1].source_attributions.shape[0]
    )
    assert (
        out_allmean_subwords.sequence_attributions[1].target_attributions.shape[0]
        < out.sequence_attributions[1].target_attributions.shape[0]
    )

    out_allmean_subwords_expanded = out.aggregate(
        aggregator=["scores", "scores", "subwords"], aggregate_fn=["mean", "mean", None]
    )
    assert out_allmean_subwords == out_allmean_subwords_expanded
