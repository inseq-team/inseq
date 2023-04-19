import random

import torch
from pytest import mark, skip

from inseq.attr.feat.ops import AggregableMixin

AGGREGATE_FN_OPTIONS = list(AggregableMixin.AGGREGATE_FN_OPTIONS.keys()) + [None]

AGGREGATE_OPTIONS = ["int", "range", "list", "none"]


@mark.parametrize("aggr_method", AGGREGATE_FN_OPTIONS)
@mark.parametrize("aggr_layers", AGGREGATE_OPTIONS)
def test_layer_aggregation(aggr_method: str, aggr_layers: str) -> None:
    layerAttention = ()

    shape = (5, 8, 7, 7)

    layers = 0

    max_layer = random.randint(4, 10)
    for _ in range(max_layer):
        attention = torch.rand(size=shape, dtype=torch.float)
        layerAttention = layerAttention + (attention,)
    layerAttention = torch.stack(layerAttention, dim=1)

    if aggr_method == "single":
        if aggr_layers != "int" and aggr_layers != "none":
            skip("only a single layer can be passed if single-layer aggregation is specified")

    if aggr_layers == "int":
        if aggr_method != "single" and aggr_method is not None:
            skip("only single-layer aggregation is possible if a single layer is passed")
        layers = random.randrange(max_layer)
    elif aggr_layers == "range":
        layers = (1, max_layer)
    elif aggr_layers == "list":
        layers = [0, 1, max_layer - 1]
    elif aggr_layers == "none":
        layers = None

    layer_aggr_attention = AggregableMixin._aggregate_layers(layerAttention, aggr_method, layers)

    assert layer_aggr_attention.shape == shape


@mark.parametrize("aggr_method", AGGREGATE_FN_OPTIONS)
@mark.parametrize("aggr_heads", AGGREGATE_OPTIONS)
def test_head_aggregation(aggr_method: str, aggr_heads: str) -> None:
    num_heads = random.randint(4, 12)

    in_shape = (5, num_heads, 7, 7)
    out_shape = (5, 7, 7)

    heads = 0

    attention = torch.rand(size=in_shape, dtype=torch.float)

    if aggr_method == "single":
        if aggr_heads != "int":
            skip("A single head has to be passed if single-head aggregation is specified")

    if aggr_heads == "int":
        if aggr_method != "single" and aggr_method is not None:
            skip("only single-head aggregation is possible if a single head is passed")
        heads = random.randrange(num_heads)
    elif aggr_heads == "range":
        heads = (1, num_heads)
    elif aggr_heads == "list":
        heads = [0, 1, num_heads - 1]
    elif aggr_heads == "none":
        heads = None

    head_aggr_attention = AggregableMixin._aggregate_units(attention, aggr_method, heads)

    assert head_aggr_attention.shape == out_shape
