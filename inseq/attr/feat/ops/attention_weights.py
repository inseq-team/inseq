# Copyright 2023 The Inseq Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import List, Optional, Tuple, Union

from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import Attribution
from captum.log import log_usage

from ....utils.typing import (
    MultiLayerMultiUnitScoreTensor,
)
from .aggregable_mixin import AggregableMixin, AggregationFunction

logger = logging.getLogger(__name__)


class AttentionWeights(Attribution, AggregableMixin):
    """
    A basic attention attribution approach.
    It will return the attention values for the specified values or aggregated across the specified ranges of heads
    and layers, given the specified aggregation functions.

    Refer to :meth:`~inseq.attr.feat.ops.AggregableMixin._aggregate_layers` and
    :meth:`~inseq.attr.feat.ops.AggregableMixin._aggregate_units` for more details on the
    aggregation procedure and default values.
    """

    @staticmethod
    def has_convergence_delta() -> bool:
        return False

    @classmethod
    @property
    def unit_name(cls) -> str:
        return "attention head"

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: TensorOrTupleOfTensorsGeneric,
        aggregate_heads_fn: Union[str, AggregationFunction, None] = None,
        aggregate_layers_fn: Union[str, AggregationFunction, None] = None,
        heads: Union[int, Tuple[int, int], List[int], None] = None,
        layers: Union[int, Tuple[int, int], List[int], None] = None,
        decoder_self_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
        cross_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """Performs basic attention attribution.

        Args:
            aggregate_heads_fn (:obj:`str` or :obj:`callable`): The method to use for aggregating across heads.
                Can be one of `average` (default if heads is list, tuple or None), `max`, `min` or `single` (default
                if heads is int), or a custom function defined by the user.
            aggregate_layers_fn (:obj:`str` or :obj:`callable`): The method to use for aggregating across layers.
                Can be one of `average` (default if layers is tuple or list), `max`, `min` or `single` (default if
                layers is int or None), or a custom function defined by the user.
            heads (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified,
                the head at the corresponding index is used. If a tuple of two indices is specified, all heads between
                the indices will be aggregated using aggregate_fn. If a list of indices is specified, the respective
                heads will be used for aggregation. If aggregate_fn is "single", a head must be specified.
                If no value specified, all heads are passed to aggregate_fn by default.
            layers (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified
                , the layer at the corresponding index is used. If a tuple of two indices is specified, all layers
                among the indices will be aggregated using aggregate_fn. If a list of indices is specified, the
                respective layers will be used for aggregation. If aggregate_fn is "single", the last layer is
                used by default. If no value is specified, all available layers are passed to aggregate_fn by default.
            decoder_self_attentions (:obj:`tuple(torch.Tensor)`, optional): Tensor of decoder self-attention weights
                computed during the forward pass with shape :obj:`(batch_size, n_layers, n_heads, seq_len, seq_len)`.
            cross_attentions (:obj:`tuple(torch.Tensor)`, optional): Tensor of cross-attention weights computed
                during the forward pass with shape :obj:`(batch_size, n_layers, n_heads, seq_len, seq_len)`.

        Returns:
            `TensorOrTupleOfTensorsGeneric`: Attribution outputs for source-only or source + target feature attribution
        """
        if self.forward_func.is_encoder_decoder:
            cross_layer_aggregation = self._aggregate_layers(cross_attentions, aggregate_layers_fn, layers)
            cross_head_aggregation = self._aggregate_units(cross_layer_aggregation, aggregate_heads_fn, heads)
            attributions = (cross_head_aggregation.select(1, -1),)
            # Encoder-decoder with attribute_target=True
            if len(inputs) > 1:
                decoder_layer_aggregation = self._aggregate_layers(
                    decoder_self_attentions, aggregate_layers_fn, layers
                )
                decoder_head_aggregation = self._aggregate_units(decoder_layer_aggregation, aggregate_heads_fn, heads)
                attributions = attributions + (decoder_head_aggregation.select(1, -1),)
        else:
            layer_aggregation = self._aggregate_layers(decoder_self_attentions, aggregate_layers_fn, layers)
            head_aggregation = self._aggregate_units(layer_aggregation, aggregate_heads_fn, heads)
            attributions = (head_aggregation.select(1, -1),)

        return attributions
