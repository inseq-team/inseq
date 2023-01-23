# Copyright 2021 The Inseq Team. All rights reserved.
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
from typing import Any, Dict, List, Protocol, Tuple, Union

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import Attribution
from captum.log import log_usage

from ....data import Batch, EncoderDecoderBatch
from ....utils.typing import AggregatedLayerAttentionTensor, FullAttentionOutput, FullLayerAttentionTensor

logger = logging.getLogger(__name__)


class AggregateAttentionFunction(Protocol):
    def __call__(self, attention: FullLayerAttentionTensor, dim: int, **kwargs) -> AggregatedLayerAttentionTensor:
        ...


class BaseAttentionAttribution(Attribution):
    """
    All attention based attribution algorithms extend this class. It requires a
    forward function, which most commonly is the forward function of the model
    that we want to interpret or the model itself.
    """

    AGGREGATE_FN_OPTIONS: Dict[str, AggregateAttentionFunction] = {
        "average": lambda x, dim: x.mean(dim),
        "max": lambda x, dim: x.max(dim)[0],
        "min": lambda x, dim: x.min(dim)[0],
        "single": lambda x, dim, idx: x.select(dim, idx),
    }

    @staticmethod
    def has_convergence_delta() -> bool:
        return False

    @staticmethod
    def _num_attention_heads(attention: FullLayerAttentionTensor) -> int:
        """Returns the number of heads contained in the attention tensor."""
        return attention.size(1)

    @staticmethod
    def _num_layers(attention: FullAttentionOutput) -> int:
        """Returns the number of layers contained in the attention tensor."""
        return len(attention)

    @classmethod
    def _aggregate_attention_heads(
        cls,
        attention: FullLayerAttentionTensor,
        aggregate_fn: Union[str, AggregateAttentionFunction, None] = None,
        heads: Union[int, Tuple[int, int], List[int], None] = None,
    ) -> AggregatedLayerAttentionTensor:
        """
        Merges the attention values across the specified attention heads for the full sequence.

        Args:
            attention (:obj:`torch.Tensor`) attention tensor of shape
                `(batch_size, num_heads, sequence_length, sequence_length)`
            aggregate_fn (:obj:`str` or :obj:`callable`): The method to use for aggregating across heads.
                Can be one of `average` (default if heads is tuple or None), `max`, `min` or `single` (default if heads
                is int), or a custom function defined by the user.
            heads (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified,
                the head at the corresponding index is used. If a tuple of two indices is specified, all heads between
                the indices will be aggregated using aggregate_fn. If a list of indices is specified, the respective
                heads will be used for aggregation. If aggregate_fn is "single", a head must be specified.
                Otherwise, all heads are passed to aggregate_fn by default.

        Returns:
            :obj:`torch.Tensor`: An aggregated attention tensor of shape
                `(batch_size, sequence_length, sequence_length)`
        """
        n_heads = cls._num_attention_heads(attention)
        aggregate_kwargs = {}

        if hasattr(heads, "__iter__"):
            if len(heads) == 0:
                raise RuntimeError("At least two heads must be specified for aggregated attention attribution.")
            if len(heads) == 1:
                heads = heads[0]

        # If heads is not specified or an tuple, average aggregation is used by default
        if aggregate_fn is None and not isinstance(heads, int):
            aggregate_fn = "average"
            logger.info("No attention head aggregation method specified. Using average aggregation by default.")
        # If a single head index is specified, single aggregation is used by default
        if aggregate_fn is None and isinstance(heads, int):
            aggregate_fn = "single"

        if aggregate_fn == "single":
            if not isinstance(heads, int):
                raise RuntimeError("A single head index must be specified for single-layer attention attribution")
            if heads not in range(-n_heads, n_heads):
                raise IndexError(f"Attention head index out of range. The model only has {n_heads} heads.")
            aggregate_kwargs = {"idx": heads}
            aggregate_fn = cls.AGGREGATE_FN_OPTIONS[aggregate_fn]
        else:
            if isinstance(aggregate_fn, str):
                if aggregate_fn not in cls.AGGREGATE_FN_OPTIONS:
                    raise RuntimeError(
                        f"Invalid aggregation method specified.Valid methods are: {cls.AGGREGATE_FN_OPTIONS.keys()}"
                    )
                aggregate_fn = cls.AGGREGATE_FN_OPTIONS[aggregate_fn]
            if heads is None:
                heads = (0, n_heads)
                logger.info("No attention heads specified for attention extraction. Using all heads by default.")
            # Convert negative indices to positive indices
            if hasattr(heads, "__iter__"):
                heads = type(heads)([h_idx if h_idx >= 0 else n_heads + h_idx for h_idx in heads])
            if not hasattr(heads, "__iter__") or (
                len(heads) == 2 and isinstance(heads, tuple) and heads[0] >= heads[1]
            ):
                raise RuntimeError(
                    "A (start, end) tuple of indices representing a span or a list of individual indices"
                    " must be specified for aggregated attention attribution."
                )
            max_idx_val = n_heads if isinstance(heads, list) else n_heads + 1
            if not all(h in range(-n_heads, max_idx_val) for h in heads):
                raise IndexError(f"One or more attention head index out of range. The model only has {n_heads} heads.")
            if len(set(heads)) != len(heads):
                raise IndexError("Duplicate head indices are not allowed.")
            if isinstance(heads, tuple):
                attention = attention[:, heads[0] : heads[1]]
            else:
                attention = torch.index_select(attention, 1, torch.tensor(heads, device=attention.device))
        return aggregate_fn(attention, 1, **aggregate_kwargs)

    @classmethod
    def _aggregate_layers(
        cls,
        attention: FullAttentionOutput,
        aggregate_fn: Union[str, AggregateAttentionFunction, None] = None,
        layers: Union[int, Tuple[int, int], List[int], None] = None,
    ) -> FullLayerAttentionTensor:
        """
        Merges the attention values of every attention head across the specified layers for the full sequence.

        Args:
            attention (:obj:`torch.Tensor`) attention tensor of shape
                `(n_layers, batch_size, num_heads, sequence_length, sequence_length)`
            aggregate_fn (:obj:`str` or :obj:`callable`): The method to use for aggregating across layers.
                Can be one of `average` (default if layers is tuple), `max`, `min` or `single` (default if layers is
                int or None), or a custom function defined by the user.
            layers (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified
                , the layer at the corresponding index is used. If a tuple of two indices is specified, all layers
                among the indices will be aggregated using aggregate_fn. If a list of indices is specified, the
                respective layers will be used for aggregation. If aggregate_fn is "single", the last layer is
                used by default. Otherwise, all available layers are passed to aggregate_fn by default.

        Returns:
            :obj:`torch.Tensor`: An aggregated attention tensor of shape
                `(batch_size, num_heads, sequence_length, sequence_length)`
        """
        n_layers = cls._num_layers(attention)
        attention = torch.stack(attention, dim=0)
        aggregate_kwargs = {}

        if hasattr(layers, "__iter__"):
            if len(layers) == 0:
                raise RuntimeError("At least two layer must be specified for aggregated attention attribution.")
            if len(layers) == 1:
                layers = layers[0]

        # If layers is not specified or an int, single layer aggregation is used by default
        if aggregate_fn is None and not hasattr(layers, "__iter__"):
            aggregate_fn = "single"
            logger.info("No layer aggregation method specified. Using single layer by default.")
        # If a tuple of indices for layers is specified, average aggregation is used by default
        if aggregate_fn is None and hasattr(layers, "__iter__"):
            aggregate_fn = "average"
            logger.info("No layer aggregation method specified. Using average across layers by default.")

        if aggregate_fn == "single":
            if layers is None:
                layers = -1
                logger.info("No layer specified for attention extraction. Using last layer by default.")
            if not isinstance(layers, int):
                raise RuntimeError("A single layer index must be specified for single-layer attention attribution")
            if layers not in range(-n_layers, n_layers):
                raise IndexError(f"Layer index out of range. The model only has {n_layers} layers.")
            aggregate_kwargs = {"idx": layers}
            aggregate_fn = cls.AGGREGATE_FN_OPTIONS[aggregate_fn]
        else:
            if isinstance(aggregate_fn, str):
                if aggregate_fn not in cls.AGGREGATE_FN_OPTIONS:
                    raise RuntimeError(
                        f"Invalid aggregation method specified.Valid methods are: {cls.AGGREGATE_FN_OPTIONS.keys()}"
                    )
                aggregate_fn = cls.AGGREGATE_FN_OPTIONS[aggregate_fn]
            if layers is None:
                layers = (0, n_layers)
                logger.info("No layer specified for attention extraction. Using all layers by default.")
            # Convert negative indices to positive indices
            if hasattr(layers, "__iter__"):
                layers = type(layers)([l_idx if l_idx >= 0 else n_layers + l_idx for l_idx in layers])
            if not hasattr(layers, "__iter__") or (
                len(layers) == 2 and isinstance(layers, tuple) and layers[0] >= layers[1]
            ):
                raise RuntimeError(
                    "A (start, end) tuple of indices representing a span or a list of individual indices"
                    " must be specified for aggregated attention attribution."
                )
            max_idx_val = n_layers if isinstance(layers, list) else n_layers + 1
            if not all(l in range(max_idx_val) for l in layers):
                raise IndexError(f"One or more layer index out of range. The model only has {n_layers} layers.")
            if len(set(layers)) != len(layers):
                raise IndexError("Duplicate layer indices are not allowed.")
            if isinstance(layers, tuple):
                attention = attention[layers[0] : layers[1]]
            else:
                attention = torch.index_select(attention, 0, torch.tensor(layers, device=attention.device))
        return aggregate_fn(attention, 0, **aggregate_kwargs)


class Attention(BaseAttentionAttribution):
    """
    A basic attention attribution approach.
    It will return the attention values for the specified values or aggregated across the specified ranges of heads
    and layers, given the specified aggregation functions.

    Refer to :meth:`~inseq.attr.feat.ops.BaseAttentionAttribution._aggregate_layers` and
    :meth:`~inseq.attr.feat.ops.BaseAttentionAttribution._aggregate_attention_heads` for more details on the
    aggregation procedure and default values.
    """

    @log_usage()
    def attribute(
        self,
        batch: Union[Batch, EncoderDecoderBatch],
        aggregate_heads_fn: Union[str, AggregateAttentionFunction, None] = None,
        aggregate_layers_fn: Union[str, AggregateAttentionFunction, None] = None,
        heads: Union[int, Tuple[int, int], List[int], None] = None,
        layers: Union[int, Tuple[int, int], List[int], None] = None,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """Performs basic attention attribution.

        Args:
            batch (`Union[Batch, EncoderDecoderBatch]`):
                The input batch used for the forward pass to extract attention scores.
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

        Returns:
            `TensorOrTupleOfTensorsGeneric`: Attribution outputs for source-only or source + target feature attribution
        """

        is_target_attribution = additional_forward_args[0]
        is_encoder_decoder = self.forward_func.is_encoder_decoder
        outputs = self.forward_func.get_forward_output(
            **self.forward_func.format_forward_args(batch), output_attentions=True
        )

        if is_encoder_decoder:
            cross_layer_aggregation = self._aggregate_layers(outputs.cross_attentions, aggregate_layers_fn, layers)
            cross_head_aggregation = self._aggregate_attention_heads(
                cross_layer_aggregation, aggregate_heads_fn, heads
            )
            attributions = (cross_head_aggregation.select(1, -1),)

            if is_target_attribution:
                decoder_layer_aggregation = self._aggregate_layers(
                    outputs.decoder_attentions, aggregate_layers_fn, layers
                )
                decoder_head_aggregation = self._aggregate_attention_heads(
                    decoder_layer_aggregation, aggregate_heads_fn, heads
                )
                attributions = attributions + (decoder_head_aggregation.select(1, -1),)
        else:
            layer_aggregation = self._aggregate_layers(outputs.attentions, aggregate_layers_fn, layers)
            head_aggregation = self._aggregate_attention_heads(layer_aggregation, aggregate_heads_fn, heads)
            attributions = (head_aggregation.select(1, -1),)

        return attributions
