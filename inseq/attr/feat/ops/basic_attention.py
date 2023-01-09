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

from typing import Any, Tuple, Union

import logging

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import Attribution
from captum.log import log_usage

from ....data import Batch, EncoderDecoderBatch


logger = logging.getLogger(__name__)


class AttentionAttribution(Attribution):
    """
    All attention based attribution algorithms extend this class. It requires a
    forward function, which most commonly is the forward function of the model
    that we want to interpret or the model itself.
    """

    merge_head_options = ["average", "max", "single"]

    def has_convergence_delta(self) -> bool:
        return False

    def _num_attention_heads(self, attention: torch.Tensor) -> int:
        """
        Returns the number of heads an attention tensor has.

        Args:
            attention: an attention tensor of shape `(batch_size, num_heads, sequence_length, sequence_length)`

        Returns:
            `int`: The number of attention heads
        """
        return attention.size(1)

    def _merge_attention_heads(
        self, attention: torch.Tensor, option: str = "average", head: int = None
    ) -> torch.Tensor:

        """
        Merges the attention values of the different heads together by either averaging across them,
        selecting the head with the maximal values or selecting a specific attention head.

        Args:
            attention: an attention tensor of shape `(batch_size, num_heads, sequence_length, sequence_length)`
            option: The method to use for merging. Should be one of `average` (default), `max`, or `single`
            head: The index of the head to use, when option is set to `single`

        Returns:
            `torch.Tensor`: The attention tensor with its attention heads merged.
        """
        num_heads = self._num_attention_heads(attention[0])

        if option == "single" and head is None:
            raise RuntimeError("An attention head has to be specified when choosing single-head attention attribution")

        if head is not None:
            if head not in range(-num_heads, num_heads):
                raise IndexError(
                    f"Attention head index for attribution out of range. The model only has {num_heads} heads."
                )

            if option != "single":
                logger.warning(
                    "Only single-head attention is possible if an attention head is specified.\n"
                    "Switching to single-head attention"
                )

            return attention.select(1, head)

        if option == "average":
            return attention.mean(1)

        elif option == "max":
            return attention.max(1)

        else:
            raise RuntimeError(
                "Invalid merge method for attention heads specified. "
                "Valid methods are: `average`, `max` and `single`"
            )

    def _get_batch_size(self, attention: torch.Tensor) -> int:
        """returns the batch size of a tensor of shape `(batch_size, heads, seq, seq)`"""
        return attention.size(0)


class AggregatedAttention(AttentionAttribution):
    """
    A basic attention attribution approach.
    It will return the attention values averaged across all layers.
    """

    @log_usage()
    def attribute(
        self,
        batch: Union[Batch, EncoderDecoderBatch],
        merge_head_option: str = "average",
        use_head: int = None,
        additional_forward_args: Any = None,
    ) -> Union[TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, torch.Tensor]]:

        is_target_attribution = additional_forward_args[0]

        is_encoder_decoder = self.forward_func.is_encoder_decoder

        outputs = self.forward_func.get_forward_output(**self.forward_func.format_forward_args(batch))

        if is_encoder_decoder:
            cross_aggregation = torch.stack(outputs.cross_attentions).mean(0)
            cross_aggregation = self._merge_attention_heads(cross_aggregation, merge_head_option, use_head)
            cross_aggregation = cross_aggregation.select(1, -1)

            attributions = (cross_aggregation,)

            if is_target_attribution:
                decoder_aggregation = torch.stack(outputs.decoder_attentions).mean(0)
                decoder_aggregation = self._merge_attention_heads(decoder_aggregation, merge_head_option, use_head)
                decoder_aggregation = decoder_aggregation.select(1, -1)

                attributions = attributions + (decoder_aggregation,)
        else:
            aggregation = torch.stack(outputs.attentions).mean(0)
            aggregation = self._merge_attention_heads(aggregation, merge_head_option, use_head)
            aggregation = aggregation.select(1, -1)

            attributions = (aggregation,)

        return attributions


class LastLayerAttention(AttentionAttribution):
    """
    A basic attention attribution approach.
    It will simply return the attention values of the last layer.
    """

    @log_usage()
    def attribute(
        self,
        batch: Union[Batch, EncoderDecoderBatch],
        merge_head_option: str = "average",
        use_head: int = None,
        additional_forward_args: Any = None,
    ) -> Union[TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, torch.Tensor]]:

        is_target_attribution = additional_forward_args[0]

        is_encoder_decoder = self.forward_func.is_encoder_decoder

        outputs = self.forward_func.get_forward_output(**self.forward_func.format_forward_args(batch))

        if is_encoder_decoder:

            last_layer_cross = outputs.cross_attentions[-1]
            last_layer_cross = self._merge_attention_heads(last_layer_cross, merge_head_option, use_head)
            last_layer_cross = last_layer_cross.select(1, -1)

            attributions = (last_layer_cross,)

            if is_target_attribution:
                last_layer_decoder = outputs.decoder_attentions[-1]
                last_layer_decoder = self._merge_attention_heads(last_layer_decoder, merge_head_option, use_head)
                last_layer_decoder = last_layer_decoder.select(1, -1)

                attributions = attributions + (last_layer_decoder,)
        else:

            aggregation = outputs.attentions[-1]
            aggregation = self._merge_attention_heads(aggregation, merge_head_option, use_head)
            aggregation = aggregation.select(1, -1)

            attributions = (aggregation,)

        return attributions
