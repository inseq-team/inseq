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
from captum._utils.common import _format_output, _is_tuple
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import Attribution
from captum.log import log_usage

from ....utils.typing import MultiStepEmbeddingsTensor
from ..attribution_utils import num_attention_heads


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

    def _merge_attention_heads(self, attention: torch.Tensor, option: str = "average", head: int = None):

        num_heads = num_attention_heads(attention[0])

        if option == "single" and head is None:
            raise RuntimeError("An attention head has to be specified when choosing single-head attention attribution")

        if head is not None:
            if head > num_heads:
                raise RuntimeError(
                    "Attention head index for attribution too high. " f"The model only has {num_heads} heads."
                )

            if option != "single":
                logger.warning(
                    "Only single-head attention is possible if an attention head is specified.\n"
                    "Switching to single-head attention"
                )

            return attention.select(1, head)

        if option == "average":
            return attention.mean(1, keepdim=True)

        # TODO: test this, I feel like this method is not doing what we want here
        elif option == "max":
            return attention.max(1, keepdim=True)

        else:
            raise RuntimeError(
                "Invalid merge method for attention heads specified. "
                "Valid methods are: `average`, `max` and `single`"
            )

    def _get_batch_size(self, attention: torch.Tensor):
        """returns the batch size of a tensor of shape `batch_size, heads, seq, seq`"""
        return attention.size(0)


class AggregatedAttention(AttentionAttribution):
    """
    A basic attention attribution approach.
    It will return the attention values averaged across all layers.
    """

    @log_usage()
    def attribute(
        self,
        inputs: MultiStepEmbeddingsTensor,
        target: TargetType = None,
        merge_head_option: str = "average",
        use_head: int = None,
        additional_forward_args: Any = None,
    ) -> Union[TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, torch.Tensor]]:

        is_inputs_tuple = _is_tuple(inputs)

        is_target_attribution = True if len(inputs) > 1 else False

        input_ids = additional_forward_args[0] if is_target_attribution else additional_forward_args[1]
        decoder_input_ids = additional_forward_args[1] if is_target_attribution else additional_forward_args[2]
        attention_mask = additional_forward_args[4] if is_target_attribution else additional_forward_args[5]
        decoder_attention_mask = additional_forward_args[5] if is_target_attribution else additional_forward_args[6]

        outputs = self.forward_func.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        cross_aggregation = torch.stack(outputs.cross_attentions).mean(0)
        cross_aggregation = self._merge_attention_heads(cross_aggregation, merge_head_option, use_head)
        cross_aggregation = torch.squeeze(cross_aggregation, 1).select(1, -1)

        attributions = (cross_aggregation,)

        if is_target_attribution:
            decoder_aggregation = torch.stack(outputs.decoder_attentions).mean(0)
            decoder_aggregation = self._merge_attention_heads(decoder_aggregation, merge_head_option, use_head)
            decoder_aggregation = torch.squeeze(decoder_aggregation, 1).select(1, -1)

            attributions = attributions + (decoder_aggregation,)

        return _format_output(is_inputs_tuple, attributions)


class LastLayerAttention(AttentionAttribution):
    """
    A basic attention attribution approach.
    It will simply return the attention values of the last layer.
    """

    @log_usage()
    def attribute(
        self,
        inputs: MultiStepEmbeddingsTensor,
        target: TargetType = None,
        merge_head_option: str = "average",
        use_head: int = None,
        additional_forward_args: Any = None,
    ) -> Union[TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, torch.Tensor]]:

        is_inputs_tuple = _is_tuple(inputs)

        is_target_attribution = True if len(inputs) > 1 else False

        input_ids = additional_forward_args[0] if is_target_attribution else additional_forward_args[1]
        decoder_input_ids = additional_forward_args[1] if is_target_attribution else additional_forward_args[2]
        attention_mask = additional_forward_args[4] if is_target_attribution else additional_forward_args[5]
        decoder_attention_mask = additional_forward_args[5] if is_target_attribution else additional_forward_args[6]

        outputs = self.forward_func.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        last_layer_cross = outputs.cross_attentions[-1]
        last_layer_cross = self._merge_attention_heads(last_layer_cross, merge_head_option, use_head)
        last_layer_cross = torch.squeeze(last_layer_cross, 1).select(1, -1)

        attributions = (last_layer_cross,)

        if is_target_attribution:
            last_layer_decoder = outputs.decoder_attentions[-1]
            last_layer_decoder = self._merge_attention_heads(last_layer_decoder, merge_head_option, use_head)
            last_layer_decoder = torch.squeeze(last_layer_decoder, 1).select(1, -1)

            attributions = attributions + (last_layer_decoder,)

        return _format_output(is_inputs_tuple, attributions)
