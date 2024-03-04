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
"""Attention-based feature attribution methods."""

import logging
from typing import Any, Optional

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from ...data import MultiDimensionalFeatureAttributionStepOutput
from ...utils import Registry
from ...utils.typing import InseqAttribution, MultiLayerMultiUnitScoreTensor
from .feature_attribution import FeatureAttribution

logger = logging.getLogger(__name__)


class InternalsAttributionRegistry(FeatureAttribution, Registry):
    r"""Model Internals-based attribution method registry."""

    pass


class AttentionWeightsAttribution(InternalsAttributionRegistry):
    """The basic attention attribution method, which retrieves the attention weights from the model."""

    method_name = "attention"

    class AttentionWeights(InseqAttribution):
        @staticmethod
        def has_convergence_delta() -> bool:
            return False

        def attribute(
            self,
            inputs: TensorOrTupleOfTensorsGeneric,
            additional_forward_args: TensorOrTupleOfTensorsGeneric,
            encoder_self_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
            decoder_self_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
            cross_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
        ) -> MultiDimensionalFeatureAttributionStepOutput:
            """Extracts the attention weights from the model.

            Args:
                inputs (`TensorOrTupleOfTensorsGeneric`):
                    Tensor or tuple of tensors that are inputs to the model. Used to match standard Captum API, and to
                    determine whether both source and target are being attributed.
                additional_forward_args (`TensorOrTupleOfTensorsGeneric`):
                    Tensor or tuple of tensors that are additional arguments to the model. Unused, but included to
                    match standard Captum API.
                encoder_self_attentions (:obj:`tuple(torch.Tensor)`, *optional*, defaults to None): Tensor of encoder
                    self-attention weights of the forward pass with shape
                    :obj:`(batch_size, n_layers, n_heads, source_seq_len, source_seq_len)`.
                decoder_self_attentions (:obj:`tuple(torch.Tensor)`, *optional*, defaults to None): Tensor of decoder
                    self-attention weights of the forward pass with shape
                    :obj:`(batch_size, n_layers, n_heads, target_seq_len, target_seq_len)`.
                cross_attentions (:obj:`tuple(torch.Tensor)`, *optional*, defaults to None):
                    Tensor of cross-attention weights computed during the forward pass with shape
                    :obj:`(batch_size, n_layers, n_heads, source_seq_len, target_seq_len)`.

            Returns:
                :class:`~inseq.data.MultiDimensionalFeatureAttributionStepOutput`: A step output containing attention
                weights for each layer and head, with shape :obj:`(batch_size, seq_len, n_layers, n_heads)`.
            """
            # We adopt the format [batch_size, sequence_length, sequence_length, num_layers, num_heads]
            # for consistency with other multi-unit methods (e.g. gradient attribution)
            decoder_self_attentions = decoder_self_attentions.to("cpu").clone().permute(0, 4, 3, 1, 2)
            decoder_self_attentions = torch.where(
                decoder_self_attentions == 0,
                (torch.ones_like(decoder_self_attentions) * float("nan")),
                decoder_self_attentions,
            )
            if self.forward_func.is_encoder_decoder:
                sequence_scores = {}
                if len(inputs) > 1:
                    target_attributions = decoder_self_attentions
                else:
                    target_attributions = None
                    sequence_scores["decoder_self_attentions"] = decoder_self_attentions
                sequence_scores["encoder_self_attentions"] = (
                    encoder_self_attentions.to("cpu").clone().permute(0, 4, 3, 1, 2)
                )
                cross_attentions = cross_attentions.to("cpu").clone().permute(0, 4, 3, 1, 2)
                return MultiDimensionalFeatureAttributionStepOutput(
                    source_attributions=cross_attentions,
                    target_attributions=target_attributions,
                    sequence_scores=sequence_scores,
                    _num_dimensions=2,  # num_layers, num_heads
                )
            else:
                return MultiDimensionalFeatureAttributionStepOutput(
                    source_attributions=None,
                    target_attributions=decoder_self_attentions,
                    _num_dimensions=2,  # num_layers, num_heads
                )

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        # Attention weights will be passed to the attribute_step method
        self.use_attention_weights = True
        # Does not rely on predicted output (i.e. decoding strategy agnostic)
        self.use_predicted_target = False
        # Needs only the final generation step to extract scores
        self.is_final_step_method = True
        self.method = self.AttentionWeights(attribution_model)

    def attribute_step(
        self,
        attribute_fn_main_args: dict[str, Any],
        attribution_args: dict[str, Any],
    ) -> MultiDimensionalFeatureAttributionStepOutput:
        return self.method.attribute(**attribute_fn_main_args, **attribution_args)
