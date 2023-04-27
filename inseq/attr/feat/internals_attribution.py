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
from typing import Any, Dict, Optional

from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import Attribution
from captum.log import log_usage

from ...data import MultiDimensionalFeatureAttributionStepOutput
from ...utils import Registry
from ...utils.typing import MultiLayerMultiUnitScoreTensor
from .feature_attribution import FeatureAttribution

logger = logging.getLogger(__name__)


class InternalsAttributionRegistry(FeatureAttribution, Registry):
    r"""Model Internals-based attribution method registry."""
    pass


class AttentionWeightsAttribution(InternalsAttributionRegistry):
    """The basic attention attribution method, which retrieves the attention weights from the model.

    Attribute Args:
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
                If no value is specified, all heads are passed to aggregate_fn by default.
            layers (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified
                , the layer at the corresponding index is used. If a tuple of two indices is specified, all layers
                among the indices will be aggregated using aggregate_fn. If a list of indices is specified, the
                respective layers will be used for aggregation. If aggregate_fn is "single", the last layer is
                used by default. If no value is specified, all available layers are passed to aggregate_fn by default.

    Example:
        - ``model.attribute(src)`` will return the average attention for all heads of the last layer.
        - ``model.attribute(src, heads=0)`` will return the attention weights for the first head of the last layer.
        - ``model.attribute(src, heads=(0, 5), aggregate_heads_fn="max", layers=[0, 2, 7])`` will return the maximum
            attention weights for the first 5 heads averaged across the first, third, and eighth layers.
    """

    method_name = "attention"

    class AttentionWeights(Attribution):
        @staticmethod
        def has_convergence_delta() -> bool:
            return False

        @log_usage()
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
            # We adopt the format [batch_size, sequence_length, num_layers, num_heads]
            # for consistency with other multi-unit methods (e.g. gradient attribution)
            decoder_self_attentions = decoder_self_attentions[..., -1, :].clone().permute(0, 3, 1, 2)
            if self.forward_func.is_encoder_decoder:
                sequence_scores = {}
                if len(inputs) > 1:
                    target_attributions = decoder_self_attentions
                else:
                    target_attributions = None
                    sequence_scores["decoder_self_attentions"] = decoder_self_attentions
                sequence_scores["encoder_self_attentions"] = (
                    encoder_self_attentions[..., -1, :].clone().permute(0, 3, 1, 2)
                )
                return MultiDimensionalFeatureAttributionStepOutput(
                    source_attributions=cross_attentions[..., -1, :].clone().permute(0, 3, 1, 2),
                    target_attributions=target_attributions,
                    step_scores={},
                    sequence_scores=sequence_scores,
                )
            else:
                return MultiDimensionalFeatureAttributionStepOutput(
                    source_attributions=None,
                    target_attributions=decoder_self_attentions,
                    step_scores={},
                    sequence_scores={},
                )

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        # Attention weights will be passed to the attribute_step method
        self.use_attention_weights = True
        # Does not rely on predicted output (i.e. decoding strategy agnostic)
        self.use_predicted_target = False
        self.method = self.AttentionWeights(attribution_model)

    def attribute_step(
        self,
        attribute_fn_main_args: Dict[str, Any],
        attribution_args: Dict[str, Any],
    ) -> MultiDimensionalFeatureAttributionStepOutput:
        return self.method.attribute(**attribute_fn_main_args, **attribution_args)
