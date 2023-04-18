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
""" Attention-based feature attribution methods. """

import logging

from ...utils import Registry
from .feature_attribution import FeatureAttribution
from .ops import AttentionWeights

logger = logging.getLogger(__name__)


class InternalsAttributionRegistry(FeatureAttribution, Registry):
    r"""Model Internals-based attribution method registry."""
    pass


class AttentionWeightsAttribution(InternalsAttributionRegistry):
    """
    The basic attention attribution method, which retrieves the attention weights from the model.

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

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        # Attention weights will be passed to the attribute_step method
        self.use_attention_weights = True
        # Does not rely on predicted output (i.e. decoding strategy agnostic)
        self.use_predicted_target = False
        self.method = AttentionWeights(attribution_model)
