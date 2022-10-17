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
from ...utils.typing import ModelIdentifier
from .feature_attribution import FeatureAttribution


logger = logging.getLogger(__name__)


class AttentionAtribution(FeatureAttribution, Registry):
    r"""Attention-based attribution method registry."""

    @classmethod
    def load(
        cls,
        method_name: str,
        attribution_model=None,
        model_name_or_path: ModelIdentifier = None,
        **kwargs,
    ) -> "FeatureAttribution":
        from inseq import AttributionModel

        if model_name_or_path is None == attribution_model is None:  # noqa
            raise RuntimeError(
                "Only one among an initialized model and a model identifier "
                "must be defined when loading the attribution method."
            )
        if model_name_or_path:
            attribution_model = AttributionModel.load(model_name_or_path)
            model_name_or_path = None

        if not attribution_model.model.config.output_attentions:
            raise RuntimeError(
                "Attention-based attribution methods require the `output_attentions` parameter to be set."
            )
        return super().load(method_name, attribution_model, model_name_or_path, **kwargs)


class AggregatedAttentionAtribution(AttentionAtribution):

    method_name = "aggregated_attention"


class LastLayerAttentionAttribution(AttentionAtribution):

    method_name = "last_layer_attention"
