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

from typing import Any, Callable, Dict, Union

import logging

from ...data import EncoderDecoderBatch, FeatureAttributionStepOutput
from ...utils import Registry
from ...utils.typing import ModelIdentifier, SingleScorePerStepTensor, TargetIdsTensor
from ..attribution_decorators import set_hook, unset_hook
from .feature_attribution import FeatureAttribution
from .ops import AggregatedAttention, LastLayerAttention


logger = logging.getLogger(__name__)


class AttentionAtribution(FeatureAttribution, Registry):
    r"""Attention-based attribution method registry."""

    @set_hook
    def hook(self, **kwargs):
        pass

    @unset_hook
    def unhook(self, **kwargs):
        pass

    def attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attribute_target: bool = False,
        attribution_args: Dict[str, Any] = {},
        attributed_fn_args: Dict[str, Any] = {},
    ) -> FeatureAttributionStepOutput:
        r"""
        Performs a single attribution step for the specified target_ids,
        given sources and targets in the batch.

        Abstract method, must be implemented by subclasses.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch of sequences on which attribution is performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size)` corresponding to tokens
                for which the attribution step must be performed.
            attributed_fn (:obj:`Callable[..., SingleScorePerStepTensor]`): The function of model outputs
                representing what should be attributed (e.g. output probits of model best prediction after softmax).
                The parameter must be a function that taking multiple keyword arguments and returns a :obj:`tensor`
                of size (batch_size,). If not provided, the default attributed function for the model will be used
                (change attribution_model.default_attributed_fn_id).
            attribute_target (:obj:`bool`, optional): Whether to attribute the target prefix or not. Defaults to False.
            attribution_args (:obj:`dict`, `optional`): Additional arguments to pass to the attribution method.
                Defaults to {}.
            attributed_fn_args (:obj:`dict`, `optional`): Additional arguments to pass to the attributed function.
                Defaults to {}.
        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: A dataclass containing attribution tensors for source
                and target attributions of size `(batch_size, source_length)` and `(batch_size, prefix length)`.
                (target optional if attribute_target=True), plus batch information and any step score present.
        """
        return FeatureAttributionStepOutput(
            source_attributions=None,
            target_attributions=None,
            step_scores=None,
        )

    @classmethod
    def load(
        cls,
        method_name: str,
        attribution_model=None,
        model_name_or_path: Union[ModelIdentifier, None] = None,
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
                "Attention-based attribution methods require the `output_attentions` parameter to be set on the model."
            )
        return super().load(method_name, attribution_model, model_name_or_path, **kwargs)


class AggregatedAttentionAtribution(AttentionAtribution):

    method_name = "aggregated_attention"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        self.method = AggregatedAttention(attribution_model)


class LastLayerAttentionAttribution(AttentionAtribution):

    method_name = "last_layer_attention"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        self.method = LastLayerAttention(attribution_model)
