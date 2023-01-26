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
from typing import Any, Callable, Dict, Union

from ...data import Batch, EncoderDecoderBatch, FeatureAttributionStepOutput
from ...utils import Registry, pretty_tensor
from ...utils.typing import SingleScorePerStepTensor, TargetIdsTensor
from ..attribution_decorators import set_hook, unset_hook
from .attribution_utils import STEP_SCORES_MAP, get_source_target_attributions
from .feature_attribution import FeatureAttribution
from .ops import Attention

logger = logging.getLogger(__name__)


class AttentionAttributionRegistry(FeatureAttribution, Registry):
    r"""Attention-based attribution method registry."""

    @set_hook
    def hook(self, **kwargs):
        pass

    @unset_hook
    def unhook(self, **kwargs):
        pass

    def format_attribute_args(
        self,
        batch: Union[Batch, EncoderDecoderBatch],
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attribute_target: bool = False,
        attributed_fn_args: Dict[str, Any] = {},
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Formats inputs for the attention attribution methods

        Args:
            batch (:class:`~inseq.data.Batch` or :class:`~inseq.data.EncoderDecoderBatch`): The batch of sequences on
                which attribution is performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size)` corresponding to tokens
                for which the attribution step must be performed.
            attributed_fn (:obj:`Callable[..., SingleScorePerStepTensor]`): The function of model outputs
                representing what should be attributed (e.g. output probits of model best prediction after softmax).
                The parameter must be a function that taking multiple keyword arguments and returns a :obj:`tensor`
                of size (batch_size,). If not provided, the default attributed function for the model will be used
                (change attribution_model.default_attributed_fn_id).
            attribute_target (:obj:`bool`, optional): Whether to attribute the target prefix or not. Defaults to False.
            attributed_fn_args (:obj:`dict`, `optional`): Additional arguments to pass to the attributed function.
                Defaults to {}.
        Returns:
            :obj:`dict`: A dictionary containing the formatted attribution arguments.
        """
        logger.debug(f"batch: {batch},\ntarget_ids: {pretty_tensor(target_ids, lpad=4)}")
        if attributed_fn != STEP_SCORES_MAP[self.attribution_model.default_attributed_fn_id]:
            logger.warning(
                "Attention-based attribution methods are output agnostic, since they do not rely on specific output"
                " targets to compute input saliency. As such, using a custom attributed function for attention"
                " attribution methods does not produce any effect of the method's results."
            )
        attribute_fn_args = {
            "batch": batch,
            "additional_forward_args": (
                attribute_target,
                attributed_fn,
                self.forward_batch_embeds,
                list(attributed_fn_args.keys()),
            )
            + tuple(attributed_fn_args.values()),
        }

        return attribute_fn_args

    def attribute_step(
        self,
        attribute_fn_main_args: Dict[str, Any],
        attribution_args: Dict[str, Any] = {},
    ) -> FeatureAttributionStepOutput:
        r"""
        Performs a single attribution step for the specified attribution arguments.

        Args:
            attribute_fn_main_args (:obj:`dict`): Main arguments used for the attribution method. These are built from
                model inputs at the current step of the feature attribution process.
            attribution_args (:obj:`dict`, `optional`): Additional arguments to pass to the attribution method.
                These can be specified by the user while calling the top level `attribute` methods. Defaults to {}.

        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: A dataclass containing a tensor of source
                attributions of size `(batch_size, source_length)`, possibly a tensor of target attributions of size
                `(batch_size, prefix length) if attribute_target=True and possibly a tensor of deltas of size
                `(batch_size)` if the attribution step supports deltas and they are requested. At this point the batch
                information is empty, and will later be filled by the enrich_step_output function.
        """
        attr = self.method.attribute(**attribute_fn_main_args, **attribution_args)

        source_attributions, target_attributions = get_source_target_attributions(
            attr, self.attribution_model.is_encoder_decoder
        )
        return FeatureAttributionStepOutput(
            source_attributions=source_attributions,
            target_attributions=target_attributions,
            step_scores={},
        )


class AttentionAttribution(AttentionAttributionRegistry):
    """
    The basic attention attribution method, which retrieves the attention weights from the model.

    Attribute Args:
        aggregate_heads_fn (:obj:`str` or :obj:`callable`): The method to use for aggregating across heads.
            Can be one of `average` (default if heads is tuple or None), `max`, or `single` (default if heads is
            int), or a custom function defined by the user.
        aggregate_layers_fn (:obj:`str` or :obj:`callable`): The method to use for aggregating across layers.
            Can be one of `average` (default if layers is tuple), `max`, or `single` (default if layers is int or
            None), or a custom function defined by the user.
        heads (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified,
                the head at the corresponding index is used. If a tuple of two indices is specified, all heads between
                the indices will be aggregated using aggregate_fn. If a list of indices is specified, the respective
                heads will be used for aggregation. If aggregate_fn is "single", a head must be specified.
                Otherwise, all heads are passed to aggregate_fn by default.
        layers (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified
                , the layer at the corresponding index is used. If a tuple of two indices is specified, all layers
                among the indices will be aggregated using aggregate_fn. If a list of indices is specified, the
                respective layers will be used for aggregation. If aggregate_fn is "single", the last layer is
                used by default. Otherwise, all available layers are passed to aggregate_fn by default.

    Example:

        - ``model.attribute(src)`` will return the average attention for all heads of the last layer.
        - ``model.attribute(src, heads=0)`` will return the attention weights for the first head of the last layer.
        - ``model.attribute(src, heads=(0, 5), aggregate_heads_fn="max", layers=[0, 2, 7])`` will return the maximum
            attention weights for the first 5 heads averaged across the first, third, and eighth layers.
    """

    method_name = "attention"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        self.method = Attention(attribution_model)
