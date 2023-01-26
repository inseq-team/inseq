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
""" Gradient-based feature attribution methods. """

import logging
from typing import Any, Dict

from captum.attr import (
    DeepLift,
    InputXGradient,
    IntegratedGradients,
    LayerDeepLift,
    LayerGradientXActivation,
    LayerIntegratedGradients,
    Saliency,
)

from ...data import GradientFeatureAttributionStepOutput
from ...utils import Registry, extract_signature_args, rgetattr
from ..attribution_decorators import set_hook, unset_hook
from .attribution_utils import get_source_target_attributions
from .feature_attribution import FeatureAttribution
from .ops import DiscretetizedIntegratedGradients

logger = logging.getLogger(__name__)


class GradientAttributionRegistry(FeatureAttribution, Registry):
    r"""Gradient-based attribution method registry."""

    @set_hook
    def hook(self, **kwargs):
        r"""
        Hooks the attribution method to the model by replacing normal :obj:`nn.Embedding` with Captum's
        `InterpretableEmbeddingBase <https://captum.ai/api/utilities.html#captum.attr.InterpretableEmbeddingBase>`__.
        """
        if self.attribute_batch_ids and not self.forward_batch_embeds:
            self.target_layer = kwargs.pop("target_layer", self.attribution_model.get_embedding_layer())
            logger.debug(f"target_layer={self.target_layer}")
            if isinstance(self.target_layer, str):
                self.target_layer = rgetattr(self.attribution_model.model, self.target_layer)
        if not self.attribute_batch_ids:
            self.attribution_model.configure_interpretable_embeddings()

    @unset_hook
    def unhook(self, **kwargs):
        r"""
        Unhook the attribution method by restoring the model's original embeddings.
        """
        if self.attribute_batch_ids and not self.forward_batch_embeds:
            self.target_layer = None
        else:
            self.attribution_model.remove_interpretable_embeddings()

    def attribute_step(
        self,
        attribute_fn_main_args: Dict[str, Any],
        attribution_args: Dict[str, Any] = {},
    ) -> GradientFeatureAttributionStepOutput:
        r"""
        Performs a single attribution step for the specified attribution arguments.

        Args:
            attribute_fn_main_args (:obj:`dict`): Main arguments used for the attribution method. These are built from
                model inputs at the current step of the feature attribution process.
            attribution_args (:obj:`dict`, `optional`): Additional arguments to pass to the attribution method.
                These can be specified by the user while calling the top level `attribute` methods. Defaults to {}.

        Returns:
            :class:`~inseq.data.GradientFeatureAttributionStepOutput`: A dataclass containing a tensor of source
                attributions of size `(batch_size, source_length)`, possibly a tensor of target attributions of size
                `(batch_size, prefix length) if attribute_target=True and possibly a tensor of deltas of size
                `(batch_size)` if the attribution step supports deltas and they are requested. At this point the batch
                information is empty, and will later be filled by the enrich_step_output function.
        """
        attr = self.method.attribute(**attribute_fn_main_args, **attribution_args)
        deltas = None
        if (
            attribution_args.get("return_convergence_delta", False)
            and hasattr(self.method, "has_convergence_delta")
            and self.method.has_convergence_delta()
        ):
            attr, deltas = attr
        source_attributions, target_attributions = get_source_target_attributions(
            attr, self.attribution_model.is_encoder_decoder
        )
        return GradientFeatureAttributionStepOutput(
            source_attributions=source_attributions,
            target_attributions=target_attributions,
            step_scores={"deltas": deltas} if deltas is not None else {},
        )


class DeepLiftAttribution(GradientAttributionRegistry):
    """DeepLIFT attribution method.

    Reference implementation:
    `https://captum.ai/api/deep_lift.html <https://captum.ai/api/deep_lift.html>`__.
    """

    method_name = "deeplift"

    def __init__(self, attribution_model, multiply_by_inputs: bool = True, **kwargs):
        super().__init__(attribution_model)
        self.method = DeepLift(self.attribution_model, multiply_by_inputs)
        self.use_baseline = True


class DiscretizedIntegratedGradientsAttribution(GradientAttributionRegistry):
    """Discretized Integrated Gradients attribution method

    Reference: https://arxiv.org/abs/2108.13654

    Original implementation: https://github.com/INK-USC/DIG
    """

    method_name = "discretized_integrated_gradients"

    def __init__(self, attribution_model, multiply_by_inputs: bool = False, **kwargs):
        super().__init__(attribution_model, hook_to_model=False)
        self.attribution_model = attribution_model
        self.attribute_batch_ids = True
        self.use_baseline = True
        self.method = DiscretetizedIntegratedGradients(
            self.attribution_model,
            multiply_by_inputs,
        )
        self.hook(**kwargs)

    @set_hook
    def hook(self, **kwargs):
        load_kwargs, other_kwargs = extract_signature_args(
            kwargs,
            self.method.load_monotonic_path_builder,
            return_remaining=True,
        )
        self.method.load_monotonic_path_builder(
            self.attribution_model.model_name,
            vocabulary_embeddings=self.attribution_model.vocabulary_embeddings.detach(),
            special_tokens=self.attribution_model.special_tokens_ids,
            embedding_scaling=self.attribution_model.embed_scale,
            **load_kwargs,
        )
        super().hook(**other_kwargs)


class IntegratedGradientsAttribution(GradientAttributionRegistry):
    """Integrated Gradients attribution method.

    Reference implementation:
    `https://captum.ai/api/integrated_gradients.html <https://captum.ai/api/integrated_gradients.html>`__.
    """

    method_name = "integrated_gradients"

    def __init__(self, attribution_model, multiply_by_inputs: bool = True, **kwargs):
        super().__init__(attribution_model)
        self.method = IntegratedGradients(self.attribution_model, multiply_by_inputs)
        self.use_baseline = True


class InputXGradientAttribution(GradientAttributionRegistry):
    """Input x Gradient attribution method.

    Reference implementation:
    `https://captum.ai/api/input_x_gradient.html <https://captum.ai/api/input_x_gradient.html>`__.
    """

    method_name = "input_x_gradient"

    def __init__(self, attribution_model):
        super().__init__(attribution_model)
        self.method = InputXGradient(self.attribution_model)


class SaliencyAttribution(GradientAttributionRegistry):
    """Saliency attribution method.

    Reference implementation:
    `https://captum.ai/api/saliency.html <https://captum.ai/api/saliency.html>`__.
    """

    method_name = "saliency"

    def __init__(self, attribution_model):
        super().__init__(attribution_model)
        self.method = Saliency(self.attribution_model)


# Layer methods


class LayerIntegratedGradientsAttribution(GradientAttributionRegistry):
    """Layer Integrated Gradients attribution method.

    Reference implementation:
    `https://captum.ai/api/layer.html#layer-integrated-gradients <https://captum.ai/api/layer.html#layer-integrated-gradients>`__.
    """  # noqa E501

    method_name = "layer_integrated_gradients"

    def __init__(self, attribution_model, multiply_by_inputs: bool = True, **kwargs):
        super().__init__(attribution_model, hook_to_model=False)
        self.attribute_batch_ids = True
        self.forward_batch_embeds = False
        self.use_baseline = True
        self.hook(**kwargs)
        self.method = LayerIntegratedGradients(
            self.attribution_model,
            self.target_layer,
            multiply_by_inputs=multiply_by_inputs,
        )


class LayerGradientXActivationAttribution(GradientAttributionRegistry):
    """Layer Integrated Gradients attribution method.

    Reference implementation:
    `https://captum.ai/api/layer.html#layer-gradient-x-activation <https://captum.ai/api/layer.html#layer-gradient-x-activation>`__.
    """  # noqa E501

    method_name = "layer_gradient_x_activation"

    def __init__(self, attribution_model, multiply_by_inputs: bool = True, **kwargs):
        super().__init__(attribution_model, hook_to_model=False)
        self.attribute_batch_ids = True
        self.forward_batch_embeds = False
        self.use_baseline = False
        self.hook(**kwargs)
        self.method = LayerGradientXActivation(
            self.attribution_model,
            self.target_layer,
            multiply_by_inputs=multiply_by_inputs,
        )


class LayerDeepLiftAttribution(GradientAttributionRegistry):
    """Layer DeepLIFT attribution method.

    Reference implementation:
    `https://captum.ai/api/layer.html#layer-deeplift <https://captum.ai/api/layer.html#layer-deeplift>`__.
    """

    method_name = "layer_deeplift"

    def __init__(self, attribution_model, multiply_by_inputs: bool = True, **kwargs):
        super().__init__(attribution_model, hook_to_model=False)
        self.attribute_batch_ids = True
        self.forward_batch_embeds = False
        self.use_baseline = True
        self.hook(**kwargs)
        self.method = LayerDeepLift(
            self.attribution_model,
            self.target_layer,
            multiply_by_inputs=multiply_by_inputs,
        )
