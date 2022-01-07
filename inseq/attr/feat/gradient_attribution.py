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

from captum.attr import (
    DeepLift,
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    LayerDeepLift,
    LayerGradientXActivation,
    LayerIntegratedGradients,
    Saliency,
)

from ...data import EncoderDecoderBatch, FeatureAttributionStepOutput
from ...utils import Registry, extract_signature_args, pretty_tensor, rgetattr, sum_normalize
from ...utils.typing import TargetIdsTensor
from ..attribution_decorators import set_hook, unset_hook
from .feature_attribution import FeatureAttribution
from .ops import DiscretetizedIntegratedGradients


logger = logging.getLogger(__name__)


class GradientAttribution(FeatureAttribution, Registry):
    r"""Gradient-based attribution method registry."""

    @set_hook
    def hook(self, **kwargs):
        r"""
        Hooks the attribution method to the model by replacing normal :obj:`nn.Embedding`
        with Captum's `InterpretableEmbeddingBase <https://captum.ai/api/utilities.html#captum.attr.InterpretableEmbeddingBase>`__.
        """  # noqa: E501
        if self.is_layer_attribution:
            self.target_layer = kwargs.pop("target_layer", self.attribution_model.get_embedding_layer())
            logger.debug(f"target_layer={self.target_layer}")
            if isinstance(self.target_layer, str):
                self.target_layer = rgetattr(self.attribution_model.model, self.target_layer)
        # For now only encoder attribution is supported
        self.attribution_model.configure_interpretable_embeddings(do_encoder=not self.is_layer_attribution)

    @unset_hook
    def unhook(self, **kwargs):
        r"""
        Unhook the attribution method by restoring the model's original embeddings.
        """
        if self.is_layer_attribution:
            self.target_layer = None
        self.attribution_model.remove_interpretable_embeddings(do_encoder=not self.is_layer_attribution)

    def attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TargetIdsTensor,
        **kwargs,
    ) -> FeatureAttributionStepOutput:
        r"""
        Performs a single attribution step for the specified target_ids,
        given sources and targets in the batch.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch of sequences on which attribution is performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size)` corresponding to tokens
                for which the attribution step must be performed.
            kwargs: Additional keyword arguments to pass to the attribution step.

        Returns:
            :obj:`FeatureAttributionStepOutput`: A tuple containing a tensor of attributions
                of size `(batch_size, source_length)` and possibly a tensor of attribution deltas
                of size `(batch_size)`, if the attribution step supports deltas and they are requested.
        """
        attribute_args = self.format_attribute_args(batch, target_ids, **kwargs)
        logger.debug(f"batch: {batch},\ntarget_ids: {pretty_tensor(target_ids, lpad=4)}")
        attr = self.method.attribute(**attribute_args)
        delta = None
        if (
            attribute_args.get("return_convergence_delta", False)
            and hasattr(self.method, "has_convergence_delta")
            and self.method.has_convergence_delta()
        ):
            attr, delta = attr
        logger.debug(f"attributions prenorm: {pretty_tensor(attr)}, summed: {attr.sum(dim=-1).squeeze(0)}\n")
        attr = sum_normalize(attr, dim_sum=-1)
        logger.debug(f"attributions: {pretty_tensor(attr)}\n" + "-" * 30)
        return (attr.detach().cpu(), delta.detach().cpu() if delta is not None else None)


class DeepLiftAttribution(GradientAttribution):
    """DeepLIFT attribution method.

    Reference implementation:
    `https://captum.ai/api/deep_lift.html <https://captum.ai/api/deep_lift.html>`__.
    """

    method_name = "deeplift"

    def __init__(self, attribution_model, **kwargs):
        from ...models import HookableModelWrapper

        super().__init__(attribution_model)
        multiply_by_inputs = kwargs.pop("multiply_by_inputs", True)
        self.method = DeepLift(HookableModelWrapper(self.attribution_model), multiply_by_inputs)
        self.use_baseline = True


class DiscretizedIntegratedGradientsAttribution(GradientAttribution):
    """Discretized Integrated Gradients attribution method
    Reference: https://arxiv.org/abs/2108.13654
    Original implementation: https://github.com/INK-USC/DIG
    """

    method_name = "discretized_integrated_gradients"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model, hook_to_model=False)
        multiply_by_inputs = kwargs.pop("multiply_by_inputs", True)
        self.attribution_model = attribution_model
        self.method = DiscretetizedIntegratedGradients(
            self.attribution_model.score_func,
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
            embedding_scaling=self.attribution_model.encoder_embed_scale,
            **load_kwargs,
        )
        super().hook(**other_kwargs)

    def format_attribute_args(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TargetIdsTensor,
        **kwargs,
    ) -> FeatureAttributionStepOutput:
        scaled_inputs = self.method.path_builder.scale_inputs(
            batch.sources.input_ids,
            batch.sources.baseline_ids,
            n_steps=kwargs.get("n_steps", None),
            scale_strategy=kwargs.get("strategy", None),
        )
        attribute_args = {
            "inputs": scaled_inputs,
            "target": target_ids,
            "additional_forward_args": (
                batch.sources.attention_mask,
                batch.targets.input_embeds,
                batch.targets.attention_mask,
                False,
            ),
        }
        return {**attribute_args, **kwargs}


class IntegratedGradientsAttribution(GradientAttribution):
    """Integrated Gradients attribution method.

    Reference implementation:
    `https://captum.ai/api/integrated_gradients.html <https://captum.ai/api/integrated_gradients.html>`__.
    """

    method_name = "integrated_gradients"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        multiply_by_inputs = kwargs.pop("multiply_by_inputs", True)
        self.method = IntegratedGradients(self.attribution_model.score_func, multiply_by_inputs)
        self.use_baseline = True


class InputXGradientAttribution(GradientAttribution):
    """Input x Gradient attribution method.

    Reference implementation:
    `https://captum.ai/api/input_x_gradient.html <https://captum.ai/api/input_x_gradient.html>`__.
    """

    method_name = "input_x_gradient"

    def __init__(self, attribution_model):
        super().__init__(attribution_model)
        self.method = InputXGradient(self.attribution_model.score_func)


class SaliencyAttribution(GradientAttribution):
    """Saliency attribution method.

    Reference implementation:
    `https://captum.ai/api/saliency.html <https://captum.ai/api/saliency.html>`__.
    """

    method_name = "saliency"

    def __init__(self, attribution_model):
        super().__init__(attribution_model)
        self.method = Saliency(self.attribution_model.score_func)


class GradientShapAttribution(GradientAttribution):
    """GradientShap attribution method.

    Reference implementation:
    `https://captum.ai/api/gradient_shap.html <https://captum.ai/api/gradient_shap.html>`__.
    """

    method_name = "gradient_shap"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        super().__init__(attribution_model)
        multiply_by_inputs = kwargs.pop("multiply_by_inputs", True)
        self.method = GradientShap(self.attribution_model.score_func, multiply_by_inputs)
        self.use_baseline = True


# Layer methods


class LayerIntegratedGradientsAttribution(GradientAttribution):
    """Layer Integrated Gradients attribution method.

    Reference implementation:
    `https://captum.ai/api/layer.html#layer-integrated-gradients <https://captum.ai/api/layer.html#layer-integrated-gradients>`__.
    """  # noqa E501

    method_name = "layer_integrated_gradients"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model, hook_to_model=False)
        self.is_layer_attribution = True
        self.use_baseline = True
        self.hook(**kwargs)
        multiply_by_inputs = kwargs.pop("multiply_by_inputs", True)
        self.method = LayerIntegratedGradients(
            self.attribution_model.score_func,
            self.target_layer,
            multiply_by_inputs=multiply_by_inputs,
        )


class LayerGradientXActivationAttribution(GradientAttribution):
    """Layer Integrated Gradients attribution method.

    Reference implementation:
    `https://captum.ai/api/layer.html#layer-integrated-gradients <https://captum.ai/api/layer.html#layer-integrated-gradients>`__.
    """  # noqa E501

    method_name = "layer_gradient_x_activation"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model, hook_to_model=False)
        self.is_layer_attribution = True
        self.use_baseline = False
        self.hook(**kwargs)
        multiply_by_inputs = kwargs.pop("multiply_by_inputs", True)
        self.method = LayerGradientXActivation(
            self.attribution_model.score_func,
            self.target_layer,
            multiply_by_inputs=multiply_by_inputs,
        )


class LayerDeepLiftAttribution(GradientAttribution):
    """Layer DeepLIFT attribution method.

    Reference implementation:
    `https://captum.ai/api/layer.html#layer-deeplift <https://captum.ai/api/layer.html#layer-deeplift>`__.
    """  # noqa E501

    method_name = "layer_deeplift"

    def __init__(self, attribution_model, **kwargs):
        from ...models import HookableModelWrapper

        super().__init__(attribution_model, hook_to_model=False)
        self.is_layer_attribution = True
        self.use_baseline = True
        self.hook(**kwargs)
        multiply_by_inputs = kwargs.pop("multiply_by_inputs", True)
        self.method = LayerDeepLift(
            HookableModelWrapper(self.attribution_model),
            self.target_layer,
            multiply_by_inputs=multiply_by_inputs,
        )
