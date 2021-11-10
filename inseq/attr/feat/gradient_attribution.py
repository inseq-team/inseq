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

from captum.attr import InputXGradient, IntegratedGradients, Saliency
from torchtyping import TensorType

from ...data import EncoderDecoderBatch, FeatureAttributionStepOutput
from ...utils import Registry, extract_signature_args, pretty_tensor, sum_normalize
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
        self.attribution_model.configure_interpretable_embeddings()

    @unset_hook
    def unhook(self, **kwargs):
        r"""
        Unhook the attribution method by restoring the model's original embeddings.
        """
        self.attribution_model.remove_interpretable_embeddings()

    def attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", int],
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
        logger.debug(f"batch: {batch},\ntarget_ids: {pretty_tensor(target_ids)}")
        delta = kwargs.get("return_convergence_delta", None)
        attribute_args = {
            "inputs": batch.sources.input_embeds if not self.use_full_batch else batch,
            "target": target_ids,
            "additional_forward_args": (
                batch.sources.attention_mask,
                batch.targets.input_embeds,
            ),
        }
        if self.use_baseline:
            attribute_args["baselines"] = (batch.sources.baseline_embeds,)
        attr = self.method.attribute(**attribute_args, **kwargs)
        if (
            delta
            and hasattr(self.method, "has_convergence_delta")
            and self.method.has_convergence_delta()
        ):
            attr, delta = attr
        attr = sum_normalize(attr, dim_sum=-1)
        logger.debug(f"attributions: {pretty_tensor(attr)}")
        return (attr, delta) if delta is not None else attr


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
        self.use_baseline = False
        self.skip_eos = False
        self.use_full_batch = True
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
            token_embeddings=self.attribution_model.token_embeddings.detach(),
            special_tokens=self.attribution_model.special_tokens_ids,
            **load_kwargs,
        )
        super().hook(**other_kwargs)


class IntegratedGradientsAttribution(GradientAttribution):
    """Integrated Gradients attribution method.

    Reference implementation:
    `https://captum.ai/api/integrated_gradients.html <https://captum.ai/api/integrated_gradients.html>`__.
    """

    method_name = "integrated_gradients"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        multiply_by_inputs = kwargs.pop("multiply_by_inputs", True)
        self.method = IntegratedGradients(
            self.attribution_model.score_func, multiply_by_inputs
        )
        self.use_baseline = True
        self.skip_eos = True


class InputXGradientAttribution(GradientAttribution):
    """Input x Gradient attribution method.

    Reference implementation:
    `https://captum.ai/api/input_x_gradient.html <https://captum.ai/api/input_x_gradient.html>`__.
    """

    method_name = "input_x_gradient"

    def __init__(self, attribution_model):
        super().__init__(attribution_model)
        self.method = InputXGradient(self.attribution_model.score_func)
        self.use_baseline = False


class SaliencyAttribution(GradientAttribution):
    """Saliency attribution method.

    Reference implementation:
    `https://captum.ai/api/saliency.html <https://captum.ai/api/saliency.html>`__.
    """

    method_name = "saliency"

    def __init__(self, attribution_model):
        super().__init__(attribution_model)
        self.method = Saliency(self.attribution_model.score_func)
        self.use_baseline = False
