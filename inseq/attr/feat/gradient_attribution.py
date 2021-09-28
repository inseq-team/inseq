import logging

from captum.attr import InputXGradient, IntegratedGradients, Saliency
from torchtyping import TensorType

from ...data import EncoderDecoderBatch, FeatureAttributionStepOutput
from ...utils import Registry, extract_signature_args, pretty_tensor, sum_normalize
from ..attribution_decorators import set_hook, unset_hook
from .feature_attribution import FeatureAttribution
from .ops import DiscretetizedIntegratedGradients, MonotonicPathBuilder

logger = logging.getLogger(__name__)


class GradientAttribution(FeatureAttribution, Registry):
    """Gradient-based attribution method."""

    @set_hook
    def hook(self, **kwargs):
        self.attribution_model.configure_interpretable_embeddings()

    @unset_hook
    def unhook(self, **kwargs):
        self.attribution_model.remove_interpretable_embeddings()

    def attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", int],
        **kwargs,
    ) -> FeatureAttributionStepOutput:
        """Attribute a single step."""
        logger.debug(f"batch: {batch},\ntarget_ids: {pretty_tensor(target_ids)}")
        delta = kwargs.get("return_convergence_delta", None)
        attribute_args = {
            "inputs": batch.sources.input_embeds,
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
        multiply_by_inputs = kwargs.pop("multiply_by_inputs", True)
        super().__init__(attribution_model, **kwargs)
        self.method = DiscretetizedIntegratedGradients(
            self.attribution_model.score_func, multiply_by_inputs
        )
        self.use_baseline = True
        self.skip_eos = True
        self.path_builder = None

    @set_hook
    def hook(self, **kwargs):
        load_args = extract_signature_args(kwargs, MonotonicPathBuilder.load)
        self.path_builder = MonotonicPathBuilder.load(
            self.attribution_model.model_name,
            self.attribution_model.token_embeddings.detach(),
            **load_args,
        )
        super().hook()

    def attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", int],
        **kwargs,
    ) -> FeatureAttributionStepOutput:
        scale_inputs_args = extract_signature_args(
            kwargs, self.path_builder.scale_inputs
        )
        batch.sources.input_embeds = self.path_builder.scale_inputs(
            batch.sources.input_ids,
            batch.sources.baseline_ids,
            special_token_ids=self.attribution_model.special_tokens_ids,
            **scale_inputs_args,
        )
        attribution_args = {k: kwargs[k] for k in set(kwargs) - set(scale_inputs_args)}
        super().attribute_step(batch, target_ids, **attribution_args)


class IntegratedGradientsAttribution(GradientAttribution):
    """Integrated Gradients attribution method."""

    method_name = "integrated_gradients"

    def __init__(self, attribution_model, **kwargs):
        multiply_by_inputs = kwargs.pop("multiply_by_inputs", True)
        super().__init__(attribution_model)
        self.method = IntegratedGradients(
            self.attribution_model.score_func, multiply_by_inputs
        )
        self.use_baseline = True
        self.skip_eos = True


class InputXGradientAttribution(GradientAttribution):
    """Input x Gradient attribution method."""

    method_name = "input_x_gradient"

    def __init__(self, attribution_model):
        super().__init__(attribution_model)
        self.method = InputXGradient(self.attribution_model.score_func)
        self.use_baseline = False


class SaliencyAttribution(GradientAttribution):
    """Saliency attribution method."""

    method_name = "saliency"

    def __init__(self, attribution_model):
        super().__init__(attribution_model)
        self.method = Saliency(self.attribution_model.score_func)
        self.use_baseline = False
