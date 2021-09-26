import logging
from collections import OrderedDict

from captum.attr import LRP, GradientShap, InputXGradient, IntegratedGradients, Saliency
from torch import long
from torchtyping import TensorType

from ...data import EncoderDecoderBatch, FeatureAttributionOutput
from ...utils import Registry, pretty_tensor, sum_normalize
from ..attribution_decorators import set_hook, unset_hook
from .feature_attribution import FeatureAttribution

logger = logging.getLogger(__name__)


class GradientAttribution(FeatureAttribution, Registry):
    """Gradient-based attribution method."""

    @set_hook
    def hook(self):
        self.attribution_model.configure_interpretable_embeddings()

    @unset_hook
    def unhook(self):
        self.attribution_model.remove_interpretable_embeddings()

    def attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", long],
        **kwargs,
    ) -> FeatureAttributionOutput:
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
        return FeatureAttributionOutput(attributions=attr, delta=delta)


class IntegratedGradientsAttribution(GradientAttribution):
    """Integrated Gradients attribution method."""

    method_name = "integrated_gradients"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        self.method = IntegratedGradients(self.attribution_model.score_func, **kwargs)
        self.use_baseline = True


class InputXGradientAttribution(GradientAttribution):
    """Input x Gradient attribution method."""

    method_name = "input_x_gradient"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        self.method = InputXGradient(self.attribution_model.score_func)
        self.use_baseline = False


class SaliencyAttribution(GradientAttribution):
    """Saliency attribution method."""

    method_name = "saliency"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        self.method = Saliency(self.attribution_model.score_func)
        self.use_baseline = False
