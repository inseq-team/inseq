import logging
from typing import Any, Dict

from captum.attr import GradientShap, Occlusion

from ...data import PerturbationFeatureAttributionStepOutput
from ...utils import Registry
from ..attribution_decorators import set_hook, unset_hook
from .attribution_utils import get_source_target_attributions
from .gradient_attribution import FeatureAttribution
from .ops.lime import Lime

logger = logging.getLogger(__name__)


class PerturbationAttributionRegistry(FeatureAttribution, Registry):
    """Perturbation-based attribution method registry."""

    @set_hook
    def hook(self, **kwargs):
        pass

    @unset_hook
    def unhook(self, **kwargs):
        pass


class OcclusionAttribution(PerturbationAttributionRegistry):
    """Occlusion-based attribution method.
    Reference implementation:
    `https://captum.ai/api/occlusion.html <https://captum.ai/api/occlusion.html>`__.

    Usages in other implementations:
    `niuzaisheng/AttExplainer <https://github.com/niuzaisheng/AttExplainer/blob/main/baseline_methods/\
    explain_baseline_captum.py>`__
    `andrewPoulton/explainable-asag <https://github.com/andrewPoulton/explainable-asag/blob/main/explanation.py>`__
    `copenlu/xai-benchmark <https://github.com/copenlu/xai-benchmark/blob/master/saliency_gen/\
    interpret_grads_occ.py>`__
    `DFKI-NLP/thermostat <https://github.com/DFKI-NLP/thermostat/blob/main/src/thermostat/explainers/occlusion.py>`__
    """

    method_name = "occlusion"

    def __init__(self, attribution_model):
        super().__init__(attribution_model)
        self.is_layer_attribution = False  # FIXME: Is this necessary?
        self.method = Occlusion(self.attribution_model)

    def attribute_step(
        self,
        attribute_fn_main_args: Dict[str, Any],
        attribution_args: Dict[str, Any] = {},
    ) -> PerturbationFeatureAttributionStepOutput:
        if "sliding_window_shapes" not in attribution_args:
            # Sliding window shapes is defined as a tuple
            # First entry is between 1 and length of input
            # Second entry is given by the embedding dimension of the underlying model
            # If not explicitly given via attribution_args, the default is (1, embedding_dim)
            embedding_layer = self.attribution_model.get_embedding_layer()
            attribution_args["sliding_window_shapes"] = (1, embedding_layer.embedding_dim)

        attr = self.method.attribute(
            **attribute_fn_main_args,
            **attribution_args,
        )

        source_attributions, target_attributions = get_source_target_attributions(
            attr, self.attribution_model.is_encoder_decoder
        )
        return PerturbationFeatureAttributionStepOutput(
            source_attributions=source_attributions,
            target_attributions=target_attributions,
        )


class LimeAttribution(PerturbationAttributionRegistry):
    """LIME-based attribution method.
    Reference implementations:
    `https://captum.ai/api/lime.html <https://captum.ai/api/lime.html>`__.
    `https://github.com/DFKI-NLP/thermostat/ <https://github.com/DFKI-NLP/thermostat/>`__.
    `https://github.com/copenlu/ALPS_2021 <https://github.com/copenlu/ALPS_2021>`__.
    """

    method_name = "lime"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        self.method = Lime(attribution_model=self.attribution_model, **kwargs)

    def attribute_step(
        self,
        attribute_fn_main_args: Dict[str, Any],
        attribution_args: Dict[str, Any] = {},
    ) -> PerturbationFeatureAttributionStepOutput:
        attr = self.method.attribute(
            **attribute_fn_main_args,
            **attribution_args,
        )

        source_attributions, target_attributions = get_source_target_attributions(
            attr, self.attribution_model.is_encoder_decoder
        )
        return PerturbationFeatureAttributionStepOutput(
            source_attributions=source_attributions,
            target_attributions=target_attributions,
        )


class ShapAttribution(PerturbationAttributionRegistry):
    """SHAP-based attribution method.
    Reference implementation:
    `https://captum.ai/api/gradient_shap.html <https://captum.ai/api/gradient_shap.html>`__.
    """

    method_name = "shap"

    def __init__(self, attribution_model, multiply_by_inputs: bool = True, **kwargs):
        super().__init__(attribution_model)
        self.use_baseline = True
        self.method = GradientShap(attribution_model, multiply_by_inputs=multiply_by_inputs)

    def attribute_step(
        self,
        attribute_fn_main_args: Dict[str, Any],
        attribution_args: Dict[str, Any] = {},
    ) -> PerturbationFeatureAttributionStepOutput:
        attr = self.method.attribute(
            **attribute_fn_main_args,
            **attribution_args,
        )

        source_attributions, target_attributions = get_source_target_attributions(
            attr, self.attribution_model.is_encoder_decoder
        )
        return PerturbationFeatureAttributionStepOutput(
            source_attributions=source_attributions,
            target_attributions=target_attributions,
        )
