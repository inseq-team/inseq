from typing import Any, Dict

import logging

from captum.attr import Occlusion

from ...data import PerturbationFeatureAttributionStepOutput
from ...utils import Registry
from ..attribution_decorators import set_hook, unset_hook
from .attribution_utils import get_source_target_attributions
from .gradient_attribution import FeatureAttribution


logger = logging.getLogger(__name__)


class PerturbationMethodRegistry(FeatureAttribution, Registry):
    """Occlusion-based attribution methods."""

    @set_hook
    def hook(self, **kwargs):
        pass

    @unset_hook
    def unhook(self, **kwargs):
        pass


class OcclusionAttribution(PerturbationMethodRegistry):
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

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        self.is_layer_attribution = False
        self.method = Occlusion(self.attribution_model)

    def attribute_step(
        self,
        attribute_fn_main_args: Dict[str, Any],
        attribution_args: Dict[str, Any] = {},
    ) -> Any:

        if "sliding_window_shapes" not in attribution_args:
            # Sliding window shapes is defined as a tuple
            # First entry is between 1 and length of input
            # Second entry is given by the max length of the underlying model
            # If not explicitly given via attribution_args, the default is (1, model_max_length)
            attribution_args["sliding_window_shapes"] = (1, self.attribution_model.model_max_length)

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
