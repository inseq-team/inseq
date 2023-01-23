import logging
from functools import partial
from typing import Any, Dict

import torch
from captum._utils.models.linear_model import SkLearnLinearModel
from captum.attr import GradientShap, LimeBase, Occlusion

from ...data import PerturbationFeatureAttributionStepOutput
from ...utils import Registry
from ..attribution_decorators import set_hook, unset_hook
from .attribution_utils import get_source_target_attributions
from .gradient_attribution import FeatureAttribution

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

    def __init__(self, attribution_model):
        super().__init__(attribution_model)
        self.method = LimeBase(
            self.attribution_model,
            interpretable_model=SkLearnLinearModel("linear_model.Ridge"),  # TODO: Make dynamic
            similarity_func=self.token_similarity_kernel,
            perturb_func=partial(
                self.perturb_func,
            ),
            perturb_interpretable_space=False,
            from_interp_rep_transform=None,
            to_interp_rep_transform=self.to_interp_rep_transform,
        )

    @staticmethod
    def token_similarity_kernel(
        original_input: tuple,
        perturbed_input: tuple,
        perturbed_interpretable_input: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        original_input_tensor = original_input[0]
        perturbed_input_tensor = perturbed_input[0]
        assert original_input_tensor.shape == perturbed_input_tensor.shape
        similarity = torch.sum(original_input_tensor == perturbed_input_tensor) / len(original_input_tensor)
        return similarity

    def perturb_func(
        self,
        original_input: tuple,  # always needs to be last argument before **kwargs due to "partial"
        **kwargs: Any,
    ) -> tuple:
        """
        Sampling function
        """
        original_input_tensor = original_input[0]
        mask = torch.randint(low=0, high=2, size=original_input_tensor.size()).to(self.attribution_model.device)
        perturbed_input = original_input_tensor * mask + (1 - mask) * self.attribution_model.tokenizer.pad_token_id
        perturbed_input_tuple = tuple({perturbed_input})
        return perturbed_input_tuple

    @staticmethod
    def to_interp_rep_transform(sample, original_input, **kwargs: Any):
        return sample

    def attribute_step(
        self,
        attribute_fn_main_args: Dict[str, Any],
        attribution_args: Dict[str, Any] = {},
    ) -> PerturbationFeatureAttributionStepOutput:
        """Run on each example in a batch at a time
        LimeBase does not accept attribution for more than one example at a time:
        https://github.com/pytorch/captum/issues/905#issuecomment-1075384565#
        """
        attrs = []
        for b, _batch in enumerate(attribute_fn_main_args["inputs"][0]):
            single_input = tuple(
                [inp[b] if type(inp) == torch.Tensor else inp for inp in attribute_fn_main_args["inputs"]]
            )
            single_additional_forward_args = tuple(
                [
                    arg[b] if type(arg) == torch.Tensor else arg
                    for arg in attribute_fn_main_args["additional_forward_args"]
                ]
            )
            single_attribute_fn_main_args = {
                "inputs": single_input,
                "additional_forward_args": single_additional_forward_args,
            }

            single_attr = self.method.attribute(
                **single_attribute_fn_main_args,
                **attribution_args,
            )
            attrs.append(single_attr)
        attr = torch.stack(list(attrs), dim=0)

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
