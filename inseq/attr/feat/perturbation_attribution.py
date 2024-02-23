import logging
from typing import Any

from captum.attr import Occlusion

from ...data import (
    CoarseFeatureAttributionStepOutput,
    GranularFeatureAttributionStepOutput,
    MultiDimensionalFeatureAttributionStepOutput,
)
from ...utils import Registry
from .attribution_utils import get_source_target_attributions
from .gradient_attribution import FeatureAttribution
from .ops import Lime, ValueZeroing

logger = logging.getLogger(__name__)


class PerturbationAttributionRegistry(FeatureAttribution, Registry):
    """Perturbation-based attribution method registry."""

    pass


class OcclusionAttribution(PerturbationAttributionRegistry):
    """Occlusion-based attribution method.
    Reference implementation:
    `https://captum.ai/api/occlusion.html <https://captum.ai/api/occlusion.html>`__.

    Usage in other implementations:
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
        self.use_baselines = True
        self.method = Occlusion(self.attribution_model)

    def attribute_step(
        self,
        attribute_fn_main_args: dict[str, Any],
        attribution_args: dict[str, Any] = {},
    ) -> CoarseFeatureAttributionStepOutput:
        r"""Sliding window shapes is defined as a tuple.
        First entry is between 1 and length of input.
        Second entry is given by the embedding dimension of the underlying model.
        If not explicitly given via attribution_args, the default is (1, embedding_dim).
        """
        if "sliding_window_shapes" not in attribution_args:
            embedding_layer = self.attribution_model.get_embedding_layer()
            attribution_args["sliding_window_shapes"] = tuple(
                (1, embedding_layer.embedding_dim) for _ in range(len(attribute_fn_main_args["inputs"]))
            )
            if len(attribution_args["sliding_window_shapes"]) == 1:
                attribution_args["sliding_window_shapes"] = attribution_args["sliding_window_shapes"][0]

        attr = self.method.attribute(**attribute_fn_main_args, **attribution_args)
        source_attributions, target_attributions = get_source_target_attributions(
            attr, self.attribution_model.is_encoder_decoder
        )

        # Make sure that the computed attributions are the same for every "embedding slice"
        attr = source_attributions if source_attributions is not None else target_attributions
        embedding_attributions = [attr[:, :, i].tolist()[0] for i in range(attr.shape[2])]
        assert all(x == embedding_attributions[0] for x in embedding_attributions)

        # Access the first embedding slice, provided it's the same result as the other slices
        if source_attributions is not None:
            source_attributions = source_attributions[:, :, 0].abs()
        if target_attributions is not None:
            target_attributions = target_attributions[:, :, 0].abs()

        return CoarseFeatureAttributionStepOutput(
            source_attributions=source_attributions.to("cpu") if source_attributions is not None else None,
            target_attributions=target_attributions.to("cpu") if target_attributions is not None else None,
        )


class LimeAttribution(PerturbationAttributionRegistry):
    """LIME-based attribution method.
    Reference implementations:
    `https://captum.ai/api/lime.html <https://captum.ai/api/lime.html>`__.
    `https://github.com/DFKI-NLP/thermostat/ <https://github.com/DFKI-NLP/thermostat/>`__.
    `https://github.com/copenlu/ALPS_2021 <https://github.com/copenlu/ALPS_2021>`__.

    The main part of the code is in Lime of ops/lime.py.
    """

    method_name = "lime"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model)
        self.method = Lime(attribution_model=self.attribution_model, **kwargs)

    def attribute_step(
        self,
        attribute_fn_main_args: dict[str, Any],
        attribution_args: dict[str, Any] = {},
    ) -> GranularFeatureAttributionStepOutput:
        if len(attribute_fn_main_args["inputs"]) > 1:
            # Captum's `_evaluate_batch` function for LIME does not account for multiple inputs when encoder-decoder
            # models and attribute_target=True are used. The model output is of length two and if the inputs are either
            # of length one (list containing a tuple) or of length two (tuple unpacked from the list), an error is
            # raised. A workaround will be added soon.
            raise NotImplementedError(
                "LIME attribution with attribute_target=True currently not supported for encoder-decoder models."
            )
        out = super().attribute_step(attribute_fn_main_args, attribution_args)
        return GranularFeatureAttributionStepOutput(
            source_attributions=out.source_attributions,
            target_attributions=out.target_attributions,
            sequence_scores=out.sequence_scores,
        )


class ValueZeroingAttribution(PerturbationAttributionRegistry):
    """Value Zeroing attribution method."""

    method_name = "value_zeroing"

    def __init__(self, attribution_model, **kwargs):
        super().__init__(attribution_model, hook_to_model=False)
        # Hidden states will be passed to the attribute_step method
        self.use_hidden_states = True
        # Does not rely on predicted output (i.e. decoding strategy agnostic)
        self.use_predicted_target = False
        # Uses model configuration to access attention module and value vector variable
        self.use_model_config = True
        # Needs only the final generation step to extract scores
        self.is_final_step_method = True
        self.method = ValueZeroing(attribution_model)
        self.hook(**kwargs)

    def attribute_step(
        self,
        attribute_fn_main_args: dict[str, Any],
        attribution_args: dict[str, Any] = {},
    ) -> MultiDimensionalFeatureAttributionStepOutput:
        attr = self.method.attribute(**attribute_fn_main_args, **attribution_args)
        source_attributions, target_attributions = get_source_target_attributions(
            attr, self.attribution_model.is_encoder_decoder
        )
        return MultiDimensionalFeatureAttributionStepOutput(
            source_attributions=source_attributions,
            target_attributions=target_attributions,
            _num_dimensions=1,  # num_layers
        )
