import logging

from captum.attr import Occlusion

from ...data import EncoderDecoderBatch, FeatureAttributionStepOutput
from ...utils import Registry, pretty_tensor
from ...utils.typing import TargetIdsTensor
from ..attribution_decorators import set_hook, unset_hook
from .gradient_attribution import FeatureAttribution


logger = logging.getLogger(__name__)


class OcclusionRegistry(FeatureAttribution, Registry):
    """Occlusion-based attribution methods."""

    @set_hook
    def hook(self, **kwargs):
        pass

    @unset_hook
    def unhook(self, **kwargs):
        pass

    def attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TargetIdsTensor,
        attribute_target: bool = False,
        **kwargs,
    ) -> FeatureAttributionStepOutput:
        r"""Performs a single attribution step for the specified target_ids,
        given sources and targets in the batch.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch of sequences on which attribution is performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size)` corresponding to tokens
                for which the attribution step must be performed.
            attribute_target (:obj:`bool`, optional): Whether to attribute the target prefix or not. Defaults to False.
            kwargs: Additional keyword arguments to pass to the attribution step.
        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: A dataclass containing a tensor of source
                and target attributions.
        """

        attribute_args = self.format_attribute_args(batch, target_ids, attribute_target, **kwargs)
        logger.debug(f"batch: {batch},\ntarget_ids: {pretty_tensor(target_ids, lpad=4)}")
        attr = self.method.attribute(
            **attribute_args, sliding_window_shapes=self.sliding_window_shapes, show_progress=True
        )
        return FeatureAttributionStepOutput(
            source_attributions=attr if not isinstance(attr, tuple) else attr[0],
            target_attributions=None
            if not isinstance(attr, tuple) or (isinstance(attr, tuple) and len(attr) == 1)
            else attr[1],
        )


class OcclusionAttribution(OcclusionRegistry):
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
        self.use_baseline = True
        self.sliding_window_shapes = (1, 1)
