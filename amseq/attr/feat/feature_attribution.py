from typing import List, NoReturn, Optional, Tuple

import logging
from abc import abstractmethod
from inspect import signature

from rich.progress import track
from torch import long
from torchtyping import TensorType

from ...data import (
    Batch,
    BatchEmbedding,
    BatchEncoding,
    EncoderDecoderBatch,
    FeatureAttributionInput,
    FeatureAttributionOutput,
    FeatureAttributionSequenceOutput,
    ModelIdentifier,
    OneOrMoreFeatureAttributionSequenceOutputs,
)
from ...utils import (
    Registry,
    UnknownAttributionMethodError,
    pretty_tensor,
    remap_from_filtered,
)
from ..attribution_decorators import set_hook, unset_hook

logger = logging.getLogger(__name__)


class FeatureAttribution(Registry):
    """Abstract base class for attribution methods."""

    attr = "method_name"
    ignore_extra_args = ["inputs", "baselines", "target", "additional_forward_args"]

    def __init__(self, attribution_model, **kwargs):
        super().__init__()
        self.attribution_model = attribution_model
        self.hook()

    @classmethod
    def load(
        cls,
        method_name: str,
        model_name_or_path: ModelIdentifier = None,
        attribution_model=None,
        **kwargs,
    ) -> "FeatureAttribution":
        from amseq import AttributionModel

        if model_name_or_path and attribution_model:
            raise OSError(
                "Only one among an initialized model and a model identifier "
                "can be defined at once when loading the attribution method."
            )
        if model_name_or_path:
            # The model is loaded with default args
            attribution_model = AttributionModel.load(model_name_or_path)
        methods = cls.available_classes()
        if method_name not in methods:
            raise UnknownAttributionMethodError(method_name)
        return methods[method_name](attribution_model, **kwargs)

    def prepare_and_attribute(
        self,
        sources: FeatureAttributionInput,
        targets: FeatureAttributionInput,
        attr_pos_start: Optional[int] = 0,
        attr_pos_end: Optional[int] = None,
        **kwargs,
    ) -> OneOrMoreFeatureAttributionSequenceOutputs:
        batch = self.prepare(sources, targets, **kwargs)
        return self.attribute(
            batch, attr_pos_start=attr_pos_start, attr_pos_end=attr_pos_end, **kwargs
        )

    def prepare(
        self,
        sources: FeatureAttributionInput,
        targets: FeatureAttributionInput,
        **kwargs,
    ) -> EncoderDecoderBatch:
        if isinstance(sources, str) or isinstance(sources, list):
            sources: BatchEncoding = self.attribution_model.encode_texts(
                sources, return_baseline=True
            )
        if isinstance(sources, BatchEncoding):
            embeds = BatchEmbedding(
                input_embeds=self.attribution_model.encoder_embed(sources.input_ids),
                baseline_embeds=self.attribution_model.encoder_embed(
                    sources.baseline_ids
                ),
            )
            sources = Batch.from_encoding_embeds(sources, embeds)
        if isinstance(targets, str) or isinstance(targets, list):
            prepend_bos_token = kwargs.pop("prepend_bos_token", True)
            targets: BatchEncoding = self.attribution_model.encode_texts(
                targets,
                as_targets=True,
                prepend_bos_token=prepend_bos_token,
                return_baseline=True,
            )
        if isinstance(targets, BatchEncoding):
            target_embeds = BatchEmbedding(
                input_embeds=self.attribution_model.decoder_embed(targets.input_ids),
                baseline_embeds=self.attribution_model.decoder_embed(
                    targets.baseline_ids
                ),
            )
            targets = Batch.from_encoding_embeds(targets, target_embeds)
        return EncoderDecoderBatch(sources, targets)

    def attribute(
        self,
        batch: EncoderDecoderBatch,
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
        **kwargs,
    ) -> OneOrMoreFeatureAttributionSequenceOutputs:
        max_generated_length = batch.targets.input_ids.shape[1]
        attr_pos_start, attr_pos_end = self.check_attribute_positions(
            max_generated_length,
            attr_pos_start,
            attr_pos_end,
        )
        attribution_steps = track(
            range(attr_pos_start, attr_pos_end), description="Attributing..."
        )
        logger.debug(f"full batch: {batch}")
        attribution_outputs: List[FeatureAttributionOutput] = [
            self.get_attribution_output(
                batch[:step],
                target_ids=batch.targets.input_ids[:, step].unsqueeze(1),
                target_attention_mask=batch.targets.attention_mask[:, step].unsqueeze(
                    1
                ),
                **kwargs,
            )
            for step in attribution_steps
        ]
        return FeatureAttributionSequenceOutput.from_attributions(attribution_outputs)

    @staticmethod
    def check_attribute_positions(
        max_length: int,
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
    ) -> Tuple[int, int]:
        if attr_pos_end is None or attr_pos_end > max_length:
            attr_pos_end = max_length
        if attr_pos_start > attr_pos_end or attr_pos_start < 1:
            raise ValueError("Invalid starting position for attribution")
        if attr_pos_start == attr_pos_end:
            raise ValueError("Start and end attribution positions cannot be the same.")
        return attr_pos_start, attr_pos_end

    def get_attribution_output(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", 1, long],
        target_attention_mask: Optional[TensorType["batch_size", 1, long]] = None,
        **kwargs,
    ) -> FeatureAttributionOutput:
        step_output = self.filtered_attribute_step(
            batch, target_ids, target_attention_mask, **kwargs
        )
        step_output = self.add_token_information(step_output, batch, target_ids)
        step_output.fix_attributions()
        step_output.check_consistency()
        return step_output

    def filtered_attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", 1, long],
        target_attention_mask: Optional[TensorType["batch_size", 1, long]] = None,
        **kwargs,
    ) -> FeatureAttributionOutput:
        logger.debug(
            f"target_ids: {pretty_tensor(target_ids)},\n"
            f"target_attention_mask: {pretty_tensor(target_attention_mask)}"
        )
        orig_batch = batch.clone()
        orig_target_ids = target_ids
        if target_attention_mask is not None and target_ids.shape[0] > 1:
            batch = batch.select_active(target_attention_mask)
            target_ids = target_ids.masked_select(target_attention_mask.bool())
            target_ids = target_ids.view(-1, 1)
        step_output = self.attribute_step(batch, target_ids.squeeze(), **kwargs)
        if target_attention_mask is not None and orig_target_ids.shape[0] > 1:
            step_output.attributions = remap_from_filtered(
                source=orig_batch.sources.input_ids,
                mask=target_attention_mask,
                filtered=step_output.attributions,
            )
            if step_output.delta is not None:
                step_output.delta = remap_from_filtered(
                    source=target_attention_mask.squeeze(),
                    mask=target_attention_mask,
                    filtered=step_output.delta,
                )
        return step_output

    def get_attribution_args(self, **kwargs):
        if hasattr(self, "method") and hasattr(self.method, "attribute"):
            return {
                k: v
                for k, v in kwargs.items()
                if k in signature(self.method.attribute).parameters
                and k not in self.ignore_extra_args
            }
        return {}

    def add_token_information(
        self,
        attribution_output: FeatureAttributionOutput,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", 1, long],
    ) -> FeatureAttributionOutput:
        source_tokens = self.attribution_model.convert_ids_to_tokens(
            batch.sources.input_ids
        )
        source_ids = self.attribution_model.convert_tokens_to_ids(source_tokens)
        prefix_tokens = self.attribution_model.convert_ids_to_tokens(
            batch.targets.input_ids
        )
        prefix_ids = self.attribution_model.convert_tokens_to_ids(prefix_tokens)
        target_tokens = self.attribution_model.convert_ids_to_tokens(
            target_ids, skip_special_tokens=False
        )
        attribution_output.source_ids = source_ids
        attribution_output.source_tokens = source_tokens
        attribution_output.prefix_ids = prefix_ids
        attribution_output.prefix_tokens = prefix_tokens
        attribution_output.target_ids = target_ids.tolist()
        attribution_output.target_tokens = target_tokens
        return attribution_output

    @abstractmethod
    def attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", long],
        **kwargs,
    ) -> FeatureAttributionOutput:
        pass

    @abstractmethod
    @set_hook
    def hook(self) -> NoReturn:
        pass

    @abstractmethod
    @unset_hook
    def unhook(self) -> NoReturn:
        pass
