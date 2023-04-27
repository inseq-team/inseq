import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch

from ..attr.feat import join_token_ids
from ..data import (
    BatchEmbedding,
    BatchEncoding,
    DecoderOnlyBatch,
    FeatureAttributionInput,
    FeatureAttributionStepOutput,
    get_batch_from_inputs,
)
from ..utils.typing import (
    AttributionForwardInputs,
    EmbeddingsTensor,
    ExpandedTargetIdsTensor,
    IdsTensor,
    LogitsTensor,
    OneOrMoreTokenSequences,
    SingleScorePerStepTensor,
    TargetIdsTensor,
    TextSequences,
    TokenWithId,
)
from .attribution_model import AttributionModel, ForwardMethod, InputFormatter, ModelOutput

CustomForwardOutput = TypeVar("CustomForwardOutput")

logger = logging.getLogger(__name__)


class DecoderOnlyInputFormatter(InputFormatter):
    @staticmethod
    def prepare_inputs_for_attribution(
        attribution_model: "DecoderOnlyAttributionModel",
        inputs: FeatureAttributionInput,
        include_eos_baseline: bool = False,
    ) -> DecoderOnlyBatch:
        batch = get_batch_from_inputs(
            attribution_model,
            inputs=inputs,
            include_eos_baseline=include_eos_baseline,
            as_targets=False,
        )
        return DecoderOnlyBatch.from_batch(batch)

    @staticmethod
    def format_attribution_args(
        batch: DecoderOnlyBatch,
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attribute_target: bool = False,  # Needed for compatibility with EncoderDecoderAttributionModel
        attributed_fn_args: Dict[str, Any] = {},
        attribute_batch_ids: bool = False,
        forward_batch_embeds: bool = True,
        use_baselines: bool = False,
    ) -> Tuple[Dict[str, Any], Tuple[Union[IdsTensor, EmbeddingsTensor, None], ...]]:
        if attribute_batch_ids:
            inputs = (batch.input_ids,)
            baselines = (batch.baseline_ids,)
        else:
            inputs = (batch.input_embeds,)
            baselines = (batch.baseline_embeds,)
        attribute_fn_args = {
            "inputs": inputs,
            "additional_forward_args": (
                # Ids are always explicitly passed as extra arguments to enable
                # usage in custom attribution functions.
                batch.input_ids,
                # Making targets 2D enables _expand_additional_forward_args
                # in Captum to preserve the expected batch dimension for methods
                # such as intergrated gradients.
                target_ids.unsqueeze(-1),
                attributed_fn,
                batch.attention_mask,
                # Defines how to treat source and target tensors
                # Maps on the use_embeddings argument of forward
                forward_batch_embeds,
                list(attributed_fn_args.keys()),
            )
            + tuple(attributed_fn_args.values()),
        }
        if use_baselines:
            attribute_fn_args["baselines"] = baselines
        return attribute_fn_args

    @staticmethod
    def enrich_step_output(
        step_output: FeatureAttributionStepOutput,
        batch: DecoderOnlyBatch,
        target_tokens: OneOrMoreTokenSequences,
        target_ids: TargetIdsTensor,
    ) -> FeatureAttributionStepOutput:
        r"""
        Enriches the attribution output with token information, producing the finished
        :class:`~inseq.data.FeatureAttributionStepOutput` object.

        Args:
            step_output (:class:`~inseq.data.FeatureAttributionStepOutput`): The output produced
                by the attribution step, with missing batch information.
            batch (:class:`~inseq.data.DecoderOnlyBatch`): The batch on which attribution was performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size, 1)` corresponding to tokens
                for which the attribution step was performed.

        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: The enriched attribution output.
        """
        if len(target_ids.shape) == 0:
            target_ids = target_ids.unsqueeze(0)
        step_output.source = None
        step_output.target = [[TokenWithId(token[0], id)] for token, id in zip(target_tokens, target_ids.tolist())]
        step_output.prefix = join_token_ids(batch.target_tokens, batch.input_ids.tolist())
        return step_output

    @staticmethod
    def format_step_function_args(
        attribution_model: "DecoderOnlyAttributionModel",
        forward_output: ModelOutput,
        target_ids: ExpandedTargetIdsTensor,
        batch: DecoderOnlyBatch,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            **kwargs,
            **{
                "attribution_model": attribution_model,
                "forward_output": forward_output,
                "encoder_input_ids": None,
                "decoder_input_ids": batch.target_ids,
                "encoder_input_embeds": None,
                "decoder_input_embeds": batch.target_embeds,
                "target_ids": target_ids,
                "encoder_attention_mask": None,
                "decoder_attention_mask": batch.target_mask,
            },
        }

    @staticmethod
    def convert_args_to_batch(
        decoder_input_ids: Optional[IdsTensor] = None,
        decoder_attention_mask: Optional[IdsTensor] = None,
        decoder_input_embeds: Optional[EmbeddingsTensor] = None,
        **kwargs,
    ) -> DecoderOnlyBatch:
        encoding = BatchEncoding(decoder_input_ids, decoder_attention_mask)
        embedding = BatchEmbedding(decoder_input_embeds)
        return DecoderOnlyBatch(encoding, embedding)

    @staticmethod
    def format_forward_args(forward_fn: ForwardMethod) -> Callable[..., CustomForwardOutput]:
        @wraps(forward_fn)
        def formatted_forward_input_wrapper(
            self: "DecoderOnlyAttributionModel",
            forward_tensor: AttributionForwardInputs,
            input_ids: IdsTensor,
            target_ids: ExpandedTargetIdsTensor,
            attributed_fn: Callable[..., SingleScorePerStepTensor],
            attention_mask: Optional[IdsTensor] = None,
            use_embeddings: bool = True,
            attributed_fn_argnames: Optional[List[str]] = None,
            *args,
            **kwargs,
        ) -> CustomForwardOutput:
            batch = self.formatter.convert_args_to_batch(
                decoder_input_ids=input_ids,
                decoder_attention_mask=attention_mask,
                decoder_input_embeds=forward_tensor if use_embeddings else None,
            )
            return forward_fn(
                self, batch, target_ids, attributed_fn, use_embeddings, attributed_fn_argnames, *args, **kwargs
            )

        return formatted_forward_input_wrapper

    @staticmethod
    def get_text_sequences(attribution_model: "DecoderOnlyAttributionModel", batch: DecoderOnlyBatch) -> TextSequences:
        return TextSequences(
            sources=None,
            targets=attribution_model.convert_tokens_to_string(batch.input_tokens, as_targets=True),
        )


class DecoderOnlyAttributionModel(AttributionModel):
    """AttributionModel class for attributing encoder-decoder models."""

    formatter = DecoderOnlyInputFormatter

    def get_forward_output(
        self,
        batch: DecoderOnlyBatch,
        use_embeddings: bool = True,
        **kwargs,
    ) -> ModelOutput:
        return self.model(
            input_ids=batch.input_ids if not use_embeddings else None,
            inputs_embeds=batch.input_embeds if use_embeddings else None,
            attention_mask=batch.attention_mask,
            **kwargs,
        )

    @formatter.format_forward_args
    def forward(self, *args, **kwargs) -> LogitsTensor:
        return self._forward(*args, **kwargs)

    @formatter.format_forward_args
    def forward_with_output(self, *args, **kwargs) -> ModelOutput:
        return self._forward_with_output(*args, **kwargs)

    def get_encoder(self) -> torch.nn.Module:
        raise NotImplementedError("Decoder-only models do not have an encoder.")

    def get_decoder(self) -> torch.nn.Module:
        return self.model
