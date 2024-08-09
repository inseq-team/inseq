import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from ..attr.feat import join_token_ids
from ..attr.step_functions import StepFunctionEncoderDecoderArgs
from ..data import (
    Batch,
    BatchEmbedding,
    BatchEncoding,
    DecoderOnlyBatch,
    EncoderDecoderBatch,
    FeatureAttributionInput,
    FeatureAttributionStepOutput,
    get_batch_from_inputs,
)
from ..utils import get_aligned_idx
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
)
from .attribution_model import AttributionModel, ForwardMethod, InputFormatter, ModelOutput

CustomForwardOutput = TypeVar("CustomForwardOutput")

logger = logging.getLogger(__name__)


class EncoderDecoderInputFormatter(InputFormatter):
    def prepare_inputs_for_attribution(
        attribution_model: "EncoderDecoderAttributionModel",
        inputs: tuple[FeatureAttributionInput, FeatureAttributionInput],
        include_eos_baseline: bool = False,
        skip_special_tokens: bool = False,
    ) -> EncoderDecoderBatch:
        r"""Prepares sources and target to produce an :class:`~inseq.data.EncoderDecoderBatch`.
        There are two stages of preparation:

            1. Raw text sources and target texts are encoded by the model.
            2. The encoded sources and targets are converted to tensors for the forward pass.

        This method is agnostic of the preparation stage of sources and targets. If they are both
        raw text, they will undergo both steps. If they are already encoded, they will only be embedded.
        If the feature attribution method works on layers, the embedding step is skipped and embeddings are
        set to None.
        The final result will be consistent in both cases.

        Args:
            inputs (:obj:`tuple` of `FeatureAttributionInput`): A tuple containing sources and targets provided to the
                :meth:`~inseq.attr.feat.FeatureAttribution.prepare` method.
            include_eos_baseline (:obj:`bool`, `optional`): Whether to include the EOS token in the baseline for
                attribution. By default the EOS token is not used for attribution. Defaults to False.

        Returns:
            :obj:`EncoderDecoderBatch`: An :class:`~inseq.data.EncoderDecoderBatch` object containing sources
                and targets in encoded and embedded formats for all inputs.
        """
        sources, targets = inputs
        source_batch = get_batch_from_inputs(
            attribution_model,
            inputs=sources,
            include_eos_baseline=include_eos_baseline,
            as_targets=False,
            skip_special_tokens=skip_special_tokens,
        )
        target_batch = get_batch_from_inputs(
            attribution_model,
            inputs=targets,
            include_eos_baseline=include_eos_baseline,
            as_targets=True,
            skip_special_tokens=skip_special_tokens,
        )
        return EncoderDecoderBatch(source_batch, target_batch)

    @staticmethod
    def format_attribution_args(
        batch: EncoderDecoderBatch,
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attribute_target: bool = False,
        attributed_fn_args: dict[str, Any] = {},
        attribute_batch_ids: bool = False,
        forward_batch_embeds: bool = True,
        use_baselines: bool = False,
    ) -> tuple[dict[str, Any], tuple[IdsTensor | EmbeddingsTensor | None, ...]]:
        if attribute_batch_ids:
            inputs = (batch.sources.input_ids,)
            baselines = (batch.sources.baseline_ids,)
        else:
            inputs = (batch.sources.input_embeds,)
            baselines = (batch.sources.baseline_embeds,)
        if attribute_target:
            if attribute_batch_ids:
                inputs += (batch.targets.input_ids,)
                baselines += (batch.targets.baseline_ids,)
            else:
                inputs += (batch.targets.input_embeds,)
                baselines += (batch.targets.baseline_embeds,)
        attribute_fn_args = {
            "inputs": inputs,
            "additional_forward_args": (
                # Ids are always explicitly passed as extra arguments to enable
                # usage in custom attribution functions.
                batch.sources.input_ids,
                batch.targets.input_ids,
                # Making targets 2D enables _expand_additional_forward_args
                # in Captum to preserve the expected batch dimension for methods
                # such as intergrated gradients.
                target_ids.unsqueeze(-1),
                attributed_fn,
                batch.sources.attention_mask,
                batch.targets.attention_mask,
                # Defines how to treat source and target tensors
                # Maps on the use_embeddings argument of forward
                forward_batch_embeds,
                list(attributed_fn_args.keys()),
            )
            + tuple(attributed_fn_args.values()),
        }
        if not attribute_target:
            attribute_fn_args["additional_forward_args"] = (batch.targets.input_embeds,) + attribute_fn_args[
                "additional_forward_args"
            ]
        if use_baselines:
            attribute_fn_args["baselines"] = baselines
        return attribute_fn_args

    @staticmethod
    def enrich_step_output(
        attribution_model: "EncoderDecoderAttributionModel",
        step_output: FeatureAttributionStepOutput,
        batch: EncoderDecoderBatch,
        target_tokens: OneOrMoreTokenSequences,
        target_ids: TargetIdsTensor,
        contrast_batch: DecoderOnlyBatch | None = None,
        contrast_targets_alignments: list[list[tuple[int, int]]] | None = None,
    ) -> FeatureAttributionStepOutput:
        r"""Enriches the attribution output with token information, producing the finished
        :class:`~inseq.data.FeatureAttributionStepOutput` object.

        Args:
            step_output (:class:`~inseq.data.FeatureAttributionStepOutput`): The output produced
                by the attribution step, with missing batch information.
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch on which attribution was performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size, 1)` corresponding to tokens
                for which the attribution step was performed.

        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: The enriched attribution output.
        """
        if target_ids.ndim == 0:
            target_ids = target_ids.unsqueeze(0)
        step_output.source = join_token_ids(batch.sources.input_tokens, batch.sources.input_ids.tolist())
        if contrast_batch is not None:
            contrast_aligned_idx = get_aligned_idx(len(batch.target_tokens[0]), contrast_targets_alignments[0])
            contrast_target_ids = contrast_batch.target_ids[:, contrast_aligned_idx]
            step_output.target = join_token_ids(
                tokens=target_tokens,
                ids=[[idx] for idx in target_ids.tolist()],
                contrast_tokens=attribution_model.convert_ids_to_tokens(
                    contrast_target_ids[None, ...], skip_special_tokens=False
                ),
            )
            step_output.prefix = join_token_ids(tokens=batch.target_tokens, ids=batch.target_ids.tolist())
        else:
            step_output.target = join_token_ids(target_tokens, [[idx] for idx in target_ids.tolist()])
            step_output.prefix = join_token_ids(batch.targets.input_tokens, batch.targets.input_ids.tolist())
        return step_output

    @staticmethod
    def format_step_function_args(
        attribution_model: "EncoderDecoderAttributionModel",
        forward_output: ModelOutput,
        target_ids: ExpandedTargetIdsTensor,
        batch: EncoderDecoderBatch,
        is_attributed_fn: bool = False,
    ) -> StepFunctionEncoderDecoderArgs:
        return StepFunctionEncoderDecoderArgs(
            attribution_model=attribution_model,
            forward_output=forward_output,
            target_ids=target_ids,
            is_attributed_fn=is_attributed_fn,
            encoder_input_ids=batch.source_ids,
            decoder_input_ids=batch.target_ids,
            encoder_input_embeds=batch.source_embeds,
            decoder_input_embeds=batch.target_embeds,
            encoder_attention_mask=batch.source_mask,
            decoder_attention_mask=batch.target_mask,
        )

    @staticmethod
    def convert_args_to_batch(
        args: StepFunctionEncoderDecoderArgs = None,
        encoder_input_ids: IdsTensor | None = None,
        decoder_input_ids: IdsTensor | None = None,
        encoder_attention_mask: IdsTensor | None = None,
        decoder_attention_mask: IdsTensor | None = None,
        encoder_input_embeds: EmbeddingsTensor | None = None,
        decoder_input_embeds: EmbeddingsTensor | None = None,
        **kwargs,
    ) -> EncoderDecoderBatch:
        if args is not None:
            encoder_input_ids = args.encoder_input_ids
            decoder_input_ids = args.decoder_input_ids
            encoder_attention_mask = args.encoder_attention_mask
            decoder_attention_mask = args.decoder_attention_mask
            encoder_input_embeds = args.encoder_input_embeds
            decoder_input_embeds = args.decoder_input_embeds
        source_encoding = BatchEncoding(encoder_input_ids, encoder_attention_mask)
        source_embedding = BatchEmbedding(encoder_input_embeds)
        source_batch = Batch(source_encoding, source_embedding)
        target_encoding = BatchEncoding(decoder_input_ids, decoder_attention_mask)
        target_embedding = BatchEmbedding(decoder_input_embeds)
        target_batch = Batch(target_encoding, target_embedding)
        return EncoderDecoderBatch(source_batch, target_batch)

    @staticmethod
    def format_forward_args(forward_fn: ForwardMethod) -> Callable[..., CustomForwardOutput]:
        @wraps(forward_fn)
        def formatted_forward_input_wrapper(
            self: "EncoderDecoderAttributionModel",
            encoder_tensors: AttributionForwardInputs,
            decoder_input_embeds: AttributionForwardInputs,
            encoder_input_ids: IdsTensor,
            decoder_input_ids: IdsTensor,
            target_ids: ExpandedTargetIdsTensor,
            attributed_fn: Callable[..., SingleScorePerStepTensor],
            encoder_attention_mask: IdsTensor | None = None,
            decoder_attention_mask: IdsTensor | None = None,
            use_embeddings: bool = True,
            attributed_fn_argnames: list[str] | None = None,
            *args,
            **kwargs,
        ) -> CustomForwardOutput:
            batch = self.formatter.convert_args_to_batch(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                encoder_input_embeds=encoder_tensors if use_embeddings else None,
                decoder_input_embeds=decoder_input_embeds,
            )
            return forward_fn(
                self, batch, target_ids, attributed_fn, use_embeddings, attributed_fn_argnames, *args, **kwargs
            )

        return formatted_forward_input_wrapper

    @staticmethod
    def get_text_sequences(
        attribution_model: "EncoderDecoderAttributionModel", batch: EncoderDecoderBatch
    ) -> TextSequences:
        return TextSequences(
            sources=attribution_model.convert_tokens_to_string(batch.sources.input_tokens),
            targets=attribution_model.decode(batch.targets.input_ids),
        )

    @staticmethod
    def get_step_function_reserved_args() -> list[str]:
        return [f.name for f in StepFunctionEncoderDecoderArgs.__dataclass_fields__.values()]


class EncoderDecoderAttributionModel(AttributionModel):
    """AttributionModel class for attributing encoder-decoder models."""

    formatter = EncoderDecoderInputFormatter

    def get_forward_output(
        self,
        batch: EncoderDecoderBatch,
        use_embeddings: bool = True,
        **kwargs,
    ) -> ModelOutput:
        return self.model(
            input_ids=None if use_embeddings else batch.source_ids,
            inputs_embeds=batch.source_embeds if use_embeddings else None,
            attention_mask=batch.source_mask,
            decoder_inputs_embeds=batch.target_embeds,
            decoder_attention_mask=batch.target_mask,
            **kwargs,
        )

    @formatter.format_forward_args
    def forward(self, *args, **kwargs) -> LogitsTensor:
        return self._forward(*args, **kwargs)

    @formatter.format_forward_args
    def forward_with_output(self, *args, **kwargs) -> ModelOutput:
        return self._forward_with_output(*args, **kwargs)
