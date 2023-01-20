import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..attr.feat import join_token_ids
from ..data import (
    Batch,
    BatchEmbedding,
    BatchEncoding,
    EncoderDecoderBatch,
    FeatureAttributionInput,
    FeatureAttributionStepOutput,
)
from ..utils import pretty_tensor
from ..utils.typing import (
    AttributionForwardInputs,
    EmbeddingsTensor,
    ExpandedTargetIdsTensor,
    FullLogitsTensor,
    IdsTensor,
    OneOrMoreTokenSequences,
    SingleScorePerStepTensor,
    TargetIdsTensor,
    TextSequences,
    TokenWithId,
)
from .attribution_model import AttributionModel, ModelOutput

logger = logging.getLogger(__name__)


class EncoderDecoderAttributionModel(AttributionModel):
    """AttributionModel class for attributing encoder-decoder models."""

    def prepare_inputs_for_attribution(
        self,
        inputs: Tuple[FeatureAttributionInput, FeatureAttributionInput],
        include_eos_baseline: bool = False,
    ) -> EncoderDecoderBatch:
        r"""
        Prepares sources and target to produce an :class:`~inseq.data.EncoderDecoderBatch`.
        There are two stages of preparation:

            1. Raw text sources and target texts are encoded by the model.
            2. The encoded sources and targets are converted to tensors for the forward pass.

        This method is agnostic of the preparation stage of sources and targets. If they are both
        raw text, they will undergo both steps. If they are already encoded, they will only be embedded.
        If the feature attribution method works on layers, the embedding step is skipped and embeddings are
        set to None.
        The final result will be consistent in both cases.

        Args:
            sources (:obj:`FeatureAttributionInput`): The sources provided to the
                :meth:`~inseq.attr.feat.FeatureAttribution.prepare` method.
            targets (:obj:`FeatureAttributionInput): The targets provided to the
                :meth:`~inseq.attr.feat.FeatureAttribution.prepare` method.
            include_eos_baseline (:obj:`bool`, `optional`): Whether to include the EOS token in the baseline for
                attribution. By default the EOS token is not used for attribution. Defaults to False.

        Returns:
            :obj:`EncoderDecoderBatch`: An :class:`~inseq.data.EncoderDecoderBatch` object containing sources
                and targets in encoded and embedded formats for all inputs.
        """
        sources, targets = inputs
        if isinstance(sources, Batch):
            source_batch = sources
        else:
            if isinstance(sources, (str, list)):
                source_encodings: BatchEncoding = self.encode(
                    sources, return_baseline=True, include_eos_baseline=include_eos_baseline
                )
            elif isinstance(sources, BatchEncoding):
                source_encodings = sources
            else:
                raise ValueError(
                    "sources must be either a string, a list of strings, a BatchEncoding or a Batch, "
                    f"not {type(sources)}"
                )
            # Even when we are performing layer attribution, we might need the embeddings
            # to compute step probabilities.
            source_embeddings = BatchEmbedding(
                input_embeds=self.embed(source_encodings.input_ids),
                baseline_embeds=self.embed(source_encodings.baseline_ids),
            )
            source_batch = Batch(source_encodings, source_embeddings)

        if isinstance(targets, Batch):
            target_batch = targets
        else:
            if isinstance(targets, (str, list)):
                target_encodings: BatchEncoding = self.encode(
                    targets,
                    as_targets=True,
                    return_baseline=True,
                    include_eos_baseline=include_eos_baseline,
                )
            elif isinstance(targets, BatchEncoding):
                target_encodings = targets
            else:
                raise ValueError(
                    "targets must be either a string, a list of strings, a BatchEncoding or a Batch, "
                    f"not {type(targets)}"
                )
            baseline_embeds = self.embed(target_encodings.baseline_ids, as_targets=True)
            target_embeddings = BatchEmbedding(
                input_embeds=self.embed(target_encodings.input_ids, as_targets=True),
                baseline_embeds=baseline_embeds,
            )
            target_batch = Batch(target_encodings, target_embeddings)
        return EncoderDecoderBatch(source_batch, target_batch)

    @staticmethod
    def format_forward_args(
        inputs: EncoderDecoderBatch,
        use_embeddings: bool = True,
    ) -> Dict[str, Any]:
        return {
            "forward_tensor": inputs.sources.input_embeds if use_embeddings else inputs.sources.input_ids,
            "decoder_input_embeds": inputs.targets.input_embeds,
            # "decoder_input_ids": inputs.targets.input_ids,
            "encoder_attention_mask": inputs.sources.attention_mask,
            "decoder_attention_mask": inputs.targets.attention_mask,
        }

    @staticmethod
    def format_attribution_args(
        batch: EncoderDecoderBatch,
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attribute_target: bool = False,
        attributed_fn_args: Dict[str, Any] = {},
        attribute_batch_ids: bool = False,
        forward_batch_embeds: bool = True,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Tuple[Union[IdsTensor, EmbeddingsTensor, None], ...]]:
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
        return attribute_fn_args, baselines

    def get_text_sequences(self, batch: EncoderDecoderBatch) -> TextSequences:
        return TextSequences(
            sources=self.convert_tokens_to_string(batch.sources.input_tokens),
            targets=self.convert_tokens_to_string(batch.targets.input_tokens, as_targets=True),
        )

    @staticmethod
    def enrich_step_output(
        step_output: FeatureAttributionStepOutput,
        batch: EncoderDecoderBatch,
        target_tokens: OneOrMoreTokenSequences,
        target_ids: TargetIdsTensor,
    ) -> FeatureAttributionStepOutput:
        r"""
        Enriches the attribution output with token information, producing the finished
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
        if len(target_ids.shape) == 0:
            target_ids = target_ids.unsqueeze(0)
        step_output.source = join_token_ids(batch.sources.input_tokens, batch.sources.input_ids.tolist())
        step_output.target = [[TokenWithId(token[0], id)] for token, id in zip(target_tokens, target_ids.tolist())]
        step_output.prefix = join_token_ids(batch.targets.input_tokens, batch.targets.input_ids.tolist())
        return step_output

    def format_step_function_args(
        self,
        forward_output: ModelOutput,
        target_ids: ExpandedTargetIdsTensor,
        encoder_input_ids: Optional[IdsTensor] = None,
        decoder_input_ids: Optional[IdsTensor] = None,
        encoder_input_embeds: Optional[EmbeddingsTensor] = None,
        decoder_input_embeds: Optional[EmbeddingsTensor] = None,
        encoder_attention_mask: Optional[IdsTensor] = None,
        decoder_attention_mask: Optional[IdsTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            **kwargs,
            **{
                "attribution_model": self,
                "forward_output": forward_output,
                "encoder_input_ids": encoder_input_ids,
                "decoder_input_ids": decoder_input_ids,
                "encoder_input_embeds": encoder_input_embeds,
                "decoder_input_embeds": decoder_input_embeds,
                "target_ids": target_ids,
                "encoder_attention_mask": encoder_attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
            },
        }

    def get_forward_output(
        self,
        forward_tensor: AttributionForwardInputs,
        encoder_attention_mask: Optional[IdsTensor] = None,
        decoder_input_embeds: Optional[EmbeddingsTensor] = None,
        decoder_attention_mask: Optional[IdsTensor] = None,
        use_embeddings: bool = True,
        **kwargs,
    ) -> ModelOutput:
        encoder_embeds = forward_tensor if use_embeddings else None
        encoder_ids = None if use_embeddings else forward_tensor
        return self.model(
            input_ids=encoder_ids,
            inputs_embeds=encoder_embeds,
            attention_mask=encoder_attention_mask,
            decoder_inputs_embeds=decoder_input_embeds,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

    def forward(
        self,
        encoder_tensors: AttributionForwardInputs,
        decoder_input_embeds: AttributionForwardInputs,
        encoder_input_ids: IdsTensor,
        decoder_input_ids: IdsTensor,
        target_ids: ExpandedTargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        encoder_attention_mask: Optional[IdsTensor] = None,
        decoder_attention_mask: Optional[IdsTensor] = None,
        use_embeddings: bool = True,
        attributed_fn_argnames: Optional[List[str]] = None,
        *args,
    ) -> FullLogitsTensor:
        assert len(args) == len(attributed_fn_argnames), "Number of arguments and number of argnames must match"
        target_ids = target_ids.squeeze(-1)
        output = self.get_forward_output(
            forward_tensor=encoder_tensors,
            encoder_attention_mask=encoder_attention_mask,
            decoder_input_embeds=decoder_input_embeds,
            decoder_attention_mask=decoder_attention_mask,
            use_embeddings=use_embeddings,
        )
        logger.debug(f"logits: {pretty_tensor(output.logits)}")
        step_function_args = self.format_step_function_args(
            attribution_model=self,
            forward_output=output,
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_input_embeds=encoder_tensors if use_embeddings else None,
            decoder_input_embeds=decoder_input_embeds,
            target_ids=target_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            **{k: v for k, v in zip(attributed_fn_argnames, args) if v is not None},
        )
        return attributed_fn(**step_function_args)
