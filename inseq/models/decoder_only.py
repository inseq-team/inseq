import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..attr.feat import join_token_ids
from ..data import (
    Batch,
    BatchEmbedding,
    BatchEncoding,
    DecoderOnlyBatch,
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


class DecoderOnlyAttributionModel(AttributionModel):
    """AttributionModel class for attributing encoder-decoder models."""

    def prepare_inputs_for_attribution(
        self,
        inputs: FeatureAttributionInput,
        include_eos_baseline: bool = False,
    ) -> DecoderOnlyBatch:
        if isinstance(inputs, Batch):
            batch = inputs
        else:
            if isinstance(inputs, (str, list)):
                # Decoder-only model do not tokenize as targets,
                # since a single tokenizer is available.
                encodings: BatchEncoding = self.encode(
                    inputs,
                    return_baseline=True,
                    include_eos_baseline=include_eos_baseline,
                )
            elif isinstance(inputs, BatchEncoding):
                encodings = inputs
            else:
                raise ValueError(
                    "targets must be either a string, a list of strings, a BatchEncoding or a Batch, "
                    f"not {type(inputs)}"
                )
            baseline_embeds = self.embed(encodings.baseline_ids)
            embeddings = BatchEmbedding(
                input_embeds=self.embed(encodings.input_ids),
                baseline_embeds=baseline_embeds,
            )
            batch = DecoderOnlyBatch(encodings, embeddings)
        return batch

    @staticmethod
    def format_forward_args(
        inputs: DecoderOnlyBatch,
        use_embeddings: bool = True,
    ) -> Dict[str, Any]:
        return {
            "forward_tensor": inputs.input_embeds if use_embeddings else inputs.input_ids,
            "attention_mask": inputs.attention_mask,
        }

    @staticmethod
    def format_attribution_args(
        batch: DecoderOnlyBatch,
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attributed_fn_args: Dict[str, Any] = {},
        attribute_batch_ids: bool = False,
        forward_batch_embeds: bool = True,
        **kwargs,
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
        return attribute_fn_args, baselines

    def get_text_sequences(self, batch: DecoderOnlyBatch) -> TextSequences:
        return TextSequences(
            sources=None,
            targets=self.convert_tokens_to_string(batch.input_tokens, as_targets=True),
        )

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

    def format_step_function_args(
        self,
        forward_output: ModelOutput,
        target_ids: ExpandedTargetIdsTensor,
        decoder_input_ids: Optional[IdsTensor] = None,
        decoder_input_embeds: Optional[EmbeddingsTensor] = None,
        decoder_attention_mask: Optional[IdsTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            **kwargs,
            **{
                "attribution_model": self,
                "forward_output": forward_output,
                "encoder_input_ids": None,
                "decoder_input_ids": decoder_input_ids,
                "encoder_input_embeds": None,
                "decoder_input_embeds": decoder_input_embeds,
                "target_ids": target_ids,
                "encoder_attention_mask": None,
                "decoder_attention_mask": decoder_attention_mask,
                **kwargs,
            },
        }

    def get_forward_output(
        self,
        forward_tensor: AttributionForwardInputs,
        attention_mask: Optional[IdsTensor] = None,
        use_embeddings: bool = True,
        **kwargs,
    ) -> ModelOutput:
        embeds = forward_tensor if use_embeddings else None
        ids = None if use_embeddings else forward_tensor
        return self.model(
            input_ids=ids,
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def forward(
        self,
        forward_tensor: AttributionForwardInputs,
        input_ids: IdsTensor,
        target_ids: ExpandedTargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attention_mask: Optional[IdsTensor] = None,
        use_embeddings: bool = True,
        attributed_fn_argnames: Optional[List[str]] = None,
        *args,
    ) -> FullLogitsTensor:
        assert len(args) == len(attributed_fn_argnames), "Number of arguments and number of argnames must match"
        target_ids = target_ids.squeeze(-1)
        output = self.get_forward_output(
            forward_tensor=forward_tensor,
            attention_mask=attention_mask,
            use_embeddings=use_embeddings,
        )
        logger.debug(f"logits: {pretty_tensor(output.logits)}")
        step_function_args = self.format_step_function_args(
            attribution_model=self,
            forward_output=output,
            decoder_input_ids=input_ids,
            decoder_input_embeds=forward_tensor if use_embeddings else None,
            target_ids=target_ids,
            decoder_attention_mask=attention_mask,
            **{k: v for k, v in zip(attributed_fn_argnames, args) if v is not None},
        )
        return attributed_fn(**step_function_args)
