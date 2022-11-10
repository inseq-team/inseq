from typing import Any, Callable, Dict, Tuple, Union

from ..attr.feat import join_token_ids
from ..data import (
    Batch,
    BatchEmbedding,
    BatchEncoding,
    DecoderOnlyBatch,
    FeatureAttributionInput,
    FeatureAttributionStepOutput,
)
from ..utils.typing import (
    EmbeddingsTensor,
    IdsTensor,
    OneOrMoreTokenSequences,
    SingleScorePerStepTensor,
    TargetIdsTensor,
    TextSequences,
    TokenWithId,
)
from .attribution_model import AttributionModel


class DecoderOnlyAttributionModel(AttributionModel):
    """AttributionModel class for attributing encoder-decoder models."""

    def prepare_inputs_for_attribution(
        self,
        inputs: FeatureAttributionInput,
        include_eos_baseline: bool = False,
        use_layer_attribution: bool = False,
    ) -> DecoderOnlyBatch:
        if isinstance(inputs, Batch):
            batch = inputs
        else:
            if isinstance(inputs, str) or isinstance(inputs, list):
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
                    f"targets must be either a string, a list of strings, a BatchEncoding or a Batch, "
                    f"not {type(inputs)}"
                )
            baseline_embeds = None
            if not use_layer_attribution:
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
    ) -> Dict[str, Any]:
        return {"attributed_tensors": inputs.input_embeds, "attention_mask": inputs.attention_mask}

    @staticmethod
    def format_attribution_args(
        batch: DecoderOnlyBatch,
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attributed_fn_args: Dict[str, Any] = {},
        is_layer_attribution: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Tuple[Union[IdsTensor, EmbeddingsTensor, None], ...]]:
        if is_layer_attribution:
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
                not is_layer_attribution,
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
