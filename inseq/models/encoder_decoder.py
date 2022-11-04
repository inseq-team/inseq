from typing import Any, Callable, Dict, Tuple, Union

from ..attr.feat import join_token_ids
from ..data import (
    Batch,
    BatchEmbedding,
    BatchEncoding,
    EncoderDecoderBatch,
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


class EncoderDecoderAttributionModel(AttributionModel):
    """AttributionModel class for attributing encoder-decoder models."""

    def prepare_inputs_for_attribution(
        self,
        inputs: Tuple[FeatureAttributionInput, FeatureAttributionInput],
        prepend_bos_token: bool = True,
        include_eos_baseline: bool = False,
        use_layer_attribution: bool = False,
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
            prepend_bos_token (:obj:`bool`, `optional`): Whether to prepend a BOS token to the
                targets, if they are to be encoded. Defaults to True.
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
            if isinstance(sources, str) or isinstance(sources, list):
                source_encodings: BatchEncoding = self.encode(
                    sources, return_baseline=True, include_eos_baseline=include_eos_baseline
                )
            elif isinstance(sources, BatchEncoding):
                source_encodings = sources
            else:
                raise ValueError(
                    f"sources must be either a string, a list of strings, a BatchEncoding or a Batch, "
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
            if isinstance(targets, str) or isinstance(targets, list):
                target_encodings: BatchEncoding = self.encode(
                    targets,
                    as_targets=True,
                    prepend_bos_token=prepend_bos_token,
                    return_baseline=True,
                    include_eos_baseline=include_eos_baseline,
                )
            elif isinstance(targets, BatchEncoding):
                target_encodings = targets
            else:
                raise ValueError(
                    f"targets must be either a string, a list of strings, a BatchEncoding or a Batch, "
                    f"not {type(targets)}"
                )
            baseline_embeds = None
            if not use_layer_attribution:
                baseline_embeds = self.embed(target_encodings.baseline_ids, as_targets=True)
            target_embeddings = BatchEmbedding(
                input_embeds=self.embed(target_encodings.input_ids, as_targets=True),
                baseline_embeds=baseline_embeds,
            )
            target_batch = Batch(target_encodings, target_embeddings)
        return EncoderDecoderBatch(source_batch, target_batch)

    @staticmethod
    def format_attribution_args(
        batch: EncoderDecoderBatch,
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attribute_target: bool = False,
        attributed_fn_args: Dict[str, Any] = {},
        is_layer_attribution: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Tuple[Union[IdsTensor, EmbeddingsTensor, None], ...]]:
        if is_layer_attribution:
            inputs = (batch.sources.input_ids,)
            baselines = (batch.sources.baseline_ids,)
        else:
            inputs = (batch.sources.input_embeds,)
            baselines = (batch.sources.baseline_embeds,)
        if attribute_target:
            inputs = inputs + (batch.targets.input_embeds,)
            baselines = baselines + (batch.targets.baseline_embeds,)
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
                not is_layer_attribution,
                list(attributed_fn_args.keys()),
            )
            + tuple(attributed_fn_args.values()),
        }
        if not attribute_target:
            attribute_fn_args["additional_forward_args"] = (batch.targets.input_embeds,) + attribute_fn_args[
                "additional_forward_args"
            ]
        return attribute_fn_args, baselines

    def get_sequences(self, batch: EncoderDecoderBatch) -> TextSequences:
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
