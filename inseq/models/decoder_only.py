from typing import Any, Callable, Dict, Tuple, Union

from ..data import Batch, BatchEmbedding, BatchEncoding, FeatureAttributionInput
from ..utils.typing import EmbeddingsTensor, IdsTensor, SingleScorePerStepTensor, TargetIdsTensor
from .attribution_model import AttributionModel


class DecoderOnlyAttributionModel(AttributionModel):
    """AttributionModel class for attributing encoder-decoder models."""

    def prepare_inputs_for_attribution(
        self,
        inputs: FeatureAttributionInput,
        prepend_bos_token: bool = True,
        include_eos_baseline: bool = False,
        use_layer_attribution: bool = False,
    ) -> Batch:
        if isinstance(inputs, Batch):
            batch = inputs
        else:
            if isinstance(inputs, str) or isinstance(inputs, list):
                # Decoder-only model do not tokenize as targets,
                # since a single tokenizer is available.
                encodings: BatchEncoding = self.encode(
                    inputs,
                    prepend_bos_token=prepend_bos_token,
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
                input_embeds=self.embed(encodings.input_ids, as_targets=True),
                baseline_embeds=baseline_embeds,
            )
            batch = Batch(encodings, embeddings)
        return batch

    @staticmethod
    def format_attribution_args(
        batch: Batch,
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
