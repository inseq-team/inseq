import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol, Union

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import ModelOutput

from ..data.attribution import DEFAULT_ATTRIBUTION_AGGREGATE_DICT
from ..utils.typing import EmbeddingsTensor, IdsTensor, SingleScorePerStepTensor, TargetIdsTensor

if TYPE_CHECKING:
    from ..models import AttributionModel

logger = logging.getLogger(__name__)


class StepScoreFunction(Protocol):
    def __call__(
        self,
        attribution_model: "AttributionModel",
        forward_output: ModelOutput,
        encoder_input_ids: IdsTensor,
        decoder_input_ids: IdsTensor,
        encoder_input_embeds: EmbeddingsTensor,
        decoder_input_embeds: EmbeddingsTensor,
        encoder_attention_mask: IdsTensor,
        decoder_attention_mask: IdsTensor,
        target_ids: TargetIdsTensor,
        **kwargs,
    ) -> SingleScorePerStepTensor:
        ...


def logit_fn(
    attribution_model: "AttributionModel", forward_output: ModelOutput, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """
    Compute the logit of the target_ids from the model's output logits.
    """
    logits = attribution_model.output2logits(forward_output)
    target_ids = target_ids.reshape(logits.shape[0], 1)
    return logits.gather(-1, target_ids).squeeze(-1)


def probability_fn(
    attribution_model: "AttributionModel", forward_output: ModelOutput, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """
    Compute the probabilty of target_ids from the model's output logits.
    """
    logits = attribution_model.output2logits(forward_output)
    target_ids = target_ids.reshape(logits.shape[0], 1)
    logits = torch.softmax(logits, dim=-1)
    # Extracts the ith score from the softmax output over the vocabulary (dim -1 of the logits)
    # where i is the value of the corresponding index in target_ids.
    return logits.gather(-1, target_ids).squeeze(-1)


def entropy_fn(
    attribution_model: "AttributionModel", forward_output: ModelOutput, **kwargs
) -> SingleScorePerStepTensor:
    """
    Compute the entropy of the model's output distribution.
    """
    logits = attribution_model.output2logits(forward_output)
    out = torch.distributions.Categorical(logits=logits).entropy()
    if len(out.shape) > 1:
        out = out.squeeze(-1)
    return out


def crossentropy_fn(
    attribution_model: "AttributionModel", forward_output: ModelOutput, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """
    Compute the cross entropy between the target_ids and the logits.
    See: https://github.com/ZurichNLP/nmtscore/blob/master/src/nmtscore/models/m2m100.py#L99
    """
    return -torch.log2(probability_fn(attribution_model, forward_output, target_ids))


def perplexity_fn(
    attribution_model: "AttributionModel", forward_output: ModelOutput, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """
    Compute perplexity of the target_ids from the logits.
    Perplexity is the weighted branching factor. If we have a perplexity of 100, it means that whenever the model is
    trying to guess the next word it is as confused as if it had to pick between 100 words.
    Reference: https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/
    """
    return 2 ** crossentropy_fn(attribution_model, forward_output, target_ids)


def contrast_prob_diff_fn(
    attribution_model: "AttributionModel",
    forward_output: ModelOutput,
    encoder_input_embeds: EmbeddingsTensor,
    encoder_attention_mask: IdsTensor,
    decoder_input_ids: IdsTensor,
    decoder_attention_mask: IdsTensor,
    target_ids: TargetIdsTensor,
    contrast_ids: IdsTensor,
    contrast_attention_mask: IdsTensor,
    **kwargs,
):
    """Returnsthe difference between next step probability for a candidate generation target vs. a contrastive
    alternative, answering the question. Can be used as attribution target to answer the question: "Which features
    were salient in the choice of picking the selected token rather than its contrastive alternative?"

    Args:
        contrast_ids (:obj:`torch.Tensor`): Tensor of shape ``[batch_size, seq_len]`` containing the ids of the
            contrastive input to be compared to the candidate.
        contrast_attention_mask (:obj:`torch.Tensor`): Tensor of shape ``[batch_size, seq_len]`` containing the
            attention mask for the contrastive input.
    """
    from ..models import DecoderOnlyAttributionModel, EncoderDecoderAttributionModel

    # We truncate contrastive ids and their attention map to the current generation step
    contrast_decoder_input_ids = contrast_ids[:, : decoder_input_ids.shape[1]].to(attribution_model.device)
    contrast_decoder_attention_mask = contrast_attention_mask[:, : decoder_attention_mask.shape[1]].to(
        attribution_model.device
    )
    # We select the next contrastive token as target
    contrast_target_ids = contrast_ids[:, decoder_input_ids.shape[1]].to(attribution_model.device)
    # Forward pass with the same model used for the main generation, but using contrastive inputs instead
    if isinstance(attribution_model, EncoderDecoderAttributionModel):
        contrast_decoder_input_embeds = attribution_model.embed_ids(contrast_decoder_input_ids, as_targets=True)
        contrast_output = attribution_model.get_forward_output(
            forward_tensor=encoder_input_embeds,
            encoder_attention_mask=encoder_attention_mask,
            decoder_input_embeds=contrast_decoder_input_embeds,
            decoder_attention_mask=contrast_decoder_attention_mask,
        )
    elif isinstance(attribution_model, DecoderOnlyAttributionModel):
        contrast_output = attribution_model.get_forward_output(
            forward_tensor=contrast_decoder_input_ids,
            attention_mask=contrast_decoder_attention_mask,
            use_embeddings=False,
        )
    else:
        raise ValueError("Unsupported attribution model type")
    # Return the prob difference as target for attribution
    model_probs = probability_fn(attribution_model, forward_output, target_ids)
    contrast_probs = probability_fn(attribution_model, contrast_output, contrast_target_ids)
    return model_probs - contrast_probs


def mc_dropout_prob_avg_fn(
    attribution_model: "AttributionModel",
    forward_output,
    encoder_input_embeds: EmbeddingsTensor,
    encoder_attention_mask: IdsTensor,
    decoder_input_ids: IdsTensor,
    decoder_input_embeds: EmbeddingsTensor,
    decoder_attention_mask: IdsTensor,
    target_ids: TargetIdsTensor,
    aux_model: Union[AutoModelForSeq2SeqLM, AutoModelForCausalLM],
    n_mcd_steps: int = 10,
    **kwargs,
):
    """Returns the average of probability scores using a pool of noisy prediction computed with MC Dropout. Can be
    used as an attribution target to compute more robust attribution scores.

    Args:
        aux_model (:obj:`transformers.AutoModelForSeq2SeqLM` or :obj:`transformers.AutoModelForCausalLM`): Model used
            to produce noisy probability predictions for target ids. Requirements:
            - Must be a model of the same category as the attribution model (e.g. encoder-decoder or decoder-only)
            - Must have the same vocabulary as the attribution model to ensure correct probability scores are computed
            - Must contain dropout layers to enable MC Dropout.
        n_mcd_steps (:obj:`int`): The number of prediction steps that should be used to normalize the original output.
    """
    from ..models import DecoderOnlyAttributionModel, EncoderDecoderAttributionModel

    noisy_probs = []
    # Compute noisy predictions using the auxiliary model
    # Important: must be in train mode to ensure noise for MCD
    aux_model.train()
    if attribution_model.model_name != aux_model.config.name_or_path:
        logger.warning(
            f"Model name mismatch between attribution model ({attribution_model.model_name}) "
            f"and auxiliary model ({aux_model.config.name_or_path}) for MC dropout averaging. "
            "mc_dropout_prob_avg expects models using the same vocabulary."
        )
    for _ in range(n_mcd_steps):
        if isinstance(attribution_model, EncoderDecoderAttributionModel):
            aux_output = aux_model(
                inputs_embeds=encoder_input_embeds,
                attention_mask=encoder_attention_mask,
                decoder_inputs_embeds=decoder_input_embeds,
                decoder_attention_mask=decoder_attention_mask,
            )
        elif isinstance(attribution_model, DecoderOnlyAttributionModel):
            aux_output = aux_model(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
            )
        else:
            raise ValueError("Unsupported model type")
        noisy_prob = probability_fn(attribution_model, aux_output, target_ids)
        noisy_probs.append(noisy_prob)
    # Original probability from the model without noise
    orig_prob = probability_fn(attribution_model, forward_output, target_ids)
    # Z-score the original based on the mean and standard deviation of MC dropout predictions
    return (orig_prob - torch.stack(noisy_probs).mean(0)).div(torch.stack(noisy_probs).std(0))


STEP_SCORES_MAP = {
    "logit": logit_fn,
    "probability": probability_fn,
    "entropy": entropy_fn,
    "crossentropy": crossentropy_fn,
    "perplexity": perplexity_fn,
    "contrast_prob_diff": contrast_prob_diff_fn,
    "mc_dropout_prob_avg": mc_dropout_prob_avg_fn,
}


def list_step_functions() -> List[str]:
    """
    Lists identifiers for all available step scores. One or more step scores identifiers can be passed to the
    :meth:`~inseq.models.AttributionModel.attribute` method either to compute scores while attributing (``step_scores``
    parameter), or as target function for the attribution, if supported by the attribution method (``attributed_fn``
    parameter).
    """
    return list(STEP_SCORES_MAP.keys())


def register_step_function(
    fn: StepScoreFunction,
    identifier: str,
    aggregate_map: Optional[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
) -> None:
    """
    Registers a function to be used to compute step scores and store them in the
    :class:`~inseq.data.attribution.FeatureAttributionOutput` object. Registered step functions can also be used as
    attribution targets by gradient-based feature attribution methods.

    Args:
        fn (:obj:`callable`): The function to be used to compute step scores. Default parameters (use kwargs to capture
        unused ones when defining your function):

            - :obj:`attribution_model`: an :class:`~inseq.models.AttributionModel` instance, corresponding to the model
                used for computing the score.

            - :obj:`forward_output`: the output of the forward pass from the attribution model.

            - :obj:`encoder_input_ids`, :obj:`decoder_input_ids`, :obj:`encoder_input_embeds`,
                :obj:`decoder_input_embeds`, :obj:`encoder_attention_mask`, :obj:`decoder_attention_mask`: all the
                elements composing the :class:`~inseq.data.Batch` used as context of the model.

            - :obj:`target_ids`: :obj:`torch.Tensor` of target token ids of size `(batch_size,)` and type long,
                corresponding to the target predicted tokens for the next generation step.

            The function can also define an arbitrary number of custom parameters that can later be provided directly
            to the `model.attribute` function call, and it must return a :obj:`torch.Tensor` of size `(batch_size,)` of
            float or long. If parameter names conflict with `model.attribute` ones, pass them as key-value pairs in the
            :obj:`step_scores_args` dict parameter.

        identifier (:obj:`str`): The identifier that will be used for the registered step score.
        aggregate_map (:obj:`dict`, `optional`): An optional dictionary mapping from :class:`~inseq.data.Aggregator`
            name identifiers to functions taking in input a tensor of shape `(batch_size, seq_len)` and producing
            tensors of shape `(batch_size, aggregated_seq_len)` in output that will be used to aggregate the
            registered step score when used in conjunction with the corresponding aggregator. E.g. the ``probability``
            step score uses the aggregate_map ``{"span_aggregate": lambda x: t.prod(dim=1, keepdim=True)}`` to
            aggregate probabilities with a product when aggregating scores over spans.
    """
    STEP_SCORES_MAP[identifier] = fn
    if isinstance(aggregate_map, dict):
        for agg_name, agg_fn in aggregate_map.items():
            if agg_name not in DEFAULT_ATTRIBUTION_AGGREGATE_DICT["step_scores"]:
                DEFAULT_ATTRIBUTION_AGGREGATE_DICT["step_scores"][agg_name] = {}
            DEFAULT_ATTRIBUTION_AGGREGATE_DICT["step_scores"][agg_name][identifier] = agg_fn
