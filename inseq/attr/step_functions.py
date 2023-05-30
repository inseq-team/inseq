import logging
from inspect import getfullargspec
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Union

import torch
from torch.nn.functional import kl_div, log_softmax
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import ModelOutput

from ..data import DecoderOnlyBatch, FeatureAttributionInput, get_batch_from_inputs
from ..data.aggregation_functions import DEFAULT_ATTRIBUTION_AGGREGATE_DICT
from ..utils.typing import EmbeddingsTensor, IdsTensor, SingleScorePerStepTensor, TargetIdsTensor

if TYPE_CHECKING:
    from ..models import AttributionModel

logger = logging.getLogger(__name__)


class StepFunction(Protocol):
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


def get_step_function_reserved_args() -> List[str]:
    return getfullargspec(StepFunction.__call__).args


def logit_fn(
    attribution_model: "AttributionModel", forward_output: ModelOutput, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """Compute the logit of the target_ids from the model's output logits."""
    logits = attribution_model.output2logits(forward_output)
    target_ids = target_ids.reshape(logits.shape[0], 1)
    return logits.gather(-1, target_ids).squeeze(-1)


def probability_fn(
    attribution_model: "AttributionModel", forward_output: ModelOutput, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """Compute the probabilty of target_ids from the model's output logits."""
    logits = attribution_model.output2logits(forward_output)
    target_ids = target_ids.reshape(logits.shape[0], 1)
    logits = logits.softmax(dim=-1)
    # Extracts the ith score from the softmax output over the vocabulary (dim -1 of the logits)
    # where i is the value of the corresponding index in target_ids.
    return logits.gather(-1, target_ids).squeeze(-1)


def entropy_fn(
    attribution_model: "AttributionModel", forward_output: ModelOutput, **kwargs
) -> SingleScorePerStepTensor:
    """Compute the entropy of the model's output distribution."""
    logits = attribution_model.output2logits(forward_output)
    out = torch.distributions.Categorical(logits=logits).entropy()
    if out.ndim > 1:
        out = out.squeeze(-1)
    return out


def crossentropy_fn(
    attribution_model: "AttributionModel", forward_output: ModelOutput, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """Compute the cross entropy between the target_ids and the logits.
    See: https://github.com/ZurichNLP/nmtscore/blob/master/src/nmtscore/models/m2m100.py#L99.
    """
    return -torch.log2(probability_fn(attribution_model, forward_output, target_ids))


def perplexity_fn(
    attribution_model: "AttributionModel", forward_output: ModelOutput, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """Compute perplexity of the target_ids from the logits.
    Perplexity is the weighted branching factor. If we have a perplexity of 100, it means that whenever the model is
    trying to guess the next word it is as confused as if it had to pick between 100 words.
    Reference: https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/.
    """
    return 2 ** crossentropy_fn(attribution_model, forward_output, target_ids)


def _get_contrast_output(
    attribution_model: "AttributionModel",
    encoder_input_ids: IdsTensor,
    decoder_input_ids: IdsTensor,
    encoder_attention_mask: IdsTensor,
    decoder_attention_mask: IdsTensor,
    encoder_input_embeds: EmbeddingsTensor,
    decoder_input_embeds: EmbeddingsTensor,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    return_contrastive_target_ids: bool = False,
) -> ModelOutput:
    """Utility function to return the output of the model for given contrastive inputs.

    Args:
        contrast_sources (:obj:`str` or :obj:`list(str)`): Source text(s) used as contrastive inputs to compute
            target probabilities for encoder-decoder models. If not specified, the source text is assumed to match the
            original source text. Defaults to :obj:`None`.
        contrast_target_prefixes (:obj:`str` or :obj:`list(str)`): Target prefix(es) used as contrastive inputs to
            compute target probabilities. If not specified, no target prefix beyond previously generated tokens is
            assumed. Defaults to :obj:`None`.
        contrast_targets (:obj:`str` or :obj:`list(str)`): Contrastive target text(s) to be compared to the original
            target text. If not specified, the original target text is used as contrastive target (will result in same
            output unless ``contrast_sources`` or ``contrast_target_prefixes`` are specified). Defaults to :obj:`None`.
        return_contrastive_target_ids (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether to return the
            contrastive target ids as well as the model output. Defaults to :obj:`False`.
    """
    c_tgt_ids = None
    if contrast_targets:
        c_batch = DecoderOnlyBatch.from_batch(
            get_batch_from_inputs(
                attribution_model=attribution_model,
                inputs=contrast_targets,
                as_targets=attribution_model.is_encoder_decoder,
            )
        )
        curr_prefix_len = decoder_input_ids.size(1)

        # We select the next contrastive token as target and truncate contrastive ids
        # and their attention map to the current generation step.
        c_tgt_ids = c_batch.target_ids[:, curr_prefix_len]
        c_batch = c_batch[:curr_prefix_len].to(attribution_model.device)

        if decoder_input_ids.size(0) != c_batch.target_ids.size(0):
            raise ValueError(
                f"Contrastive batch size ({c_batch.target_ids.size(0)}) must match candidate batch size "
                f"({decoder_input_ids.size(0)}). Multi-sentence attribution and methods expanding inputs to multiple "
                "steps (e.g. Integrated Gradients) are currently not supported for contrastive feature attribution."
            )

        decoder_input_ids = c_batch.target_ids
        decoder_input_embeds = c_batch.target_embeds
        decoder_attention_mask = c_batch.target_mask
    if contrast_target_prefixes:
        c_dec_in = attribution_model.encode(
            contrast_target_prefixes, as_targets=attribution_model.is_encoder_decoder, add_special_tokens=False
        )
        if attribution_model.is_encoder_decoder:
            # Remove the first token of the decoder input ids and attention mask if it's BOS
            if torch.all(torch.eq(decoder_input_ids[:, 0], attribution_model.bos_token_id)):
                decoder_input_ids = decoder_input_ids[:, 1:]
                decoder_attention_mask = decoder_attention_mask[:, 1:]
        c_dec_ids = torch.cat((c_dec_in.input_ids, decoder_input_ids), dim=1)
        c_dec_mask = torch.cat((c_dec_in.attention_mask, decoder_attention_mask), dim=1)
        c_dec_embeds = attribution_model.embed(c_dec_ids, as_targets=attribution_model.is_encoder_decoder)
    else:
        c_dec_ids = decoder_input_ids
        c_dec_mask = decoder_attention_mask
        c_dec_embeds = decoder_input_embeds
    if contrast_sources:
        if not attribution_model.is_encoder_decoder:
            raise ValueError(
                "Contrastive source inputs can only be used with encoder-decoder models. "
                "Use `contrast_target_prefixes` to set a contrastive target prefix for decoder-only models."
            )
        c_enc_in = attribution_model.encode(contrast_sources)
        c_enc_ids = c_enc_in.input_ids
        c_enc_mask = c_enc_in.attention_mask
        c_enc_embeds = attribution_model.embed(c_enc_ids, as_targets=False)
    else:
        c_enc_ids = encoder_input_ids
        c_enc_mask = encoder_attention_mask
        c_enc_embeds = encoder_input_embeds
    c_batch = attribution_model.formatter.convert_args_to_batch(
        encoder_input_ids=c_enc_ids,
        decoder_input_ids=c_dec_ids,
        encoder_input_embeds=c_enc_embeds,
        decoder_input_embeds=c_dec_embeds,
        encoder_attention_mask=c_enc_mask,
        decoder_attention_mask=c_dec_mask,
    )
    c_out = attribution_model.get_forward_output(c_batch, use_embeddings=attribution_model.is_encoder_decoder)
    if return_contrastive_target_ids:
        return c_out, c_tgt_ids
    return c_out


def contrast_prob_fn(
    attribution_model: "AttributionModel",
    target_ids: TargetIdsTensor,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    **kwargs,
):
    """Returns the probability of a generation target given contrastive context or target prediction alternative.
    If only ``contrast_targets`` are specified, the probability of the contrastive prediction is computed given same
    context. The probability for the same token given contrastive source/target preceding context can also be computed
    using ``contrast_sources`` and ``contrast_target_prefixes`` without specifying ``contrast_targets``.

    Args:
        contrast_sources (:obj:`str` or :obj:`list(str)`): Source text(s) used as contrastive inputs to compute
            target probabilities for encoder-decoder models. If not specified, the source text is assumed to match the
            original source text. Defaults to :obj:`None`.
        contrast_target_prefixes (:obj:`str` or :obj:`list(str)`): Target prefix(es) used as contrastive inputs to
            compute target probabilities. If not specified, no target prefix beyond previously generated tokens is
            assumed. Defaults to :obj:`None`.
        contrast_targets (:obj:`str` or :obj:`list(str)`): Contrastive target text(s) to be compared to the original
            target text. If not specified, the original target text is used as contrastive target (will result in same
            output unless ``contrast_sources`` or ``contrast_target_prefixes`` are specified). Defaults to :obj:`None`.
    """
    kwargs.pop("forward_output", None)
    c_output, c_tgt_ids = _get_contrast_output(
        attribution_model=attribution_model,
        contrast_sources=contrast_sources,
        contrast_target_prefixes=contrast_target_prefixes,
        contrast_targets=contrast_targets,
        return_contrastive_target_ids=True,
        **kwargs,
    )
    if c_tgt_ids is None:
        c_tgt_ids = target_ids
    return probability_fn(attribution_model, c_output, c_tgt_ids)


def pcxmi_fn(
    attribution_model: "AttributionModel",
    forward_output: ModelOutput,
    target_ids: TargetIdsTensor,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    **kwargs,
) -> SingleScorePerStepTensor:
    """Compute the pointwise conditional cross-mutual information (P-CXMI) of target ids given original and contrastive
    input options. The P-CXMI is defined as the negative log-ratio between the conditional probability of the target
    given the original input and the conditional probability of the target given the contrastive input, as defined
    by `Yin et al. (2021) <https://arxiv.org/abs/2109.07446>`__.

    Args:
        contrast_sources (:obj:`str` or :obj:`list(str)`): Source text(s) used as contrastive inputs to compute
            the P-CXMI for encoder-decoder models. If not specified, the source text is assumed to match the original
            source text. Defaults to :obj:`None`.
        contrast_target_prefixes (:obj:`str` or :obj:`list(str)`): Target prefix(es) used as contrastive inputs to
            compute the P-CXMI. If not specified, no target prefix beyond previously generated tokens is assumed.
            Defaults to :obj:`None`.
    """
    original_probs = probability_fn(attribution_model, forward_output, target_ids)
    contrast_probs = contrast_prob_fn(
        attribution_model=attribution_model,
        contrast_sources=contrast_sources,
        contrast_target_prefixes=contrast_target_prefixes,
        target_ids=target_ids,
        **kwargs,
    )
    return -torch.log2(torch.div(original_probs, contrast_probs))


def kl_divergence_fn(
    attribution_model: "AttributionModel",
    forward_output: ModelOutput,
    target_ids: TargetIdsTensor,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    top_k: int = 0,
    **kwargs,
) -> SingleScorePerStepTensor:
    """Compute the pointwise Kullback-Leibler divergence of target ids given original and contrastive input options.
    The KL divergence is the expectation of the log difference between the probabilities of regular (P) and contrastive
    (Q) inputs.

    Args:
        contrast_sources (:obj:`str` or :obj:`list(str)`): Source text(s) used as contrastive inputs to compute
            the KL divergence for encoder-decoder models. If not specified, the source text is assumed to match the
            original source text. Defaults to :obj:`None`.
        contrast_target_prefixes (:obj:`str` or :obj:`list(str)`): Target prefix(es) used as contrastive inputs to
            compute the KL divergence. If not specified, no target prefix beyond previously generated tokens is
            assumed. Defaults to :obj:`None`.
        top_k (:obj:`int`): If set to a value > 0, only the top :obj:`top_k` tokens will be considered for
            computing the KL divergence. Defaults to :obj:`0`.
    """

    original_logits = attribution_model.output2logits(forward_output)
    contrast_output = _get_contrast_output(
        attribution_model=attribution_model,
        contrast_sources=contrast_sources,
        contrast_target_prefixes=contrast_target_prefixes,
        return_contrastive_target_ids=False,
        **kwargs,
    )
    contrast_logits = attribution_model.output2logits(contrast_output)
    top_k = min(top_k, contrast_logits.size(-1))
    if top_k > 0:
        filtered_contrast_logits = torch.zeros(contrast_logits.size(0), top_k)
        filtered_original_logits = torch.zeros(original_logits.size(0), top_k)
        indices_to_remove = contrast_logits < contrast_logits.topk(top_k).values[..., -1, None]
        for i in range(contrast_logits.size(0)):
            filtered_contrast_logits[i] = contrast_logits[i].masked_select(~indices_to_remove[i])
            filtered_original_logits[i] = original_logits[i].masked_select(~indices_to_remove[i])
    else:
        filtered_contrast_logits = contrast_logits
        filtered_original_logits = original_logits
    original_logprobs = log_softmax(filtered_original_logits, dim=-1)
    contrast_logprobs = log_softmax(filtered_contrast_logits, dim=-1)
    kl_divergence = torch.zeros(original_logprobs.size(0))
    for i in range(original_logprobs.size(0)):
        kl_divergence[i] = kl_div(contrast_logprobs[i], original_logprobs[i], reduction="sum", log_target=True)
    return kl_divergence


def contrast_prob_diff_fn(
    attribution_model: "AttributionModel",
    forward_output: ModelOutput,
    target_ids: TargetIdsTensor,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    **kwargs,
):
    """Returns the difference between next step probability for a candidate generation target vs. a contrastive
    alternative. Can be used as attribution target to answer the question: "Which features were salient in the
    choice of picking the selected token rather than its contrastive alternative?". Follows the implementation
    of `Yin and Neubig (2022) <https://aclanthology.org/2022.emnlp-main.14>`__. Can also be used to compute the
    difference in probability for the same token given contrastive source/target preceding context using
    ``contrast_sources`` and ``contrast_target_prefixes`` without specifying ``contrast_targets``.

    Args:
        contrast_sources (:obj:`str` or :obj:`list(str)`): Source text(s) used as contrastive inputs to compute
            target probabilities for encoder-decoder models. If not specified, the source text is assumed to match the
            original source text. Defaults to :obj:`None`.
        contrast_target_prefixes (:obj:`str` or :obj:`list(str)`): Target prefix(es) used as contrastive inputs to
            compute target probabilities. If not specified, no target prefix beyond previously generated tokens is
            assumed. Defaults to :obj:`None`.
        contrast_targets (:obj:`str` or :obj:`list(str)`): Contrastive target text(s) to be compared to the original
            target text. If not specified, the original target text is used as contrastive target (will result in same
            output unless ``contrast_sources`` or ``contrast_target_prefixes`` are specified). Defaults to :obj:`None`.
    """
    model_probs = probability_fn(attribution_model, forward_output, target_ids)
    contrast_probs = contrast_prob_fn(
        attribution_model=attribution_model,
        target_ids=target_ids,
        contrast_sources=contrast_sources,
        contrast_target_prefixes=contrast_target_prefixes,
        contrast_targets=contrast_targets,
        **kwargs,
    )
    # Return the prob difference as target for attribution
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
    n_mcd_steps: int = 5,
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
        aux_batch = attribution_model.formatter.convert_args_to_batch(
            encoder_input_ids=None,
            decoder_input_ids=decoder_input_ids,
            encoder_input_embeds=encoder_input_embeds,
            decoder_input_embeds=decoder_input_embeds,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        aux_output = attribution_model.get_forward_output(
            aux_batch, use_embeddings=attribution_model.is_encoder_decoder
        )
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
    "contrast_prob": contrast_prob_fn,
    "pcxmi": pcxmi_fn,
    "kl_divergence": kl_divergence_fn,
    "contrast_prob_diff": contrast_prob_diff_fn,
    "mc_dropout_prob_avg": mc_dropout_prob_avg_fn,
}


def list_step_functions() -> List[str]:
    """Lists identifiers for all available step scores. One or more step scores identifiers can be passed to the
    :meth:`~inseq.models.AttributionModel.attribute` method either to compute scores while attributing (``step_scores``
    parameter), or as target function for the attribution, if supported by the attribution method (``attributed_fn``
    parameter).
    """
    return list(STEP_SCORES_MAP.keys())


def register_step_function(
    fn: StepFunction,
    identifier: str,
    aggregate_map: Optional[Dict[str, str]] = None,
    overwrite: bool = False,
) -> None:
    """Registers a function to be used to compute step scores and store them in the
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
            name identifiers to aggregation function identifiers. A list of available aggregation functions is
            available using :func:`~inseq.list_aggregation_functions`.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether to overwrite an existing function
            registered with the same identifier.
    """
    if identifier in STEP_SCORES_MAP:
        if not overwrite:
            raise ValueError(
                f"{identifier} is already registered in step functions map. Override with overwrite=True."
            )
        logger.warning(f"Overwriting {identifier} step function.")
    STEP_SCORES_MAP[identifier] = fn
    if isinstance(aggregate_map, dict):
        for agg_name, aggregation_fn_identifier in aggregate_map.items():
            if agg_name not in DEFAULT_ATTRIBUTION_AGGREGATE_DICT["step_scores"]:
                DEFAULT_ATTRIBUTION_AGGREGATE_DICT["step_scores"][agg_name] = {}
            DEFAULT_ATTRIBUTION_AGGREGATE_DICT["step_scores"][agg_name][identifier] = aggregation_fn_identifier
