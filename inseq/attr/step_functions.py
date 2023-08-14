import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

import torch
from transformers.modeling_outputs import ModelOutput

from ..data import DecoderOnlyBatch, FeatureAttributionInput, get_batch_from_inputs, slice_batch_from_position
from ..data.aggregation_functions import DEFAULT_ATTRIBUTION_AGGREGATE_DICT
from ..utils import extract_signature_args, logits_kl_divergence, top_p_logits_mask
from ..utils.typing import EmbeddingsTensor, IdsTensor, SingleScorePerStepTensor, TargetIdsTensor

if TYPE_CHECKING:
    from ..models import AttributionModel

logger = logging.getLogger(__name__)


@dataclass
class StepFunctionArgs:
    """Base class for step function base arguments. These arguments are passed to all step functions and are
    complemented by the ones defined in the step function signature.

    Attributes:
        attribution_model (:class:`~inseq.models.AttributionModel`): The attribution model used in the current step.
        forward_output (:class:`~inseq.models.ModelOutput`): The output of the model's forward pass.
        target_ids (:obj:`torch.Tensor`): Tensor of target token ids of size :obj:`(batch_size,)` corresponding to
            the target predicted tokens for the next generation step.
        encoder_input_ids (:obj:`torch.Tensor`): Tensor of ids of encoder input tokens of size
            :obj:`(batch_size, source_seq_len)`, representing encoder inputs at the present step. Available only for
            encoder-decoder models.
        decoder_input_ids (:obj:`torch.Tensor`): Tensor of ids of decoder input tokens of size
            :obj:`(batch_size, target_seq_len)`, representing decoder inputs at the present step.
        encoder_input_embeds (:obj:`torch.Tensor`): Tensor of embeddings of encoder input tokens of size
            :obj:`(batch_size, source_seq_len, hidden_size)`, representing encoder inputs at the present step.
            Available only for encoder-decoder models.
        decoder_input_embeds (:obj:`torch.Tensor`): Tensor of embeddings of decoder input tokens of size
            :obj:`(batch_size, target_seq_len, hidden_size)`, representing decoder inputs at the present step.
        encoder_attention_mask (:obj:`torch.Tensor`): Tensor of attention mask of encoder input tokens of size
            :obj:`(batch_size, source_seq_len)`, used for masking padding tokens in the encoder input. Available only
            for encoder-decoder models.
        decoder_attention_mask (:obj:`torch.Tensor`): Tensor of attention mask of decoder input tokens of size
            :obj:`(batch_size, target_seq_len)`, used for masking padding tokens in the decoder input.
    """

    attribution_model: "AttributionModel"
    forward_output: ModelOutput
    target_ids: TargetIdsTensor
    decoder_input_ids: IdsTensor
    decoder_input_embeds: EmbeddingsTensor
    decoder_attention_mask: IdsTensor


@dataclass
class StepFunctionEncoderDecoderArgs(StepFunctionArgs):
    encoder_input_ids: IdsTensor
    encoder_input_embeds: EmbeddingsTensor
    encoder_attention_mask: IdsTensor


@dataclass
class StepFunctionDecoderOnlyArgs(StepFunctionArgs):
    pass


class StepFunction(Protocol):
    def __call__(
        self,
        args: StepFunctionArgs,
        **kwargs,
    ) -> SingleScorePerStepTensor:
        ...


def logit_fn(args: StepFunctionArgs) -> SingleScorePerStepTensor:
    """Compute the logit of the target_ids from the model's output logits."""
    logits = args.attribution_model.output2logits(args.forward_output)
    target_ids = args.target_ids.reshape(logits.shape[0], 1)
    return logits.gather(-1, target_ids).squeeze(-1)


def probability_fn(args: StepFunctionArgs) -> SingleScorePerStepTensor:
    """Compute the probabilty of target_ids from the model's output logits."""
    logits = args.attribution_model.output2logits(args.forward_output)
    target_ids = args.target_ids.reshape(logits.shape[0], 1)
    logits = logits.softmax(dim=-1)
    # Extracts the ith score from the softmax output over the vocabulary (dim -1 of the logits)
    # where i is the value of the corresponding index in target_ids.
    return logits.gather(-1, target_ids).squeeze(-1)


def entropy_fn(args: StepFunctionArgs) -> SingleScorePerStepTensor:
    """Compute the entropy of the model's output distribution."""
    logits = args.attribution_model.output2logits(args.forward_output)
    out = torch.distributions.Categorical(logits=logits).entropy()
    if out.ndim > 1:
        out = out.squeeze(-1)
    return out


def crossentropy_fn(args: StepFunctionArgs) -> SingleScorePerStepTensor:
    """Compute the cross entropy between the target_ids and the logits.
    See: https://github.com/ZurichNLP/nmtscore/blob/master/src/nmtscore/models/m2m100.py#L99.
    """
    return -torch.log2(probability_fn(args))


def perplexity_fn(args: StepFunctionArgs) -> SingleScorePerStepTensor:
    """Compute perplexity of the target_ids from the logits.
    Perplexity is the weighted branching factor. If we have a perplexity of 100, it means that whenever the model is
    trying to guess the next word it is as confused as if it had to pick between 100 words.
    Reference: https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/.
    """
    return 2 ** crossentropy_fn(args)


def _get_contrast_output(
    args: StepFunctionArgs,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    contrast_targets_alignments: Optional[List[List[Tuple[int, int]]]] = None,
    return_contrastive_target_ids: bool = False,
    **forward_kwargs,
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
        contrast_targets_alignments (:obj:`list(tuple(int, int))`, `optional`): A list of tuples of indices, where the
            first element is the index of the original target token and the second element is the index of the
            contrastive target token, used only if :obj:`contrast_targets` is specified. If an explicit alignment is
            not specified, the alignment of the original and contrastive target texts is assumed to be 1:1 for all
            available tokens. Defaults to :obj:`None`.
        return_contrastive_target_ids (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether to return the
            contrastive target ids as well as the model output. Defaults to :obj:`False`.
        **forward_kwargs: Additional keyword arguments to be passed to the model's forward pass.
    """
    c_tgt_ids = None
    is_enc_dec = args.attribution_model.is_encoder_decoder
    if contrast_targets:
        c_batch = DecoderOnlyBatch.from_batch(
            get_batch_from_inputs(
                attribution_model=args.attribution_model,
                inputs=contrast_targets,
                as_targets=is_enc_dec,
            )
        )
        curr_prefix_len = args.decoder_input_ids.size(1)
        if len(contrast_targets_alignments) > 0 and isinstance(contrast_targets_alignments[0], list):
            contrast_targets_alignments = contrast_targets_alignments[0]
        c_batch, c_tgt_ids = slice_batch_from_position(c_batch, curr_prefix_len, contrast_targets_alignments)

        if args.decoder_input_ids.size(0) != c_batch.target_ids.size(0):
            raise ValueError(
                f"Contrastive batch size ({c_batch.target_ids.size(0)}) must match candidate batch size"
                f" ({args.decoder_input_ids.size(0)}). Multi-sentence attribution and methods expanding inputs to"
                " multiple steps (e.g. Integrated Gradients) are not supported for contrastive feature attribution."
            )

        args.decoder_input_ids = c_batch.target_ids
        args.decoder_input_embeds = c_batch.target_embeds
        args.decoder_attention_mask = c_batch.target_mask
    if contrast_target_prefixes:
        c_dec_in = args.attribution_model.encode(
            contrast_target_prefixes, as_targets=is_enc_dec, add_special_tokens=False
        )
        if is_enc_dec:
            # Remove the first token of the decoder input ids and attention mask if it's BOS
            if torch.all(torch.eq(args.decoder_input_ids[:, 0], args.attribution_model.bos_token_id)):
                args.decoder_input_ids = args.decoder_input_ids[:, 1:]
                args.decoder_attention_mask = args.decoder_attention_mask[:, 1:]
        args.decoder_input_ids = torch.cat((c_dec_in.input_ids, args.decoder_input_ids), dim=1)
        args.decoder_attention_mask = torch.cat((c_dec_in.attention_mask, args.decoder_attention_mask), dim=1)
        args.decoder_input_embeds = args.attribution_model.embed(args.decoder_input_ids, as_targets=is_enc_dec)
    if contrast_sources:
        if not is_enc_dec:
            raise ValueError(
                "Contrastive source inputs can only be used with encoder-decoder models. "
                "Use `contrast_target_prefixes` to set a contrastive target prefix for decoder-only models."
            )
        c_enc_in = args.attribution_model.encode(contrast_sources)
        if is_enc_dec and isinstance(args, StepFunctionEncoderDecoderArgs):
            args.encoder_input_ids = c_enc_in.input_ids
            args.encoder_attention_mask = c_enc_in.attention_mask
            args.encoder_input_embeds = args.attribution_model.embed(args.encoder_input_ids, as_targets=False)
    c_batch = args.attribution_model.formatter.convert_args_to_batch(args)
    c_out = args.attribution_model.get_forward_output(c_batch, use_embeddings=is_enc_dec, **forward_kwargs)
    if return_contrastive_target_ids:
        return c_out, c_tgt_ids
    return c_out


def contrast_prob_fn(
    args: StepFunctionArgs,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    contrast_targets_alignments: Optional[List[List[Tuple[int, int]]]] = None,
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
        contrast_targets_alignments (:obj:`list(tuple(int, int))`, `optional`): A list of tuples of indices, where the
            first element is the index of the original target token and the second element is the index of the
            contrastive target token, used only if :obj:`contrast_targets` is specified. If an explicit alignment is
            not specified, the alignment of the original and contrastive target texts is assumed to be 1:1 for all
            available tokens. Defaults to :obj:`None`.
    """
    c_output, c_tgt_ids = _get_contrast_output(
        args,
        contrast_sources=contrast_sources,
        contrast_target_prefixes=contrast_target_prefixes,
        contrast_targets=contrast_targets,
        contrast_targets_alignments=contrast_targets_alignments,
        return_contrastive_target_ids=True,
    )
    if c_tgt_ids:
        args.target_ids = c_tgt_ids
    args.forward_output = c_output
    return probability_fn(args)


def pcxmi_fn(
    args: StepFunctionArgs,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    contrast_targets_alignments: Optional[List[List[Tuple[int, int]]]] = None,
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
        contrast_targets (:obj:`str` or :obj:`list(str)`): Contrastive target text(s) to be compared to the original
            target text. If not specified, the original target text is used as contrastive target (will result in same
            output unless ``contrast_sources`` or ``contrast_target_prefixes`` are specified). Defaults to :obj:`None`.
        contrast_targets_alignments (:obj:`list(tuple(int, int))`, `optional`): A list of tuples of indices, where the
            first element is the index of the original target token and the second element is the index of the
            contrastive target token, used only if :obj:`contrast_targets` is specified. If an explicit alignment is
            not specified, the alignment of the original and contrastive target texts is assumed to be 1:1 for all
            available tokens. Defaults to :obj:`None`.
    """
    original_probs = probability_fn(args)
    contrast_probs = contrast_prob_fn(
        args=args,
        contrast_sources=contrast_sources,
        contrast_target_prefixes=contrast_target_prefixes,
        contrast_targets=contrast_targets,
        contrast_targets_alignments=contrast_targets_alignments,
    )
    return -torch.log2(torch.div(original_probs, contrast_probs))


def kl_divergence_fn(
    args: StepFunctionArgs,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    contrast_targets_alignments: Optional[List[List[Tuple[int, int]]]] = None,
    top_k: int = 0,
    top_p: float = 1.0,
    min_tokens_to_keep: int = 1,
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
        contrast_targets (:obj:`str` or :obj:`list(str)`): Contrastive target text(s) to be compared to the original
            target text. If not specified, the original target text is used as contrastive target (will result in same
            output unless ``contrast_sources`` or ``contrast_target_prefixes`` are specified). Defaults to :obj:`None`.
        contrast_targets_alignments (:obj:`list(tuple(int, int))`, `optional`): A list of tuples of indices, where the
            first element is the index of the original target token and the second element is the index of the
            contrastive target token, used only if :obj:`contrast_targets` is specified. If an explicit alignment is
            not specified, the alignment of the original and contrastive target texts is assumed to be 1:1 for all
            available tokens. Defaults to :obj:`None`.
        top_k (:obj:`int`): If set to a value > 0, only the top :obj:`top_k` tokens will be considered for
            computing the KL divergence. Defaults to :obj:`0` (no top-k selection).
        top_p (:obj:`float`): If set to a value > 0 and < 1, only the tokens with cumulative probability above
            :obj:`top_p` will be considered for computing the KL divergence. Defaults to :obj:`1.0` (no filtering),
            applied before :obj:`top_k` filtering.
        min_tokens_to_keep (:obj:`int`): Minimum number of tokens to keep with :obj:`top_p` filtering. Defaults to
            :obj:`1`.
    """

    original_logits: torch.Tensor = args.attribution_model.output2logits(args.forward_output)
    contrast_output = _get_contrast_output(
        args=args,
        contrast_sources=contrast_sources,
        contrast_targets=contrast_targets,
        contrast_target_prefixes=contrast_target_prefixes,
        contrast_targets_alignments=contrast_targets_alignments,
        return_contrastive_target_ids=False,
    )
    contrast_logits: torch.Tensor = args.attribution_model.output2logits(contrast_output)
    return logits_kl_divergence(
        original_logits=original_logits,
        contrast_logits=contrast_logits,
        top_p=top_p,
        top_k=top_k,
        min_tokens_to_keep=min_tokens_to_keep,
    )


def contrast_prob_diff_fn(
    args: StepFunctionArgs,
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_target_prefixes: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    contrast_targets_alignments: Optional[List[List[Tuple[int, int]]]] = None,
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
        contrast_targets_alignments (:obj:`list(tuple(int, int))`, `optional`): A list of tuples of indices, where the
            first element is the index of the original target token and the second element is the index of the
            contrastive target token, used only if :obj:`contrast_targets` is specified. If an explicit alignment is
            not specified, the alignment of the original and contrastive target texts is assumed to be 1:1 for all
            available tokens. Defaults to :obj:`None`.
    """
    model_probs = probability_fn(args)
    contrast_probs = contrast_prob_fn(
        args=args,
        contrast_sources=contrast_sources,
        contrast_target_prefixes=contrast_target_prefixes,
        contrast_targets=contrast_targets,
        contrast_targets_alignments=contrast_targets_alignments,
    )
    # Return the prob difference as target for attribution
    return model_probs - contrast_probs


def mc_dropout_prob_avg_fn(
    args: StepFunctionArgs,
    n_mcd_steps: int = 5,
):
    """Returns the average of probability scores using a pool of noisy prediction computed with MC Dropout. Can be
    used as an attribution target to compute more robust attribution scores.

    Note:
        In order to obtain meaningful results, the :obj:`attribution_model` must contain dropout layers or other
        sources of noise in the forward pass.

    Args:
        n_mcd_steps (:obj:`int`): The number of prediction steps that should be used to normalize the original output.
    """
    # Original probability from the model without noise
    orig_prob = probability_fn(args)

    # Compute noisy predictions using the noisy model
    # Important: must be in train mode to ensure noise for MCD
    args.attribution_model.train()
    noisy_probs = []
    for _ in range(n_mcd_steps):
        aux_batch = args.attribution_model.formatter.convert_args_to_batch(args)
        aux_output = args.attribution_model.get_forward_output(
            aux_batch, use_embeddings=args.attribution_model.is_encoder_decoder
        )
        args.forward_output = aux_output
        noisy_prob = probability_fn(args)
        noisy_probs.append(noisy_prob)
    # Z-score the original based on the mean and standard deviation of MC dropout predictions
    return (orig_prob - torch.stack(noisy_probs).mean(0)).div(torch.stack(noisy_probs).std(0))


def top_p_size_fn(
    args: StepFunctionArgs,
    top_p: float,
):
    """Returns the number of tokens that have cumulative probability above :obj:`top_p` in the model's output logits.

    Args:
        top_p (:obj:`float`): The cumulative probability threshold to use for filtering the logits.
    """
    logits: torch.Tensor = args.attribution_model.output2logits(args.forward_output)
    indices_to_remove = top_p_logits_mask(logits, top_p, 1)
    logits = logits.masked_select(~indices_to_remove)[None, ...]
    return torch.tensor(logits.size(-1))[None, ...]


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
    "top_p_size": top_p_size_fn,
}


def check_is_step_function(identifier: str) -> None:
    if identifier not in STEP_SCORES_MAP:
        raise AttributeError(
            f"Step score {identifier} not found. Available step scores are: "
            f"{', '.join(list(STEP_SCORES_MAP.keys()))}. Use the inseq.register_step_function"
            "function to register a custom step score."
        )


def get_step_function(score_identifier: str) -> StepFunction:
    """Returns the step function corresponding to the provided identifier."""
    check_is_step_function(score_identifier)
    return STEP_SCORES_MAP[score_identifier]


def get_step_scores(
    score_identifier: str,
    step_fn_args: StepFunctionArgs,
    step_fn_extra_args: Dict[str, Any] = {},
) -> SingleScorePerStepTensor:
    """Returns step scores for the target tokens in the batch."""
    return get_step_function(score_identifier)(step_fn_args, **step_fn_extra_args)


def get_step_scores_args(
    score_identifiers: List[str], kwargs: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    step_scores_args = {}
    for step_fn_id in score_identifiers:
        step_fn = get_step_function(step_fn_id)
        step_scores_args.update(
            **extract_signature_args(
                kwargs,
                step_fn,
                exclude_args=default_args,
                return_remaining=False,
            )
        )
    return step_scores_args


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
