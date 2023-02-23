import logging
from typing import TYPE_CHECKING, Callable, Union

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from ..data import DecoderOnlyBatch, EncoderDecoderBatch
from ..utils.typing import EmbeddingsTensor, IdsTensor, SingleScorePerStepTensor, TargetIdsTensor

if TYPE_CHECKING:
    from ..models import AttributionModel

logger = logging.getLogger(__name__)

StepScoreInput = Callable[
    [
        "AttributionModel",
        Union[EncoderDecoderBatch, DecoderOnlyBatch],
        IdsTensor,
        IdsTensor,
        EmbeddingsTensor,
        EmbeddingsTensor,
        IdsTensor,
        IdsTensor,
        TargetIdsTensor,
    ],
    SingleScorePerStepTensor,
]


def probability_fn(
    attribution_model: "AttributionModel", forward_output, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """
    Compute the probabilty of target_ids from the logits.
    """
    logits = attribution_model.output2logits(forward_output)
    target_ids = target_ids.reshape(logits.shape[0], 1)
    logits = torch.softmax(logits, dim=-1)
    # Extracts the ith score from the softmax output over the vocabulary (dim -1 of the logits)
    # where i is the value of the corresponding index in target_ids.
    return logits.gather(-1, target_ids).squeeze(-1)


def entropy_fn(attribution_model: "AttributionModel", forward_output, **kwargs) -> SingleScorePerStepTensor:
    """
    Compute the entropy of the outputs from the logits.
    Target id is not used in the computation, but kept for consistency with
    the other functions.
    """
    logits = attribution_model.output2logits(forward_output)
    out = torch.distributions.Categorical(logits=logits).entropy()
    if len(out.shape) > 1:
        out = out.squeeze(-1)
    return out


def crossentropy_fn(
    attribution_model: "AttributionModel", forward_output, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """
    Compute the cross entropy between the target_ids and the logits.
    See: https://github.com/ZurichNLP/nmtscore/blob/master/src/nmtscore/models/m2m100.py#L99
    """
    return -torch.log2(probability_fn(attribution_model, forward_output, target_ids))


def perplexity_fn(
    attribution_model: "AttributionModel", forward_output, target_ids: TargetIdsTensor, **kwargs
) -> SingleScorePerStepTensor:
    """
    Compute perplexity of the target_ids from the logits.
    Perplexity is the weighted branching factor. If we have a perplexity of 100,
    it means that whenever the model is trying to guess the next word it is as
    confused as if it had to pick between 100 words.
    Reference: https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/
    """
    return 2 ** crossentropy_fn(attribution_model, forward_output, target_ids)


def contrast_prob_diff_fn(
    attribution_model: "AttributionModel",
    forward_output,
    encoder_input_embeds: EmbeddingsTensor,
    encoder_attention_mask: IdsTensor,
    decoder_input_ids: IdsTensor,
    decoder_attention_mask: IdsTensor,
    target_ids: TargetIdsTensor,
    contrast_ids: IdsTensor,
    contrast_attention_mask: IdsTensor,
    **kwargs,
):
    """Custom attribution function returning the difference between next step probability for
    candidate generation vs. a contrastive alternative, answering the question: Which features
    were salient in deciding to pick the selected token rather than its contrastive alternative?

    Args:
        contrast_ids: Tensor containing the ids of the contrastive input to be compared to the
            regular one.
        contrast_attention_mask: Tensor containing the attention mask for the contrastive input.
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
    """Custom attribution function normalizing probabilities normally used as attribution
    targets over a pool of noisy prediction computed with MC Dropout, to increase their robustness.

    Args:
        aux_model: Model in train mode to produce noisy probability predictions thanks to dropout
            modules.
        n_mcd_steps: The number of predictions that should be used to normalize the original output.
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
    "probability": probability_fn,
    "entropy": entropy_fn,
    "crossentropy": crossentropy_fn,
    "perplexity": perplexity_fn,
    "contrast_prob_diff": contrast_prob_diff_fn,
    "mc_dropout_prob_avg": mc_dropout_prob_avg_fn,
}
