import torch
from pytest import fixture
from transformers import AutoModelForSeq2SeqLM

import inseq
from inseq.models.huggingface_model import HuggingfaceEncoderDecoderModel
from inseq.utils import output2prob


@fixture(scope="session")
def saliency_mt_model() -> HuggingfaceEncoderDecoderModel:
    return inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency")


@fixture(scope="session")
def auxiliary_saliency_mt_model():
    aux_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")
    aux_model.train()
    return aux_model


def attribute_contrast_logits_diff(
    attribution_model,
    forward_output,
    encoder_input_embeds,
    encoder_attention_mask,
    decoder_input_ids,
    decoder_attention_mask,
    target_ids,
    contrast_ids,
    contrast_attention_mask,
    **kwargs,
):
    """Custom attribution function returning the difference between next step probability for
    candidate generation vs. a contrastive alternative, answering the question "Which features
    were salient in deciding to pick the selected token rather than its contrastive alternative?"

    Extra args:
        contrast_ids: Tensor containing the ids of the contrastive input to be compared to the
            regular one.
        contrast_attention_mask: Tensor containing the attention mask for the contrastive input
    """
    # We truncate contrastive ids and their attention map to the current generation step
    contrast_decoder_input_ids = contrast_ids[:, : decoder_input_ids.shape[1]].to(attribution_model.device)
    contrast_decoder_attention_mask = contrast_attention_mask[:, : decoder_attention_mask.shape[1]].to(
        attribution_model.device
    )
    # We select the next contrastive token as target
    contrast_target_ids = contrast_ids[:, decoder_input_ids.shape[1]].to(attribution_model.device)
    # Forward pass with the same model used for the main generation, but using contrastive inputs instead
    contrast_output = attribution_model.model(
        inputs_embeds=encoder_input_embeds,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=contrast_decoder_input_ids,
        decoder_attention_mask=contrast_decoder_attention_mask,
    )
    # Return the prob difference as target for attribution
    model_probs = output2prob(attribution_model, forward_output, target_ids)
    contrast_probs = output2prob(attribution_model, contrast_output, contrast_target_ids)
    return model_probs - contrast_probs


def test_contrastive_attribution(saliency_mt_model: HuggingfaceEncoderDecoderModel):
    """Runs a contrastive feature attribution using the method relying on logits difference
    introduced by [Yin and Neubig '22](https://arxiv.org/pdf/2202.10419.pdf), taking advantage of
    the custom feature attribution target function module.
    """
    # Register the function defined above
    # Since outputs are still probabilities, contiguous tokens can still be aggregated using product
    inseq.register_step_score(
        fn=attribute_contrast_logits_diff,
        identifier="contrast_logits_diff",
        aggregate_map={"span_aggregate": lambda x: x.prod(dim=1, keepdim=True)},
    )

    # Pre-compute ids and attention map for the contrastive target
    contrast = saliency_mt_model.encode("Non posso crederlo.", as_targets=True, prepend_bos_token=True)

    # Perform the contrastive attribution:
    # Regular (forced) target -> "Non posso crederci."
    # Contrastive target      -> "Non posso crederlo."
    # contrast_ids & contrast_attention_mask are kwargs defined in the function definition
    out = saliency_mt_model.attribute(
        "I can't believe it",
        "Non posso crederci.",
        attributed_fn="contrast_logits_diff",
        contrast_ids=contrast.input_ids,
        contrast_attention_mask=contrast.attention_mask,
    )
    attribution_scores = out.sequence_attributions[0].source_attributions

    # Since the two target strings are identical for the first three tokens (Non posso creder)
    # the resulting contrastive source attributions should be all 0
    assert attribution_scores[:, :3].sum().eq(0)

    # Starting at the following token in which they differ, scores should diverge
    assert not attribution_scores[:, :4].sum().eq(0)


def attribute_mcd_uncertainty_weighted(
    attribution_model,
    forward_output,
    encoder_input_embeds,
    encoder_attention_mask,
    decoder_input_embeds,
    decoder_attention_mask,
    target_ids,
    aux_model,
    n_mcd_steps: int = 10,
    **kwargs,
):
    """Custom attribution function normalizing probabilities normally used as attribution
    targets over a pool of noisy prediction computed with MC Dropout, to increase their robustness.

    Extra args:
        aux_model: Model in train mode to produce noisy probability predictions thanks to dropout
            modules.
        n_mcd_steps: The number of predictions that should be used to normalize the original output.
    """
    noisy_probs = []
    # Compute noisy predictions using the auxiliary model
    # Important: must be in train mode to ensure noise for MCD
    for _ in range(n_mcd_steps):
        aux_marian_out = aux_model(
            inputs_embeds=encoder_input_embeds,
            attention_mask=encoder_attention_mask,
            decoder_inputs_embeds=decoder_input_embeds,
            decoder_attention_mask=decoder_attention_mask,
        )
        noisy_prob = output2prob(attribution_model, aux_marian_out, target_ids)
        noisy_probs.append(noisy_prob)
    # Original probability from the model without noise
    orig_prob = output2prob(attribution_model, forward_output, target_ids)
    # Z-score the original based on the mean and standard deviation of MC dropout predictions
    return (orig_prob - torch.stack(noisy_probs).mean(0)).div(torch.stack(noisy_probs).std(0))


def test_mcd_weighted_attribution(saliency_mt_model, auxiliary_saliency_mt_model):
    """Runs a MCD-weighted feature attribution taking advantage of
    the custom feature attribution target function module.
    """
    # Register the function defined above
    # Since outputs are still probabilities, contiguous tokens can still be aggregated using product
    inseq.register_step_score(
        fn=attribute_mcd_uncertainty_weighted,
        identifier="mcd_uncertainty_weighted",
        aggregate_map={"span_aggregate": lambda x: x.prod(dim=1, keepdim=True)},
    )

    out = saliency_mt_model.attribute(
        "Hello ladies and badgers!",
        attributed_fn="mcd_uncertainty_weighted",
        attributed_fn_args={"n_mcd_steps": 5, "aux_model": auxiliary_saliency_mt_model.to(saliency_mt_model.device)},
    )
    attribution_scores = out.sequence_attributions[0].source_attributions
    assert isinstance(attribution_scores, torch.Tensor)
