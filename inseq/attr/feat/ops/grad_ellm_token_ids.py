"""Exact-token replay avoids generation text decoding/re-tokenization changes."""

import torch

from inseq.data import BatchEncoding, FeatureAttributionSequenceOutput

from .grad_ellm_device import attribution_device, model_input_device


def attribute_token_ids(wrapped_model, token_ids, prompt_length, attention_mask=None, **kwargs):
    """Attribute one full prompt+continuation sequence without re-tokenization.

    This uses Inseq's lower-level batch API because model.attribute validates
    string prefixes. Batch size one is intentional for checkpoint parity.
    prompt_length is measured in IDs, not characters. EOS is retained.
    """
    device = model_input_device(wrapped_model.model)
    with attribution_device(device):
        return _attribute_token_ids(wrapped_model, token_ids, prompt_length, attention_mask, **kwargs)


def _attribute_token_ids(wrapped_model, token_ids, prompt_length, attention_mask=None, **kwargs):
    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("Exact-token helper currently accepts one sequence: [1, total_length].")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Token IDs must be integer tensors (int32/int64).")
    token_ids = token_ids.detach().clone().long()
    if isinstance(prompt_length, bool) or not isinstance(prompt_length, int):
        raise TypeError("prompt_length must be an integer measured in token IDs.")
    if not 1 <= prompt_length < token_ids.shape[1]:
        raise ValueError("Require a non-empty prompt and at least one continuation token.")
    if attention_mask is None:
        attention_mask = torch.ones_like(token_ids)
    if (
        not isinstance(attention_mask, torch.Tensor)
        or attention_mask.shape != token_ids.shape
        or not (attention_mask == 1).all()
    ):
        raise ValueError("Exact-token helper expects an unpadded, fully valid sequence.")
    if wrapped_model.attribution_method is None:
        raise ValueError("Load the wrapper with an attribution method before exact-token replay.")
    vocab_size = wrapped_model.model.get_input_embeddings().num_embeddings
    if (token_ids < 0).any() or (token_ids >= vocab_size).any():
        raise ValueError("Token IDs must lie within the model embedding vocabulary.")
    # The baseline is unused by Grad-ELLM, but Inseq still embeds it. Do not
    # assume every tokenizer defines UNK (e.g. tokenizers used for Llama).
    baseline_id = next(
        (
            value
            for value in (
                wrapped_model.tokenizer.unk_token_id,
                wrapped_model.tokenizer.pad_token_id,
                wrapped_model.tokenizer.eos_token_id,
                wrapped_model.tokenizer.bos_token_id,
                0,
            )
            if isinstance(value, int) and 0 <= value < vocab_size
        ),
        0,
    )
    encoded = BatchEncoding(
        input_ids=token_ids,
        attention_mask=attention_mask,
        # Inseq prepares baseline embeddings even for methods that do not use a baseline.
        baseline_ids=torch.full_like(token_ids, baseline_id),
        input_tokens=wrapped_model.convert_ids_to_tokens(token_ids, skip_special_tokens=False),
    )
    # Low-level formatters embed BatchEncoding as-is, unlike model.encode().
    # Move IDs, mask AND baseline before either embedding lookup.
    encoded = encoded.to(wrapped_model.get_embedding_layer().weight.device)
    batch = wrapped_model.formatter.prepare_inputs_for_attribution(wrapped_model, encoded)
    attributed_fn = wrapped_model.get_attributed_fn(kwargs.pop("attributed_fn", "logit"))
    keep_steps = kwargs.pop("output_step_attributions", False)
    out = wrapped_model.attribution_method.attribute(
        batch, attributed_fn=attributed_fn, attr_pos_start=prompt_length, output_step_attributions=True, **kwargs
    )
    # Every ID here is valid: rebuild from retained steps without token-value
    # padding removal. Otherwise PAD == EOS silently deletes real EOS scores.
    target_tokens = wrapped_model.get_token_with_ids(batch)
    if kwargs.get("clean_special_chars", False):
        target_tokens = wrapped_model.clean_tokens(target_tokens, as_targets=True)
    out.sequence_attributions = FeatureAttributionSequenceOutput.from_step_attributions(
        out.step_attributions,
        target_tokens,
        pad_token=None,
        attr_pos_end=out.info["attr_pos_end"],
    )
    if not keep_steps:
        out.step_attributions = None
    out.info["output_step_attributions"] = keep_steps
    out.info["attributed_fn"] = attributed_fn.__name__
    return out
