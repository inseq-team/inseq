import pytest
import torch

import inseq
from inseq.attr.step_functions import get_step_scores

from ...inference_commons import get_example_batches


@pytest.fixture(scope="session")
def encoder_decoder_batches():
    return get_example_batches()


@pytest.fixture(scope="session")
def m2m100_model():
    return inseq.load_model(
        "facebook/m2m100_418M",
        "integrated_gradients",
        tokenizer_kwargs={"src_lang": "en", "tgt_lang": "en", "use_fast": False},
    )


def test_get_step_prediction_probabilities(m2m100_model, encoder_decoder_batches):
    # fmt: off
    probas = [
        0.622, 0.008, 0.006, 0.841, 0.002, 0.127, 0.003, 0.087, 0.0,
        0.843, 0.744, 0.865, 0.012,  0.27, 0.085, 0.739, 0.749, 0.9,
    ]
    # fmt: on
    for i, (batch, next_batch) in enumerate(
        zip(encoder_decoder_batches["batches"][1:], encoder_decoder_batches["batches"][2:])
    ):
        output = m2m100_model.get_forward_output(
            batch.to(m2m100_model.device), use_embeddings=m2m100_model.attribution_method.forward_batch_embeds
        )
        target_ids = next_batch.targets.encoding.input_ids[0, -1].to(m2m100_model.device)
        step_fn_args = m2m100_model.formatter.format_step_function_args(
            attribution_model=m2m100_model, forward_output=output, target_ids=target_ids, batch=batch
        )
        pred_proba = get_step_scores("probability", step_fn_args)
        assert float(pred_proba) == pytest.approx(probas[i], abs=1e-3)


def test_crossentropy_nlll_equivalence(m2m100_model, encoder_decoder_batches):
    for batch, next_batch in zip(encoder_decoder_batches["batches"][1:], encoder_decoder_batches["batches"][2:]):
        batch.to(m2m100_model.device)
        next_batch.to(m2m100_model.device)
        output = m2m100_model.model(
            input_ids=None,
            inputs_embeds=batch.sources.input_embeds,
            attention_mask=batch.sources.attention_mask,
            decoder_inputs_embeds=batch.targets.input_embeds,
            decoder_attention_mask=batch.targets.attention_mask,
        )
        # Full logits for last position of every sentence:
        # (batch_size, tgt_seq_len, vocab_size) => (batch_size, vocab_size)
        logits = output.logits[:, -1, :].squeeze(1).detach().to("cpu")
        batch.detach().to("cpu")
        next_batch.detach().to("cpu")
        cross_entropy = torch.nn.functional.cross_entropy(logits, next_batch.targets.encoding.input_ids[:, -1])
        nlll = torch.nn.functional.nll_loss(
            torch.log(torch.softmax(logits, dim=-1)), next_batch.targets.encoding.input_ids[:, -1]
        )
        assert cross_entropy == pytest.approx(nlll, abs=1e-3)
