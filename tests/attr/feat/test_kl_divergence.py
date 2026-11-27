from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from inseq.attr.step_functions import StepFunctionDecoderOnlyArgs, kl_divergence_fn


def _make_kl_args(original_logits: torch.Tensor, contrast_logits: torch.Tensor) -> StepFunctionDecoderOnlyArgs:
    mock_model = MagicMock()
    mock_model.is_encoder_decoder = False
    mock_model.output2logits.side_effect = lambda out: out.logits
    mock_model.get_forward_output.return_value = SimpleNamespace(logits=contrast_logits)
    mock_model.formatter.convert_args_to_batch.return_value = "dummy_batch"
    batch_size = original_logits.size(0)
    return StepFunctionDecoderOnlyArgs(
        attribution_model=mock_model,
        forward_output=SimpleNamespace(logits=original_logits),
        target_ids=torch.zeros(batch_size, dtype=torch.long),
        decoder_input_ids=torch.ones(batch_size, 2, dtype=torch.long),
        decoder_input_embeds=torch.zeros(batch_size, 2, 4),
        decoder_attention_mask=torch.ones(batch_size, 2, dtype=torch.long),
        is_attributed_fn=False,
    )


def test_kl_divergence_top_k_finite_and_matches_support():
    orig_logits = torch.tensor([[1.0, 2.0, 3.0, 10.0, 12.0]], requires_grad=True)
    contrast_logits = torch.tensor([[1.0, 2.0, 3.0, 12.0, 10.0]])
    args = _make_kl_args(orig_logits, contrast_logits)

    kl = kl_divergence_fn(args, top_k=2)

    expected = F.kl_div(
        F.log_softmax(contrast_logits[:, 3:], dim=-1),
        F.log_softmax(orig_logits.detach()[:, 3:], dim=-1),
        reduction="sum",
        log_target=True,
    )
    assert torch.isfinite(kl).all()
    assert torch.allclose(kl, expected.unsqueeze(0))

    kl.sum().backward()
    assert orig_logits.grad is not None
    assert torch.isfinite(orig_logits.grad).all()
    assert torch.all(orig_logits.grad[:, :3] == 0.0)


def test_kl_divergence_top_p_batched_variable_support():
    orig_logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 10.0, 12.0],
            [0.0, 0.0, 5.0, 5.0, 5.0],
        ]
    )
    contrast_logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 12.0, 10.0],
            [0.0, 0.0, 4.0, 5.0, 6.0],
        ]
    )
    args = _make_kl_args(orig_logits, contrast_logits)

    kl = kl_divergence_fn(args, top_p=0.9)

    assert kl.shape == (2,)
    assert kl.device == orig_logits.device
    assert torch.isfinite(kl).all()
    assert (kl > 0).all()


def test_kl_divergence_unfiltered_matches_pytorch():
    orig_logits = torch.tensor([[1.0, 2.0, 3.0, 10.0, 12.0], [0.5, -1.0, 2.5, 0.0, 1.5]])
    contrast_logits = torch.tensor([[1.0, 2.0, 3.0, 12.0, 10.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
    args = _make_kl_args(orig_logits, contrast_logits)

    kl = kl_divergence_fn(args)

    expected = torch.stack(
        [
            F.kl_div(
                F.log_softmax(contrast_logits[i], dim=-1),
                F.log_softmax(orig_logits[i], dim=-1),
                reduction="sum",
                log_target=True,
            )
            for i in range(orig_logits.size(0))
        ]
    )
    assert kl.shape == (2,)
    assert torch.allclose(kl, expected)
