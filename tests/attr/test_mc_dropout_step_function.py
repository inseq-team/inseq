from types import SimpleNamespace

import torch

from inseq.attr.step_functions import mc_dropout_prob_avg_fn


class _DropoutModel(torch.nn.Module):
    """Minimal stand-in for an AttributionModel whose forward output is the logits."""

    is_encoder_decoder = False

    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.logits = torch.randn(2, 11)
        self.formatter = SimpleNamespace(convert_args_to_batch=lambda args: None)

    def output2logits(self, forward_output):
        return forward_output

    def get_forward_output(self, batch, use_embeddings=False):
        return self.dropout(self.logits)


def test_mc_dropout_prob_avg_restores_eval_mode():
    torch.manual_seed(0)
    model = _DropoutModel().eval()
    args = SimpleNamespace(attribution_model=model, forward_output=model.logits, target_ids=torch.tensor([1, 2]))
    mc_dropout_prob_avg_fn(args, n_mcd_steps=3)
    assert not model.training
    assert not model.dropout.training
