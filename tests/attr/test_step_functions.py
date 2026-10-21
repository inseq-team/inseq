import math
from types import SimpleNamespace

import torch

from inseq.attr.step_functions import crossentropy_fn, perplexity_fn


def _uniform_args(vocab_size: int, batch_size: int = 3) -> SimpleNamespace:
    logits = torch.zeros(batch_size, vocab_size)
    model = SimpleNamespace(output2logits=lambda output: output)
    return SimpleNamespace(attribution_model=model, forward_output=logits, target_ids=torch.zeros(batch_size).long())


def test_perplexity_of_uniform_distribution_is_vocab_size():
    # A uniform prediction over V tokens has perplexity V
    args = _uniform_args(vocab_size=100)
    torch.testing.assert_close(perplexity_fn(args), torch.full((3,), 100.0))


def test_perplexity_is_exp_of_crossentropy():
    args = _uniform_args(vocab_size=7)
    torch.testing.assert_close(crossentropy_fn(args), torch.full((3,), math.log(7)))
    torch.testing.assert_close(perplexity_fn(args), crossentropy_fn(args).exp())
