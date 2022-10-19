import pytest
import torch

from inseq.utils.misc import pretty_tensor


@pytest.mark.parametrize(
    ("tensor", "output"),
    [
        (None, "None"),
        (torch.tensor([1, 1], dtype=torch.long), "torch.int64 tensor of shape [2]"),
        (
            torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
            "torch.int64 tensor of shape [2, 3]",
        ),
        (torch.randn(4, 1, 1, 10), "torch.float32 tensor of shape [4, 1, 1, 10]"),
        (torch.randn(2, 21, 1), "torch.float32 tensor of shape [2, 21, 1]"),
    ],
)
def test_pretty_tensor(tensor: torch.Tensor, output: str) -> None:
    assert pretty_tensor(tensor).startswith(output)


def test_probits2prob():
    # Test with batch of size > 1
    probits = torch.stack(
        [
            torch.arange(0, 30000, 1.0),
            torch.arange(30000, 60000, 1.0),
            torch.arange(60000, 90000, 1.0),
            torch.arange(90000, 120000, 1.0),
        ]
    )
    target_ids = torch.tensor([10, 77, 999, 1765]).unsqueeze(-1)
    probs = torch.gather(probits, -1, target_ids.T)
    assert probs.shape == (1, 4)
    assert torch.eq(probs, torch.tensor([10.0, 77.0, 999.0, 1765.0])).all()

    # Test with batch of size 1
    probits = torch.stack(
        [
            torch.arange(0, 30000, 1.0),
        ]
    )
    target_ids = torch.tensor([23456]).unsqueeze(-1)
    probs = torch.gather(probits, -1, target_ids.T)
    assert probs.shape == (1, 1)
    assert torch.eq(probs, torch.tensor([23456.0])).all()
