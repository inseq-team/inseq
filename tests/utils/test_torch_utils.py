import pytest
import torch

from inseq.utils.misc import pretty_tensor
from inseq.utils.torch_utils import filter_logits


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


def test_filter_logits():
    original_logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
        ]
    )
    filtered_logits = filter_logits(original_logits, top_k=2)
    topk2 = original_logits.clone().index_fill(1, torch.tensor([0, 1, 2]), float("-inf"))
    assert torch.eq(filtered_logits, topk2).all()
    filtered_logits = filter_logits(original_logits, top_p=0.9)
    topp90 = original_logits.clone().index_fill(1, torch.tensor([0, 1]), float("-inf"))
    assert torch.eq(filtered_logits, topp90).all()
    contrast_logits = torch.tensor(
        [
            [15.0, 13.0, 11.0, 9.0, 7.0],
            [13.0, 11.0, 9.0, 7.0, 5.0],
            [11.0, 9.0, 7.0, 5.0, 3.0],
            [9.0, 7.0, 5.0, 3.0, 1.0],
        ]
    )
    filtered_logits, contrast_logits = filter_logits(original_logits, contrast_logits=contrast_logits, top_k=2)
    top2merged = original_logits.clone().index_fill(1, torch.tensor([2, 3, 4]), float("-inf"))
    assert torch.eq(filtered_logits, top2merged).all()
