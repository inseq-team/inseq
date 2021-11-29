import pytest
import torch

from inseq.utils.misc import pretty_tensor


@pytest.mark.parametrize(
    ("tensor", "output"),
    [
        (None, "None"),
        (torch.tensor([1, 1], dtype=torch.long), "tensor of shape [2]"),
        (
            torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
            "tensor of shape [2, 3]",
        ),
        (torch.randn(4, 1, 1, 10), "tensor of shape [4, 1, 1, 10]"),
        (torch.randn(2, 21, 1), "tensor of shape [2, 21, 1]"),
    ],
)
def test_pretty_tensor(tensor: torch.Tensor, output: str) -> None:
    assert pretty_tensor(tensor).startswith(output)
