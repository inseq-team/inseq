"""Tests for hello function."""
import pytest

from amseq.outputs import GradientAttributionOutput


@pytest.mark.parametrize(
    ("source_tokens", "target_tokens", "attributions", "deltas", "expected"),
    [
        (
            ["▁I", "▁like", "▁to", "▁eat", "▁cake", ".", "</s>"],
            ["▁Mi", "▁piace", "▁mangiare", "▁la", "▁tor", "ta", "." "</s>"],
            [[1], [2], [3]],
            [1, 2, 3],
            "GradientAttributionOutput(\n"
            "  source_tokens=['▁I', '▁like', '▁to', '▁eat', '▁cake', '.', '</s>'],\n"
            "  target_tokens=['▁Mi', '▁piace', '▁mangiare', '▁la', '▁tor', 'ta', '.</s>'],\n"
            "  attributions=[[1], [2], [3]],\n"
            "  deltas=[1, 2, 3]\n"
            ")",
        ),
    ],
)
def test_gradient_attribution_str(
    source_tokens, target_tokens, attributions, deltas, expected
):
    """Test gradient attribution string output."""
    output = GradientAttributionOutput(
        source_tokens, target_tokens, attributions, deltas
    )
    assert str(output) == expected
