from typing import Tuple

import pytest
import torch

from inseq.attr.feat.ops import MonotonicPathBuilder
from inseq.utils import euclidean_distance


def original_monotonic(vec1, vec2, vec3):
    """Taken verbatim from https://github.com/INK-USC/DIG/blob/main/monotonic_paths.py"""
    increasing_dims = vec1 > vec2  # dims where baseline > input
    decreasing_dims = vec1 < vec2  # dims where baseline < input
    equal_dims = vec1 == vec2  # dims where baseline == input
    vec3_greater_vec1 = vec3 >= vec1
    vec3_greater_vec2 = vec3 >= vec2
    vec3_lesser_vec1 = vec3 <= vec1
    vec3_lesser_vec2 = vec3 <= vec2
    vec3_equal_vec1 = vec3 == vec1
    vec3_equal_vec2 = vec3 == vec2
    valid = (
        increasing_dims * vec3_lesser_vec1 * vec3_greater_vec2
        + decreasing_dims * vec3_greater_vec1 * vec3_lesser_vec2
        + equal_dims * vec3_equal_vec1 * vec3_equal_vec2
    )
    return valid  # vec condition


def original_dummy_find_word_path(wrd_idx: int, n_steps: int):
    word_path = [wrd_idx]
    last_idx = wrd_idx
    for _ in range(n_steps):
        # The original code finds the next word
        # We only want to test the walrus operator variant, so any method is ok. We use hash.
        next_idx = hash(last_idx + 0.01 + len(word_path) / 1000)
        word_path.append(next_idx)
        last_idx = next_idx
    return word_path


def walrus_operator_find_word_path(wrd_idx: int, n_steps: int):
    word_path = [wrd_idx]
    for _ in range(n_steps):
        word_path.append((wrd_idx := hash(wrd_idx + 0.01 + len(word_path) / 1000)))
    return word_path


@pytest.mark.parametrize(
    ("input_dims"),
    [
        ((512,)),
        ((512, 12)),
        ((512, 12, 3)),
    ],
)
def test_equivalent_monotonic_method(input_dims: Tuple[int, ...]) -> None:
    torch.manual_seed(42)
    baseline_embeds = torch.randn(input_dims)
    input_embeds = torch.randn(input_dims)
    interpolated_embeds = torch.randn(input_dims)
    out = MonotonicPathBuilder.get_monotonic_dims(
        interpolated_embeds, baseline_embeds, input_embeds
    )
    orig_out = original_monotonic(baseline_embeds, input_embeds, interpolated_embeds)
    assert torch.equal(out.int(), orig_out.int())


def test_valid_distance_multidim_tensors() -> None:
    torch.manual_seed(42)
    vec_a = torch.randn((512,))
    vec_b = torch.randn((512,))
    vec_a_multi = torch.stack([vec_a, vec_a, vec_a], dim=0)
    vec_b_multi = torch.stack([vec_b, vec_b, vec_b], dim=0)
    assert list(vec_a.shape) == list(vec_b.shape) == [512]
    assert list(vec_a_multi.shape) == list(vec_b_multi.shape) == [3, 512]
    dist = euclidean_distance(vec_a, vec_b)
    dist_multi = euclidean_distance(vec_a_multi, vec_b_multi)
    assert not list(dist.shape)  # scalar
    assert list(dist_multi.shape)[0] == vec_a_multi.shape[0] == vec_b_multi.shape[0]
    assert (
        torch.equal(dist_multi[0], dist)
        and torch.equal(dist_multi[1], dist)
        and torch.equal(dist_multi[2], dist)
    )


@pytest.mark.parametrize(
    ("wrd_idx", "n_steps"),
    [
        (0, 10),
        (10, 100),
        (100, 1000),
    ],
)
def test_walrus_find_word_path(wrd_idx: int, n_steps: int) -> None:
    assert original_dummy_find_word_path(
        wrd_idx, n_steps
    ) == walrus_operator_find_word_path(wrd_idx, n_steps)
