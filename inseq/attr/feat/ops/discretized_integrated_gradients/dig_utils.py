# Adapted from https://github.com/INK-USC/DIG/blob/main/monotonic_paths.py, licensed MIT:
# Copyright © 2021 Intelligence and Knowledge Discovery (INK) Research Lab at University of Southern California

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the “Software”), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Any, List, Optional, Tuple

import logging
import os
from collections import defaultdict
from pathlib import Path

import torch
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from torchtyping import TensorType

from .....utils.cache import INSEQ_ARTIFACTS_CACHE
from .....utils.misc import cache_results

logger = logging.getLogger(__name__)

TokenEmbeddings = TensorType["vocab_size", "embed_size", float]


class UnknownDIGStrategy(Exception):
    """Raised when a strategy for DIG is not valid"""

    def __init__(
        self,
        strategy: str,
        *args: Tuple[Any],
    ) -> None:
        super().__init__(
            f"Unknown strategy: {strategy}.\nAvailable strategies: greedy, maxcount",
            *args,
        )


class MonotonicPathBuilder:
    def __init__(
        self,
        token_embeddings: TokenEmbeddings,
        knn_graph: csr_matrix,
    ) -> None:
        self.token_embeddings = token_embeddings
        self.knn_graph = knn_graph

    @staticmethod
    @cache_results
    def compute_embeddings_knn(
        token_embeddings: Optional[TokenEmbeddings],
        n_neighbors: int = 50,
        mode: str = "distance",
        n_jobs: int = -1,
    ) -> Tuple[TokenEmbeddings, csr_matrix]:
        """
        Etiher loads or computes the knn graph for token embeddings.
        """
        knn_graph = kneighbors_graph(
            token_embeddings,
            n_neighbors=n_neighbors,
            mode=mode,
            n_jobs=n_jobs,
        )
        return [token_embeddings, knn_graph]

    @classmethod
    def load(
        cls,
        model_name: str,
        n_neighbors: int = 50,
        mode: str = "distance",
        n_jobs: int = -1,
        save_cache: bool = True,
        overwrite_cache: bool = False,
        cache_dir: Path = INSEQ_ARTIFACTS_CACHE / "dig_knn",
        token_embeddings: Optional[
            TensorType["vocab_size", "embed_size", float]
        ] = None,
    ) -> "MonotonicPathBuilder":
        cache_filename = os.path.join(
            cache_dir, f"{model_name.replace('/', '__')}_{n_neighbors}.pkl"
        )
        if token_embeddings is None:
            logger.warning(
                "Since no token embeddings are passed, a cached file is expected. "
                "If the file is not found, an exception will be raised."
            )
        # Cache parameters are passed to the cache_results decorator
        [token_embeddings, knn_graph] = cls.compute_embeddings_knn(
            cache_dir,
            cache_filename,
            save_cache,
            overwrite_cache,
            token_embeddings=token_embeddings,
            n_neighbors=n_neighbors,
            mode=mode,
            n_jobs=n_jobs,
        )
        return cls(token_embeddings, knn_graph)

    @staticmethod
    def get_monotonic_dims(
        interpolated: torch.Tensor,
        baseline: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if the interpolation vector is monotonic w.r.t. the baseline and the input.
        """
        return torch.where(
            (baseline > original)
            * (baseline >= interpolated)
            * (interpolated >= original)
            + (baseline < original)
            * (original >= interpolated)
            * (interpolated >= baseline)
            + (baseline == original)
            * (baseline == interpolated)
            * (interpolated == original),
            1,
            0,
        ).bool()

    @classmethod
    def make_monotonic_vec(
        cls,
        anchor: torch.Tensor,
        baseline: torch.Tensor,
        input: torch.Tensor,
        steps: Optional[int] = 30,
    ) -> torch.Tensor:
        """Create a new vector from an existing anchor that is monotonic w.r.t. input and baseline"""
        non_monotonic_dims = ~cls.get_monotonic_dims(anchor, baseline, input)
        if non_monotonic_dims.sum() == 0:
            return anchor
        # make the new vector monotonic
        new_vec = anchor.clone()
        new_vec[non_monotonic_dims] = input[non_monotonic_dims] - (1.0 / steps) * (
            input[non_monotonic_dims] - baseline[non_monotonic_dims]
        )
        return new_vec

    @staticmethod
    def euclidean_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
        """Euclidean distance between two points."""
        return (vec_a - vec_b).pow(2).sum(-1).sqrt()

    def get_next_word(
        self, word_idx, reference_idx, word_path, strategy="greedy", steps=5
    ):
        # If (for some reason) we do select the ref_idx as the previous anchor word,
        # then all further anchor words should be ref_idx
        if word_idx == reference_idx:
            return reference_idx
        anchor_map = defaultdict(list)
        cx = self.knn_graph[word_idx].tocoo()
        for j in cx.col:
            # we should not consider the anchor word to be the ref_idx [baseline]
            # unless forced to.
            if j == reference_idx:
                continue
            if strategy == "greedy":
                # calculate the distance of the monotonized vec from the anchor point
                monotonic_vec = self.make_monotonic_vec(
                    self.token_embeddings[j],
                    self.token_embeddings[reference_idx],
                    self.token_embeddings[word_idx],
                    steps,
                )
                anchor_map[j] = [
                    self.euclidean_distance(self.token_embeddings[j], monotonic_vec)
                ]
            elif strategy == "maxcount":
                # count the number of non-monotonic dimensions
                # 10000 is an arbitrarily high to be agnostic
                # of token_embeddings dimensionality
                monotonic_dims = self.get_monotonic_dims(
                    self.token_embeddings[j],
                    self.token_embeddings[reference_idx],
                    self.token_embeddings[word_idx],
                )
                non_monotonic_count = 10000 - monotonic_dims.sum()
                anchor_map[j] = [non_monotonic_count]
            else:
                raise UnknownDIGStrategy(strategy)
        if len(anchor_map) == 0:
            return reference_idx
        sorted_dist_map = {
            k: v for k, v in sorted(anchor_map.items(), key=lambda item: item[1][0])
        }
        # remove words that are already selected in the path
        for key in word_path:
            sorted_dist_map.pop(key, None)
        if len(sorted_dist_map) == 0:
            return reference_idx
        # return the top key
        return next(iter(sorted_dist_map))

    def find_word_path(
        self,
        word_idx,
        reference_idx,
        steps=5,
        strategy="greedy",
        special_token_ids: List[int] = [],
    ):
        print(word_idx)
        # if word_idx is a special token copy it and return
        if word_idx in special_token_ids:
            return [word_idx] * (steps + 1)
        word_path = [word_idx]
        last_idx = word_idx
        for _ in range(steps):
            next_idx = self.get_next_word(
                word_idx=last_idx,
                reference_idx=reference_idx,
                word_path=word_path,
                strategy=strategy,
                steps=steps,
            )
            word_path.append(next_idx)
            last_idx = next_idx
        return word_path

    def make_monotonic_path(self, word_path_ids, reference_idx, steps: int = 5):
        monotonic_embs = [self.token_embeddings[word_path_ids[0]]]
        baseline_vec = self.token_embeddings[reference_idx]
        for idx in range(len(word_path_ids) - 1):
            input_vec = monotonic_embs[-1]
            anchor_vec = self.token_embeddings[word_path_ids[idx + 1]]
            monotonic_vec = self.make_monotonic_vec(
                anchor_vec, baseline_vec, input_vec, steps
            )
            monotonic_embs.append(monotonic_vec)
        monotonic_embs.append(baseline_vec)
        # reverse the list so that baseline is the first and input word is the last
        monotonic_embs.reverse()
        final_embs = monotonic_embs
        # verify monotonicity
        check = True
        for i in range(len(final_embs) - 1):
            monotonic_dims = self.get_monotonic_dims(
                final_embs[i + 1], final_embs[-1], final_embs[i]
            )
            is_fully_monotonic = monotonic_dims.sum() == final_embs[-1].shape[0]
            check *= is_fully_monotonic
        assert check, "The embeddings are not monotonic"
        return final_embs

    def scale_inputs(
        self,
        input_ids,
        baseline_ids,
        scale_steps=30,
        strategy="greedy",
        special_token_ids: List[int] = [],
    ):
        """Generate paths required by DIG."""
        all_path_embs = []
        for seq_idx in range(len(input_ids.squeeze().tolist())):
            #    seq_path_embs = []
            #    for tok_idx in range(len(input_ids[seq_idx])):
            print("Find word path")
            word_path = self.find_word_path(
                input_ids.squeeze()[seq_idx],  # [tok_idx],
                baseline_ids.squeeze()[seq_idx],  # [tok_idx],
                steps=scale_steps,
                strategy=strategy,
                special_token_ids=special_token_ids,
            )
            print("Make monotonic path")
            monotonic_embs = self.make_monotonic_path(
                word_path,
                baseline_ids.squeeze()[seq_idx],  # [tok_idx],
                steps=scale_steps,
            )
            # seq_path_embs.append(monotonic_embs)
            # all_path_embs.append(seq_path_embs)
            all_path_embs.append(monotonic_embs)
        # all_path_embs = torch.tensor(
        #    np.stack(all_path_embs, axis=1),
        #    dtype=torch.float,
        #    device=input_ids.device,
        #    requires_grad=True,
        # )
        # print(all_path_embs.shape)
        return all_path_embs
