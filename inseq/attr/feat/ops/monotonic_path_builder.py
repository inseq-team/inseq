# Adapted from https://github.com/INK-USC/DIG/blob/main/monotonic_paths.py, licensed MIT:
# Copyright © 2021 The Inseq Team and the Intelligence and Knowledge Discovery (INK) Research Lab
# at the University of Southern California

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

import logging
import os
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from scipy.sparse import csr_matrix
from torchtyping import TensorType

from ....utils import is_joblib_available, is_scikitlearn_available

if is_joblib_available():
    from joblib import Parallel, delayed

if is_scikitlearn_available():
    from sklearn.neighbors import kneighbors_graph

from ....utils import INSEQ_ARTIFACTS_CACHE, cache_results, euclidean_distance
from ....utils.typing import MultiStepEmbeddingsTensor, VocabularyEmbeddingsTensor

logger = logging.getLogger(__name__)


class PathBuildingStrategies(Enum):
    GREEDY = "greedy"  # Based on the Euclidean distance between embeddings.
    MAXCOUNT = "maxcount"  # Based on the number of monotonic dimensions


class UnknownPathBuildingStrategy(Exception):
    """Raised when a strategy for pathbuilding is not valid"""

    def __init__(
        self,
        strategy: str,
        *args: Tuple[Any],
    ) -> None:
        super().__init__(
            (
                f"Unknown strategy: {strategy}.\nAvailable strategies: "
                f"{','.join([s.value for s in PathBuildingStrategies])}"
            ),
            *args,
        )


class MonotonicPathBuilder:
    def __init__(
        self,
        vocabulary_embeddings: VocabularyEmbeddingsTensor,
        knn_graph: csr_matrix,
        special_tokens: List[int] = [],
    ) -> None:
        self.vocabulary_embeddings = vocabulary_embeddings
        self.knn_graph = knn_graph
        self.special_tokens = special_tokens

    @staticmethod
    @cache_results
    def compute_embeddings_knn(
        vocabulary_embeddings: Optional[VocabularyEmbeddingsTensor],
        n_neighbors: int = 50,
        mode: str = "distance",
        n_jobs: int = -1,
    ) -> csr_matrix:
        """
        Etiher loads or computes the knn graph for token embeddings.
        """
        if not is_scikitlearn_available():
            raise ImportError("scikit-learn is not available. Please install it to use MonotonicPathBuilder.")
        knn_graph = kneighbors_graph(
            vocabulary_embeddings,
            n_neighbors=n_neighbors,
            mode=mode,
            n_jobs=n_jobs,
        )
        return knn_graph

    @classmethod
    def load(
        cls,
        model_name: str,
        n_neighbors: int = 50,
        mode: str = "distance",
        n_jobs: int = -1,
        save_cache: bool = True,
        overwrite_cache: bool = False,
        cache_dir: Path = INSEQ_ARTIFACTS_CACHE / "path_knn",
        vocabulary_embeddings: Optional[VocabularyEmbeddingsTensor] = None,
        special_tokens: List[int] = [],
        embedding_scaling: int = 1,
    ) -> "MonotonicPathBuilder":
        cache_filename = os.path.join(cache_dir, f"{model_name.replace('/', '__')}_{n_neighbors}.pkl")
        if vocabulary_embeddings is None:
            logger.warning(
                "Since no token embeddings are passed, a cached file is expected. "
                "If the file is not found, an exception will be raised."
            )
        vocabulary_embeddings = vocabulary_embeddings * embedding_scaling
        # Cache parameters are passed to the cache_results decorator
        knn_graph = cls.compute_embeddings_knn(
            cache_dir,
            cache_filename,
            save_cache,
            overwrite_cache,
            vocabulary_embeddings=vocabulary_embeddings,
            n_neighbors=n_neighbors,
            mode=mode,
            n_jobs=n_jobs,
        )
        return cls(vocabulary_embeddings, knn_graph, special_tokens)

    def scale_inputs(
        self,
        input_ids: TensorType["batch_size", "seq_len", int],
        baseline_ids: TensorType["batch_size", "seq_len", int],
        n_steps: Optional[int] = None,
        scale_strategy: Optional[str] = None,
    ) -> MultiStepEmbeddingsTensor:
        """Generate paths required by DIG."""
        if n_steps is None:
            n_steps = 30
        if scale_strategy is None:
            scale_strategy = "greedy"
        if not is_joblib_available():
            raise ImportError("joblib is not available. Please install it to use MonotonicPathBuilder.")
        word_paths_flat = Parallel(n_jobs=3, prefer="threads")(
            delayed(self.find_path)(
                int(input_ids[seq_idx, tok_idx]),
                int(baseline_ids[seq_idx, tok_idx]),
                n_steps=n_steps,
                strategy=scale_strategy,
            )
            for seq_idx in range(input_ids.shape[0])
            for tok_idx in range(input_ids.shape[1])
        )
        # Unflatten word paths
        word_paths_iter = iter(word_paths_flat)
        word_paths = [list(islice(word_paths_iter, input_ids.shape[1])) for _ in range(input_ids.shape[0])]
        # Fill embeddings list
        lst_all_seq_embeds = []
        for seq_idx in range(input_ids.shape[0]):
            lst_curr_seq_embeds = []
            for tok_idx in range(input_ids.shape[1]):
                lst_curr_seq_embeds.append(
                    self.build_monotonic_path_embedding(
                        word_path=word_paths[seq_idx][tok_idx],
                        baseline_idx=int(baseline_ids[seq_idx, tok_idx]),
                        n_steps=n_steps,
                    )
                )
            # out shape: n_steps x seq_len x hidden_size
            t_curr_seq_embeds = torch.stack(lst_curr_seq_embeds, axis=1).float()
            lst_all_seq_embeds.append(t_curr_seq_embeds)
        # concat sequences on batch dimension
        t_all_seq_embeds = torch.cat(lst_all_seq_embeds).to(input_ids.device).requires_grad_()
        return t_all_seq_embeds

    def find_path(
        self,
        word_idx: int,
        baseline_idx: int,
        n_steps: Optional[int] = 30,
        strategy: Optional[str] = "greedy",
    ) -> List[int]:
        # if word_idx is a special token copy it and return
        if word_idx in self.special_tokens:
            return [word_idx] * (n_steps - 1)
        word_path = [word_idx]
        for _ in range(n_steps - 2):
            word_path.append(
                word_idx := self.get_closest_word(
                    word_idx=word_idx,
                    baseline_idx=baseline_idx,
                    word_path=word_path,
                    strategy=strategy,
                    n_steps=n_steps,
                )
            )
        return word_path

    def build_monotonic_path_embedding(
        self, word_path: List[int], baseline_idx: int, n_steps: int = 30
    ) -> TensorType["n_steps", "embed_size", float]:
        baseline_vec = self.vocabulary_embeddings[baseline_idx]
        monotonic_embs = [self.vocabulary_embeddings[word_path[0]]]
        for idx in range(len(word_path) - 1):
            monotonic_embs.append(
                self.make_monotonic_vec(
                    anchor=self.vocabulary_embeddings[word_path[idx + 1]],
                    baseline=baseline_vec,
                    input=monotonic_embs[-1],
                    n_steps=n_steps,
                )
            )
        monotonic_embs += [baseline_vec]
        # reverse the list so that baseline is the first and input word is the last
        monotonic_embs.reverse()
        assert self.check_monotonic(monotonic_embs), "The embeddings are not monotonic"
        return torch.stack(monotonic_embs)

    def get_closest_word(
        self,
        word_idx: int,
        baseline_idx: int,
        word_path: List[int],
        strategy: str = "greedy",
        n_steps: int = 30,
    ) -> int:
        # If (for some reason) we do select the ref_idx as the previous anchor word,
        # then all further anchor words should be ref_idx
        if word_idx == baseline_idx:
            return baseline_idx
        cx = self.knn_graph[word_idx].tocoo()
        # ignore anchor word if equals the baseline (padding, special tokens)
        # remove words that are already selected in the path
        anchor_map = {
            anchor_idx: self.get_word_distance(strategy, anchor_idx, baseline_idx, word_idx, n_steps)
            for anchor_idx in cx.col
            if anchor_idx not in word_path + [baseline_idx]
        }
        if len(anchor_map) == 0:
            return baseline_idx
        # return the top key
        return [k for k, _ in sorted(anchor_map.items(), key=lambda pair: pair[1])].pop(0)

    def get_word_distance(
        self,
        strategy: str,
        anchor_idx: int,
        baseline_idx: int,
        original_idx: int,
        n_steps: int,
    ) -> Union[float, int]:
        if strategy == PathBuildingStrategies.GREEDY.value:
            # calculate the distance of the monotonized vec from the interpolated point
            monotonic_vec = self.make_monotonic_vec(
                self.vocabulary_embeddings[anchor_idx],
                self.vocabulary_embeddings[baseline_idx],
                self.vocabulary_embeddings[original_idx],
                n_steps,
            )
            return euclidean_distance(self.vocabulary_embeddings[anchor_idx], monotonic_vec)
        elif strategy == PathBuildingStrategies.MAXCOUNT.value:
            # count the number of non-monotonic dimensions
            monotonic_dims = self.get_monotonic_dims(
                self.vocabulary_embeddings[anchor_idx],
                self.vocabulary_embeddings[baseline_idx],
                self.vocabulary_embeddings[original_idx],
            )
            # 10000 is an arbitrarily high to be agnostic of embeddings dimensionality
            return 10000 - monotonic_dims.sum()
        else:
            raise UnknownPathBuildingStrategy(strategy)

    @classmethod
    def check_monotonic(cls, input: torch.Tensor) -> bool:
        """Return true if input dimensions are monotonic, false otherwise."""
        check = True
        for i in range(len(input) - 1):
            monotonic_dims = cls.get_monotonic_dims(input[i + 1], input[-1], input[i])
            is_fully_monotonic = monotonic_dims.sum() == input[-1].shape[0]
            check *= is_fully_monotonic
        return check

    @classmethod
    def make_monotonic_vec(
        cls,
        anchor: torch.Tensor,
        baseline: torch.Tensor,
        input: torch.Tensor,
        n_steps: Optional[int] = 30,
    ) -> torch.Tensor:
        """Create a new monotonic vector w.r.t. input and baseline from an existing anchor"""
        non_monotonic_dims = ~cls.get_monotonic_dims(anchor, baseline, input)
        if non_monotonic_dims.sum() == 0:
            return anchor
        # make the anchor monotonic
        monotonic_vec = anchor.clone()
        monotonic_vec[non_monotonic_dims] = input[non_monotonic_dims] - (1.0 / n_steps) * (
            input[non_monotonic_dims] - baseline[non_monotonic_dims]
        )
        return monotonic_vec

    @staticmethod
    def get_monotonic_dims(
        anchor: torch.Tensor,
        baseline: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if the anchor vector is monotonic w.r.t. the baseline and the input.
        """
        # fmt: off
        return torch.where(
            (baseline > input)  * (baseline >= anchor) * (anchor >= input) + # noqa E211 W504
            (baseline < input)  * (baseline <= anchor) * (anchor <= input) + # noqa E211 W504
            (baseline == input) * (baseline == anchor) * (anchor == input),
            1, 0
        ).bool()
        # fmt: on
