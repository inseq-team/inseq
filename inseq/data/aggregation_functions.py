# Copyright 2023 The Inseq Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from abc import abstractmethod
from typing import List, Tuple, Union

import torch
from torch.linalg import vector_norm

from ..utils import Registry, available_classes
from ..utils.typing import (
    ScoreTensor,
)

logger = logging.getLogger(__name__)


class AggregationFunction(Registry):
    registry_attr = "aggregation_function_name"

    def __init__(self):
        self.takes_single_tensor: bool = True
        self.takes_sequence_scores: bool = False

    @abstractmethod
    def __call__(
        self,
        scores: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        dim: int,
        **kwargs,
    ) -> ScoreTensor:
        pass


class MeanAggregationFunction(AggregationFunction):
    aggregation_function_name = "mean"

    def __call__(self, scores: torch.Tensor, dim: int) -> ScoreTensor:
        return scores.mean(dim)


class MaxAggregationFunction(AggregationFunction):
    aggregation_function_name = "max"

    def __call__(self, scores: torch.Tensor, dim: int) -> ScoreTensor:
        return scores.max(dim)


class MinAggregationFunction(AggregationFunction):
    aggregation_function_name = "min"

    def __call__(self, scores: torch.Tensor, dim: int) -> ScoreTensor:
        return scores.min(dim)


class SingleAggregationFunction(AggregationFunction):
    aggregation_function_name = "single"

    def __call__(self, scores: torch.Tensor, dim: int, single_idx: int) -> ScoreTensor:
        return scores.select(dim, single_idx)


class SumAggregationFunction(AggregationFunction):
    aggregation_function_name = "sum"

    def __call__(self, scores: torch.Tensor, dim: int) -> ScoreTensor:
        return scores.sum(dim)


class ProdAggregationFunction(AggregationFunction):
    aggregation_function_name = "prod"

    def __call__(self, scores: torch.Tensor, dim: int) -> ScoreTensor:
        return scores.prod(dim)


class AbsMaxAggregationFunction(AggregationFunction):
    aggregation_function_name = "absmax"

    def __call__(self, scores: torch.Tensor, dim: int) -> ScoreTensor:
        return scores.gather(dim, scores.abs().argmax(dim, keepdim=True)).squeeze(dim)


class VectorNormAggregationFunction(AggregationFunction):
    aggregation_function_name = "vnorm"

    def __call__(self, scores: torch.Tensor, dim: int, vnorm_ord: int = 2) -> ScoreTensor:
        return vector_norm(scores, ord=vnorm_ord, dim=dim)


DEFAULT_ATTRIBUTION_AGGREGATE_DICT = {
    "source_attributions": {"spans": "absmax"},
    "target_attributions": {"spans": "absmax"},
    "step_scores": {
        "spans": {
            "probability": "prod",
            "entropy": "sum",
            "crossentropy": "sum",
            "perplexity": "prod",
            "contrast_prob_diff": "prod",
            "contrast_prob": "prod",
            "pcxmi": "sum",
            "kl_divergence": "sum",
            "mc_dropout_prob_avg": "prod",
        }
    },
}


def list_aggregation_functions() -> List[str]:
    """Lists identifiers for all available aggregation functions scores."""
    return available_classes(AggregationFunction)
