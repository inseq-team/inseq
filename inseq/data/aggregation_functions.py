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
from typing import Dict, List, Tuple, Union

import torch
from torch.linalg import vector_norm

from ..attr.feat.ops import rollout_fn
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
        return scores.max(dim).values


class MinAggregationFunction(AggregationFunction):
    aggregation_function_name = "min"

    def __call__(self, scores: torch.Tensor, dim: int) -> ScoreTensor:
        return scores.min(dim).values


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
        return scores.gather(dim, torch.nan_to_num(scores).abs().argmax(dim, keepdim=True)).squeeze(dim)


class VectorNormAggregationFunction(AggregationFunction):
    aggregation_function_name = "vnorm"

    def __call__(self, scores: torch.Tensor, dim: int, vnorm_ord: int = 2) -> ScoreTensor:
        return vector_norm(scores, ord=vnorm_ord, dim=dim)


class RolloutAggregationFunction(AggregationFunction):
    aggregation_function_name = "rollout"

    def __init__(self):
        super().__init__()
        self.takes_single_tensor: bool = False
        self.takes_sequence_scores: bool = True

    def __call__(
        self,
        scores: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        dim: int,
        sequence_scores: Dict[str, torch.Tensor] = {},
    ) -> ScoreTensor:
        dec_self_prefix = "decoder_self"
        enc_self_prefix = "encoder_self"
        dec_match = [name for name in sequence_scores.keys() if name.startswith(dec_self_prefix)]
        enc_match = [name for name in sequence_scores.keys() if name.startswith(enc_self_prefix)]
        if isinstance(scores, torch.Tensor):
            # If no matching prefix is found, we assume the decoder-only target-only rollout case
            if not dec_match or not enc_match:
                return rollout_fn(scores, dim=dim)
            # If both prefixes are found, we assume the encoder-decoder source-only rollout case
            else:
                enc_match = sequence_scores[enc_match[0]]
                dec_match = sequence_scores[dec_match[0]]
                return rollout_fn((enc_match, scores, dec_match), dim=dim)[0]
        else:
            if not enc_match:
                raise KeyError(
                    "Could not find encoder self-importance scores in sequence scores. "
                    "Encoder self-importance scores are required for encoder-decoder rollout. They should be provided "
                    f"as an entry in the sequence scores dictionary with key starting with '{enc_self_prefix}', and "
                    "value being a tensor of shape (src_seq_len, src_seq_len, ..., rollout_dim)."
                )
            else:
                enc_match = sequence_scores[enc_match[0]]
                return rollout_fn((enc_match,) + scores, dim=dim)


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
    """Lists identifiers for all available aggregation functions."""
    return available_classes(AggregationFunction)
