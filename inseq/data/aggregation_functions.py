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
from typing import Dict, List, Protocol, Sequence, Union

import torch
from torch.linalg import vector_norm

from ..attr.feat.ops import rollout_fn
from ..utils import normalize_attributions, sum_normalize_attributions
from ..utils.typing import (
    ScoreTensor,
)

logger = logging.getLogger(__name__)


class AggregationFunction(Protocol):
    def __call__(
        self,
        scores: Union[torch.Tensor, Sequence[torch.Tensor]],
        dim: int,
        **kwargs,
    ) -> ScoreTensor:
        ...


AGGREGATION_FN_MAP: Dict[str, AggregationFunction] = {
    "mean": lambda scores, dim: scores.mean(dim),
    "max": lambda scores, dim: scores.max(dim).values,
    "min": lambda scores, dim: scores.min(dim).values,
    "single": lambda scores, dim, idx: scores.select(dim, idx),
    "sum": lambda scores, dim: scores.sum(dim),
    "prod": lambda scores, dim: scores.prod(dim),
    "absmax": lambda scores, dim: scores.gather(dim, scores.abs().argmax(dim, keepdim=True)).squeeze(dim),
    "identity": lambda scores, dim: scores,
    "vnorm": lambda scores, dim, ord=2: vector_norm(scores, ord=ord, dim=dim),
    "normalize": lambda scores, dim: normalize_attributions(scores),
    "sum_normalize": lambda scores, dim: sum_normalize_attributions(scores),
    "rollout": lambda scores, dim, add_residual=False: rollout_fn(scores, dim, add_residual),
}

DEFAULT_ATTRIBUTION_AGGREGATE_DICT = {
    "source_attributions": {"scores": "identity", "spans": "absmax"},
    "target_attributions": {"scores": "identity", "spans": "absmax"},
    "step_scores": {
        "spans": {
            "probability": "prod",
            "entropy": "sum",
            "crossentropy": "sum",
            "perplexity": "prod",
            "contrast_prob_diff": "prod",
            "mc_dropout_prob_avg": "prod",
        }
    },
}


def list_aggregation_functions() -> List[str]:
    """
    Lists identifiers for all available aggregation functions scores.
    """
    return list(AGGREGATION_FN_MAP.keys())


def register_aggregation_function(fn: AggregationFunction, identifier: str, overwrite: bool = False) -> None:
    """
    Registers a function to be used for aggregation purposes.

    Args:
        fn (:obj:`callable`): The function to be used to compute step scores. Default parameters are:

            - :obj:`scores`: an :obj:`torch.Tensor` or a sequence of :obj:`torch.Tensor` objects to be aggregated.

            - :obj:`dim`: An integer specifying the dimension along which tensors should be aggregated.

            The function can also define an arbitrary number of custom parameters, and it must return a
            :obj:`torch.Tensor` or a tuple of :obj:`torch.Tensor` objects containing aggregated scores alongside the
            corresponding aggregation dimensions.

        identifier (:obj:`str`): The identifier that will be used for the registered step score.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether to overwrite an existing function
            registered with the same identifier.
    """
    if identifier in AGGREGATION_FN_MAP:
        if not overwrite:
            raise ValueError(
                f"{identifier} is already registered in aggregation functions map. Override with overwrite=True."
            )
        logger.warning(f"Overwriting {identifier} aggregation function.")
    AGGREGATION_FN_MAP[identifier] = fn


def get_aggregation_fns_from_ids(aggregation_dict: Dict[str, Union[Dict, str]]) -> Dict[str, AggregationFunction]:
    """
    Returns a dictionary mapping aggregation identifiers to aggregation functions.

    Args:
        aggregation_dict (:obj:`dict`): A dictionary mapping aggregator identifiers to aggregation function
            identifiers.

    Returns:
        :obj:`Dict[str, AggregationFunction]`: A dictionary mapping aggregation identifiers to aggregation functions.
    """
    fn_dict = {}
    for k, v in aggregation_dict.items():
        if isinstance(v, dict):
            fn_dict[k] = get_aggregation_fns_from_ids(v)
        elif isinstance(v, str):
            if v not in AGGREGATION_FN_MAP:
                raise ValueError(
                    f"Aggregation function {v} is not registered."
                    "Register it with :func:`~inseq.register_aggregation_function`."
                )
            fn_dict[k] = AGGREGATION_FN_MAP[v]
    return fn_dict
