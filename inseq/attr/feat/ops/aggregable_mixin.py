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
from typing import Dict, List, Protocol, Tuple, Union

import torch

from ....utils.typing import (
    MultiLayerMultiUnitScoreTensor,
    MultiLayerScoreTensor,
    MultiUnitScoreTensor,
    ScoreTensor,
)

logger = logging.getLogger(__name__)


class AggregationFunction(Protocol):
    def __call__(self, attention: MultiUnitScoreTensor, dim: int, **kwargs) -> ScoreTensor:
        ...


class AggregableMixin:
    """A mixin class for all attribution algorithms aggregating across layers and units."""

    @classmethod
    @property
    def unit_name(cls) -> str:
        return "unit"

    AGGREGATE_FN_OPTIONS: Dict[str, AggregationFunction] = {
        "average": lambda x, dim: x.mean(dim),
        "max": lambda x, dim: x.max(dim)[0],
        "min": lambda x, dim: x.min(dim)[0],
        "single": lambda x, dim, idx: x.select(dim, idx),
    }

    @staticmethod
    def _num_units(scores: MultiUnitScoreTensor) -> int:
        """Returns the number of units contained in the score tensor."""
        return scores.size(1)

    @staticmethod
    def _num_layers(scores: MultiLayerScoreTensor) -> int:
        """Returns the number of layers contained in the scores tensor."""
        return scores.size(1)

    @classmethod
    def _aggregate_units(
        cls,
        scores: MultiUnitScoreTensor,
        aggregate_fn: Union[str, AggregationFunction, None] = None,
        units: Union[int, Tuple[int, int], List[int], None] = None,
    ) -> ScoreTensor:
        """
        Merges the scores values across the specified units for the full sequence.

        Args:
            scores (:obj:`torch.Tensor`) Tensor of shape
                `(batch_size, num_units, sequence_length, sequence_length)`
            aggregate_fn (:obj:`str` or :obj:`callable`): The method to use for aggregating across units.
                Can be one of `average` (default if units is list, tuple or None), `max`, `min` or `single` (default
                if units is int), or a custom function defined by the user.
            units (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified,
                the unit at the corresponding index is used. If a tuple of two indices is specified, all units between
                the indices will be aggregated using aggregate_fn. If a list of indices is specified, the respective
                units will be used for aggregation. If aggregate_fn is "single", a unit must be specified.
                Otherwise, all units are passed to aggregate_fn by default.

        Returns:
            :obj:`torch.Tensor`: An aggregated score tensor of shape
                `(batch_size, sequence_length, sequence_length)`
        """
        n_units = cls._num_units(scores)
        aggregate_kwargs = {}

        if hasattr(units, "__iter__"):
            if len(units) == 0:
                raise RuntimeError(f"At least two {cls.unit_name}s must be specified for aggregation.")
            if len(units) == 1:
                units = units[0]

        # If units is not specified or an tuple, average aggregation is used by default
        if aggregate_fn is None and not isinstance(units, int):
            aggregate_fn = "average"
            logger.info(f"No {cls.unit_name} aggregation method specified. Using average aggregation by default.")
        # If a single head index is specified, single aggregation is used by default
        if aggregate_fn is None and isinstance(units, int):
            aggregate_fn = "single"

        if aggregate_fn == "single":
            if not isinstance(units, int):
                raise RuntimeError(f"A single {cls.unit_name} index must be specified for single-layer attribution")
            if units not in range(-n_units, n_units):
                raise IndexError(
                    f"{cls.unit_name.capitalize()} index out of range.The model only has {n_units} {cls.unit_name}s."
                )
            aggregate_kwargs = {"idx": units}
            aggregate_fn = cls.AGGREGATE_FN_OPTIONS[aggregate_fn]
        else:
            if isinstance(aggregate_fn, str):
                if aggregate_fn not in cls.AGGREGATE_FN_OPTIONS:
                    raise RuntimeError(
                        f"Invalid aggregation method specified.Valid methods are: {cls.AGGREGATE_FN_OPTIONS.keys()}"
                    )
                aggregate_fn = cls.AGGREGATE_FN_OPTIONS[aggregate_fn]
            if units is None:
                units = (0, n_units)
                logger.info(f"No {cls.unit_name}s specified for extraction. Using all {cls.unit_name}s by default.")
            # Convert negative indices to positive indices
            if hasattr(units, "__iter__"):
                units = type(units)([h_idx if h_idx >= 0 else n_units + h_idx for h_idx in units])
            if not hasattr(units, "__iter__") or (
                len(units) == 2 and isinstance(units, tuple) and units[0] >= units[1]
            ):
                raise RuntimeError(
                    "A (start, end) tuple of indices representing a span or a list of individual indices"
                    " must be specified for aggregation."
                )
            max_idx_val = n_units if isinstance(units, list) else n_units + 1
            if not all(h in range(-n_units, max_idx_val) for h in units):
                raise IndexError(
                    f"One or more {cls.unit_name} index out of range.The model only has {n_units} {cls.unit_name}s."
                )
            if len(set(units)) != len(units):
                raise IndexError(f"Duplicate {cls.unit_name} indices are not allowed.")
            if isinstance(units, tuple):
                scores = scores[:, units[0] : units[1]]
            else:
                scores = torch.index_select(scores, 1, torch.tensor(units, device=scores.device))
        return aggregate_fn(scores, 1, **aggregate_kwargs)

    @classmethod
    def _aggregate_layers(
        cls,
        scores: Union[MultiLayerMultiUnitScoreTensor, MultiLayerScoreTensor],
        aggregate_fn: Union[str, AggregationFunction, None] = None,
        layers: Union[int, Tuple[int, int], List[int], None] = None,
    ) -> Union[MultiUnitScoreTensor, ScoreTensor]:
        """
        Merges the scores of every unit across the specified layers for the full sequence.

        Args:
            scores (:obj:`torch.Tensor`) Tensor of shape `(batch_size, num_layers, ...)`
            aggregate_fn (:obj:`str` or :obj:`callable`): The method to use for aggregating across layers.
                Can be one of `average` (default if layers is tuple or list), `max`, `min` or `single` (default if
                layers is int or None), or a custom function defined by the user.
            layers (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified
                , the layer at the corresponding index is used. If a tuple of two indices is specified, all layers
                among the indices will be aggregated using aggregate_fn. If a list of indices is specified, the
                respective layers will be used for aggregation. If aggregate_fn is "single", the last layer is
                used by default. Otherwise, all available layers are passed to aggregate_fn by default.

        Returns:
            :obj:`torch.Tensor`: An aggregated scores tensor of shape `(batch_size, ...)`
        """
        n_layers = cls._num_layers(scores)
        aggregate_kwargs = {}

        if hasattr(layers, "__iter__"):
            if len(layers) == 0:
                raise RuntimeError("At least two layers must be specified for aggregation.")
            if len(layers) == 1:
                layers = layers[0]

        # If layers is not specified or an int, single layer aggregation is used by default
        if aggregate_fn is None and not hasattr(layers, "__iter__"):
            aggregate_fn = "single"
            logger.info("No layer aggregation method specified. Using single layer by default.")
        # If a tuple of indices for layers is specified, average aggregation is used by default
        if aggregate_fn is None and hasattr(layers, "__iter__"):
            aggregate_fn = "average"
            logger.info("No layer aggregation method specified. Using average across layers by default.")

        if aggregate_fn == "single":
            if layers is None:
                layers = -1
                logger.info("No layer specified for scores extraction. Using last layer by default.")
            if not isinstance(layers, int):
                raise RuntimeError("A single layer index must be specified for single-layer attribution")
            if layers not in range(-n_layers, n_layers):
                raise IndexError(f"Layer index out of range. The model only has {n_layers} layers.")
            aggregate_kwargs = {"idx": layers}
            aggregate_fn = cls.AGGREGATE_FN_OPTIONS[aggregate_fn]
        else:
            if isinstance(aggregate_fn, str):
                if aggregate_fn not in cls.AGGREGATE_FN_OPTIONS:
                    raise RuntimeError(
                        f"Invalid aggregation method specified.Valid methods are: {cls.AGGREGATE_FN_OPTIONS.keys()}"
                    )
                aggregate_fn = cls.AGGREGATE_FN_OPTIONS[aggregate_fn]
            if layers is None:
                layers = (0, n_layers)
                logger.info("No layer specified for scores extraction. Using all layers by default.")
            # Convert negative indices to positive indices
            if hasattr(layers, "__iter__"):
                layers = type(layers)([l_idx if l_idx >= 0 else n_layers + l_idx for l_idx in layers])
            if not hasattr(layers, "__iter__") or (
                len(layers) == 2 and isinstance(layers, tuple) and layers[0] >= layers[1]
            ):
                raise RuntimeError(
                    "A (start, end) tuple of indices representing a span or a list of individual indices"
                    " must be specified for aggregation."
                )
            max_idx_val = n_layers if isinstance(layers, list) else n_layers + 1
            if not all(l in range(max_idx_val) for l in layers):
                raise IndexError(f"One or more layer index out of range. The model only has {n_layers} layers.")
            if len(set(layers)) != len(layers):
                raise IndexError("Duplicate layer indices are not allowed.")
            if isinstance(layers, tuple):
                scores = scores[:, layers[0] : layers[1], ...]
            else:
                scores = torch.index_select(scores, 1, torch.tensor(layers, device=scores.device))
        return aggregate_fn(scores, 1, **aggregate_kwargs)
