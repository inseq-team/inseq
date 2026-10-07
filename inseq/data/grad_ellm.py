"""Token-level outputs using Inseq's standard score normalization pipeline."""

from copy import deepcopy
from dataclasses import dataclass, replace

import torch

from inseq.data import AggregationFunction, FeatureAttributionSequenceOutput, FeatureAttributionStepOutput
from inseq.data import attribution as inseq_attribution
from inseq.data.aggregation_functions import DEFAULT_ATTRIBUTION_AGGREGATE_DICT


class GradELLMIdentityAggregation(AggregationFunction):
    """Keep the two token axes; Inseq applies its usual L1 normalization next."""

    aggregation_function_name = "grad_ellm_identity"

    def __call__(self, scores, dim):
        return scores


@dataclass(eq=False, repr=False)
class GradELLMSequenceOutput(FeatureAttributionSequenceOutput):
    def __post_init__(self):
        # Inseq 0.6/0.7's base initializer updates shared global dictionaries.
        # Own the configuration instead: reconstructing, loading, or aggregating
        # Grad-ELLM must never change the defaults used by another method.
        functions = deepcopy(DEFAULT_ATTRIBUTION_AGGREGATE_DICT)
        functions.update(deepcopy(self._dict_aggregate_fn or {}))
        functions.setdefault("target_attributions", {})["scores"] = "grad_ellm_identity"
        self._dict_aggregate_fn = functions
        if hasattr(self, "_attribution_dim_names"):
            dimensions = deepcopy(getattr(inseq_attribution, "DEFAULT_ATTRIBUTION_DIM_NAMES", {}))
            dimensions.update(deepcopy(self._attribution_dim_names or {}))
            self._attribution_dim_names = dimensions
        if self._aggregator is None:
            self._aggregator = "scores"
        if hasattr(self, "_num_dimensions") and self._num_dimensions is None:
            self._num_dimensions = 0
        if self.attr_pos_end is None or self.attr_pos_end > len(self.target):
            self.attr_pos_end = len(self.target)
        if self.step_scores is None:
            self.step_scores = {}
        if self.sequence_scores is None:
            self.sequence_scores = {}
        # Inseq's HTML renderer converts step/sequence scores to NumPy, which
        # does not support bfloat16. Preserve values in a supported dtype.
        self.step_scores = {
            name: values.float() if isinstance(values, torch.Tensor) and values.dtype == torch.bfloat16 else values
            for name, values in self.step_scores.items()
        }
        self.sequence_scores = {
            name: values.float() if isinstance(values, torch.Tensor) and values.dtype == torch.bfloat16 else values
            for name, values in self.sequence_scores.items()
        }

    def __getitem__(self, selection):
        # Inseq slices select prefix rows and retain generated tokens/columns.
        # Normalize open/negative bounds. Construct directly so this also works
        # on Inseq 0.6, which does not expose the slices aggregator.
        length = self.attr_pos_start
        if isinstance(selection, int) and not isinstance(selection, bool):
            index = selection + length if selection < 0 else selection
            if not 0 <= index < length:
                raise IndexError("Prompt token index out of range.")
            selection = slice(index, index + 1)
        if not isinstance(selection, slice):
            raise TypeError("Use an integer or a contiguous slice of prompt tokens.")
        start, stop, stride = selection.indices(length)
        if stride != 1 or start >= stop:
            raise ValueError(
                "Inseq output slices require a non-empty contiguous prompt span; slice the tensor otherwise."
            )
        sequence_scores = {
            name: torch.cat((values[start:stop], values[length:]), dim=0)
            if name.startswith("decoder")
            else values[start:stop]
            for name, values in self.sequence_scores.items()
        }
        return replace(
            self,
            source=self.source[start:stop],
            target=self.target[start:stop] + self.target[length:],
            target_attributions=torch.cat(
                (self.target_attributions[start:stop], self.target_attributions[length:]), dim=0
            ),
            step_scores=self.step_scores.copy(),
            sequence_scores=sequence_scores,
            attr_pos_start=stop - start,
            attr_pos_end=self.attr_pos_end - length + stop - start,
        )


@dataclass(eq=False, repr=False)
class GradELLMStepOutput(FeatureAttributionStepOutput):
    _sequence_cls: type[FeatureAttributionSequenceOutput] = GradELLMSequenceOutput

    def _recover_from_safetensors(self):
        # Inseq 0.7 serializes steps as NumPy arrays, but its loader calls this
        # method on both step and sequence outputs. Steps have no safetensor
        # payload: restore their NumPy arrays using the standard wrapper API.
        return self.torch()
