# Adapted from https://github.com/INK-USC/DIG/blob/main/dig.py, licensed MIT:
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

from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._utils.batching import _batch_attribution
from captum.attr._utils.common import _format_input_baseline, _reshape_and_sum, _validate_input
from captum.log import log_usage
from torch import Tensor

from ....utils import INSEQ_ARTIFACTS_CACHE
from ....utils.typing import MultiStepEmbeddingsTensor, VocabularyEmbeddingsTensor
from .monotonic_path_builder import MonotonicPathBuilder


class DiscretetizedIntegratedGradients(IntegratedGradients):
    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs: bool = False,
    ) -> None:
        super().__init__(forward_func, multiply_by_inputs)
        self.path_builder = None

    def load_monotonic_path_builder(
        self,
        model_name: str,
        vocabulary_embeddings: VocabularyEmbeddingsTensor,
        special_tokens: List[int],
        cache_dir: Path = INSEQ_ARTIFACTS_CACHE / "dig_knn",
        embedding_scaling: int = 1,
        **kwargs,
    ) -> None:
        """Loads the Discretized Integrated Gradients (DIG) path builder."""
        self.path_builder = MonotonicPathBuilder.load(
            model_name,
            vocabulary_embeddings=vocabulary_embeddings.to("cpu"),
            special_tokens=special_tokens,
            cache_dir=cache_dir,
            embedding_scaling=embedding_scaling,
            **kwargs,
        )

    @staticmethod
    def get_inputs_baselines(scaled_features_tpl: Tuple[Tensor, ...], n_steps: int) -> Tuple[Tensor, ...]:
        # Baseline and inputs are reversed in the path builder
        # For every element in the batch, the first embedding of the sub-tensor
        # of shape (n_steps x embedding_dim) is the baseline, the last is the input.
        n_examples = scaled_features_tpl[0].shape[0] // n_steps
        baselines = tuple(
            torch.cat(
                [features[i, :, :].unsqueeze(0) for i in range(0, n_steps * n_examples, n_steps)],
            )
            for features in scaled_features_tpl
        )
        inputs = tuple(
            torch.cat(
                [features[i, :, :].unsqueeze(0) for i in range(n_steps - 1, n_steps * n_examples, n_steps)],
            )
            for features in scaled_features_tpl
        )
        return inputs, baselines

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: MultiStepEmbeddingsTensor,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "greedy",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
    ) -> Union[TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]]:
        n_examples = inputs[0].shape[0]
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        _validate_input(inputs, baselines, n_steps)
        scaled_features_tpl = tuple(
            self.path_builder.scale_inputs(
                input_tensor,
                baseline_tensor,
                n_steps=n_steps,
                scale_strategy=method,
            )
            for input_tensor, baseline_tensor in zip(inputs, baselines)
        )
        if internal_batch_size is not None:
            attributions = _batch_attribution(
                self,
                n_examples,
                internal_batch_size,
                n_steps,
                scaled_features_tpl=scaled_features_tpl,
                target=target,
                additional_forward_args=additional_forward_args,
            )
        else:
            attributions = self._attribute(
                scaled_features_tpl=scaled_features_tpl,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
            )
        if return_convergence_delta:
            start_point, end_point = self.get_inputs_baselines(scaled_features_tpl, n_steps)
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_output(is_inputs_tuple, attributions), delta
        return _format_output(is_inputs_tuple, attributions)

    def _attribute(
        self,
        scaled_features_tpl: Tuple[Tensor, ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
    ) -> Tuple[Tensor, ...]:
        additional_forward_args = _format_additional_forward_args(additional_forward_args)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)
        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )
        # calculate (x - x') for each interpolated point
        shifted_inputs_tpl = tuple(
            torch.cat(
                [
                    torch.cat([features[idx + 1 : idx + n_steps], features[idx + n_steps - 1].unsqueeze(0)])
                    for idx in range(0, scaled_features_tpl[0].shape[0], n_steps)
                ]
            )
            for features in scaled_features_tpl
        )
        steps = tuple(shifted_inputs_tpl[i] - scaled_features_tpl[i] for i in range(len(shifted_inputs_tpl)))
        scaled_grads = tuple(grads[i] * steps[i] for i in range(len(grads)))
        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as the original inputs
        total_grads = tuple(
            _reshape_and_sum(scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:])
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )
        # computes attribution for each tensor in input_tuple
        # attributions has the same dimensionality as the original inputs
        if not self.multiplies_by_inputs:
            return total_grads
        else:
            inputs, baselines = self.get_inputs_baselines(scaled_features_tpl, n_steps)
            return tuple(
                total_grad * (input - baseline)
                for (total_grad, input, baseline) in zip(total_grads, inputs, baselines)
            )
