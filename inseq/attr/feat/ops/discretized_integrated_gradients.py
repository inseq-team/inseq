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

from typing import Any, Callable, List, Tuple, Union

from pathlib import Path

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._utils.batching import _batch_attribution
from captum.attr._utils.common import _format_input, _reshape_and_sum
from captum.log import log_usage
from torch import Tensor
from torchtyping import TensorType

from ....utils import INSEQ_ARTIFACTS_CACHE
from ....utils.typing import VocabularyEmbeddingsTensor
from .monotonic_path_builder import MonotonicPathBuilder


class DiscretetizedIntegratedGradients(IntegratedGradients):
    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs: bool = True,
    ) -> None:
        super().__init__(forward_func, multiply_by_inputs)
        self.path_builder = None

    def load_monotonic_path_builder(
        self,
        model_name: str,
        vocabulary_embeddings: VocabularyEmbeddingsTensor,
        special_tokens: List[int],
        cache_dir: Path = INSEQ_ARTIFACTS_CACHE / "dig_knn",
        **kwargs,
    ) -> None:
        """Loads the Discretized Integrated Gradients (DIG) path builder."""
        self.path_builder = MonotonicPathBuilder.load(
            model_name,
            vocabulary_embeddings=vocabulary_embeddings,
            special_tokens=special_tokens,
            cache_dir=cache_dir,
            **kwargs,
        )

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorType["batch_size_x_n_steps", "seq_len", float],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        is_inputs_tuple = _is_tuple(inputs)
        scaled_features_tpl = _format_input(inputs)
        if internal_batch_size is not None:
            num_examples = inputs[0].shape[0] // n_steps
            attributions = _batch_attribution(
                self,
                num_examples,
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
            assert (
                len(scaled_features_tpl) == 1
            ), "More than one tuple not supported in this code!"
            start_point = _format_input(
                torch.cat(
                    [
                        scaled_features_tpl[0][i, :, :].unsqueeze(0)
                        for i in range(0, n_steps * num_examples, n_steps)
                    ],
                    dim=0,
                )
            )
            end_point = _format_input(
                torch.cat(
                    [
                        scaled_features_tpl[0][i, :, :].unsqueeze(0)
                        for i in range(n_steps - 1, n_steps * num_examples, n_steps)
                    ],
                    dim=0,
                )
            )
            # fmt: on
            print("Attributions", attributions[0].shape)
            print("Start", start_point[0].shape)
            print("End", end_point[0].shape)
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
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
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
            torch.cat([scaled_features[1:], scaled_features[-1].unsqueeze(0)])
            for scaled_features in scaled_features_tpl
        )
        steps = tuple(
            shifted_inputs_tpl[i] - scaled_features_tpl[i]
            for i in range(len(shifted_inputs_tpl))
        )
        scaled_grads = tuple(grads[i] * steps[i] for i in range(len(grads)))
        # aggregates across all steps for each tensor in the input tuple
        attributions = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )
        return attributions
