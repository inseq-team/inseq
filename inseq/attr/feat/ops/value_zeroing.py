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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import Attribution
from captum.log import log_usage
from torch import nn

from ....utils import find_block_stack, get_post_variable_assignment_hook
from ....utils.typing import EmbeddingsTensor, MultiLayerEmbeddingsTensor
from .aggregable_mixin import AggregableMixin, AggregationFunction

if TYPE_CHECKING:
    from ....models import AttributionModel

logger = logging.getLogger(__name__)


class ValueZeroing(Attribution, AggregableMixin):
    SIMILARITY_METRICS = {
        "cosine_distance": nn.CosineSimilarity(dim=-1),
    }

    """Value Zeroing method for feature attribution."""

    def __init__(self, forward_func: "AttributionModel") -> None:
        super().__init__(forward_func)
        self.clean_hidden_states: Dict[int, EmbeddingsTensor] = {}
        self.corrupted_hidden_states: Dict[int, EmbeddingsTensor] = {}

    @staticmethod
    def get_value_zeroing_hook(varname: str = "value") -> Callable[..., None]:
        def value_zeroing_forward_mid_hook(frame, value_zeroing_index: Optional[int] = None) -> None:
            # Zeroing value vectors corresponding to the given token index
            if value_zeroing_index is not None:
                value_size = torch.zeros(frame.f_locals[varname][:, :, value_zeroing_index].size())
                frame.f_locals[varname][:, :, value_zeroing_index] = value_size

        return value_zeroing_forward_mid_hook

    def get_states_extract_and_patch_hook(self, layer: int, hidden_state_index: int = 0) -> Callable[..., None]:
        def states_extract_and_patch_forward_hook(module, args, output) -> None:
            device = output[hidden_state_index].device
            self.corrupted_hidden_states[layer] = output[hidden_state_index].clone().detach().cpu()
            output = (
                output[:hidden_state_index]
                + (self.clean_hidden_states[layer].to(device),)
                + output[hidden_state_index + 1 :]
            )
            return output

        return states_extract_and_patch_forward_hook

    @staticmethod
    def has_convergence_delta() -> bool:
        return False

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: TensorOrTupleOfTensorsGeneric,
        layers: Union[int, Tuple[int, int], List[int], None] = None,
        aggregate_layers_fn: Union[str, AggregationFunction, None] = None,
        metric: str = "cosine_distance",
        encoder_hidden_states: Optional[MultiLayerEmbeddingsTensor] = None,
        decoder_hidden_states: Optional[MultiLayerEmbeddingsTensor] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """Perform attribution using the Value Zeroing method.

        Args:
            layers (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified
                , the layer at the corresponding index is used. If a tuple of two indices is specified, all layers
                among the indices will be aggregated using aggregate_fn. If a list of indices is specified, the
                respective layers will be used for aggregation. If aggregate_fn is "single", the last layer is
                used by default. If no value is specified, all available layers are passed to aggregate_fn by default.
            encoder_hidden_states (:obj:`torch.Tensor`, optional): A tensor of shape ``[num_layers+1, batch_size,
                source_seq_len, hidden_size] containing hidden states of the encoder. Available only for
                encoder-decoders models. Default: None.
            decoder_hidden_states (:obj:`torch.Tensor`, optional): A tensor of shape ``[num_layers, batch_size,
                target_seq_len, hidden_size]`` containing hidden states of the decoder.

        Returns:
            `TensorOrTupleOfTensorsGeneric`: Attribution outputs for source-only or source + target feature attribution
        """
        batch_size = decoder_hidden_states.size(0)
        num_layers = self._num_layers(decoder_hidden_states)
        tgt_seq_len = decoder_hidden_states.size(2)
        decoder_stack: nn.ModuleList = find_block_stack(self.forward_func.get_decoder())
        self.clean_hidden_states = {
            layer: decoder_hidden_states[:, layer, ...].clone().detach().cpu() for layer in range(len(decoder_stack))
        }
        scores = torch.zeros(batch_size, num_layers, tgt_seq_len, tgt_seq_len)

        # Hooks:
        #   1. forward_hook on transformer block is used to hijack "hidden_states" (config var depending on
        #      model) and force original hidden states ("decoder_hidden_states" variable). It also records the
        #      output of the block (after value-zeroing) in a dictionary. This output would normally be passed
        #      to the next layer, but gets swapped out by the step above.
        #   2. pre_forward_hook to perform the value zeroing by dynamically replacing the intermediate "value"
        #      tensor in the forward (name is config-dependent) with a zeroed version.
        states_extraction_hook_handles = []
        for block_idx, block in enumerate(decoder_stack):
            states_extract_and_patch_hook = self.get_states_extract_and_patch_hook(block_idx, hidden_state_index=0)
            states_extraction_hook_handles.append(block.register_forward_hook(states_extract_and_patch_hook))

        for token_idx in range(tgt_seq_len):
            value_zeroing_hook_handles = []
            for block in decoder_stack:
                attention_module = block.get_submodule(self.forward_func.config.attention_module)
                value_zeroing_block_hook = get_post_variable_assignment_hook(
                    attention_module,
                    hook_fn=self.get_value_zeroing_hook(self.forward_func.config.value_vector),
                    varname=self.forward_func.config.value_vector,
                    value_zeroing_index=token_idx,
                )
                value_zeroing_hook_handle = attention_module.register_forward_pre_hook(value_zeroing_block_hook)
                value_zeroing_hook_handles.append(value_zeroing_hook_handle)
            with torch.no_grad():
                self.forward_func(*inputs, *additional_forward_args)
            for handle in value_zeroing_hook_handles:
                handle.remove()
            for block_idx in range(len(decoder_stack)):
                similarity_scores = self.SIMILARITY_METRICS[metric](
                    self.clean_hidden_states[block_idx], self.corrupted_hidden_states[block_idx]
                )
                scores[:, block_idx, :, token_idx] = torch.ones_like(similarity_scores) - similarity_scores
        for handle in states_extraction_hook_handles:
            handle.remove()

        # Normalize scores to sum to 1
        per_token_sum_score = scores.sum(dim=-1, keepdim=True)
        per_token_sum_score[per_token_sum_score == 0] = 1
        scores = scores / per_token_sum_score
        # [batch_size, tgt_seq_len, tgt_seq_len]
        scores = self._aggregate_layers(scores, aggregate_layers_fn, layers)
        # Since presently Inseq attribution is constrained by the attribution loop over the full generation, the
        # extraction of value zeroing scores is done inefficiently by picking only the last token scores at every step.
        # This makes the complexity of calling this method O(n^2), when it could be O(n) if the scores were extracted
        # only at the final step. Since other methods (e.g. attention) face the same issue, this will be addressed in
        # future releases.
        return scores[:, :, -1]
        # TODO: Add support for encoder-decoder models
        # if is_encoder_decoder:
        #    encoder_hidden_states = torch.stack(outputs.encoder_hidden_states)
        #    encoder = self.forward_func.get_encoder()
        #    encoder_stack = find_block_stack(encoder)
