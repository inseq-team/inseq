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
    """Value Zeroing method for feature attribution.

    Introduced by `Mohebbi et al. (2023) <https://arxiv.org/abs/2301.12971>`__ to quantify context mixing inside
    Transformer models. The method is based on the observation that context mixing is regulated by the value vectors
    of the attention mechanism. The method consists of two steps:

    1. Zeroing the value vectors of the attention mechanism for a given token index at a given layer of the model.
    2. Computing the similarity between hidden states produced with and without the zeroing operation, and using it
         as a measure of context mixing for the given token at the given layer.

    The method is converted into a feature attribution method by allowing for extraction of value zeroing scores at
    specific layers, or by aggregating them across layers.

    Attributes:
        SIMILARITY_METRICS (:obj:`Dict[str, Callable]`):
            Dictionary of available similarity metrics to be used forvcomputing the distance between hidden states
            produced with and without the zeroing operation. Converted to distances as 1 - produced values.
        forward_func (:obj:`AttributionModel`):
            The attribution model to be used for value zeroing.
        clean_block_output_states (:obj:`Dict[int, torch.Tensor]`):
            Dictionary to store the hidden states produced by the model without the zeroing operation.
        corrupted_block_output_states (:obj:`Dict[int, torch.Tensor]`):
            Dictionary to store the hidden states produced by the model with the zeroing operation.
    """

    SIMILARITY_METRICS = {
        "cosine": nn.CosineSimilarity(dim=-1),
    }

    def __init__(self, forward_func: "AttributionModel") -> None:
        super().__init__(forward_func)
        self.clean_block_output_states: Dict[int, EmbeddingsTensor] = {}
        self.corrupted_block_output_states: Dict[int, EmbeddingsTensor] = {}

    @staticmethod
    def get_value_zeroing_hook(varname: str = "value") -> Callable[..., None]:
        """Returns a hook to zero the value vectors of the attention mechanism.

        Args:
            varname (:obj:`str`, optional): The name of the variable containing the value vectors. The variable
                is expected to be a 3D tensor of shape (batch_size, num_heads, seq_len) and is retrieved from the
                local variables of the execution frame during the forward pass.
        """

        def value_zeroing_forward_mid_hook(frame, value_zeroing_index: Optional[int] = None) -> None:
            # Zeroing value vectors corresponding to the given token index
            if value_zeroing_index is not None:
                value_size = torch.zeros(frame.f_locals[varname][:, :, value_zeroing_index].size())
                frame.f_locals[varname][:, :, value_zeroing_index] = value_size

        return value_zeroing_forward_mid_hook

    def get_states_extract_and_patch_hook(self, block_idx: int, hidden_state_idx: int = 0) -> Callable[..., None]:
        """Returns a hook to extract the produced hidden states (corrupted by value zeroing)
          and patch them with pre-computed clean states that will be passed onwards in the model forward.

        Args:
            block_idx (:obj:`int`): The idx of the block at which the hook is applied, used to store extracted states.
            hidden_state_idx (:obj:`int`, optional): The index of the hidden state in the model output tuple.
        """

        def states_extract_and_patch_forward_hook(module, args, output) -> None:
            self.corrupted_block_output_states[block_idx] = output[hidden_state_idx].clone().detach().cpu()

            # Rebuild the output tuple patching the clean states at the place of the corrupted ones
            output = (
                output[:hidden_state_idx]
                + (self.clean_block_output_states[block_idx].to(output[hidden_state_idx].device),)
                + output[hidden_state_idx + 1 :]
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
        similarity_metric: str = "cosine",
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
            aggregate_layers_fn (:obj:`str` or :obj:`callable`): The method to use for aggregating across layers.
                Can be one of `average` (default if layers is tuple or list), `max`, `min` or `single` (default if
                layers is int or None), or a custom function defined by the user.
            metric (:obj:`str`, optional): The similarity metric to use for computing the distance between hidden
                states produced with and without the zeroing operation. Default: cosine.
            encoder_hidden_states (:obj:`torch.Tensor`, optional): A tensor of shape ``[batch_size, num_layers + 1,
                source_seq_len, hidden_size]`` containing hidden states of the encoder. Available only for
                encoder-decoders models. Default: None.
            decoder_hidden_states (:obj:`torch.Tensor`, optional): A tensor of shape ``[batch_size, num_layers + 1,
                target_seq_len, hidden_size]`` containing hidden states of the decoder.

        Returns:
            `TensorOrTupleOfTensorsGeneric`: Attribution outputs for source-only or source + target feature attribution
        """
        if similarity_metric not in self.SIMILARITY_METRICS:
            raise ValueError(
                f"Similarity metric {similarity_metric} not available."
                f"Available metrics: {','.join(self.SIMILARITY_METRICS.keys())}"
            )
        batch_size = decoder_hidden_states.size(0)
        tgt_seq_len = decoder_hidden_states.size(2)
        decoder_stack: nn.ModuleList = find_block_stack(self.forward_func.get_decoder())

        # Store clean hidden states for later use. We use idx + 1 since the first element of the decoder stack is the
        # embedding layer, and we are only interested in the transformer blocks outputs.
        self.clean_block_output_states = {
            idx: decoder_hidden_states[:, idx + 1, ...].clone().detach().cpu() for idx, _ in enumerate(decoder_stack)
        }
        scores = torch.zeros(batch_size, len(decoder_stack), tgt_seq_len, tgt_seq_len)

        # Hooks:
        #   1. states_extract_and_patch_hook on the transformer block stores corrupted states and force clean states
        #      as the output of the block forward pass, i.e. the zeroing is done independently across layers.
        #   2. value_zeroing_hook on the attention module performs the value zeroing by replacing the "value" tensor
        #      during the forward (name is config-dependent) with a zeroed version for the specified token index.
        #
        # State extraction hooks can be registered only once since they are token-independent
        # Skip last block since its states are not used raw, but may have further transformations applied to them
        # (e.g. LayerNorm, Dropout). These are extracted separately from the model outputs.
        states_extraction_hook_handles = []
        for block_idx in range(len(decoder_stack) - 1):
            states_extract_and_patch_hook = self.get_states_extract_and_patch_hook(block_idx, hidden_state_idx=0)
            states_extraction_hook_handles.append(
                decoder_stack[block_idx].register_forward_hook(states_extract_and_patch_hook)
            )

        # Zeroing is done for every token in the target sequence separately (O(n) complexity)
        for token_idx in range(tgt_seq_len):
            value_zeroing_hook_handles = []
            # Value zeroing hooks are registered for every token separately since they are token-dependent
            for block in decoder_stack:
                attention_module = block.get_submodule(self.forward_func.config.attention_module)
                value_zeroing_hook = get_post_variable_assignment_hook(
                    attention_module,
                    hook_fn=self.get_value_zeroing_hook(self.forward_func.config.value_vector),
                    varname=self.forward_func.config.value_vector,
                    value_zeroing_index=token_idx,
                )
                value_zeroing_hook_handle = attention_module.register_forward_pre_hook(value_zeroing_hook)
                value_zeroing_hook_handles.append(value_zeroing_hook_handle)

            # Run forward pass with hooks. Fills self.corrupted_hidden_states with corrupted states across layers
            # when zeroing the specified token index.
            with torch.no_grad():
                output = self.forward_func.forward_with_output(
                    *inputs, *additional_forward_args, output_hidden_states=True
                )
                # Extract last layer states directly from the model outputs
                corrupted_states_dict = self.forward_func.get_hidden_states_dict(output)
                corrupted_decoder_last_hidden_state = (
                    corrupted_states_dict["decoder_hidden_states"][:, -1, ...].clone().detach().cpu()
                )
                self.corrupted_block_output_states[len(decoder_stack) - 1] = corrupted_decoder_last_hidden_state
            for handle in value_zeroing_hook_handles:
                handle.remove()
            for block_idx in range(len(decoder_stack)):
                similarity_scores = self.SIMILARITY_METRICS[similarity_metric](
                    self.clean_block_output_states[block_idx], self.corrupted_block_output_states[block_idx]
                )
                scores[:, block_idx, :, token_idx] = torch.ones_like(similarity_scores) - similarity_scores
        for handle in states_extraction_hook_handles:
            handle.remove()

        # Normalize scores to sum to 1
        per_token_sum_score = scores.sum(dim=-1, keepdim=True)
        per_token_sum_score[per_token_sum_score == 0] = 1
        scores = scores / per_token_sum_score

        # Aggregate scores across layers to obtain final attribution scores
        # Aggregation output has shape: [batch_size, tgt_seq_len, tgt_seq_len]
        scores = self._aggregate_layers(scores, aggregate_layers_fn, layers)

        # Since presently Inseq attribution is constrained by the attribution loop over the full generation, the
        # extraction of value zeroing scores is done inefficiently by picking only the last token scores at every step.
        # This makes the complexity of calling this method O(n^2), when it could be O(n) if the scores were extracted
        # only at the final step. Since other methods (e.g. attention) face the same issue, this will be addressed in
        # future releases.
        return scores[:, -1, :]
        # TODO: Add support for encoder-decoder models
        # if is_encoder_decoder:
        #    encoder_hidden_states = torch.stack(outputs.encoder_hidden_states)
        #    encoder = self.forward_func.get_encoder()
        #    encoder_stack = find_block_stack(encoder)
