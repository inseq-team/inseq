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
from typing import Any, List, Tuple, Union

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import Attribution
from captum.log import log_usage

from ....data import Batch, EncoderDecoderBatch
from ....utils import find_block_stack, get_nn_submodule, get_post_variable_assignment_hook

logger = logging.getLogger(__name__)


def value_zeroing(frame, value_zeroing_index=None):
    # Zeroing value vectors corresponding to the given token index
    if value_zeroing_index is not None:
        value_size = torch.zeros(frame.f_locals["value"][:, :, value_zeroing_index].size())
        frame.f_locals["value"][:, :, value_zeroing_index] = value_size


class ValueZeroing(Attribution):
    @staticmethod
    def has_convergence_delta() -> bool:
        return False

    @log_usage()
    def attribute(
        self,
        batch: Union[Batch, EncoderDecoderBatch],
        layers: Union[int, Tuple[int, int], List[int], None] = None,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """Perform attribution using the Value Zeroing method.

        Args:
            batch (`Union[Batch, EncoderDecoderBatch]`):
                The input batch used for the forward pass to extract attention scores.
            layers (:obj:`int` or :obj:`tuple[int, int]` or :obj:`list(int)`, optional): If a single value is specified
                , the layer at the corresponding index is used. If a tuple of two indices is specified, all layers
                among the indices will be aggregated using aggregate_fn. If a list of indices is specified, the
                respective layers will be used for aggregation. If aggregate_fn is "single", the last layer is
                used by default. If no value is specified, all available layers are passed to aggregate_fn by default.

        Returns:
            `TensorOrTupleOfTensorsGeneric`: Attribution outputs for source-only or source + target feature attribution
        """

        outputs = self.forward_func.get_forward_output(
            **self.forward_func.format_forward_args(batch), output_hidden_states=True
        )
        # num_decoder_layers = len(outputs.decoder_hidden_states)
        sequence_length = outputs.decoder_hidden_states[0].shape[1]
        # decoder_value_zeroing_scores = torch.zeros(num_decoder_layers, sequence_length, sequence_length)
        decoder_hidden_states = torch.stack(outputs.hidden_states)
        decoder = self.forward_func.get_decoder()
        decoder_stack = find_block_stack(decoder)
        for block_idx, block in enumerate(decoder_stack):
            for token_idx in range(sequence_length):
                value_zeroing_hook = get_post_variable_assignment_hook(
                    get_nn_submodule(block, "attn"),
                    hook_fn=value_zeroing,
                    varname="value",
                    value_zeroing_index=token_idx,
                )
                handle = block.attn.register_forward_pre_hook(value_zeroing_hook)
                with torch.no_grad():
                    outputs = block(
                        hidden_states=decoder_hidden_states[block_idx],
                        # attention_mask=attention_mask,
                    )
                handle.remove()
                # zeroed_hidden_states = outputs[0]
        # if is_encoder_decoder:
        #    encoder_hidden_states = torch.stack(outputs.encoder_hidden_states)
        #    encoder = self.forward_func.get_encoder()
        #    encoder_stack = find_block_stack(encoder)
