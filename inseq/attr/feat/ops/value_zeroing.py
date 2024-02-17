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
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import Attribution
from captum.log import log_usage
from torch import nn

from ....utils import find_block_stack, get_post_variable_assignment_hook, StackFrame, validate_indices
from ....utils.typing import EmbeddingsTensor, MultiLayerEmbeddingsTensor, MultiLayerScoreTensor

if TYPE_CHECKING:
    from ....models import AttributionModel

logger = logging.getLogger(__name__)


class ValueZeroingSimilarityMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class ValueZeroingModule(Enum):
    DECODER = "decoder"
    ENCODER = "encoder"
    CROSS = "cross"


class ValueZeroing(Attribution):
    """Value Zeroing method for feature attribution.

    Introduced by `Mohebbi et al. (2023) <https://aclanthology.org/2023.eacl-main.245/>`__ to quantify context mixing inside
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
        "euclidean": lambda x, y: torch.cdist(x, y, p=2),
    }

    def __init__(self, forward_func: "AttributionModel") -> None:
        super().__init__(forward_func)
        self.clean_block_output_states: Dict[int, EmbeddingsTensor] = {}
        self.corrupted_block_output_states: Dict[int, EmbeddingsTensor] = {}
        self.zeroing_indices: Optional[Union[int, list[int], tuple[int, int]]] = None

    @staticmethod
    def get_value_zeroing_hook(varname: str = "value") -> Callable[..., None]:
        """Returns a hook to zero the value vectors of the attention mechanism.

        Args:
            varname (:obj:`str`, optional): The name of the variable containing the value vectors. The variable
                is expected to be a 3D tensor of shape (batch_size, num_heads, seq_len) and is retrieved from the
                local variables of the execution frame during the forward pass.
        """

        def value_zeroing_forward_mid_hook(
            frame: StackFrame,
            value_zeroing_index: Optional[int] = None,
            zeroing_indices: Union[int, list[int], tuple[int, int], None] = None,
            batch_size: int = 1,
        ) -> None:
            # Zeroing value vectors corresponding to the given token index
            if value_zeroing_index is not None:
                values_size = frame.f_locals[varname].size()
                if len(values_size) == 3: # Assume merged shape (bsz * num_heads, seq_len, hidden_size) e.g. Whisper
                    per_head_values = frame.f_locals[varname].view(batch_size, -1, *values_size[1:])
                elif len(values_size) == 4: # Assume per-head shape (bsz, num_heads, seq_len, hidden_size) e.g. GPT-2
                    per_head_values = frame.f_locals[varname]
                else:
                    raise ValueError(
                        f"Value vector shape {frame.f_locals[varname].size()} not supported. "
                        "Supported shapes: (batch_size, num_heads, seq_len, hidden_size) or "
                        "(batch_size * num_heads, seq_len, hidden_size)"
                    )
                zeroing_indices = validate_indices(per_head_values, 1, zeroing_indices)
                per_head_values.index_fill_(1, zeroing_indices.to(per_head_values.device), 0)
                if len(values_size) == 3:
                    frame.f_locals[varname] = per_head_values.view(-1, *values_size[1:])
                elif len(values_size) == 4:
                    frame.f_locals[varname] = per_head_values

        return value_zeroing_forward_mid_hook

    def get_states_extract_and_patch_hook(self, block_idx: int, hidden_state_idx: int = 0) -> Callable[..., None]:
        """Returns a hook to extract the produced hidden states (corrupted by value zeroing)
          and patch them with pre-computed clean states that will be passed onwards in the model forward.

        Args:
            block_idx (:obj:`int`): The idx of the block at which the hook is applied, used to store extracted states.
            hidden_state_idx (:obj:`int`, optional): The index of the hidden state in the model output tuple.
        """

        def states_extract_and_patch_forward_hook(module, args, output) -> None:
            self.corrupted_block_output_states[block_idx] = output[hidden_state_idx].clone().float().detach().cpu()

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
    
    def compute_modules_post_zeroing_similarity(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: TensorOrTupleOfTensorsGeneric,
        modules: nn.ModuleList,
        hidden_states: MultiLayerEmbeddingsTensor,
        similarity_scores_shape: torch.Size,
        similarity_metric: ValueZeroingSimilarityMetric = ValueZeroingSimilarityMetric.COSINE,
        mode: ValueZeroingModule = ValueZeroingModule.DECODER,
        zeroing_indices: Union[int, list[int], tuple[int, int], None] = None,
    ) -> MultiLayerScoreTensor:
        """Given a ``nn.ModuleList``, computes the similarity between the clean and corrupted states for each block.
        
        Args:
            modules (:obj:`nn.ModuleList`): The list of modules to compute the similarity for.
            hidden_states (:obj:`MultiLayerEmbeddingsTensor`): The cached hidden states of the modules to use as clean
                counterparts when computing the similarity.
            similarity_scores_shape (:obj:`torch.Size`): The shape of the similarity scores tensor to be returned.
            similarity_metric (:obj:`str`): The name of the similarity metric used. Default: "cosine".
            mode (:obj:`ValueZeroingModule`): The mode of the model to compute the similarity for. Default: "decoder".
            zeroing_indices (:obj:`Union[int, list[int], tuple[int, int], None]`, optional): The indices of the
                attention heads that should be zeroed to compute corrupted states. If None, all attention heads are
                zeroed. Default: None.

        Returns:
            :obj:`MultiLayerScoreTensor`: The similarity scores for each layer of the model.
        """
        # Store clean hidden states for later use. Starts at 1 since the first element of the modules stack is the
        # embedding layer, and we are only interested in the transformer blocks outputs.
        self.clean_block_output_states = {
            block_idx: hidden_states[:, block_idx + 1, ...].clone().detach().cpu() for block_idx in range(len(modules))
        }
        # Scores for every layer of the model
        all_scores = torch.zeros(*similarity_scores_shape)
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
        for block_idx in range(len(modules) - 1):
            states_extract_and_patch_hook = self.get_states_extract_and_patch_hook(block_idx, hidden_state_idx=0)
            states_extraction_hook_handles.append(
                modules[block_idx].register_forward_hook(states_extract_and_patch_hook)
            )
        # Zeroing is done for every token in the sequence separately (O(n) complexity)
        for token_idx in range(similarity_scores_shape[-1]):
            value_zeroing_hook_handles = []
            # Value zeroing hooks are registered for every token separately since they are token-dependent
            for block in modules:
                attention_module = block.get_submodule(self.forward_func.config.attention_module)
                value_zeroing_hook = get_post_variable_assignment_hook(
                    attention_module,
                    hook_fn=self.get_value_zeroing_hook(self.forward_func.config.value_vector),
                    varname=self.forward_func.config.value_vector,
                    value_zeroing_index=token_idx,
                    zeroing_indices=zeroing_indices,
                    batch_size=similarity_scores_shape[0],
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
                    corrupted_states_dict[f"{mode.value}_hidden_states"][:, -1, ...].clone().detach().cpu()
                )
                self.corrupted_block_output_states[len(modules) - 1] = corrupted_decoder_last_hidden_state
            for handle in value_zeroing_hook_handles:
                handle.remove()
            for block_idx in range(len(modules)):
                similarity_scores = self.SIMILARITY_METRICS[similarity_metric](
                    self.clean_block_output_states[block_idx].float(), self.corrupted_block_output_states[block_idx]
                )
                all_scores[:, block_idx, :, token_idx] = torch.ones_like(similarity_scores) - similarity_scores
        for handle in states_extraction_hook_handles:
            handle.remove()
        # Normalize all_scores to sum to 1
        per_token_sum_score = all_scores.sum(dim=-1, keepdim=True)
        per_token_sum_score[per_token_sum_score == 0] = 1
        all_scores = all_scores / per_token_sum_score
        # Since presently Inseq attribution is constrained by the attribution loop over the full generation, the
        # extraction of value zeroing scores is done inefficiently by picking only the last token scores at every step.
        # This makes the complexity of calling this method O(n^2), when it could be O(n) if the scores were extracted
        # only at the final step. Since other methods (e.g. attention) face the same issue, this will be addressed in
        # future releases.
        # Final shape: [batch_size, seq_len, num_layers]
        return all_scores[..., -1, :].mT.clone()
        



    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: TensorOrTupleOfTensorsGeneric,
        similarity_metric: str = "cosine",
        zeroing_indices: Union[int, list[int], tuple[int, int], None] = None,
        encoder_hidden_states: Optional[MultiLayerEmbeddingsTensor] = None,
        decoder_hidden_states: Optional[MultiLayerEmbeddingsTensor] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """Perform attribution using the Value Zeroing method.

        Args:
            similarity_metric (:obj:`str`, optional): The similarity metric to use for computing the distance between 
                hidden states produced with and without the zeroing operation. Default: cosine similarity.
            zeroing_indices (:obj:`Union[int, Sequence[int]]`, optional): The indices of the attention heads
                that should be zeroed to compute corrupted states. If None, all attention heads are zeroed.
                Default: None.
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
        decoder_scores_size = (batch_size, len(decoder_stack), tgt_seq_len, tgt_seq_len)
        decoder_scores = self.compute_modules_post_zeroing_similarity(
            inputs=inputs,
            additional_forward_args=additional_forward_args,
            modules=decoder_stack,
            hidden_states=decoder_hidden_states,
            similarity_scores_shape=decoder_scores_size,
            similarity_metric=similarity_metric,
            mode=ValueZeroingModule.DECODER,
            zeroing_indices=zeroing_indices,
        )
        return decoder_scores
        # Encoder-decoder models also perform zeroing on the encoder self-attention and cross-attention values
        # Adapted from https://github.com/hmohebbi/ContextMixingASR/blob/master/scoring/valueZeroing.py
        #if is_encoder_decoder:
        #    encoder_hidden_states = torch.stack(outputs.encoder_hidden_states)
        #    encoder = self.forward_func.get_encoder()
        #    encoder_stack = find_block_stack(encoder)
