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
from typing import TYPE_CHECKING, Callable, Optional

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from torch import nn
from torch.utils.hooks import RemovableHandle

from ....utils import (
    StackFrame,
    find_block_stack,
    get_post_variable_assignment_hook,
    recursive_get_submodule,
    validate_indices,
)
from ....utils.typing import (
    EmbeddingsTensor,
    InseqAttribution,
    MultiLayerEmbeddingsTensor,
    MultiLayerScoreTensor,
    OneOrMoreIndices,
    OneOrMoreIndicesDict,
)

if TYPE_CHECKING:
    from ....models import HuggingfaceModel

logger = logging.getLogger(__name__)


class ValueZeroingSimilarityMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class ValueZeroingModule(Enum):
    DECODER = "decoder"
    ENCODER = "encoder"


class ValueZeroing(InseqAttribution):
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

    def __init__(self, forward_func: "HuggingfaceModel") -> None:
        super().__init__(forward_func)
        self.clean_block_output_states: dict[int, EmbeddingsTensor] = {}
        self.corrupted_block_output_states: dict[int, EmbeddingsTensor] = {}

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
            zeroed_token_index: Optional[int] = None,
            zeroed_units_indices: Optional[OneOrMoreIndices] = None,
            batch_size: int = 1,
        ) -> None:
            if varname not in frame.f_locals:
                raise ValueError(
                    f"Variable {varname} not found in the local frame."
                    f"Other variable names: {', '.join(frame.f_locals.keys())}"
                )
            # Zeroing value vectors corresponding to the given token index
            if zeroed_token_index is not None:
                values_size = frame.f_locals[varname].size()
                if len(values_size) == 3:  # Assume merged shape (bsz * num_heads, seq_len, hidden_size) e.g. Whisper
                    values = frame.f_locals[varname].view(batch_size, -1, *values_size[1:])
                elif len(values_size) == 4:  # Assume per-head shape (bsz, num_heads, seq_len, hidden_size) e.g. GPT-2
                    values = frame.f_locals[varname].clone()
                else:
                    raise ValueError(
                        f"Value vector shape {frame.f_locals[varname].size()} not supported. "
                        "Supported shapes: (batch_size, num_heads, seq_len, hidden_size) or "
                        "(batch_size * num_heads, seq_len, hidden_size)"
                    )
                zeroed_units_indices = validate_indices(values, 1, zeroed_units_indices).to(values.device)
                zeroed_token_index = torch.tensor(zeroed_token_index, device=values.device)
                # Mask heads corresponding to zeroed units and tokens corresponding to zeroed tokens
                values[:, zeroed_units_indices, zeroed_token_index] = 0
                if len(values_size) == 3:
                    frame.f_locals[varname] = values.view(-1, *values_size[1:])
                elif len(values_size) == 4:
                    frame.f_locals[varname] = values

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
        hidden_states: MultiLayerEmbeddingsTensor,
        attention_module_name: str,
        attributed_seq_len: Optional[int] = None,
        similarity_metric: str = ValueZeroingSimilarityMetric.COSINE.value,
        mode: str = ValueZeroingModule.DECODER.value,
        zeroed_units_indices: Optional[OneOrMoreIndicesDict] = None,
        min_score_threshold: float = 1e-5,
        use_causal_mask: bool = False,
    ) -> MultiLayerScoreTensor:
        """Given a ``nn.ModuleList``, computes the similarity between the clean and corrupted states for each block.

        Args:
            modules (:obj:`nn.ModuleList`): The list of modules to compute the similarity for.
            hidden_states (:obj:`MultiLayerEmbeddingsTensor`): The cached hidden states of the modules to use as clean
                counterparts when computing the similarity.
            attention_module_name (:obj:`str`): The name of the attention module to zero the values for.
            attributed_seq_len (:obj:`int`): The length of the sequence to attribute. If not specified, it is assumed
                to be the same as the length of the hidden states.
            similarity_metric (:obj:`str`): The name of the similarity metric used. Default: "cosine".
            mode (:obj:`str`): The mode of the model to compute the similarity for. Default: "decoder".
            zeroed_units_indices (:obj:`Union[int, tuple[int, int], list[int]]` or :obj:`dict` with :obj:`int` keys and
                `Union[int, tuple[int, int], list[int]]` values, optional): The indices of the attention heads
                that should be zeroed to compute corrupted states.
                    - If None, all attention heads across all layers are zeroed.
                    - If an integer, the same attention head is zeroed across all layers.
                    - If a tuple of two integers, the attention heads in the range are zeroed across all layers.
                    - If a list of integers, the attention heads in the list are zeroed across all layers.
                    - If a dictionary, the keys are the layer indices and the values are the zeroed attention heads for
                      the corresponding layer. Any missing layer will not be zeroed.
                Default: None.
            min_score_threshold (:obj:`float`, optional): The minimum score threshold to consider when computing the
                similarity. Default: 1e-5.
            use_causal_mask (:obj:`bool`, optional): Whether a causal mask is applied to zeroing scores Default: False.

        Returns:
            :obj:`MultiLayerScoreTensor`: A tensor of shape ``[batch_size, seq_len, num_layer]`` containing distances
                (1 - similarity score) between original and corrupted states for each layer.
        """
        if mode == ValueZeroingModule.DECODER.value:
            modules: nn.ModuleList = find_block_stack(self.forward_func.get_decoder())
        elif mode == ValueZeroingModule.ENCODER.value:
            modules: nn.ModuleList = find_block_stack(self.forward_func.get_encoder())
        else:
            raise NotImplementedError(f"Mode {mode} not implemented for value zeroing.")
        if attributed_seq_len is None:
            attributed_seq_len = hidden_states.size(2)
        batch_size = hidden_states.size(0)
        generated_seq_len = hidden_states.size(2)
        num_layers = len(modules)

        # Store clean hidden states for later use. Starts at 1 since the first element of the modules stack is the
        # embedding layer, and we are only interested in the transformer blocks outputs.
        self.clean_block_output_states = {
            block_idx: hidden_states[:, block_idx + 1, ...].clone().detach().cpu() for block_idx in range(len(modules))
        }
        # Scores for every layer of the model
        all_scores = torch.ones(
            batch_size, num_layers, generated_seq_len, attributed_seq_len, device=hidden_states.device
        ) * float("nan")

        # Hooks:
        #   1. states_extract_and_patch_hook on the transformer block stores corrupted states and force clean states
        #      as the output of the block forward pass, i.e. the zeroing is done independently across layers.
        #   2. value_zeroing_hook on the attention module performs the value zeroing by replacing the "value" tensor
        #      during the forward (name is config-dependent) with a zeroed version for the specified token index.
        #
        # State extraction hooks can be registered only once since they are token-independent
        # Skip last block since its states are not used raw, but may have further transformations applied to them
        # (e.g. LayerNorm, Dropout). These are extracted separately from the model outputs.
        states_extraction_hook_handles: list[RemovableHandle] = []
        for block_idx in range(len(modules) - 1):
            states_extract_and_patch_hook = self.get_states_extract_and_patch_hook(block_idx, hidden_state_idx=0)
            states_extraction_hook_handles.append(
                modules[block_idx].register_forward_hook(states_extract_and_patch_hook)
            )
        # Zeroing is done for every token in the sequence separately (O(n) complexity)
        for token_idx in range(attributed_seq_len):
            value_zeroing_hook_handles: list[RemovableHandle] = []
            # Value zeroing hooks are registered for every token separately since they are token-dependent
            for block_idx, block in enumerate(modules):
                attention_module = recursive_get_submodule(block, attention_module_name)
                if attention_module is None:
                    raise ValueError(f"Attention module {attention_module_name} not found in block {block_idx}.")
                if isinstance(zeroed_units_indices, dict):
                    if block_idx not in zeroed_units_indices:
                        continue
                    zeroed_units_indices_block = zeroed_units_indices[block_idx]
                else:
                    zeroed_units_indices_block = zeroed_units_indices
                value_zeroing_hook = get_post_variable_assignment_hook(
                    module=attention_module,
                    varname=self.forward_func.config.value_vector,
                    hook_fn=self.get_value_zeroing_hook(self.forward_func.config.value_vector),
                    zeroed_token_index=token_idx,
                    zeroed_units_indices=zeroed_units_indices_block,
                    batch_size=batch_size,
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
                # This allows us to handle the presence of additional transformations (e.g. LayerNorm, Dropout)
                # in the last layer automatically.
                corrupted_states_dict = self.forward_func.get_hidden_states_dict(output)
                corrupted_decoder_last_hidden_state = (
                    corrupted_states_dict[f"{mode}_hidden_states"][:, -1, ...].clone().detach().cpu()
                )
                self.corrupted_block_output_states[len(modules) - 1] = corrupted_decoder_last_hidden_state
            for handle in value_zeroing_hook_handles:
                handle.remove()
            for block_idx in range(len(modules)):
                similarity_scores = self.SIMILARITY_METRICS[similarity_metric](
                    self.clean_block_output_states[block_idx].float(), self.corrupted_block_output_states[block_idx]
                )
                if use_causal_mask:
                    all_scores[:, block_idx, token_idx:, token_idx] = 1 - similarity_scores[:, token_idx:]
                else:
                    all_scores[:, block_idx, :, token_idx] = 1 - similarity_scores
            self.corrupted_block_output_states = {}
        for handle in states_extraction_hook_handles:
            handle.remove()
        self.clean_block_output_states = {}
        all_scores = torch.where(all_scores < min_score_threshold, torch.zeros_like(all_scores), all_scores)
        # Normalize scores to sum to 1
        per_token_sum_score = all_scores.nansum(dim=-1, keepdim=True)
        per_token_sum_score[per_token_sum_score == 0] = 1
        all_scores = all_scores / per_token_sum_score

        # Final shape: [batch_size, attributed_seq_len, generated_seq_len, num_layers]
        return all_scores.permute(0, 3, 2, 1)

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: TensorOrTupleOfTensorsGeneric,
        similarity_metric: str = ValueZeroingSimilarityMetric.COSINE.value,
        encoder_zeroed_units_indices: Optional[OneOrMoreIndicesDict] = None,
        decoder_zeroed_units_indices: Optional[OneOrMoreIndicesDict] = None,
        cross_zeroed_units_indices: Optional[OneOrMoreIndicesDict] = None,
        encoder_hidden_states: Optional[MultiLayerEmbeddingsTensor] = None,
        decoder_hidden_states: Optional[MultiLayerEmbeddingsTensor] = None,
        output_decoder_self_scores: bool = True,
        output_encoder_self_scores: bool = True,
    ) -> TensorOrTupleOfTensorsGeneric:
        """Perform attribution using the Value Zeroing method.

        Args:
            similarity_metric (:obj:`str`, optional): The similarity metric to use for computing the distance between
                hidden states produced with and without the zeroing operation. Default: cosine similarity.
            zeroed_units_indices (:obj:`Union[int, tuple[int, int], list[int]]` or :obj:`dict` with :obj:`int` keys and
                `Union[int, tuple[int, int], list[int]]` values, optional): The indices of the attention heads
                that should be zeroed to compute corrupted states.
                    - If None, all attention heads across all layers are zeroed.
                    - If an integer, the same attention head is zeroed across all layers.
                    - If a tuple of two integers, the attention heads in the range are zeroed across all layers.
                    - If a list of integers, the attention heads in the list are zeroed across all layers.
                    - If a dictionary, the keys are the layer indices and the values are the zeroed attention heads for
                        the corresponding layer.

                Default: None (all heads are zeroed for every layer).
            encoder_hidden_states (:obj:`torch.Tensor`, optional): A tensor of shape ``[batch_size, num_layers + 1,
                source_seq_len, hidden_size]`` containing hidden states of the encoder. Available only for
                encoder-decoders models. Default: None.
            decoder_hidden_states (:obj:`torch.Tensor`, optional): A tensor of shape ``[batch_size, num_layers + 1,
                target_seq_len, hidden_size]`` containing hidden states of the decoder.
            output_decoder_self_scores (:obj:`bool`, optional): Whether to produce scores derived from zeroing the
                decoder self-attention value vectors in encoder-decoder models. Cannot be false for decoder-only, or
                if target-side attribution is requested using `attribute_target=True`. Default: True.
            output_encoder_self_scores (:obj:`bool`, optional): Whether to produce scores derived from zeroing the
                encoder self-attention value vectors in encoder-decoder models. Default: True.

        Returns:
            `TensorOrTupleOfTensorsGeneric`: Attribution outputs for source-only or source + target feature attribution
        """
        if similarity_metric not in self.SIMILARITY_METRICS:
            raise ValueError(
                f"Similarity metric {similarity_metric} not available."
                f"Available metrics: {','.join(self.SIMILARITY_METRICS.keys())}"
            )
        decoder_scores = None
        if not self.forward_func.is_encoder_decoder or output_decoder_self_scores or len(inputs) > 1:
            decoder_scores = self.compute_modules_post_zeroing_similarity(
                inputs=inputs,
                additional_forward_args=additional_forward_args,
                hidden_states=decoder_hidden_states,
                attention_module_name=self.forward_func.config.self_attention_module,
                similarity_metric=similarity_metric,
                mode=ValueZeroingModule.DECODER.value,
                zeroed_units_indices=decoder_zeroed_units_indices,
                use_causal_mask=True,
            )
        # Encoder-decoder models also perform zeroing on the encoder self-attention and cross-attention values
        # Adapted from https://github.com/hmohebbi/ContextMixingASR/blob/master/scoring/valueZeroing.py
        if self.forward_func.is_encoder_decoder:
            encoder_scores = None
            if output_encoder_self_scores:
                encoder_scores = self.compute_modules_post_zeroing_similarity(
                    inputs=inputs,
                    additional_forward_args=additional_forward_args,
                    hidden_states=encoder_hidden_states,
                    attention_module_name=self.forward_func.config.self_attention_module,
                    similarity_metric=similarity_metric,
                    mode=ValueZeroingModule.ENCODER.value,
                    zeroed_units_indices=encoder_zeroed_units_indices,
                )
            cross_scores = self.compute_modules_post_zeroing_similarity(
                inputs=inputs,
                additional_forward_args=additional_forward_args,
                hidden_states=decoder_hidden_states,
                attributed_seq_len=encoder_hidden_states.size(2),
                attention_module_name=self.forward_func.config.cross_attention_module,
                similarity_metric=similarity_metric,
                mode=ValueZeroingModule.DECODER.value,
                zeroed_units_indices=cross_zeroed_units_indices,
            )
            return encoder_scores, cross_scores, decoder_scores
        elif encoder_zeroed_units_indices is not None or cross_zeroed_units_indices is not None:
            logger.warning(
                "Zeroing indices for encoder and cross-attentions were specified, but the model is not an "
                "encoder-decoder. Use `decoder_zeroed_units_indices` to parametrize zeroing for the decoder module."
            )
        return (decoder_scores,)
