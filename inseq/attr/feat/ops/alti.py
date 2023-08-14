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
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import Attribution
from captum.log import log_usage

from ....data import MultiDimensionalFeatureAttributionStepOutput
from ....utils.typing import MultiLayerEmbeddingsTensor, MultiLayerMultiUnitScoreTensor

if TYPE_CHECKING:
    from ....models import AttributionModel

logger = logging.getLogger(__name__)


class Alti(Attribution):
    """ALTI method for feature attribution.

    Introduced by `Ferrando et al. (2022) <https://aclanthology.org/2022.emnlp-main.595/>`__ to quantify context mixing
    inside Transformer models. The method is based on previous work by `Kobayashi et al. (2021)
    <https://aclanthology.org/2021.emnlp-main.373/>`__ estimating token-to-token contributions at every layer of a
    Transformer model. In ALTI, per-layer importance scores are quantified as the L1 norm of the difference between
    the output of the attention block and the transformed vector obtained by the decomposed attention block (see
    `Ferrando et al. (2022) <https://aclanthology.org/2022.emnlp-main.595/>`__ for details).

    Attributes:
        forward_func (:obj:`AttributionModel`):
            The attribution model to be used for ALTI attribution.
    """

    def __init__(self, forward_func: "AttributionModel") -> None:
        super().__init__(forward_func)

    @staticmethod
    def has_convergence_delta() -> bool:
        return False

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: TensorOrTupleOfTensorsGeneric,
        encoder_self_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
        decoder_self_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
        cross_attentions: Optional[MultiLayerMultiUnitScoreTensor] = None,
        encoder_hidden_states: Optional[MultiLayerEmbeddingsTensor] = None,
        decoder_hidden_states: Optional[MultiLayerEmbeddingsTensor] = None,
        modules_inputs: Optional[Dict[str, List[torch.Tensor]]] = None,
        modules_outputs: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """Perform attribution using the ALTI method.
        Args:
            encoder_hidden_states (:obj:`torch.Tensor`, optional): A tensor of shape ``[batch_size, num_layers + 1,
                source_seq_len, hidden_size]`` containing hidden states of the encoder. Available only for
                encoder-decoders models. Default: None.
            decoder_hidden_states (:obj:`torch.Tensor`, optional): A tensor of shape ``[batch_size, num_layers + 1,
                target_seq_len, hidden_size]`` containing hidden states of the decoder.
            encoder_self_attentions (:obj:`torch.Tensor`, *optional*, defaults to None): Tensor of encoder
                self-attention weights of the forward pass with shape
                :obj:`(batch_size, n_layers, n_heads, source_seq_len, source_seq_len)`.
            decoder_self_attentions (:obj:`torch.Tensor`, *optional*, defaults to None): Tensor of decoder
                self-attention weights of the forward pass with shape
                :obj:`(batch_size, n_layers, n_heads, target_seq_len, target_seq_len)`.
            cross_attentions (:obj:`torch.Tensor`, *optional*, defaults to None):
                Tensor of cross-attention weights computed during the forward pass with shape
                :obj:`(batch_size, n_layers, n_heads, target_seq_len, source_seq_len)`.
            modules_inputs (:obj:`Dict[str, List[torch.Tensor]]`, *optional*, defaults to None): Dictionary of
                module names mapped to a list of inputs to that module.
            modules_outputs (:obj:`Dict[str, List[torch.Tensor]]`, *optional*, defaults to None): Dictionary of
                module names mapped to a list of outputs to that module.

        Returns:
            `TensorOrTupleOfTensorsGeneric`: Attribution outputs for a single generation step with source-only or
            source + target feature attribution
        """
        # TODO: Implement decoder-only and encoder-decoder variants
        # Resulting tensors (pre-rollout):
        # `decoder_contributions` is a tensor of ALTI contributions for each token in the target sequence
        # with shape (batch_size, n_layers, target_seq_len, target_seq_len)
        decoder_contributions = torch.zeros_like(decoder_self_attentions[:, :, 0, ...])
        if self.forward_func.is_encoder_decoder:
            # `encoder_contributions` is a tensor of ALTI contributions for each token in the source sequence
            # with shape (batch_size, n_layers, source_seq_len, source_seq_len)
            encoder_contributions = torch.zeros_like(encoder_self_attentions[:, :, 0, ...])
            # `cross_contributions` is a tensor of ALTI contributions of shape
            # (batch_size, n_layers, target_seq_len, source_seq_len)
            cross_contributions = torch.zeros_like(cross_attentions[:, :, 0, ...])
        else:
            encoder_contributions = None
            cross_contributions = None

        # We adopt the format [batch_size, sequence_length, num_layers]
        # for consistency with other multi-unit methods (e.g. gradient attribution)
        decoder_contributions = decoder_contributions[..., -1, :].clone().permute(0, 2, 1)
        if self.forward_func.is_encoder_decoder:
            sequence_scores = {}
            # Source and target contribution performed for encoder-decoder models
            if len(inputs) > 1:
                target_attributions = decoder_contributions
            else:
                target_attributions = None
                sequence_scores["decoder_contributions"] = decoder_contributions
            sequence_scores["encoder_contributions"] = encoder_contributions[..., -1, :].clone().permute(0, 2, 1)
            return MultiDimensionalFeatureAttributionStepOutput(
                source_attributions=cross_contributions[..., -1, :].clone().permute(0, 2, 1),
                target_attributions=target_attributions,
                sequence_scores=sequence_scores,
                _num_dimensions=1,  # num_layers
            )
        else:
            return MultiDimensionalFeatureAttributionStepOutput(
                source_attributions=None,
                target_attributions=decoder_contributions,
                _num_dimensions=1,  # num_layers
            )
