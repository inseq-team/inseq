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
import torch.nn.functional as F
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import Attribution

from ....data import MultiDimensionalFeatureAttributionStepOutput
from ....utils.torch_utils import get_submodule
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

    @staticmethod
    def l_transform(x, w_ln, ln_eps, pre_ln_states):
        """Computes mean and performs hadamard product with layernorm weights (w_ln) as a linear transformation,
        finally dividing by the standard deviation of the pre layernorm states.

        out = number of positions in forwards-pass (number of residuals)
        Input:
            x (tensor): tensor to which apply l_transform on its out dimension
            w_ln (tensor): weights of layer norm
            ln_eps (float): epsilon of layer norm
            pre_ln_states (tensor) -> [batch_size, out, dim]: the states (or values) of the tensor just
                            before LN is applied
        Output:
            output (tensor) -> [batch_size, out, int_dim, dim]
        """

        # Create square matrix with γ parameters in diagonal
        # ln_param_transf -> [dim, dim]
        ln_param_transf = torch.diag(w_ln)
        ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - 1 / w_ln.size(0) * torch.ones_like(
            ln_param_transf
        ).to(w_ln.device)

        # Compute variance of pre final layernorm states (variance computed individual for each out position)
        # var_pre_ln_states -> [out]
        var_pre_ln_states = torch.var(pre_ln_states, unbiased=False, dim=-1)

        # Add epsilon value to each position [out]
        # Square root (element-wise) -> [out]
        # ln_std_coef -> [out]
        ln_std_coef = 1 / torch.sqrt(var_pre_ln_states + ln_eps)

        # Compute main operation
        # out [batch_size, out, int_dim, dim] , int_dim is the intermediate dimension of MLP
        output = torch.einsum("... e , e f , f g -> ... g", x, ln_mean_transf, ln_param_transf)
        # Move ln_std_coef values to first dim to multiply elementwise with out dimension of out
        # ln_std_coef -> [out, 1, 1]
        ln_std_coef = ln_std_coef.view(-1, 1, 1)
        output = output * ln_std_coef

        return output

    @torch.no_grad()
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
        # TODO: post-layer norm variant
        # TODO: check if there is a better way to do this
        # model: AttributionModel = self.forward_func
        # n_layers = decoder_self_attentions.size(1)
        # n_heads = decoder_self_attentions.size(2)
        # head_dim = embed_dim // n_heads
        config = self.forward_func.attribution_method.attribution_model.config
        model = self.forward_func.model
        n_layers = decoder_self_attentions.size(1)
        n_heads = decoder_self_attentions.size(2)
        embed_dim = modules_inputs[config.layer + ".0." + config.dense][0][0].size(-1)
        seq_len = modules_inputs[config.layer + ".0." + config.dense][0][0].size(-2)
        head_dim = embed_dim // n_heads

        list_importance_matrix = []
        list_resultant_norm = []
        for layer in range(n_layers):
            ln1 = get_submodule(model, config.ln1, layer)
            dense = get_submodule(model, config.dense, layer)
            w_o = dense.weight
            # TODO: find a better way to handle this in config yaml
            if model.config.model_type == "gpt2":
                w_o = w_o.transpose(0, 1)
            # Get W^{l,h}_V
            values_module = get_submodule(model, config.values, layer)
            # TODO: Maybe we can work directly with values from hooks in modules_outputs and avoid this
            if model.config.model_type == "gpt2":
                w_v = values_module.weight[:, -embed_dim:]
                w_v = w_v.view(-1, n_heads, head_dim)
                # b^l_v -> n_heads, dim_head
                b_v = values_module.bias[-embed_dim:].view(n_heads, -1)
            elif model.config.model_type == "opt":
                w_v = values_module.weight.transpose(0, 1)
                w_v = w_v.view(-1, n_heads, head_dim)
                # b^l_v -> n_heads, dim_head
                b_v = values_module.bias[-embed_dim:].view(n_heads, -1)
            elif model.config.model_type == "bloom":
                # BLOOM has a weird way of computing the values
                # https://github.com/huggingface/transformers/blob/cf11493dce0a1d22446efe0d6c4ade02fd928e50/src/transformers/models/bloom/modeling_bloom.py#LL238C9-L238C21
                w_v_big = values_module.weight.transpose(0, 1)
                split_w_v_big = w_v_big.view(-1, n_heads, 3, head_dim)
                w_v = split_w_v_big[:, :, 2, :].reshape(embed_dim, embed_dim)
                w_v = w_v.view(-1, n_heads, head_dim)
                # b^l_v -> n_heads, dim_head
                b_v = values_module.bias.view(n_heads, 3, head_dim)[:, 2, :]
            else:
                w_v = values_module.weight.transpose(0, 1)
                w_v = w_v.view(-1, n_heads, head_dim)
                # b^l_v -> n_heads, dim_head
                b_v = values_module.bias[-embed_dim:].view(n_heads, -1)

            # Get Get W^{l,h}_O partitioned by head -> [embed_dim, n_heads, head_dim]
            w_o_h = w_o.view(embed_dim, n_heads, head_dim)
            b_o = dense.bias

            ln1_eps = ln1.eps
            w_ln1 = ln1.weight.data.detach()
            b_ln1 = ln1.bias
            a_mat = decoder_self_attentions[:, layer]

            # Compute linear component, left term in Eq. 29
            # in `Ferrando et al. (2023) <https://aclanthology.org/2023.acl-long.301.pdf/>`
            # Head-level transformed vectors
            # layer_input -> [batch, seq_len, 1, embed_dim]
            layer_input = modules_inputs[config.layer + "." + str(layer) + "." + config.ln1][0][0].unsqueeze(2)
            # l_layer_input -> [s, d]
            l_layer_input = self.l_transform(layer_input, w_ln1, ln1_eps, layer_input).squeeze(-2)
            # v_j_heads -> [batch, n_heads, seq_len, head_dim]
            v_j_heads = torch.einsum("bsd,dhz->bhsz", l_layer_input, w_v)  # + b_v.unsqueeze(1)
            # lin_x_j_w_o_heads -> [batch, n_heads, seq_len, embed_dim]
            lin_x_j_w_o_heads = torch.einsum("bhsz,dhz->bhsd", v_j_heads, w_o_h)
            lin_x_j_heads = torch.einsum("bhsd,bhts->bhtsd", lin_x_j_w_o_heads, a_mat)

            # Compute biases term A⋅θ, right term (translation) in Eq. 29
            # in `Ferrando et al. (2023) <https://aclanthology.org/2023.acl-long.301.pdf/>`
            b_ln1_w_v = torch.einsum("d,dhz->hz", b_ln1, w_v) + b_v
            gamma_l_h = torch.einsum("hz,dhz->hd", b_ln1_w_v, w_o_h)
            biases_term_attn_heads = torch.einsum("bhts,hd->bhtsd", a_mat, gamma_l_h)
            biases_term_src = biases_term_attn_heads.sum(1)

            # Layer-level transformed vectors
            # Linear component, sum over heads lin_x_j -> [batch, seq_len, seq_len, embed_dim]
            lin_x_j = lin_x_j_heads.sum(1)
            # Affine component, linear component + translation
            aff_x_j = lin_x_j + biases_term_src

            # Make residual matrix -> [batch, seq_len, seq_len, embed_dim]
            layer_input.squeeze(-2).size()
            device = layer_input.device
            residual = torch.einsum(
                "sk,bsd->bskd", torch.eye(seq_len, dtype=layer_input.dtype).to(device), layer_input.squeeze(-2)
            )
            # Add residual to transformed vector
            res_and_aff_x_j = aff_x_j + residual

            # Add attention output and bias W^{l,h}_O
            attn_res_output = res_and_aff_x_j.sum(2) + b_o

            # Get actual output attention + res (resultant) from forward-pass activations
            # TODO: This only works for pre-layernorm, which corresponds to the input of MLP layernorm
            pre_ln2_states = modules_inputs[config.layer + "." + str(layer) + "." + config.ln2][0][0]
            real_attn_res_output = pre_ln2_states

            assert torch.dist(attn_res_output, real_attn_res_output).item() < 1e-3 * real_attn_res_output.numel()

            # TODO: add as argument in function
            # Distance between transformed attention vectors and resultant
            importance_matrix = -F.pairwise_distance(res_and_aff_x_j, attn_res_output.unsqueeze(2), p=1)

            list_importance_matrix.append(importance_matrix)
            # Save norm of resultant
            list_resultant_norm.append(torch.norm(attn_res_output, p=1, dim=-1))

        model_contributions = torch.stack(list_importance_matrix, dim=1)
        model_resultant_norm = torch.stack(list_resultant_norm, dim=1)

        # Normalization, Eq. 9 in `Ferrando et al. (2022) <https://aclanthology.org/2022.emnlp-main.595/>`
        normalized_model_contributions = torch.zeros(model_contributions.size())
        for l in range(n_layers):
            normalized_model_contributions[:, l] = model_contributions[:, l] + model_resultant_norm[:, l].unsqueeze(-1)
            normalized_model_contributions[:, l] = torch.clip(normalized_model_contributions[:, l], min=0)
            normalized_model_contributions[:, l] = normalized_model_contributions[
                :, l
            ] / normalized_model_contributions[:, l].sum(dim=-1, keepdim=True)

        # TODO: handle layers using the aggregation function of rollout
        decoder_contributions = normalized_model_contributions

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
