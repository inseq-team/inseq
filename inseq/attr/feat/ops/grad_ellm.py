"""Grad-ELLM scores, with explicit choices for reproducing the supplied code.

Q/K are raw linear projections before RoPE. No attention probability tensors
are needed. Hooks exist only during one attribution forward/backward.
"""

from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor

from .grad_ellm_device import attribution_device, model_input_device


def expand_kv(x: Tensor, query_width: int, head_dim: int, mode: str) -> Tensor:
    """Expand flattened K/V, retaining the original implementation as a mode."""
    if mode not in {"legacy_repeat", "grouped"}:
        raise ValueError("kv_expansion must be 'legacy_repeat' or 'grouped'.")
    width = x.shape[-1]
    if query_width % width or width % head_dim:
        raise ValueError("Q/K/V widths must form an integer GQA ratio with equal head dimensions.")
    ratio = query_width // width
    if mode == "legacy_repeat":
        # Original code repeats the entire flattened vector, not each KV head.
        return x.repeat(1, 1, ratio) if ratio != 1 else x
    shape = x.shape
    return (
        x.reshape(*shape[:-1], width // head_dim, head_dim)
        .repeat_interleave(ratio, dim=-2)
        .reshape(*shape[:-1], query_width)
    )


def layer_scores(q: Tensor, k: Tensor, v: Tensor, grad: Tensor, mask: Tensor, normalize: bool) -> Tensor:
    """One score per prefix token. Min/max is per example, excluding padding."""
    if q.shape != grad.shape or k.shape != v.shape or k.shape[-1] != q.shape[-1]:
        raise ValueError("Q, expanded K/V, and the attention gradient have incompatible dimensions.")
    if q.dtype in (torch.float16, torch.bfloat16):
        # Avoid fp16 epsilon underflow at zero norms and overflow in grad * V.
        # Cast captured values only; model forward/backward retains its dtype.
        q, k, v, grad = (tensor.float() for tensor in (q, k, v, grad))
    if normalize:
        similarity = (F.normalize(q[:, None, :], dim=-1) * F.normalize(k, dim=-1)).sum(-1)
        lower = similarity.masked_fill(~mask, float("inf")).amin(-1, keepdim=True)
        upper = similarity.masked_fill(~mask, -float("inf")).amax(-1, keepdim=True)
        span = upper - lower
        # The supplied implementation yields NaNs for constant similarities.
        # Here constant similarity yields zero attribution, without random fallback.
        similarity = (similarity - lower) / torch.where(span == 0, torch.ones_like(span), span)
        similarity = similarity.masked_fill(~mask, 0)
    else:
        similarity = (q[:, None, :] * k).sum(-1) / (k.shape[-1] ** 0.5)
        similarity = similarity.masked_fill(~mask, -float("inf")).softmax(-1)
    return F.relu((grad[:, None, :] * v * similarity[:, :, None]).sum(-1)).masked_fill(~mask, 0)


class GradELLMCore:
    """Core for Llama/Mistral decoder-only Hugging Face models.

    Defaults reproduce the supplied batch-size-one algorithm (apart from the
    constant-similarity NaN guard). 'grouped' and 'pre_o' are explicit alternative
    semantics and must be evaluated separately from the published results.
    """

    def __init__(
        self,
        model,
        n_layers: int = 1,
        normalize: bool = True,
        kv_expansion: str = "legacy_repeat",
        gradient_source: str = "attention_output",
    ):
        if getattr(model.config, "is_encoder_decoder", False):
            raise NotImplementedError("Grad-ELLM currently supports decoder-only Llama/Mistral models, not T5.")
        if getattr(model.config, "model_type", None) not in {"llama", "mistral"}:
            raise NotImplementedError("Validated model families: Llama and Mistral; other families need adapters.")
        if (
            getattr(model, "is_quantized", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
        ):
            raise NotImplementedError("Quantized models are not validated. Use a standard floating-point model.")
        model_input_device(model)
        layers = model.model.layers
        if isinstance(n_layers, bool) or not isinstance(n_layers, int) or not 1 <= n_layers <= len(layers):
            raise ValueError(f"n_layers must be an integer between 1 and {len(layers)}.")
        if kv_expansion not in {"legacy_repeat", "grouped"}:
            raise ValueError("kv_expansion must be 'legacy_repeat' or 'grouped'.")
        if gradient_source not in {"attention_output", "pre_o"}:
            raise ValueError("gradient_source must be 'attention_output' or 'pre_o'.")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool, not a string or number.")
        self.model = model
        self.layers = list(layers[-n_layers:])
        self.n_layers = n_layers
        self.normalize = normalize
        self.kv_expansion = kv_expansion
        self.gradient_source = gradient_source
        self._active = False
        self._captures = {}
        self.head_dims = []
        for layer in self.layers:
            attention = layer.self_attn
            if not all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
                raise NotImplementedError("Expected separate q_proj/k_proj/v_proj/o_proj modules.")
            heads = model.config.num_attention_heads
            width = attention.q_proj.out_features
            if width % heads:
                raise ValueError("Query projection width is not divisible by the number of attention heads.")
            if gradient_source == "attention_output" and width != attention.o_proj.out_features:
                raise NotImplementedError(
                    "Post-o_proj gradient width must match Q width; this architecture needs an adapter."
                )
            self.head_dims.append(width // heads)

    @contextmanager
    def _capture(self):
        if self._active:
            raise RuntimeError("Concurrent or reentrant attribution on one model is unsupported.")
        self._active = True
        handles = []
        captures = [{} for _ in self.layers]
        self._captures = captures
        try:
            for idx, layer in enumerate(self.layers):
                attention = layer.self_attn

                def save_projection(name, index):
                    def hook(module, args, output):
                        if name in captures[index]:
                            raise NotImplementedError(
                                "Grad-ELLM expects one model forward per attribution objective; "
                                "multi-forward/contrastive objectives require a separate adapter."
                            )
                        captures[index][name] = (output[:, -1, :] if name == "q" else output).detach()

                    return hook

                for name in ("q", "k", "v"):
                    handles.append(
                        getattr(attention, f"{name}_proj").register_forward_hook(save_projection(name, idx))
                    )

                if self.gradient_source == "pre_o":

                    def save_pre_o(index):
                        def hook(module, args):
                            activation = args[0]
                            if not activation.requires_grad:
                                activation.requires_grad_(True)
                            captures[index]["activation"] = activation

                        return hook

                    handles.append(attention.o_proj.register_forward_pre_hook(save_pre_o(idx)))
                else:

                    def save_output(index):
                        def hook(module, args, output):
                            activation = output[0] if isinstance(output, (tuple, list)) else output
                            if not activation.requires_grad:
                                # Enables a local gradient even if all model parameters are frozen.
                                activation.requires_grad_(True)
                            captures[index]["activation"] = activation

                        return hook

                    handles.append(attention.register_forward_hook(save_output(idx)))
            yield captures
        finally:
            for handle in handles:
                handle.remove()
            captures.clear()
            self._captures = {}
            self._active = False

    def explain(self, forward_call: Callable[[], Tensor], attention_mask: Tensor) -> Tensor:
        """Differentiate a per-example objective and return [batch, prefix]."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Run attribution outside torch.inference_mode(); torch.no_grad() is supported.")
        if self.model.training:
            raise ValueError("Use model.eval() for reproducible attribution without dropout/checkpoint recomputation.")
        model_input_device(self.model)
        if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2 or 0 in attention_mask.shape:
            raise ValueError("Expected a non-empty [batch, prefix_length] attention mask.")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("Attention masks must contain only zero or one.")
        mask = attention_mask.bool()
        if not mask.any(-1).all() or not mask[:, -1].all():
            raise ValueError("Expected a non-empty prefix and left padding (last token must be valid).")
        if ((mask[:, :-1]) & (~mask[:, 1:])).any():
            raise ValueError("Attention masks must contain a contiguous valid suffix (left padding only).")
        with torch.enable_grad(), self._capture() as captures:
            objective = forward_call()
            if not isinstance(objective, Tensor) or not objective.requires_grad:
                raise ValueError("The attribution objective must be a differentiable tensor from one model forward.")
            if objective.numel() != mask.shape[0]:
                raise ValueError("The attribution objective must return one scalar per example.")
            if any("activation" not in item for item in captures):
                raise ValueError("The objective did not run every selected attention layer.")
            activations = [item["activation"] for item in captures]
            gradients = torch.autograd.grad(objective.sum(), activations, retain_graph=False, create_graph=False)
            scores = []
            for idx, (item, gradient) in enumerate(zip(captures, gradients, strict=False)):
                q = item["q"]
                k = expand_kv(item["k"], q.shape[-1], self.head_dims[idx], self.kv_expansion)
                v = expand_kv(item["v"], q.shape[-1], self.head_dims[idx], self.kv_expansion)
                scores.append(layer_scores(q, k, v, gradient[:, -1, :].detach(), mask.to(q.device), self.normalize))
            # Multi-device shards are not validated, but sum on the last selected layer's device.
            destination = scores[-1].device
            return torch.stack([score.to(destination) for score in scores]).sum(0).detach()

    def attribute(self, input_ids: Tensor, target_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Convenience next-token logit attribution on exact token IDs."""
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or 0 in input_ids.shape:
            raise ValueError("Expected non-empty input_ids with shape [batch, prefix_length].")
        if (
            not isinstance(target_ids, Tensor)
            or input_ids.dtype not in (torch.int32, torch.int64)
            or target_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Input and target token IDs must be integer tensors (int32/int64).")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if not isinstance(attention_mask, Tensor) or attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        targets = target_ids.reshape(-1, 1)
        if targets.shape[0] != input_ids.shape[0]:
            raise ValueError("Expected one target token ID per example.")
        device = model_input_device(self.model)
        input_ids = input_ids.to(device=device, dtype=torch.long)
        attention_mask = attention_mask.to(device=device)

        def forward_call():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            return output.logits[:, -1, :].gather(-1, targets.to(output.logits.device, dtype=torch.long)).squeeze(-1)

        with attribution_device(device):
            return self.explain(forward_call, attention_mask)
