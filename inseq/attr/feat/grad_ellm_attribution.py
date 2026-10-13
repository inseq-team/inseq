"""Native Inseq registry adapter for the target-dependent Grad-ELLM method."""

import torch

from inseq.data.grad_ellm import GradELLMStepOutput

from .internals_attribution import InternalsAttributionRegistry
from .ops.grad_ellm import GradELLMCore


class GradELLMOperation:
    def __init__(self, attribution_model, core):
        self.attribution_model = attribution_model
        self.core = core

    def attribute(self, inputs, additional_forward_args):
        if len(inputs) != 1:
            raise NotImplementedError("Expected one decoder-only input tensor.")
        mask = additional_forward_args[3]
        if mask is None:
            import torch

            mask = torch.ones_like(inputs[0])

        def forward_call():
            # Respect Inseq's logit/probability/custom objective, retaining its full computation graph.
            return self.attribution_model(inputs[0], *additional_forward_args, use_cache=False)

        return self.core.explain(forward_call, mask)


class GradELLMAttribution(InternalsAttributionRegistry):
    """Per-token Grad-ELLM with standard Inseq output, generation, and step scores.

    Pass attributed_fn='logit' to model.attribute to match the supplied implementation.
    Options passed to model.attribute override constructor settings for that call only.
    """

    method_name = "grad_ellm"
    _option_names = ("n_layers", "normalize", "kv_expansion", "gradient_source")

    def __init__(
        self,
        attribution_model,
        n_layers=None,
        normalize=None,
        kv_expansion=None,
        gradient_source=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unknown Grad-ELLM options: {sorted(kwargs)}")
        # Inseq constructs a fresh instance when attribute(method=...) is used.
        # Keep explicitly configured options on this wrapper across such calls.
        options = {
            "n_layers": 1,
            "normalize": True,
            "kv_expansion": "grouped",
            "gradient_source": "pre_o",
        }
        options.update(getattr(attribution_model, "_grad_ellm_options", {}))
        supplied = {
            "n_layers": n_layers,
            "normalize": normalize,
            "kv_expansion": kv_expansion,
            "gradient_source": gradient_source,
        }
        options.update({key: value for key, value in supplied.items() if value is not None})
        self.core = GradELLMCore(attribution_model.model, **options)
        attribution_model._grad_ellm_options = options.copy()
        super().__init__(attribution_model, hook_to_model=False)
        self.attribute_batch_ids = True
        self.forward_batch_embeds = False
        self.use_predicted_target = True
        self.is_final_step_method = False
        self.method = GradELLMOperation(attribution_model, self.core)
        self.hook()

    def get_attribution_args(self, **kwargs):
        # Route call-time options even though the token-level operation only
        # accepts inputs and forward arguments. Configure the core once per batch.
        options = {name: kwargs.pop(name) for name in self._option_names if name in kwargs}
        attribution_args, unused_args = super().get_attribution_args(**kwargs)
        attribution_args.update(options)
        return attribution_args, unused_args

    def attribute(self, *args, attribution_args=None, **kwargs):
        attribution_args = dict(attribution_args or {})
        options = {name: getattr(self.core, name) for name in self._option_names}
        overrides = {}
        for name in self._option_names:
            if name in attribution_args:
                value = attribution_args.pop(name)
                if value is not None:
                    overrides[name] = value
        options.update(overrides)

        previous_core = self.core
        if overrides:
            # Validate before changing either reference. Avoid constructing a core
            # for every output token, and leave saved constructor options intact.
            call_core = GradELLMCore(self.attribution_model.model, **options)
            self.core = self.method.core = call_core
        try:
            out = super().attribute(*args, attribution_args=attribution_args, **kwargs)
            for step in out.step_attributions or []:
                step.step_scores = {
                    name: values.float() if values.dtype == torch.bfloat16 else values
                    for name, values in step.step_scores.items()
                }
            out.info["grad_ellm_options"] = options
            return out
        finally:
            self.core = self.method.core = previous_core

    def attribute_step(self, attribute_fn_main_args, attribution_args=None):
        scores = self.method.attribute(**attribute_fn_main_args, **(attribution_args or {}))
        # Inseq uses target_attributions for the entire decoder-only prefix,
        # including the prompt. Raw scores stay two-dimensional; the output's
        # default aggregator applies Inseq's usual per-column L1 normalization.
        return GradELLMStepOutput(target_attributions=scores, step_scores={})
