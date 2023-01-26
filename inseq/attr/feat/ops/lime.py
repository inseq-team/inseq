import inspect
import math
import warnings
from functools import partial
from typing import Any, Callable, Optional, cast

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
)
from captum._utils.models.linear_model import SkLearnLinearModel
from captum._utils.models.model import Model
from captum._utils.progress import progress
from captum._utils.typing import (
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr import LimeBase
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class Lime(LimeBase):
    def __init__(
        self,
        attribution_model: Callable,
        interpretable_model: Model = None,
        similarity_func: Callable = None,
        perturb_func: Callable = None,
        perturb_interpretable_space: bool = False,
        from_interp_rep_transform: Optional[Callable] = None,
        to_interp_rep_transform: Optional[Callable] = None,
        mask_prob: float = 0.3,
    ) -> None:
        if interpretable_model is None:
            interpretable_model = SkLearnLinearModel("linear_model.Ridge")

        if similarity_func is None:
            similarity_func = self.token_similarity_kernel

        if perturb_func is None:
            perturb_func = partial(
                self.perturb_func,
                mask_prob=mask_prob,
            )

        super().__init__(
            forward_func=attribution_model,
            interpretable_model=interpretable_model,
            similarity_func=similarity_func,
            perturb_func=perturb_func,
            perturb_interpretable_space=perturb_interpretable_space,
            from_interp_rep_transform=None,
            to_interp_rep_transform=self.to_interp_rep_transform,
        )
        self.attribution_model = attribution_model
        assert self.attribution_model.model.device is not None
        assert self.attribution_model.tokenizer.pad_token_id is not None

    # @log_usage
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_samples: int = 50,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above.
        It trains an interpretable model and returns a representation of the
        interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME is generally
        used for sample-based interpretability, training a separate interpretable
        model to explain a model's prediction on each individual example.

        A batch of inputs can be provided as inputs only if forward_func
        returns a single value per batch (e.g. loss).
        The interpretable feature representation should still have shape
        1 x num_interp_features, corresponding to the interpretable
        representation for the full batch, and perturbations_per_eval
        must be set to 1.

        Args:

            inputs (tensor or tuple of tensors):  Input for which LIME
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            target (int, tuple, tensor or list, optional):  Output indices for
                        which surrogate model is trained
                        (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. For all other types,
                        the given argument is used for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_samples (int, optional):  The number of samples of the original
                        model used to train the surrogate interpretable model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
                        to be processed simultaneously in one call to forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function returns a single scalar per batch,
                        perturbations_per_eval must be set to 1.
                        Default: 1
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False
            **kwargs (Any, optional): Any additional arguments necessary for
                        sampling and transformation functions (provided to
                        constructor).
                        Default: None

        Returns:
            **interpretable model representation**:
            - **interpretable model representation* (*Any*):
                    A representation of the interpretable model trained. The return
                    type matches the return type of train_interpretable_model_func.
                    For example, this could contain coefficients of a
                    linear surrogate model.
        """
        with torch.no_grad():
            inp_tensor = cast(Tensor, inputs) if isinstance(inputs, Tensor) else inputs[0]
            device = inp_tensor.device

            interpretable_inps = []
            similarities = []
            outputs = []

            curr_model_inputs = []
            expanded_additional_args = None
            expanded_target = None
            perturb_generator = None
            if inspect.isgeneratorfunction(self.perturb_func):
                perturb_generator = self.perturb_func(inputs, **kwargs)

            if show_progress:
                attr_progress = progress(
                    total=math.ceil(n_samples / perturbations_per_eval),
                    desc=f"{self.get_name()} attribution",
                )
                attr_progress.update(0)

            batch_count = 0
            for _ in range(n_samples):
                if perturb_generator:
                    try:
                        curr_sample = next(perturb_generator)
                    except StopIteration:
                        warnings.warn("Generator completed prior to given n_samples iterations!")
                        break
                else:
                    curr_sample = self.perturb_func(inputs, **kwargs)
                batch_count += 1
                if self.perturb_interpretable_space:
                    interpretable_inps.append(curr_sample)
                    curr_model_inputs.append(
                        self.from_interp_rep_transform(curr_sample, inputs, **kwargs)  # type: ignore
                    )
                else:
                    curr_model_inputs.append(curr_sample)
                    interpretable_inps.append(
                        self.to_interp_rep_transform(curr_sample, inputs, **kwargs)  # type: ignore
                    )
                curr_sim = self.similarity_func(inputs, curr_model_inputs[-1], interpretable_inps[-1], **kwargs)
                similarities.append(
                    curr_sim.flatten() if isinstance(curr_sim, Tensor) else torch.tensor([curr_sim], device=device)
                )

                if len(curr_model_inputs) == perturbations_per_eval:
                    if expanded_additional_args is None:
                        expanded_additional_args = _expand_additional_forward_args(
                            additional_forward_args, len(curr_model_inputs)
                        )
                    if expanded_target is None:
                        expanded_target = _expand_target(target, len(curr_model_inputs))

                    model_out = self._evaluate_batch(
                        curr_model_inputs,
                        expanded_target,
                        expanded_additional_args,
                        device,
                    )

                    if show_progress:
                        attr_progress.update()

                    outputs.append(model_out)

                    curr_model_inputs = []

            if len(curr_model_inputs) > 0:
                expanded_additional_args = _expand_additional_forward_args(
                    additional_forward_args, len(curr_model_inputs)
                )
                expanded_target = _expand_target(target, len(curr_model_inputs))
                model_out = self._evaluate_batch(
                    curr_model_inputs,
                    expanded_target,
                    expanded_additional_args,
                    device,
                )
                if show_progress:
                    attr_progress.update()
                outputs.append(model_out)

            if show_progress:
                attr_progress.close()

            """ Modification of original attribute function:
            Squeeze the batch dimension out of interpretable_inps
            -> 2D tensor (n_samples ✕ (input_dim * embedding_dim))
            """
            combined_interp_inps = torch.cat([i.view(-1).unsqueeze(dim=0) for i in interpretable_inps]).double()

            combined_outputs = (torch.cat(outputs) if len(outputs[0].shape) > 0 else torch.stack(outputs)).double()
            combined_sim = (
                torch.cat(similarities) if len(similarities[0].shape) > 0 else torch.stack(similarities)
            ).double()
            dataset = TensorDataset(combined_interp_inps, combined_outputs, combined_sim)
            self.interpretable_model.fit(DataLoader(dataset, batch_size=batch_count))

            """ Second modification:
            Reshape of the learned representation
            -> 3D tensor (b=1 ✕ input_dim ✕ embedding_dim)
            """
            return self.interpretable_model.representation().reshape(inp_tensor.shape)

    @staticmethod
    def token_similarity_kernel(
        original_input: tuple,
        perturbed_input: tuple,
        perturbed_interpretable_input: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        original_input_tensor = original_input[0]  # [0]
        perturbed_input_tensor = perturbed_input[0]
        assert original_input_tensor.shape == perturbed_input_tensor.shape
        similarity = torch.sum(original_input_tensor == perturbed_input_tensor) / len(original_input_tensor)
        return similarity

    def perturb_func(
        self,
        mask_prob: float = 0.3,
        original_input_tuple: tuple = (),  # always needs to be last argument before **kwargs due to "partial"
        **kwargs: Any,
    ) -> tuple:
        """
        Sampling function
        """
        original_input = original_input_tuple[0]

        # Build mask for replacing random tokens with [PAD] token
        mask_value_probs = torch.tensor([mask_prob, 1 - mask_prob])
        mask_multinomial_binary = torch.multinomial(mask_value_probs, len(original_input[0]), replacement=True)

        def detach_to_list(t):
            return t.detach().cpu().numpy().tolist() if type(t) == torch.Tensor else t

        # Additionally remove special_token_ids
        mask_special_token_ids = torch.Tensor(
            [1 if id_ in self.attribution_model.special_token_ids else 0 for id_ in detach_to_list(original_input[0])]
        ).int()

        # Merge the binary mask (12.5% masks) with the special_token_ids mask
        torch.tensor([m + s if s == 0 else s for m, s in zip(mask_multinomial_binary, mask_special_token_ids)]).to(
            self.attribution_model.device
        )

        # Apply mask to original input

    @staticmethod
    def to_interp_rep_transform(sample, original_input, **kwargs: Any):
        return sample[0]  # [0]  # FIXME: Access first entry of tuple
