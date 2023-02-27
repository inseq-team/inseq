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

        if to_interp_rep_transform is None:
            to_interp_rep_transform_func = self.to_interp_rep_transform
        else:
            # Use custom function
            to_interp_rep_transform_func = to_interp_rep_transform

        super().__init__(
            forward_func=attribution_model,
            interpretable_model=interpretable_model,
            similarity_func=similarity_func,
            perturb_func=perturb_func,
            perturb_interpretable_space=perturb_interpretable_space,
            from_interp_rep_transform=from_interp_rep_transform,
            to_interp_rep_transform=to_interp_rep_transform_func,
        )
        self.attribution_model = attribution_model

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
        r"""Adapted from Captum: Two modifications at the end ensure that 3D
        tensors (needed for transformers inference) are reshaped as 2D tensors
        before being passed to the linear surrogate model, and reshaped again
        back to their 3D equivalents.

        See the LimeBase (super class) docstring for a proper description of
        LIME's functionality. What follows is an abbreviated docstring.

        Args:

            inputs (tensor or tuple of tensors):  Input for which LIME
                        is computed.
            target (int, tuple, tensor or list, optional):  Output indices for
                        which surrogate model is trained
                        (for classification cases,
                        this is usually the target class).
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided.
            n_samples (int, optional):  The number of samples of the original
                        model used to train the surrogate interpretable model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
                        to be processed simultaneously in one call to forward_fn.
            show_progress (bool, optional): Displays the progress of computation.
            **kwargs (Any, optional): Any additional arguments necessary for
                        sampling and transformation functions (provided to
                        constructor).

        Returns:
            **interpretable model representation**:
            - **interpretable model representation* (*Any*):
                    A representation of the interpretable model trained.
                    In this adaptation, the return is a 3D tensor.
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
            Zero-indexed interpretable_inps elements for unpacking the tuples.
            """
            combined_interp_inps = torch.cat([i[0].view(-1).unsqueeze(dim=0) for i in interpretable_inps]).double()

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
        perturbed_interpretable_input: tuple,
        **kwargs,
    ) -> torch.Tensor:
        r"""Calculates the similarity between original and perturbed input"""

        if len(original_input) == 1:
            original_input_tensor = original_input[0][0]
            perturbed_input_tensor = perturbed_input[0][0]
        elif len(original_input) == 2:
            original_input_tensor = torch.cat(original_input, dim=1)
            perturbed_input_tensor = torch.cat(perturbed_input, dim=1)
        else:
            raise ValueError("Original input tuple has to be of either length 1 or 2.")

        assert original_input_tensor.shape == perturbed_input_tensor.shape
        similarity = torch.sum(original_input_tensor == perturbed_input_tensor)
        return similarity

    def perturb_func(
        self,
        original_input_tuple: tuple = (),
        mask_prob: float = 0.3,
        mask_token: str = "unk",
        **kwargs: Any,
    ) -> tuple:
        r"""Sampling function:

        Args:

            original_input_tuple (tuple): Tensor tuple where its first element
                is a 3D tensor (b=1, seq_len, emb_dim)
            mask_prob (float): probability of the MASK token (no information)
                in the mask that the original input tensor is being multiplied
                with.
            mask_token (str): What kind of special token to use for masking the
                input. Options: "unk" and "pad"
        """
        perturbed_inputs = []
        for original_input_tensor in original_input_tuple:
            # Build mask for replacing random tokens with [PAD] token
            mask_value_probs = torch.tensor([mask_prob, 1 - mask_prob])
            mask_multinomial_binary = torch.multinomial(
                mask_value_probs, len(original_input_tensor[0]), replacement=True
            )

            def detach_to_list(t):
                return t.detach().cpu().numpy().tolist() if type(t) == torch.Tensor else t

            # Additionally remove special_token_ids
            mask_special_token_ids = torch.Tensor(
                [
                    1 if id_ in self.attribution_model.special_tokens_ids else 0
                    for id_ in detach_to_list(original_input_tensor[0])
                ]
            ).int()

            # Merge the binary mask with the special_token_ids mask
            mask = (
                torch.tensor([m + s if s == 0 else s for m, s in zip(mask_multinomial_binary, mask_special_token_ids)])
                .to(self.attribution_model.device)
                .unsqueeze(-1)  # 1D -> 2D
            )

            # Set special token for masking
            if mask_token == "unk":
                tokenizer_mask_token = self.attribution_model.tokenizer.unk_token_id
            elif mask_token == "pad":
                tokenizer_mask_token = self.attribution_model.tokenizer.pad_token_id
            else:
                raise ValueError(f"Invalid mask token {mask_token} for tokenizer: {self.attribution_model.tokenizer}")

            # Apply mask to original input
            perturbed_inputs.append(original_input_tensor * mask + (1 - mask) * tokenizer_mask_token)

        return tuple(perturbed_inputs)

    @staticmethod
    def to_interp_rep_transform(sample, original_input, **kwargs: Any):
        return sample
