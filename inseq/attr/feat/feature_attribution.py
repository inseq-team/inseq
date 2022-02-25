# Copyright 2021 The Inseq Team. All rights reserved.
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
""" Feature attribution methods registry.

Todo:
    * 🟡: Allow custom arguments for model loading in the :class:`FeatureAttribution` :meth:`load` method.
"""

from typing import Any, Dict, List, Literal, NoReturn, Optional, Tuple, Union, overload

import logging
from abc import abstractmethod

from torchtyping import TensorType

from ...data import (
    Batch,
    BatchEmbedding,
    BatchEncoding,
    EncoderDecoderBatch,
    FeatureAttributionInput,
    FeatureAttributionOutput,
    FeatureAttributionSequenceOutput,
    FeatureAttributionStepOutput,
    OneOrMoreFeatureAttributionSequenceOutputs,
    OneOrMoreFeatureAttributionSequenceOutputsWithStepOutputs,
)
from ...data.viz import close_progress_bar, get_progress_bar, update_progress_bar
from ...utils import (
    Registry,
    UnknownAttributionMethodError,
    drop_padding,
    extract_signature_args,
    find_char_indexes,
    pretty_tensor,
    remap_from_filtered,
)
from ...utils.typing import ModelIdentifier, TargetIdsTensor
from ..attribution_decorators import set_hook, unset_hook
from .attribution_utils import get_attribution_sentences, get_split_targets, rescale_attributions_to_tokens


logger = logging.getLogger(__name__)


class FeatureAttribution(Registry):
    r"""
    Abstract registry for feature attribution methods.

    Attributes:
        attr (:obj:`str`): Attribute of child classes that will act as lookup name
            for the registry.
        ignore_extra_args (:obj:`list` of :obj:`str`): Arguments used by default in the
            attribute step and thus ignored as extra arguments during attribution.
            The selection of defaults follows the `Captum <https://captum.ai/api/integrated_gradients.html>`__
            naming convention.
    """

    attr = "method_name"
    ignore_extra_args = ["inputs", "baselines", "target", "additional_forward_args"]

    def __init__(self, attribution_model, hook_to_model: bool = True, **kwargs):
        r"""
        Common instantiation steps for FeatureAttribution methods. Hooks the attribution method
        to the model calling the :meth:`~inseq.attr.feat.FeatureAttribution.hook` method of the child class.

        Args:
            attribution_model (:class:`~inseq.models.AttributionModel`): The attribution model
                that is used to obtain predictions and on which attribution is performed.
            hook_to_model (:obj:`bool`, default `True`): Whether the attribution method should be
                hooked to the attribution model during initialization.
            **kwargs: Additional keyword arguments to pass to the hook method.
        Attributes:
            is_layer_attribution (:obj:`bool`, default `False`): If True, the attribution method maps saliency
                scores to the output of a layer instead of model inputs. Layer attribution methods do not require
                interpretable embeddings unless intermediate features before the embedding layer are attributed.
            target_layer (:obj:`torch.nn.Module`, default `None`): The layer on which attribution should be
                performed if is_layer_attribution is True.
            use_baseline (:obj:`bool`, default `False`): Whether a baseline should be used for the attribution method.
        """
        super().__init__()
        self.attribution_model = attribution_model
        self.is_layer_attribution: bool = False
        self.target_layer = None
        self.use_baseline: bool = False
        if hook_to_model:
            self.hook(**kwargs)

    @classmethod
    def load(
        cls,
        method_name: str,
        attribution_model=None,
        model_name_or_path: ModelIdentifier = None,
        **kwargs,
    ) -> "FeatureAttribution":
        r"""
        Load the selected method and hook it to an existing or available
        attribution model.

        Args:
            method_name (:obj:`str`): The name of the attribution method to load.
            attribution_model (:class:`~inseq.models.AttributionModel`, `optional`): An instance of an
                :class:`~inseq.models.AttributionModel` child class. If not provided, the method
                will try to load the model from the model_name_or_path argument. Defaults to None.
            model_name_or_path (:obj:`ModelIdentifier`, `optional`): The name of the model to load or its
                path on disk. If not provided, an instantiated model must be provided. If the model is loaded
                in this way, the model will be created with default arguments. Defaults to None.
            **kwargs: Additional arguments to pass to the attribution method :obj:`__init__` function.

        Raises:
            :obj:`RuntimeError`: Raised if both or neither model_name_or_path and attribution_model are
                provided.
            :obj:`UnknownAttributionMethodError`: Raised if the method_name is not found in the registry.

        Returns:
            :class:`~inseq.attr.feat.FeatureAttribution`: The loaded attribution method.
        """
        from inseq import AttributionModel

        if model_name_or_path is None == attribution_model is None:  # noqa
            raise RuntimeError(
                "Only one among an initialized model and a model identifier "
                "must be defined when loading the attribution method."
            )
        if model_name_or_path:
            attribution_model = AttributionModel.load(model_name_or_path)
        methods = cls.available_classes()
        if method_name not in methods:
            raise UnknownAttributionMethodError(method_name)
        return methods[method_name](attribution_model, **kwargs)

    def prepare_and_attribute(
        self,
        sources: FeatureAttributionInput,
        targets: FeatureAttributionInput,
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        output_step_attributions: bool = False,
        attribute_target: bool = False,
        **kwargs,
    ) -> OneOrMoreFeatureAttributionSequenceOutputsWithStepOutputs:
        r"""
        Prepares inputs and performs attribution.

        Wraps the attribution method :meth:`~inseq.attr.feat.FeatureAttribution.attribute` method
        and the :meth:`~inseq.attr.feat.FeatureAttribution.prepare` method.

        Args:
            sources (:obj:`FeatureAttributionInput`): The sources provided to the
                :meth:`~inseq.attr.feat.FeatureAttribution.prepare` method.
            targets (:obj:`FeatureAttributionInput): The targets provided to the
                :meth:`~inseq.attr.feat.FeatureAttribution.prepare` method.
            attr_pos_start (:obj:`int`, `optional`): The initial position for performing
                sequence attribution. Defaults to 0.
            attr_pos_end (:obj:`int`, `optional`): The final position for performing sequence
                attribution. Defaults to None (full string).
            show_progress (:obj:`bool`, `optional`): Whether to show a progress bar. Defaults to True.
            pretty_progress (:obj:`bool`, `optional`): Whether to use a pretty progress bar. Defaults to True.
            output_step_attributions (:obj:`bool`, `optional`): Whether to output a list of FeatureAttributionOutput
                objects for each step. Defaults to False.
            attribute_target (:obj:`bool`, `optional`): Whether to include target prefix for feature attribution.
                Defaults to False.

        Returns:
            :obj:`OneOrMoreFeatureAttributionSequenceOutputsWithStepOutputs`: One or more
                :class:`~inseq.data.FeatureAttributionSequenceOutput` depending on the number of inputs, with an
                optional added list of single :class:`~inseq.data.FeatureAttributionOutput` for each step.
        """
        prepend_bos_token = kwargs.get("prepend_bos_token", True)
        batch = self.prepare(sources, targets, prepend_bos_token)
        return self.attribute(
            batch,
            attr_pos_start=attr_pos_start,
            attr_pos_end=attr_pos_end,
            show_progress=show_progress,
            pretty_progress=pretty_progress,
            output_step_attributions=output_step_attributions,
            attribute_target=attribute_target,
            **kwargs,
        )

    def prepare(
        self,
        sources: FeatureAttributionInput,
        targets: FeatureAttributionInput,
        prepend_bos_token: bool = True,
    ) -> EncoderDecoderBatch:
        r"""
        Prepares sources and target to produce an :class:`~inseq.data.EncoderDecoderBatch`.
        There are two stages of preparation:

            1. Raw text sources and target texts are encoded by the model.
            2. The encoded sources and targets are converted to tensors for the forward pass.

        This method is agnostic of the preparation stage of sources and targets. If they are both
        raw text, they will undergo both steps. If they are already encoded, they will only be embedded.
        If the feature attribution method works on layers, the embedding step is skipped and embeddings are
        set to None.
        The final result will be consistent in both cases.

        Args:
            sources (:obj:`FeatureAttributionInput`): The sources provided to the
                :meth:`~inseq.attr.feat.FeatureAttribution.prepare` method.
            targets (:obj:`FeatureAttributionInput): The targets provided to the
                :meth:`~inseq.attr.feat.FeatureAttribution.prepare` method.
            prepend_bos_token (:obj:`bool`, `optional`): Whether to prepend a BOS token to the
                targets, if they are to be encoded. Defaults to True.

        Returns:
            :obj:`EncoderDecoderBatch`: An :class:`~inseq.data.EncoderDecoderBatch` object containing sources
                and targets in encoded and embedded formats for all inputs.
        """
        if isinstance(sources, str) or isinstance(sources, list):
            sources: BatchEncoding = self.attribution_model.encode(sources, return_baseline=True)
        if isinstance(sources, BatchEncoding):
            # Layer attribution methods use token ids as inputs instead of embeddings since they
            # don't need to attribute scores to embedding vectors.
            if self.is_layer_attribution:
                embeds = BatchEmbedding()
            else:
                embeds = BatchEmbedding(
                    input_embeds=self.attribution_model.embed(sources.input_ids),
                    baseline_embeds=self.attribution_model.embed(sources.baseline_ids),
                )
            sources = Batch(sources, embeds)
        if isinstance(targets, str) or isinstance(targets, list):
            targets: BatchEncoding = self.attribution_model.encode(
                targets,
                as_targets=True,
                prepend_bos_token=prepend_bos_token,
                return_baseline=True,
            )
        if isinstance(targets, BatchEncoding):
            baseline_embeds = None
            if not self.is_layer_attribution:
                baseline_embeds = self.attribution_model.embed(targets.baseline_ids, as_targets=True)
            target_embeds = BatchEmbedding(
                input_embeds=self.attribution_model.embed(targets.input_ids, as_targets=True),
                baseline_embeds=baseline_embeds,
            )
            targets = Batch(targets, target_embeds)
        sources_targets = EncoderDecoderBatch(sources, targets)
        return sources_targets.to(self.attribution_model.device)

    @overload
    def attribute(
        self,
        texts: str,
        output_step_attributions: Literal[False],
    ) -> OneOrMoreFeatureAttributionSequenceOutputs:
        ...

    @overload
    def attribute(
        self,
        output_step_attributions: Literal[True],
    ) -> Union[OneOrMoreFeatureAttributionSequenceOutputs, List[FeatureAttributionOutput]]:
        ...

    def attribute(
        self,
        batch: EncoderDecoderBatch,
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        output_step_attributions: bool = False,
        attribute_target: bool = False,
        **kwargs,
    ) -> OneOrMoreFeatureAttributionSequenceOutputsWithStepOutputs:
        r"""
        Attributes each target token to each source token for every sequence in the batch.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch of sequences to attribute.
            attr_pos_start (:obj:`int`, `optional`): The initial position for performing
                sequence attribution. Defaults to 1 (0 is the default BOS token).
            attr_pos_end (:obj:`int`, `optional`): The final position for performing sequence
                attribution. Defaults to None (full string).
            show_progress (:obj:`bool`, `optional`): Whether to show a progress bar. Defaults to True.
            pretty_progress (:obj:`bool`, `optional`): Whether to use a pretty progress bar. Defaults to True.
            output_step_attributions (:obj:`bool`, `optional`): Whether to output a list of FeatureAttributionOutput
                objects for each step. Defaults to False.
            attribute_target (:obj:`bool`, `optional`): Whether to include target prefix for feature attribution.
                Defaults to False.
            kwargs: Additional keyword arguments to pass to the attribution step.

        Returns:
            :obj:`OneOrMoreFeatureAttributionSequenceOutputsWithStepOutputs`: One or more
                :class:`~inseq.data.FeatureAttributionSequenceOutput` depending on the number of inputs, with an
                optional added list of single :class:`~inseq.data.FeatureAttributionOutput` for each step.
        """
        max_generated_length = batch.targets.input_ids.shape[1]
        attr_pos_start, attr_pos_end = self.check_attribute_positions(
            max_generated_length,
            attr_pos_start,
            attr_pos_end,
        )
        logger.debug("=" * 30 + f"\nfull batch: {batch}\n" + "=" * 30)
        sources, targets, lengths = get_attribution_sentences(
            self.attribution_model, batch, attr_pos_start, attr_pos_end
        )
        pbar = get_progress_bar(
            all_sentences=(sources, targets, lengths),
            method_name=self.method_name,
            show=show_progress,
            pretty=pretty_progress,
        )
        whitespace_indexes = find_char_indexes(targets, " ")
        attribution_outputs = []
        for step in range(attr_pos_start, attr_pos_end):
            step_output = self.filtered_attribute_step(
                batch[:step],
                batch.targets.input_ids[:, step].unsqueeze(1),
                batch.targets.attention_mask[:, step].unsqueeze(1),
                attribute_target=attribute_target,
                **kwargs,
            )
            attribution_outputs.append(
                self.make_attribution_output(
                    step_output,
                    batch[:step],
                    batch.targets.input_ids[:, step].unsqueeze(1),
                    kwargs.get("prepend_bos_token", True),
                )
            )
            if pretty_progress:
                split_targets = get_split_targets(
                    self.attribution_model, batch.targets.input_tokens, attr_pos_start, attr_pos_end, step
                )
                update_progress_bar(
                    pbar,
                    split_targets,
                    whitespace_indexes,
                    show=show_progress,
                    pretty=pretty_progress,
                )
            else:
                update_progress_bar(pbar, show=show_progress, pretty=pretty_progress)
        close_progress_bar(pbar, show=show_progress, pretty=pretty_progress)
        batch.to("cpu")
        sequence_attribution = FeatureAttributionSequenceOutput.from_attributions(attribution_outputs)
        if output_step_attributions:
            return sequence_attribution, attribution_outputs
        return sequence_attribution

    @staticmethod
    def check_attribute_positions(
        max_length: int,
        attr_pos_start: Optional[int] = None,
        attr_pos_end: Optional[int] = None,
    ) -> Tuple[int, int]:
        r"""
        Checks whether the combination of start/end positions for attribution is valid.

        Args:
            max_length (:obj:`int`): The maximum length of sequences in the batch.
            attr_pos_start (:obj:`int`, `optional`): The initial position for performing
                sequence attribution. Defaults to 1 (0 is the default BOS token).
            attr_pos_end (:obj:`int`, `optional`): The final position for performing sequence
                attribution. Defaults to None (full string).

        Raises:
            ValueError: If the start position is greater or equal than the end position or < 0.

        Returns:
            `tuple[int, int]`: The start and end positions for attribution.
        """
        if attr_pos_start is None:
            attr_pos_start = 1
        if attr_pos_end is None or attr_pos_end > max_length:
            attr_pos_end = max_length
        if attr_pos_start > attr_pos_end or attr_pos_start < 1:
            raise ValueError("Invalid starting position for attribution")
        if attr_pos_start == attr_pos_end:
            raise ValueError("Start and end attribution positions cannot be the same.")
        return attr_pos_start, attr_pos_end

    def filtered_attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", 1, int],
        target_attention_mask: Optional[TensorType["batch_size", 1, int]] = None,
        attribute_target: bool = False,
        **kwargs: Dict[str, Any],
    ) -> FeatureAttributionStepOutput:
        r"""
        Performs a single attribution step for all the sequences in the batch that
        still have valid target_ids, as identified by the target_attention_mask.
        Finished sentences are temporarily filtered out to make the attribution step
        faster and then reinserted before returning.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch of sequences to attribute.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size, 1)` corresponding to tokens
                for which the attribution step must be performed.
            target_attention_mask (:obj:`torch.Tensor`, `optional`): Boolean attention mask of size `(batch_size, 1)`
                specifying which target_ids are valid for attribution and which are padding.
            attribute_target (:obj:`bool`, `optional`): Whether to include target prefix for feature attribution.
                Defaults to False.
            kwargs: Additional keyword arguments to pass to the attribution step.

        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: A dataclass containing a tensor of source-side
                attributions of size `(batch_size, source_length)`, possibly a tensor of target attributions of size
                `(batch_size, prefix length) if attribute_target=True and possibly a tensor of deltas of size
                `(batch_size)` if the attribution step supports deltas and they are requested.
        """
        orig_batch = batch.clone()
        orig_target_ids = target_ids
        # Filter out finished sentences
        if target_attention_mask is not None and target_ids.shape[0] > 1:
            batch = batch.select_active(target_attention_mask)
            target_ids = target_ids.masked_select(target_attention_mask.bool())
            target_ids = target_ids.view(-1, 1)
        logger.debug(
            f"\ntarget_ids: {pretty_tensor(target_ids)},\n"
            f"target_attention_mask: {pretty_tensor(target_attention_mask)}"
        )
        # Perform attribution step
        step_output = self.attribute_step(batch, target_ids.squeeze(), attribute_target, **kwargs)
        # Reinsert finished sentences
        if target_attention_mask is not None and orig_target_ids.shape[0] > 1:
            step_output.source_attributions = remap_from_filtered(
                source=orig_batch.sources.input_ids,
                mask=target_attention_mask,
                filtered=step_output.source_attributions,
            )
            if step_output.target_attributions is not None:
                step_output.target_attributions = remap_from_filtered(
                    source=orig_batch.targets.input_ids,
                    mask=target_attention_mask,
                    filtered=step_output.target_attributions,
                )
            if step_output.deltas is not None:
                step_output.deltas = remap_from_filtered(
                    source=target_attention_mask.squeeze(),
                    mask=target_attention_mask,
                    filtered=step_output.deltas,
                )
        return step_output.detach().to("cpu")

    def get_attribution_args(self, **kwargs):
        if hasattr(self, "method") and hasattr(self.method, "attribute"):
            return extract_signature_args(kwargs, self.method.attribute, self.ignore_extra_args)
        return {}

    def make_attribution_output(
        self,
        step_output: FeatureAttributionStepOutput,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", 1, int],
        has_bos_token: bool = True,
    ) -> FeatureAttributionOutput:
        r"""
        Enriches the attribution output with token information and builds the final
        :class:`~inseq.data.FeatureAttributionOutput` object.

        Args:
            step_output (:class:`~inseq.data.FeatureAttributionStepOutput`): The output produced
                by the attribution step.
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch on which attribution was performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size, 1)` corresponding to tokens
                for which the attribution step was performed.
            has_bos_token (:obj:`bool`): Whether the target sequence contains a BOS token.

        Returns:
            :class:`~inseq.data.FeatureAttributionOutput`: The enriched attribution output.
        """
        source_tokens = [drop_padding(seq, self.attribution_model.pad_token) for seq in batch.sources.input_tokens]
        prefix_tokens = [drop_padding(seq, self.attribution_model.pad_token) for seq in batch.targets.input_tokens]
        if has_bos_token:
            prefix_tokens = [tokens[1:] for tokens in prefix_tokens]
        target_tokens = self.attribution_model.convert_ids_to_tokens(target_ids, skip_special_tokens=False)
        source_ids = self.attribution_model.convert_tokens_to_ids(source_tokens)
        prefix_ids = self.attribution_model.convert_tokens_to_ids(prefix_tokens)
        source_attributions = step_output.source_attributions.tolist()
        source_attributions = rescale_attributions_to_tokens(source_attributions, source_tokens)
        target_attributions = None
        if step_output.target_attributions is not None:
            target_attributions = step_output.target_attributions.tolist()
            if has_bos_token:
                target_attributions = [attr[1:] for attr in target_attributions]
            target_attributions = rescale_attributions_to_tokens(target_attributions, prefix_tokens)
        delta = None
        if step_output.deltas is not None:
            delta = step_output.deltas.squeeze().tolist()
            if not isinstance(delta, list):
                delta = [delta]
        return FeatureAttributionOutput(
            source_attributions=source_attributions,
            target_attributions=target_attributions,
            delta=delta,
            source_ids=source_ids,
            prefix_ids=prefix_ids,
            target_ids=target_ids.tolist(),
            source_tokens=source_tokens,
            prefix_tokens=prefix_tokens,
            target_tokens=target_tokens,
        )

    def format_attribute_args(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TargetIdsTensor,
        attribute_target: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        if self.is_layer_attribution:
            inputs = (batch.sources.input_ids,)
            baselines = (batch.sources.baseline_ids,)
            if attribute_target:
                inputs = inputs + (batch.targets.input_ids,)
                baselines = baselines + (batch.targets.baseline_ids,)
        else:
            inputs = (batch.sources.input_embeds,)
            baselines = (batch.sources.baseline_embeds,)
            if attribute_target:
                inputs = inputs + (batch.targets.input_embeds,)
                baselines = baselines + (batch.targets.baseline_embeds,)
        attribute_args = {
            "inputs": inputs,
            "target": target_ids,
            "additional_forward_args": (
                batch.sources.attention_mask,
                batch.targets.attention_mask,
                # Defines how to treat source and target tensors
                # Maps on the use_embeddings argument of score_func
                not self.is_layer_attribution,
                attribute_target,
            ),
        }
        if not attribute_target:
            attribute_args["additional_forward_args"] = (batch.targets.input_embeds,) + attribute_args[
                "additional_forward_args"
            ]
        if self.use_baseline:
            attribute_args["baselines"] = baselines
        return {**attribute_args, **kwargs}

    @abstractmethod
    def attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", int],
        attribute_target: bool = False,
        **kwargs: Dict[str, Any],
    ) -> FeatureAttributionStepOutput:
        r"""
        Performs a single attribution step for the specified target_ids,
        given sources and targets in the batch.

        Abstract method, must be implemented by subclasses.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch of sequences on which attribution is performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size)` corresponding to tokens
                for which the attribution step must be performed.
            attribute_target (:obj:`bool`, optional): Whether to attribute the target prefix or not. Defaults to False.
            kwargs: Additional keyword arguments to pass to the attribution step.

        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: A dataclass containing a tensor of source-side
                attributions of size `(batch_size, source_length)`, possibly a tensor of target attributions of size
                `(batch_size, prefix length) if attribute_target=True and possibly a tensor of deltas of size
                `(batch_size)` if the attribution step supports deltas and they are requested.
        """
        pass

    @abstractmethod
    @set_hook
    def hook(self, **kwargs) -> NoReturn:
        r"""
        Hooks the attribution method to the model. Useful to implement pre-attribution logic
        (e.g. freezing layers, replacing embeddings, raise warnings, etc.).

        Abstract method, must be implemented by subclasses.
        """
        pass

    @abstractmethod
    @unset_hook
    def unhook(self, **kwargs) -> NoReturn:
        r"""
        Unhooks the attribution method from the model. If the model was modified in any way, this
        should restore its initial state.

        Abstract method, must be implemented by subclasses.
        """
        pass
