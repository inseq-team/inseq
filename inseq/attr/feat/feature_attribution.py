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

from typing import Any, Dict, NoReturn, Optional, Tuple

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
)
from ...data.viz import close_progress_bar, get_progress_bar, update_progress_bar
from ...utils import (
    Registry,
    UnknownAttributionMethodError,
    extract_signature_args,
    find_char_indexes,
    pretty_tensor,
    remap_from_filtered,
)
from ...utils.typing import ModelIdentifier, TargetIdsTensor
from ..attribution_decorators import set_hook, unset_hook

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
            skip_eos (:obj:`bool`, default `False`): Whether the EOS token is considered as a
                valid token during attribution.
            target_layer (:obj:`torch.nn.Module`, default `None`): The layer on which attribution should be
                performed if is_layer_attribution is True.
            use_baseline (:obj:`bool`, default `False`): Whether a baseline should be used for the attribution method.
        """
        super().__init__()
        self.attribution_model = attribution_model
        self.is_layer_attribution: bool = False
        self.skip_eos: bool = False
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
        **kwargs,
    ) -> OneOrMoreFeatureAttributionSequenceOutputs:
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
            show_progress (:obj:`bool`, `optional`): Whether to show a progress bar. Defaults to False.
            pretty_progress (:obj:`bool`, `optional`): Whether to use a pretty progress bar. Defaults to False.

        Returns:
            :obj:`OneOrMoreFeatureAttributionSequenceOutputs`: One or more sequence attribution outputs,
                depending on the number of inputs.
        """
        prepend_bos_token = kwargs.pop("prepend_bos_token", True)
        batch = self.prepare(sources, targets, prepend_bos_token)
        return self.attribute(
            batch,
            attr_pos_start=attr_pos_start,
            attr_pos_end=attr_pos_end,
            show_progress=show_progress,
            pretty_progress=pretty_progress,
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
            :obj:`OneOrMoreFeatureAttributionSequenceOutputs`: One or more
                :class:`~inseq.data.FeatureAttributionSequenceOutput`,
                depending on the number of inputs.
        """
        if isinstance(sources, str) or isinstance(sources, list):
            sources: BatchEncoding = self.attribution_model.encode_texts(
                sources, return_baseline=True
            )
        if isinstance(sources, BatchEncoding):
            if self.is_layer_attribution:
                embeds = BatchEmbedding(None, None)
            else:
                embeds = BatchEmbedding(
                    input_embeds=self.attribution_model.encoder_embed(
                        sources.input_ids
                    ),
                    baseline_embeds=self.attribution_model.encoder_embed(
                        sources.baseline_ids
                    ),
                )
            sources = Batch.from_encoding_embeds(sources, embeds)
        if isinstance(targets, str) or isinstance(targets, list):
            targets: BatchEncoding = self.attribution_model.encode_texts(
                targets,
                as_targets=True,
                prepend_bos_token=prepend_bos_token,
                return_baseline=True,
            )
        if isinstance(targets, BatchEncoding):
            baseline_embeds = None
            if not self.is_layer_attribution:
                baseline_embeds = self.attribution_model.decoder_embed(
                    targets.baseline_ids
                )
            target_embeds = BatchEmbedding(
                input_embeds=self.attribution_model.decoder_embed(targets.input_ids),
                baseline_embeds=baseline_embeds,
            )
            targets = Batch.from_encoding_embeds(targets, target_embeds)
        return EncoderDecoderBatch(sources, targets)

    def attribute(
        self,
        batch: EncoderDecoderBatch,
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        **kwargs,
    ) -> OneOrMoreFeatureAttributionSequenceOutputs:
        r"""
        Attributes each target token to each source token for every sequence in the batch.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch of sequences to attribute.
            attr_pos_start (:obj:`int`, `optional`): The initial position for performing
                sequence attribution. Defaults to 1 (0 is the default BOS token).
            attr_pos_end (:obj:`int`, `optional`): The final position for performing sequence
                attribution. Defaults to None (full string).
            kwargs: Additional keyword arguments to pass to the attribution step.

        Returns:
            :obj:`OneOrMoreFeatureAttributionSequenceOutputs`: One or more
                :class:`~inseq.data.FeatureAttributionSequenceOutput`,
                depending on the number of inputs.
        """
        max_generated_length = batch.targets.input_ids.shape[1]
        attr_pos_start, attr_pos_end = self.check_attribute_positions(
            max_generated_length,
            attr_pos_start,
            attr_pos_end,
        )

        def tok2string(token_lists, start=None, end=None, as_targets=True):
            start = [0 if start is None else start for tokens in token_lists]
            end = [len(tokens) if end is None else end for tokens in token_lists]
            # fmt: off
            return self.attribution_model.convert_tokens_to_string(
                [tokens[start[i]:end[i]] for i, tokens in enumerate(token_lists)],
                as_targets=as_targets,
            )
            # fmt: on

        logger.debug("=" * 30 + f"\nfull batch: {batch}\n" + "=" * 30)
        source_sentences = tok2string(batch.sources.input_tokens, as_targets=False)
        target_sentences = tok2string(batch.targets.input_tokens)
        if isinstance(source_sentences, str):
            source_sentences = [source_sentences]
            target_sentences = [target_sentences]
        whitespace_indexes = find_char_indexes(target_sentences, " ")
        tokenized_target_sentences = [
            self.attribution_model.convert_string_to_tokens(sent, as_targets=True)
            for sent in target_sentences
        ]
        lengths = [
            min(attr_pos_end, len(tts) + 1) - attr_pos_start
            for tts in tokenized_target_sentences
        ]
        targets = zip(source_sentences, target_sentences, lengths)
        pbar = get_progress_bar(
            target_sentences=list(targets),
            method_name=self.method_name,
            show=show_progress,
            pretty=pretty_progress,
        )
        attribution_outputs = []
        for step in range(attr_pos_start, attr_pos_end):
            attribution_outputs.append(
                self.get_attribution_output(
                    batch[:step],
                    batch.targets.input_ids[:, step].unsqueeze(1),
                    batch.targets.attention_mask[:, step].unsqueeze(1),
                    **kwargs,
                )
            )
            if pretty_progress:
                skipped_prefixes = tok2string(
                    batch.targets.input_tokens, end=attr_pos_start
                )
                attributed_sentences = tok2string(
                    batch.targets.input_tokens, attr_pos_start, step + 1
                )
                unattributed_suffixes = tok2string(
                    batch.targets.input_tokens, step + 1, attr_pos_end
                )
                skipped_suffixes = tok2string(
                    batch.targets.input_tokens, start=attr_pos_end
                )
                update_progress_bar(
                    pbar,
                    skipped_prefixes,
                    attributed_sentences,
                    unattributed_suffixes,
                    skipped_suffixes,
                    whitespace_indexes,
                    show=show_progress,
                    pretty=pretty_progress,
                )
            else:
                update_progress_bar(pbar, show=show_progress, pretty=pretty_progress)
        close_progress_bar(pbar, show=show_progress, pretty=pretty_progress)
        return FeatureAttributionSequenceOutput.from_attributions(attribution_outputs)

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

    def get_attribution_output(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", 1, int],
        target_attention_mask: Optional[TensorType["batch_size", 1, int]] = None,
        **kwargs,
    ) -> FeatureAttributionOutput:
        r"""
        Performs a single attribution step for all sequences in the batch and fixes the
        format of generated :class:`~inseq.data.FeatureAttributionOutput`.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch of sequences to attribute.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size, 1)` corresponding to tokens
                for which the attribution step must be performed.
            target_attention_mask (:obj:`torch.Tensor`, `optional`): Boolean attention mask of size `(batch_size, 1)`
                specifying which target_ids are valid for attribution and which are padding.
            kwargs: Additional keyword arguments to pass to the attribution step.
        """
        step_output = self.filtered_attribute_step(
            batch, target_ids, target_attention_mask, **kwargs
        )
        attribution_output = FeatureAttributionOutput()
        attribution_output = self.add_token_information(
            attribution_output, batch, target_ids
        )
        attribution_output.set_attributions(*step_output)
        return attribution_output

    def filtered_attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", 1, int],
        target_attention_mask: Optional[TensorType["batch_size", 1, int]] = None,
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
            kwargs: Additional keyword arguments to pass to the attribution step.

        Returns:
            :obj:`FeatureAttributionStepOutput`: A tuple containing a tensor of attributions
                of size `(batch_size, source_length)` and possibly a tensor of attribution deltas
                of size `(batch_size)`, if the attribution step supports deltas and they are requested.
        """
        logger.debug(
            f"\ntarget_ids: {pretty_tensor(target_ids)},\n"
            f"target_attention_mask: {pretty_tensor(target_attention_mask)}"
        )
        orig_batch = batch.clone()
        orig_target_ids = target_ids
        # Filter out finished sentences
        if target_attention_mask is not None and target_ids.shape[0] > 1:
            batch = batch.select_active(target_attention_mask)
            target_ids = target_ids.masked_select(target_attention_mask.bool())
            target_ids = target_ids.view(-1, 1)
        # Perform attribution step
        step_output = self.attribute_step(batch, target_ids.squeeze(), **kwargs)
        attributions, deltas = (
            step_output if isinstance(step_output, tuple) else (step_output, None)
        )
        # Reinsert finished sentences
        if target_attention_mask is not None and orig_target_ids.shape[0] > 1:
            attributions = remap_from_filtered(
                source=orig_batch.sources.input_ids,
                mask=target_attention_mask,
                filtered=attributions,
            )
            if deltas is not None:
                deltas = remap_from_filtered(
                    source=target_attention_mask.squeeze(),
                    mask=target_attention_mask,
                    filtered=deltas,
                )
        if deltas is not None:
            return (attributions, deltas)
        return (attributions,)

    def get_attribution_args(self, **kwargs):
        if hasattr(self, "method") and hasattr(self.method, "attribute"):
            return extract_signature_args(
                kwargs, self.method.attribute, self.ignore_extra_args
            )
        return {}

    def add_token_information(
        self,
        attribution_output: FeatureAttributionOutput,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", 1, int],
    ) -> FeatureAttributionOutput:
        r"""
        Enriches the attribution output with token information and builds the final
        :class:`~inseq.data.FeatureAttributionOutput` object.

        Args:
            attribution_output (:class:`~inseq.data.FeatureAttributionOutput`): The attribution output to enrich.
            batch (:class:`~inseq.data.EncoderDecoderBatch`): The batch on which attribution was performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size, 1)` corresponding to tokens
                for which the attribution step was performed.

        Returns:
            :class:`~inseq.data.FeatureAttributionOutput`: The enriched attribution output.
        """
        source_tokens = [
            [tok for tok in seq if tok != self.attribution_model.pad_token]
            for seq in batch.sources.input_tokens
        ]
        prefix_tokens = [
            [tok for tok in seq if tok != self.attribution_model.pad_token]
            for seq in batch.targets.input_tokens
        ]
        attribution_output.source_ids = self.attribution_model.convert_tokens_to_ids(
            source_tokens
        )
        attribution_output.source_tokens = source_tokens
        attribution_output.prefix_ids = self.attribution_model.convert_tokens_to_ids(
            prefix_tokens
        )
        attribution_output.prefix_tokens = prefix_tokens
        attribution_output.target_ids = target_ids.tolist()
        attribution_output.target_tokens = self.attribution_model.convert_ids_to_tokens(
            target_ids, skip_special_tokens=False
        )
        return attribution_output

    def format_attribute_args(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TargetIdsTensor,
        **kwargs,
    ) -> Dict[str, Any]:
        logger.debug(f"batch: {batch},\ntarget_ids: {pretty_tensor(target_ids)}")
        # For now only encoder attribution is supported
        if self.is_layer_attribution:
            inputs = batch.sources.input_ids
            baselines = batch.sources.baseline_ids
        else:
            inputs = batch.sources.input_embeds
            baselines = batch.sources.baseline_embeds
        attribute_args = {
            "inputs": inputs,
            "target": target_ids,
            "additional_forward_args": (
                batch.sources.attention_mask,
                batch.targets.input_embeds,
                batch.targets.attention_mask,
                self.is_layer_attribution,  # Defines how to treat source and target tensors
            ),
        }
        if self.use_baseline:
            attribute_args["baselines"] = baselines
        return {**attribute_args, **kwargs}

    @abstractmethod
    def attribute_step(
        self,
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", int],
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
            kwargs: Additional keyword arguments to pass to the attribution step.

        Returns:
            :obj:`FeatureAttributionStepOutput`: A tuple containing a tensor of attributions
                of size `(batch_size, source_length)` and possibly a tensor of attribution deltas
                of size `(batch_size)`, if the attribution step supports deltas and they are requested.
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
