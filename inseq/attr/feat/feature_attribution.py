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
"""Feature attribution methods registry.

Todo:
    * 🟡: Allow custom arguments for model loading in the :class:`FeatureAttribution` :meth:`load` method.
"""
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from jaxtyping import Int

from ...data import (
    DecoderOnlyBatch,
    EncoderDecoderBatch,
    FeatureAttributionInput,
    FeatureAttributionOutput,
    FeatureAttributionSequenceOutput,
    FeatureAttributionStepOutput,
    get_batch_from_inputs,
)
from ...data.viz import close_progress_bar, get_progress_bar, update_progress_bar
from ...utils import (
    Registry,
    UnknownAttributionMethodError,
    available_classes,
    extract_signature_args,
    find_char_indexes,
    get_front_padding,
    pretty_tensor,
)
from ...utils.typing import ModelIdentifier, OneOrMoreTokenSequences, SingleScorePerStepTensor, TextSequences
from ..attribution_decorators import batched, set_hook, unset_hook
from ..step_functions import get_step_function, get_step_scores, get_step_scores_args
from .attribution_utils import (
    check_attribute_positions,
    get_source_target_attributions,
    tok2string,
)

if TYPE_CHECKING:
    from ...models import AttributionModel


logger = logging.getLogger(__name__)


class FeatureAttribution(Registry):
    r"""Abstract registry for feature attribution methods.

    Attributes:
        attr (:obj:`str`): Attribute of child classes that will act as lookup name
            for the registry.
        ignore_extra_args (:obj:`list` of :obj:`str`): Arguments used by default in the
            attribute step and thus ignored as extra arguments during attribution.
            The selection of defaults follows the `Captum <https://captum.ai/api/integrated_gradients.html>`__
            naming convention.
    """

    registry_attr = "method_name"
    ignore_extra_args = ["inputs", "baselines", "target", "additional_forward_args"]

    def __init__(self, attribution_model: "AttributionModel", hook_to_model: bool = True, **kwargs):
        r"""Common instantiation steps for FeatureAttribution methods. Hooks the attribution method
        to the model calling the :meth:`~inseq.attr.feat.FeatureAttribution.hook` method of the child class.

        Args:
            attribution_model (:class:`~inseq.models.AttributionModel`): The attribution model
                that is used to obtain predictions and on which attribution is performed.
            hook_to_model (:obj:`bool`, default `True`): Whether the attribution method should be
                hooked to the attribution model during initialization.
            **kwargs: Additional keyword arguments to pass to the hook method.

        Attributes:
            attribute_batch_ids (:obj:`bool`, default `False`): If True, the attribution method will receive batch ids
                instead of batch embeddings for attribution. Used by layer gradient-based attribution methods mapping
                saliency scores to the output of a layer instead of model inputs.
            forward_batch_embeds (:obj:`bool`, default `True`): If True, the model will use embeddings in the
                forward pass instead of token ids. Using this in combination with `attribute_batch_ids` will allow for
                custom conversion of ids into embeddings inside the attribution method.
            target_layer (:obj:`torch.nn.Module`, default `None`): The layer on which attribution should be
                performed for layer attribution methods.
            use_baselines (:obj:`bool`, default `False`): Whether a baseline should be used for the attribution method.
            use_attention_weights (:obj:`bool`, default `False`): Whether attention weights are used in the attribution
                method.
            use_hidden_states (:obj:`bool`, default `False`): Whether hidden states are used in the attribution method.
            use_predicted_target (:obj:`bool`, default `True`): Whether the attribution method uses the predicted
                target for attribution. In case it doesn't, a warning message will be shown if the target is not
                the default one.
            use_model_config (:obj:`bool`, default `False`): Whether the attribution method uses the model config. If
                True, the method will try to load the config matching the model when hooking to the model. Missing
                configurations can be registered using :meth:`~inseq.models.register_model_config`.
        """
        super().__init__()
        self.attribution_model = attribution_model
        self.attribute_batch_ids: bool = False
        self.forward_batch_embeds: bool = True
        self.target_layer = None
        self.use_baselines: bool = False
        self.use_attention_weights: bool = False
        self.use_hidden_states: bool = False
        self.use_predicted_target: bool = True
        self.use_model_config: bool = False
        self.is_final_step_method: bool = False
        if hook_to_model:
            self.hook(**kwargs)

    @classmethod
    def load(
        cls,
        method_name: str,
        attribution_model: Optional["AttributionModel"] = None,
        model_name_or_path: Optional[ModelIdentifier] = None,
        **kwargs,
    ) -> "FeatureAttribution":
        r"""Load the selected method and hook it to an existing or available
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
        from ...models import load_model

        methods = cls.available_classes()
        if method_name not in methods:
            raise UnknownAttributionMethodError(method_name)
        if model_name_or_path is not None:
            model = load_model(model_name_or_path)
        elif attribution_model is not None:
            model = attribution_model
        else:
            raise RuntimeError(
                "Only one among an initialized model and a model identifier "
                "must be defined when loading the attribution method."
            )
        return methods[method_name](model, **kwargs)

    @batched
    def prepare_and_attribute(
        self,
        sources: FeatureAttributionInput,
        targets: FeatureAttributionInput,
        attr_pos_start: Optional[int] = None,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        output_step_attributions: bool = False,
        attribute_target: bool = False,
        step_scores: list[str] = [],
        include_eos_baseline: bool = False,
        attributed_fn: Union[str, Callable[..., SingleScorePerStepTensor], None] = None,
        attribution_args: dict[str, Any] = {},
        attributed_fn_args: dict[str, Any] = {},
        step_scores_args: dict[str, Any] = {},
    ) -> FeatureAttributionOutput:
        r"""Prepares inputs and performs attribution.

        Wraps the attribution method :meth:`~inseq.attr.feat.FeatureAttribution.attribute` method
        and the :meth:`~inseq.models.InputFormatter.prepare_inputs_for_attribution` method.

        Args:
            sources (:obj:`FeatureAttributionInput`): The sources provided to the
                :meth:`~inseq.attr.feat.FeatureAttribution.prepare` method.
            targets (:obj:`FeatureAttributionInput`): The targets provided to the
                :meth:`~inseq.attr.feat.FeatureAttribution.prepare` method.
            attr_pos_start (:obj:`int`, `optional`): The initial position for performing
                sequence attribution. Defaults to 0.
            attr_pos_end (:obj:`int`, `optional`): The final position for performing sequence
                attribution. Defaults to None (full string).
            show_progress (:obj:`bool`, `optional`): Whether to show a progress bar. Defaults to True.
            pretty_progress (:obj:`bool`, `optional`): Whether to use a pretty progress bar. Defaults to True.
            output_step_attributions (:obj:`bool`, `optional`): Whether to output a list of
                FeatureAttributionStepOutput objects for each step. Defaults to False.
            attribute_target (:obj:`bool`, `optional`): Whether to include target prefix for feature attribution.
                Defaults to False.
            step_scores (:obj:`list` of `str`): List of identifiers for step scores that need to be computed during
                attribution. The available step scores are defined in :obj:`inseq.attr.feat.STEP_SCORES_MAP` and new
                step scores can be added by using the :meth:`~inseq.register_step_function` function.
            include_eos_baseline (:obj:`bool`, `optional`): Whether to include the EOS token in the baseline for
                attribution. By default the EOS token is not used for attribution. Defaults to False.
            attributed_fn (:obj:`str` or :obj:`Callable[..., SingleScorePerStepTensor]`, `optional`): The identifier or
                function of model outputs representing what should be attributed (e.g. output probits of model best
                prediction after softmax). If it is a string, it must be a valid function.
                Otherwise, it must be a function that taking multiple keyword arguments and returns a :obj:`tensor`
                of size (batch_size,). If not provided, the default attributed function for the model will be used
                (change attribution_model.default_attributed_fn_id).
            attribution_args (:obj:`dict`, `optional`): Additional arguments to pass to the attribution method.
            attributed_fn_args (:obj:`dict`, `optional`): Additional arguments to pass to the attributed function.
            step_scores_args (:obj:`dict`, `optional`): Additional arguments to pass to the step scores functions.

        Returns:
            :class:`~inseq.data.FeatureAttributionOutput`: An object containing a list of sequence attributions, with
                an optional added list of single :class:`~inseq.data.FeatureAttributionStepOutput` for each step and
                extra information regarding the attribution parameters.
        """
        inputs = (sources, targets)
        if not self.attribution_model.is_encoder_decoder:
            inputs = targets
            encoded_sources = self.attribution_model.encode(sources, return_baseline=True)
            # We do this here to support separate attr_pos_start for different sentences when batching
            if attr_pos_start is None or attr_pos_start < encoded_sources.input_ids.shape[1]:
                attr_pos_start = encoded_sources.input_ids.shape[1]
        batch = self.attribution_model.formatter.prepare_inputs_for_attribution(
            self.attribution_model, inputs, include_eos_baseline
        )
        # If prepare_and_attribute was called from AttributionModel.attribute,
        # attributed_fn is already a Callable. Keep here to allow for usage independently
        # of AttributionModel.attribute.
        attributed_fn = self.attribution_model.get_attributed_fn(attributed_fn)
        attribution_output = self.attribute(
            batch,
            attributed_fn=attributed_fn,
            attr_pos_start=attr_pos_start,
            attr_pos_end=attr_pos_end,
            show_progress=show_progress,
            pretty_progress=pretty_progress,
            output_step_attributions=output_step_attributions,
            attribute_target=attribute_target,
            step_scores=step_scores,
            attribution_args=attribution_args,
            attributed_fn_args=attributed_fn_args,
            step_scores_args=step_scores_args,
        )
        # Same here, repeated from AttributionModel.attribute
        # to allow independent usage
        attribution_output.info["include_eos_baseline"] = include_eos_baseline
        attribution_output.info["attributed_fn"] = attributed_fn.__name__
        attribution_output.info["attribution_args"] = attribution_args
        attribution_output.info["attributed_fn_args"] = attributed_fn_args
        attribution_output.info["step_scores_args"] = step_scores_args
        return attribution_output

    def _run_compatibility_checks(self, attributed_fn) -> None:
        default_attributed_fn = get_step_function(self.attribution_model.default_attributed_fn_id)
        if not self.use_predicted_target and attributed_fn != default_attributed_fn:
            logger.warning(
                "Internals attribution methods are output agnostic, since they do not rely on specific output"
                " targets to compute importance scores. Using a custom attributed function in this context does not"
                " influence in any way the method's results."
            )
        if self.use_model_config and self.attribution_model.is_distributed:
            raise RuntimeError(
                "Distributed models are incompatible with attribution methods requiring access to models' internals "
                "for storing or intervention purposes. Please use a non-distributed model with the current attribution"
                " method."
            )

    @staticmethod
    def _build_multistep_output_from_single_step(
        single_step_output: FeatureAttributionStepOutput,
        attr_pos_start: int,
        attr_pos_end: int,
    ) -> list[FeatureAttributionStepOutput]:
        if single_step_output.step_scores:
            raise ValueError("step_scores are not supported for final step attribution methods.")
        num_seq = len(single_step_output.prefix)
        steps = []
        for pos_idx in range(attr_pos_start, attr_pos_end):
            step_output = single_step_output.clone_empty()
            step_output.source = single_step_output.source
            step_output.prefix = [single_step_output.prefix[seq_idx][:pos_idx] for seq_idx in range(num_seq)]
            step_output.target = (
                single_step_output.target
                if pos_idx == attr_pos_end - 1
                else [[single_step_output.prefix[seq_idx][pos_idx]] for seq_idx in range(num_seq)]
            )
            if single_step_output.source_attributions is not None:
                step_output.source_attributions = single_step_output.source_attributions[:, :, pos_idx - 1]
            if single_step_output.target_attributions is not None:
                step_output.target_attributions = single_step_output.target_attributions[:, :pos_idx, pos_idx - 1]
            single_step_output.step_scores = {}
            if single_step_output.sequence_scores is not None:
                step_output.sequence_scores = single_step_output.sequence_scores
            steps.append(step_output)
        return steps

    def format_contrastive_targets(
        self,
        target_sequences: TextSequences,
        target_tokens: OneOrMoreTokenSequences,
        attributed_fn_args: dict[str, Any],
        step_scores_args: dict[str, Any],
        attr_pos_start: int,
        attr_pos_end: int,
    ) -> tuple[Optional[DecoderOnlyBatch], Optional[list[list[tuple[int, int]]]], dict[str, Any], dict[str, Any]]:
        contrast_batch, contrast_targets_alignments = None, None
        contrast_targets = attributed_fn_args.get("contrast_targets", None)
        if contrast_targets is None:
            contrast_targets = step_scores_args.get("contrast_targets", None)
        contrast_targets_alignments = attributed_fn_args.get("contrast_targets_alignments", None)
        if contrast_targets_alignments is None:
            contrast_targets_alignments = step_scores_args.get("contrast_targets_alignments", None)
        if contrast_targets_alignments is not None and contrast_targets is None:
            raise ValueError("contrast_targets_alignments requires contrast_targets to be specified.")
        contrast_targets = [contrast_targets] if isinstance(contrast_targets, str) else contrast_targets
        if contrast_targets is not None:
            as_targets = self.attribution_model.is_encoder_decoder
            contrast_batch = get_batch_from_inputs(
                attribution_model=self.attribution_model,
                inputs=contrast_targets,
                as_targets=as_targets,
            )
            contrast_batch = DecoderOnlyBatch.from_batch(contrast_batch)
            clean_tgt_tokens = self.attribution_model.clean_tokens(target_tokens, as_targets=as_targets)
            clean_c_tokens = self.attribution_model.clean_tokens(contrast_batch.target_tokens, as_targets=as_targets)
            contrast_targets_alignments = self.attribution_model.formatter.format_contrast_targets_alignments(
                contrast_targets_alignments=contrast_targets_alignments,
                target_sequences=target_sequences,
                target_tokens=clean_tgt_tokens,
                contrast_sequences=contrast_targets,
                contrast_tokens=clean_c_tokens,
                special_tokens=self.attribution_model.special_tokens,
                start_pos=attr_pos_start,
                end_pos=attr_pos_end,
            )
            if "contrast_targets" in step_scores_args:
                step_scores_args["contrast_targets_alignments"] = contrast_targets_alignments
            if "contrast_targets" in attributed_fn_args:
                attributed_fn_args["contrast_targets_alignments"] = contrast_targets_alignments
        return contrast_batch, contrast_targets_alignments, attributed_fn_args, step_scores_args

    def attribute(
        self,
        batch: Union[DecoderOnlyBatch, EncoderDecoderBatch],
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attr_pos_start: Optional[int] = None,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        output_step_attributions: bool = False,
        attribute_target: bool = False,
        step_scores: list[str] = [],
        attribution_args: dict[str, Any] = {},
        attributed_fn_args: dict[str, Any] = {},
        step_scores_args: dict[str, Any] = {},
    ) -> FeatureAttributionOutput:
        r"""Performs the feature attribution procedure using the specified attribution method.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch` or :class:`~inseq.data.DecoderOnlyBatch`): The batch of
                sequences to attribute.
            attributed_fn (:obj:`Callable[..., SingleScorePerStepTensor]`): The function of model
                outputs representing what should be attributed (e.g. output probits of model best
                prediction after softmax). It must be a function that taking multiple keyword
                arguments and returns a :obj:`tensor` of size (batch_size,). If not provided,
                the default attributed function for the model will be used.
            attr_pos_start (:obj:`int`, `optional`): The initial position for performing
                sequence attribution. Defaults to 1 (0 is the default BOS token).
            attr_pos_end (:obj:`int`, `optional`): The final position for performing sequence
                attribution. Defaults to None (full string).
            show_progress (:obj:`bool`, `optional`): Whether to show a progress bar. Defaults to True.
            pretty_progress (:obj:`bool`, `optional`): Whether to use a pretty progress bar. Defaults to True.
            output_step_attributions (:obj:`bool`, `optional`): Whether to output a list of
                FeatureAttributionStepOutput objects for each step. Defaults to False.
            attribute_target (:obj:`bool`, `optional`): Whether to include target prefix for feature attribution.
                Defaults to False.
            step_scores (:obj:`list` of `str`): List of identifiers for step scores that need to be computed during
                attribution. The available step scores are defined in :obj:`inseq.attr.feat.STEP_SCORES_MAP` and new
                step scores can be added by using the :meth:`~inseq.register_step_function` function.
            attribution_args (:obj:`dict`, `optional`): Additional arguments to pass to the attribution method.
            attributed_fn_args (:obj:`dict`, `optional`): Additional arguments to pass to the attributed function.
            step_scores_args (:obj:`dict`, `optional`): Additional arguments to pass to the step scores function.

        Returns:
            :class:`~inseq.data.FeatureAttributionOutput`: An object containing a list of sequence attributions, with
                an optional added list of single :class:`~inseq.data.FeatureAttributionStepOutput` for each step and
                extra information regarding the attribution parameters.
        """
        if self.attribute_batch_ids and not self.forward_batch_embeds and attribute_target:
            raise ValueError(
                "Layer attribution methods do not support attribute_target=True. Use regular attributions instead."
            )
        self._run_compatibility_checks(attributed_fn)
        attr_pos_start, attr_pos_end = check_attribute_positions(
            batch.max_generation_length,
            attr_pos_start,
            attr_pos_end,
        )
        logger.debug("=" * 30 + f"\nfull batch: {batch}\n" + "=" * 30)
        # Sources are empty for decoder-only models
        sequences = self.attribution_model.formatter.get_text_sequences(self.attribution_model, batch)
        (
            contrast_batch,
            contrast_targets_alignments,
            attributed_fn_args,
            step_scores_args,
        ) = self.format_contrastive_targets(
            sequences.targets,
            batch.target_tokens,
            attributed_fn_args,
            step_scores_args,
            attr_pos_start,
            attr_pos_end,
        )
        target_tokens_with_ids = self.attribution_model.get_token_with_ids(
            batch,
            contrast_target_tokens=contrast_batch.target_tokens if contrast_batch is not None else None,
            contrast_targets_alignments=contrast_targets_alignments,
        )

        # Manages front padding for decoder-only models, using 0 as lower bound
        # when attr_pos_start exceeds target length.
        targets_lengths = [
            max(
                0,
                min(attr_pos_end, len(target_tokens_with_ids[idx]))
                - (attr_pos_start + 1)
                + get_front_padding(batch.target_mask)[idx],
            )
            for idx in range(len(target_tokens_with_ids))
        ]
        if self.attribution_model.is_encoder_decoder:
            iter_pos_end = min(attr_pos_end + 1, batch.max_generation_length)
        else:
            iter_pos_end = attr_pos_end
        pbar = get_progress_bar(
            sequences=sequences,
            target_lengths=targets_lengths,
            method_name=self.method_name,
            show=show_progress,
            pretty=False if self.is_final_step_method else pretty_progress,
            attr_pos_start=attr_pos_start,
            attr_pos_end=1 if self.is_final_step_method else attr_pos_end,
        )
        whitespace_indexes = find_char_indexes(sequences.targets, " ")
        attribution_outputs = []

        start = datetime.now()

        # Attribution loop for generation
        for step in range(attr_pos_start, iter_pos_end):
            if self.is_final_step_method and step != iter_pos_end - 1:
                continue
            tgt_ids, tgt_mask = batch.get_step_target(step, with_attention=True)
            step_output = self.filtered_attribute_step(
                batch[:step],
                target_ids=tgt_ids.unsqueeze(1),
                attributed_fn=attributed_fn,
                target_attention_mask=tgt_mask.unsqueeze(1),
                attribute_target=attribute_target,
                step_scores=step_scores,
                attribution_args=attribution_args,
                attributed_fn_args=attributed_fn_args,
                step_scores_args=step_scores_args,
            )
            # Add batch information to output
            step_output = self.attribution_model.formatter.enrich_step_output(
                self.attribution_model,
                step_output,
                batch[:step],
                self.attribution_model.convert_ids_to_tokens(tgt_ids.unsqueeze(1), skip_special_tokens=False),
                tgt_ids.detach().to("cpu"),
                contrast_batch=contrast_batch,
                contrast_targets_alignments=contrast_targets_alignments,
            )
            attribution_outputs.append(step_output)
            if pretty_progress and not self.is_final_step_method:
                tgt_tokens = batch.target_tokens
                skipped_prefixes = tok2string(self.attribution_model, tgt_tokens, end=attr_pos_start)
                attributed_sentences = tok2string(self.attribution_model, tgt_tokens, attr_pos_start, step + 1)
                unattributed_suffixes = tok2string(self.attribution_model, tgt_tokens, step + 1, attr_pos_end)
                skipped_suffixes = tok2string(self.attribution_model, tgt_tokens, start=attr_pos_end)
                update_progress_bar(
                    pbar,
                    skipped_prefixes,
                    attributed_sentences,
                    unattributed_suffixes,
                    skipped_suffixes,
                    whitespace_indexes,
                    show=show_progress,
                    pretty=True,
                )
            else:
                update_progress_bar(pbar, show=show_progress, pretty=False)
        end = datetime.now()
        close_progress_bar(pbar, show=show_progress, pretty=False if self.is_final_step_method else pretty_progress)
        batch.detach().to("cpu")
        if self.is_final_step_method:
            attribution_outputs = self._build_multistep_output_from_single_step(
                attribution_outputs[0],
                attr_pos_start=attr_pos_start,
                attr_pos_end=iter_pos_end,
            )
        out = FeatureAttributionOutput(
            sequence_attributions=FeatureAttributionSequenceOutput.from_step_attributions(
                attributions=attribution_outputs,
                tokenized_target_sentences=target_tokens_with_ids,
                pad_token=self.attribution_model.pad_token,
                attr_pos_end=attr_pos_end,
            ),
            step_attributions=attribution_outputs if output_step_attributions else None,
            info={
                "attribution_method": self.method_name,
                "attr_pos_start": attr_pos_start,
                "attr_pos_end": attr_pos_end,
                "output_step_attributions": output_step_attributions,
                "attribute_target": attribute_target,
                "step_scores": step_scores,
                # Convert to datetime.timedelta as timedelta(seconds=exec_time)
                "exec_time": (end - start).total_seconds(),
            },
        )
        out.info.update(self.attribution_model.info)
        return out

    def filtered_attribute_step(
        self,
        batch: Union[DecoderOnlyBatch, EncoderDecoderBatch],
        target_ids: Int[torch.Tensor, "batch_size 1"],
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        target_attention_mask: Optional[Int[torch.Tensor, "batch_size 1"]] = None,
        attribute_target: bool = False,
        step_scores: list[str] = [],
        attribution_args: dict[str, Any] = {},
        attributed_fn_args: dict[str, Any] = {},
        step_scores_args: dict[str, Any] = {},
    ) -> FeatureAttributionStepOutput:
        r"""Performs a single attribution step for all the sequences in the batch that
        still have valid target_ids, as identified by the target_attention_mask.
        Finished sentences are temporarily filtered out to make the attribution step
        faster and then reinserted before returning.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch` or :class:`~inseq.data.DecoderOnlyBatch`): The batch of
                sequences to attribute.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size, 1)` corresponding to tokens
                for which the attribution step must be performed.
            attributed_fn (:obj:`Callable[..., SingleScorePerStepTensor]`): The function of model outputs
                representing what should be attributed (e.g. output probits of model best prediction after softmax).
                The parameter must be a function that taking multiple keyword arguments and returns a :obj:`tensor`
                of size (batch_size,). If not provided, the default attributed function for the model will be used
                (change attribution_model.default_attributed_fn_id).
            target_attention_mask (:obj:`torch.Tensor`, `optional`): Boolean attention mask of size `(batch_size, 1)`
                specifying which target_ids are valid for attribution and which are padding.
            attribute_target (:obj:`bool`, `optional`): Whether to include target prefix for feature attribution.
                Defaults to False.
            step_scores (:obj:`list` of `str`): List of identifiers for step scores that need to be computed during
                attribution. The available step scores are defined in :obj:`inseq.attr.feat.STEP_SCORES_MAP` and new
                step scores can be added by using the :meth:`~inseq.register_step_function` function.
            attribution_args (:obj:`dict`, `optional`): Additional arguments to pass to the attribution method.
            attributed_fn_args (:obj:`dict`, `optional`): Additional arguments to pass to the attributed function.
            step_scores_args (:obj:`dict`, `optional`): Additional arguments to pass to the step scores functions.

        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: A dataclass containing attribution tensors for source
                and target attributions of size `(batch_size, source_length)` and `(batch_size, prefix length)`.
                (target optional if attribute_target=True), plus batch information and any step score present.
        """
        orig_batch = batch.clone().detach().to("cpu")
        is_filtered = False
        # Filter out finished sentences
        if target_attention_mask is not None and int(target_attention_mask.sum()) < target_ids.shape[0]:
            batch = batch.select_active(target_attention_mask)
            target_ids = target_ids.masked_select(target_attention_mask.bool())
            target_ids = target_ids.view(-1, 1)
            is_filtered = True
        target_ids = target_ids.squeeze()
        logger.debug(
            f"\ntarget_ids: {pretty_tensor(target_ids)},\n"
            f"target_attention_mask: {pretty_tensor(target_attention_mask)}"
        )
        logger.debug(f"batch: {batch},\ntarget_ids: {pretty_tensor(target_ids, lpad=4)}")
        attribute_main_args = self.attribution_model.formatter.format_attribution_args(
            batch=batch,
            target_ids=target_ids,
            attributed_fn=attributed_fn,
            attribute_target=attribute_target,
            attributed_fn_args=attributed_fn_args,
            attribute_batch_ids=self.attribute_batch_ids,
            forward_batch_embeds=self.forward_batch_embeds,
            use_baselines=self.use_baselines,
        )
        if len(step_scores) > 0 or self.use_attention_weights or self.use_hidden_states:
            with torch.no_grad():
                output = self.attribution_model.get_forward_output(
                    batch,
                    use_embeddings=self.forward_batch_embeds,
                    output_attentions=self.use_attention_weights,
                    output_hidden_states=self.use_hidden_states,
                )
            if self.use_attention_weights:
                attentions_dict = self.attribution_model.get_attentions_dict(output)
                attribution_args = {**attribution_args, **attentions_dict}
            if self.use_hidden_states:
                hidden_states_dict = self.attribution_model.get_hidden_states_dict(output)
                attribution_args = {**attribution_args, **hidden_states_dict}
        # Perform attribution step
        step_output = self.attribute_step(
            attribute_main_args,
            attribution_args,
        )
        # Format step scores arguments and calculate step scores
        for score in step_scores:
            step_fn_args = self.attribution_model.formatter.format_step_function_args(
                attribution_model=self.attribution_model,
                forward_output=output,
                target_ids=target_ids,
                is_attributed_fn=False,
                batch=batch,
            )
            step_fn_extra_args = get_step_scores_args([score], step_scores_args)
            step_output.step_scores[score] = get_step_scores(score, step_fn_args, step_fn_extra_args).to("cpu")
        # Reinsert finished sentences
        if target_attention_mask is not None and is_filtered:
            step_output.remap_from_filtered(target_attention_mask, orig_batch, self.is_final_step_method)
        step_output = step_output.detach().to("cpu")
        return step_output

    def get_attribution_args(self, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        if hasattr(self, "method") and hasattr(self.method, "attribute"):
            return extract_signature_args(kwargs, self.method.attribute, self.ignore_extra_args, return_remaining=True)
        return {}, {}

    def attribute_step(
        self,
        attribute_fn_main_args: dict[str, Any],
        attribution_args: dict[str, Any] = {},
    ) -> FeatureAttributionStepOutput:
        r"""Performs a single attribution step for the specified attribution arguments.

        Args:
            attribute_fn_main_args (:obj:`dict`): Main arguments used for the attribution method. These are built from
                model inputs at the current step of the feature attribution process.
            attribution_args (:obj:`dict`, `optional`): Additional arguments to pass to the attribution method.
                These can be specified by the user while calling the top level `attribute` methods. Defaults to {}.

        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: A dataclass containing a tensor of source
                attributions of size `(batch_size, source_length)`. At this point the batch
                information is empty, and will later be filled by the enrich_step_output function.
        """
        attr = self.method.attribute(**attribute_fn_main_args, **attribution_args)
        source_attributions, target_attributions = get_source_target_attributions(
            attr, self.attribution_model.is_encoder_decoder
        )
        return FeatureAttributionStepOutput(
            source_attributions=source_attributions,
            target_attributions=target_attributions,
            step_scores={},
        )

    @set_hook
    def hook(self, **kwargs) -> None:
        r"""Hooks the attribution method to the model. Useful to implement pre-attribution logic
        (e.g. freezing layers, replacing embeddings, raise warnings, etc.).
        """
        from ...models.model_config import get_model_config

        if self.use_model_config and self.attribution_model is not None:
            self.attribution_model.config = get_model_config(self.attribution_model.info["model_class"])

    @unset_hook
    def unhook(self, **kwargs) -> None:
        r"""Unhooks the attribution method from the model. If the model was modified in any way, this
        should restore its initial state.
        """
        if self.use_model_config and self.attribution_model is not None:
            self.attribution_model.config = None


def list_feature_attribution_methods():
    """Lists identifiers for all available feature attribution methods. A feature attribution method identifier (e.g.
    `integrated_gradients`) can be passed to :class:`~inseq.models.AttributionModel` or :meth:`~inseq.load_model`
    to define a model for attribution.
    """
    return available_classes(FeatureAttribution)


class DummyAttribution(FeatureAttribution):
    """Dummy attribution method that returns empty attributions."""

    method_name = "dummy"

    def attribute_step(
        self, attribute_fn_main_args: dict[str, Any], attribution_args: dict[str, Any] = {}
    ) -> FeatureAttributionStepOutput:
        return FeatureAttributionStepOutput(
            source_attributions=None,
            target_attributions=None,
            step_scores={},
        )
