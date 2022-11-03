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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import logging
from abc import abstractmethod
from datetime import datetime

from torchtyping import TensorType

from ...data import (
    Batch,
    EncoderDecoderBatch,
    FeatureAttributionInput,
    FeatureAttributionOutput,
    FeatureAttributionSequenceOutput,
    FeatureAttributionStepOutput,
)
from ...data.viz import close_progress_bar, get_progress_bar, update_progress_bar
from ...utils import (
    Registry,
    UnknownAttributionMethodError,
    extract_signature_args,
    find_char_indexes,
    get_available_methods,
    pretty_tensor,
)
from ...utils.typing import (
    ModelIdentifier,
    SingleScorePerStepTensor,
    TargetIdsTensor,
    TokenWithId,
)
from ..attribution_decorators import batched, set_hook, unset_hook
from .attribution_utils import (
    STEP_SCORES_MAP,
    check_attribute_positions,
    enrich_step_output,
    get_split_targets,
    get_step_scores,
)


if TYPE_CHECKING:
    from ...models import AttributionModel


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

    registry_attr = "method_name"
    ignore_extra_args = ["inputs", "baselines", "target", "additional_forward_args"]

    def __init__(self, attribution_model: "AttributionModel", hook_to_model: bool = True, **kwargs):
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
        attribution_model: Optional["AttributionModel"] = None,
        model_name_or_path: Optional[ModelIdentifier] = None,
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
        from ...models import load_model

        if model_name_or_path is not None:
            model = load_model(model_name_or_path)
        elif attribution_model is not None:
            model = attribution_model
        else:
            raise RuntimeError(
                "Only one among an initialized model and a model identifier "
                "must be defined when loading the attribution method."
            )
        methods = cls.available_classes()
        if method_name not in methods:
            raise UnknownAttributionMethodError(method_name)
        return methods[method_name](model, **kwargs)

    @batched
    def prepare_and_attribute(
        self,
        inputs: Union[FeatureAttributionInput, Tuple[FeatureAttributionInput, FeatureAttributionInput]],
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        output_step_attributions: bool = False,
        attribute_target: bool = False,
        step_scores: List[str] = [],
        include_eos_baseline: bool = False,
        prepend_bos_token: bool = True,
        attributed_fn: Union[str, Callable[..., SingleScorePerStepTensor], None] = None,
        attribution_args: Dict[str, Any] = {},
        attributed_fn_args: Dict[str, Any] = {},
        step_scores_args: Dict[str, Any] = {},
    ) -> FeatureAttributionOutput:
        r"""
        Prepares inputs and performs attribution.

        Wraps the attribution method :meth:`~inseq.attr.feat.FeatureAttribution.attribute` method
        and the :meth:`~inseq.models.AttributionModel.prepare_inputs_for_attribution` method.

        Args:
            inputs (:obj:`FeatureAttributionInput` or :obj:`tuple` of :obj:`FeatureAttributionInput`): The inputs
                provided to the :meth:`~inseq.models.AttributionModel.prepare_inputs_for_attribution` method.
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
                step scores can be added by using the :meth:`~inseq.register_step_score` function.
            include_eos_baseline (:obj:`bool`, `optional`): Whether to include the EOS token in the baseline for
                attribution. By default the EOS token is not used for attribution. Defaults to False.
            prepend_bos_token (:obj:`bool`, `optional`): Whether to prepend the BOS token to the input sequence.
                Defaults to True.
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
        batch = self.attribution_model.prepare_inputs_for_attribution(
            inputs,
            prepend_bos_token,
            include_eos_baseline,
            self.is_layer_attribution,
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
            prepend_bos_token=prepend_bos_token,
            attribution_args=attribution_args,
            attributed_fn_args=attributed_fn_args,
            step_scores_args=step_scores_args,
        )
        # Same here, repeated from AttributionModel.attribute
        # to allow independent usage
        attribution_output.info["prepend_bos_token"] = prepend_bos_token
        attribution_output.info["include_eos_baseline"] = include_eos_baseline
        attribution_output.info["attributed_fn"] = attributed_fn.__name__
        attribution_output.info["attribution_args"] = attribution_args
        attribution_output.info["attributed_fn_args"] = attributed_fn_args
        attribution_output.info["step_scores_args"] = step_scores_args
        return attribution_output

    def attribute(
        self,
        batch: Union[Batch, EncoderDecoderBatch],
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attr_pos_start: Optional[int] = 1,
        attr_pos_end: Optional[int] = None,
        show_progress: bool = True,
        pretty_progress: bool = True,
        output_step_attributions: bool = False,
        attribute_target: bool = False,
        step_scores: List[str] = [],
        prepend_bos_token: bool = True,
        attribution_args: Dict[str, Any] = {},
        attributed_fn_args: Dict[str, Any] = {},
        step_scores_args: Dict[str, Any] = {},
    ) -> FeatureAttributionOutput:
        r"""
        Performs the feature attribution procedure using the specified attribution method.

        Args:
            batch (:class:`~inseq.data.EncoderDecoderBatch` or :class:`~inseq.data.Batch`): The batch of sequences to
                attribute.
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
                step scores can be added by using the :meth:`~inseq.register_step_score` function.
            prepend_bos_token (:obj:`bool`, `optional`): Whether to prepend a BOS token to the
                targets, if they are to be encoded. Defaults to True.
            attribution_args (:obj:`dict`, `optional`): Additional arguments to pass to the attribution method.
            attributed_fn_args (:obj:`dict`, `optional`): Additional arguments to pass to the attributed function.
            step_scores_args (:obj:`dict`, `optional`): Additional arguments to pass to the step scores function.
        Returns:
            :class:`~inseq.data.FeatureAttributionOutput`: An object containing a list of sequence attributions, with
                an optional added list of single :class:`~inseq.data.FeatureAttributionStepOutput` for each step and
                extra information regarding the attribution parameters.
        """
        if self.is_layer_attribution and attribute_target:
            raise ValueError(
                "Layer attribution methods do not support attribute_target=True. Use regular ones instead."
            )
        attr_pos_start, attr_pos_end = check_attribute_positions(
            batch.max_generation_length,
            attr_pos_start,
            attr_pos_end,
        )
        logger.debug("=" * 30 + f"\nfull batch: {batch}\n" + "=" * 30)
        sources = self.attribution_model.convert_tokens_to_string(batch.sources.input_tokens)
        targets = self.attribution_model.convert_tokens_to_string(batch.targets.input_tokens, as_targets=True)
        tokenized_target_sentences = [
            self.attribution_model.convert_string_to_tokens(sent, as_targets=True, skip_special_tokens=False)
            for sent in targets
        ]
        ids_target_sentences = self.attribution_model.convert_tokens_to_ids(tokenized_target_sentences)
        target_tokens_with_ids = [
            [
                TokenWithId(tokenized_target_sentences[sent_idx][tok_idx], ids_target_sentences[sent_idx][tok_idx])
                for tok_idx in range(len(tokenized_target_sentences[sent_idx]))
            ]
            for sent_idx in range(len(tokenized_target_sentences))
        ]
        targets_lengths = [min(attr_pos_end, len(tts)) - attr_pos_start for tts in tokenized_target_sentences]
        pbar = get_progress_bar(
            all_sentences=(sources, targets, targets_lengths),
            method_name=self.method_name,
            show=show_progress,
            pretty=pretty_progress,
            attr_pos_start=attr_pos_start,
            attr_pos_end=attr_pos_end,
        )
        whitespace_indexes = find_char_indexes(targets, " ")
        attribution_outputs = []
        start = datetime.now()

        # Attribution loop for generation
        for step in range(attr_pos_start, attr_pos_end):
            step_output = self.filtered_attribute_step(
                batch[:step],
                target_ids=batch.targets.input_ids[:, step].unsqueeze(1),
                attributed_fn=attributed_fn,
                target_attention_mask=batch.targets.attention_mask[:, step].unsqueeze(1),
                attribute_target=attribute_target,
                step_scores=step_scores,
                attribution_args=attribution_args,
                attributed_fn_args=attributed_fn_args,
                step_scores_args=step_scores_args,
            )
            attribution_outputs.append(step_output)
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
        end = datetime.now()
        close_progress_bar(pbar, show=show_progress, pretty=pretty_progress)
        batch.to("cpu")
        out = FeatureAttributionOutput(
            sequence_attributions=FeatureAttributionSequenceOutput.from_step_attributions(
                attributions=attribution_outputs,
                tokenized_target_sentences=target_tokens_with_ids,
                pad_id=self.attribution_model.pad_token,
                has_bos_token=prepend_bos_token,
                attr_pos_start=attr_pos_start,
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
        batch: EncoderDecoderBatch,
        target_ids: TensorType["batch_size", 1, int],
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        target_attention_mask: Optional[TensorType["batch_size", 1, int]] = None,
        attribute_target: bool = False,
        step_scores: List[str] = [],
        attribution_args: Dict[str, Any] = {},
        attributed_fn_args: Dict[str, Any] = {},
        step_scores_args: Dict[str, Any] = {},
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
                step scores can be added by using the :meth:`~inseq.register_step_score` function.
            attribution_args (:obj:`dict`, `optional`): Additional arguments to pass to the attribution method.
            attributed_fn_args (:obj:`dict`, `optional`): Additional arguments to pass to the attributed function.
            step_scores_args (:obj:`dict`, `optional`): Additional arguments to pass to the step scores functions.
        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: A dataclass containing attribution tensors for source
                and target attributions of size `(batch_size, source_length)` and `(batch_size, prefix length)`.
                (target optional if attribute_target=True), plus batch information and any step score present.
        """
        orig_batch = batch.clone().detach().to("cpu")
        orig_target_ids = target_ids.clone()
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
        attribute_main_args = self.format_attribute_args(
            batch=batch,
            target_ids=target_ids,
            attributed_fn=attributed_fn,
            attribute_target=attribute_target,
            attributed_fn_args=attributed_fn_args,
        )
        # Perform attribution step
        step_output = self.attribute_step(
            attribute_main_args,
            attribution_args,
        )
        # Calculate step scores
        for step_score in step_scores:
            if step_score not in STEP_SCORES_MAP:
                raise AttributeError(
                    f"Step score {step_score} not found. Available step scores are: "
                    f"{', '.join([x for x in STEP_SCORES_MAP.keys()])}. Use the inseq.register_step_score"
                    f"function to register a custom step score."
                )
            step_output.step_scores[step_score] = get_step_scores(
                self.attribution_model, batch, target_ids, step_score, step_scores_args
            )
        # Add batch information to output
        step_output = enrich_step_output(
            step_output,
            orig_batch,
            self.attribution_model.convert_ids_to_tokens(orig_target_ids, skip_special_tokens=False),
            orig_target_ids.squeeze().detach().to("cpu"),
        )
        # Reinsert finished sentences
        if target_attention_mask is not None and is_filtered:
            step_output.remap_from_filtered(target_attention_mask)
        step_output = step_output.detach().to("cpu")
        return step_output

    def get_attribution_args(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if hasattr(self, "method") and hasattr(self.method, "attribute"):
            return extract_signature_args(kwargs, self.method.attribute, self.ignore_extra_args, return_remaining=True)
        return {}

    def format_attribute_args(
        self,
        batch: Union[Batch, EncoderDecoderBatch],
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attributed_fn_args: Dict[str, Any] = {},
        **kwargs,
    ) -> Dict[str, Any]:
        r"""
        Formats inputs for the attribution method based on the model type and the attribution method requirements.

        Args:
            batch (:class:`~inseq.data.Batch` or :class:`~inseq.data.EncoderDecoderBatch`): The batch of sequences on
                which attribution is performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size)` corresponding to tokens
                for which the attribution step must be performed.
            attributed_fn (:obj:`Callable[..., SingleScorePerStepTensor]`): The function of model outputs
                representing what should be attributed (e.g. output probits of model best prediction after softmax).
                The parameter must be a function that taking multiple keyword arguments and returns a :obj:`tensor`
                of size (batch_size,). If not provided, the default attributed function for the model will be used
                (change attribution_model.default_attributed_fn_id).
            attribute_target (:obj:`bool`, optional): Whether to attribute the target prefix or not. Defaults to False.
            attributed_fn_args (:obj:`dict`, `optional`): Additional arguments to pass to the attributed function.
                Defaults to {}.
            **kwargs: Additional arguments to pass to the model-specific
                :meth:`inseq.models.AttributionModel.format_attribution_args` method.
        Returns:
            :obj:`dict`: A dictionary containing the formatted attribution arguments.
        """
        logger.debug(f"batch: {batch},\ntarget_ids: {pretty_tensor(target_ids, lpad=4)}")
        attribute_fn_args, baselines = self.attribution_model.format_attribution_args(
            batch=batch,
            target_ids=target_ids,
            attributed_fn=attributed_fn,
            attributed_fn_args=attributed_fn_args,
            is_layer_attribution=self.is_layer_attribution,
            **kwargs,
        )
        if self.use_baseline:
            attribute_fn_args["baselines"] = baselines
        return attribute_fn_args

    def attribute_step(
        self,
        attribute_fn_main_args: Dict[str, Any],
        attribution_args: Dict[str, Any] = {},
    ) -> Any:
        r"""
        Performs a single attribution step for the specified attribution arguments.

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
        return FeatureAttributionStepOutput(source_attributions=attr, step_scores={})

    @abstractmethod
    @set_hook
    def hook(self, **kwargs) -> None:
        r"""
        Hooks the attribution method to the model. Useful to implement pre-attribution logic
        (e.g. freezing layers, replacing embeddings, raise warnings, etc.).

        Abstract method, must be implemented by subclasses.
        """
        pass

    @abstractmethod
    @unset_hook
    def unhook(self, **kwargs) -> None:
        r"""
        Unhooks the attribution method from the model. If the model was modified in any way, this
        should restore its initial state.

        Abstract method, must be implemented by subclasses.
        """
        pass


def list_feature_attribution_methods():
    """
    Lists identifiers for all available feature attribution methods.
    """
    return get_available_methods(FeatureAttribution)
