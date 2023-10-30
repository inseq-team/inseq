import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from transformers.modeling_outputs import ModelOutput

from ..data import (
    DecoderOnlyBatch,
    EncoderDecoderBatch,
    FeatureAttributionInput,
    get_batch_from_inputs,
    slice_batch_from_position,
)
from ..utils.typing import TargetIdsTensor

if TYPE_CHECKING:
    from ..attr.step_functions import StepFunction, StepFunctionArgs

logger = logging.getLogger(__name__)

CONTRAST_FN_ARGS_DOCSTRING = """Args:
        contrast_sources (:obj:`str` or :obj:`list(str)`): Source text(s) used as contrastive inputs to compute
            the contrastive step function for encoder-decoder models. If not specified, the source text is assumed to
            match the original source text. Defaults to :obj:`None`.
        contrast_targets (:obj:`str` or :obj:`list(str)`): Contrastive target text(s) to be compared to the original
            target text. If not specified, the original target text is used as contrastive target (will result in same
            output unless ``contrast_sources`` are specified). Defaults to :obj:`None`.
        contrast_targets_alignments (:obj:`list(tuple(int, int))`, `optional`): A list of tuples of indices, where the
            first element is the index of the original target token and the second element is the index of the
            contrastive target token, used only if :obj:`contrast_targets` is specified. If an explicit alignment is
            not specified, the alignment of the original and contrastive target texts is assumed to be 1:1 for all
            available tokens. Defaults to :obj:`None`.
        contrast_force_inputs (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether to force the contrastive
            inputs to be used for attribution. Defaults to :obj:`False`.
"""


def contrast_fn_docstring() -> Callable[..., "StepFunction"]:
    def docstring_decorator(fn: "StepFunction") -> "StepFunction":
        """Returns the docstring for the contrastive step functions."""
        if fn.__doc__ is not None:
            if "Args:\n" in fn.__doc__:
                fn.__doc__ = fn.__doc__.replace("Args:\n", CONTRAST_FN_ARGS_DOCSTRING)
            else:
                fn.__doc__ = fn.__doc__ + "\n    " + CONTRAST_FN_ARGS_DOCSTRING
        else:
            fn.__doc__ = CONTRAST_FN_ARGS_DOCSTRING
        return fn

    return docstring_decorator


@dataclass
class ContrastOutput:
    forward_output: ModelOutput
    batch: Union[EncoderDecoderBatch, DecoderOnlyBatch, None] = None
    target_ids: Optional[TargetIdsTensor] = None


def _get_contrast_output(
    args: "StepFunctionArgs",
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    contrast_targets_alignments: Optional[List[List[Tuple[int, int]]]] = None,
    return_contrastive_target_ids: bool = False,
    return_contrastive_batch: bool = False,
    **forward_kwargs,
) -> ContrastOutput:
    """Utility function to return the output of the model for given contrastive inputs.

    Args:
        return_contrastive_target_ids (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether to return the
            contrastive target ids as well as the model output. Defaults to :obj:`False`.
        **forward_kwargs: Additional keyword arguments to be passed to the model's forward pass.
    """
    c_tgt_ids = None
    is_enc_dec = args.attribution_model.is_encoder_decoder
    if contrast_targets:
        c_batch = DecoderOnlyBatch.from_batch(
            get_batch_from_inputs(
                attribution_model=args.attribution_model,
                inputs=contrast_targets,
                as_targets=is_enc_dec,
            )
        )
        curr_prefix_len = args.decoder_input_ids.size(1)
        c_batch, c_tgt_ids = slice_batch_from_position(c_batch, curr_prefix_len, contrast_targets_alignments)

        if args.decoder_input_ids.size(0) != c_batch.target_ids.size(0):
            raise ValueError(
                f"Contrastive batch size ({c_batch.target_ids.size(0)}) must match candidate batch size"
                f" ({args.decoder_input_ids.size(0)}). Multi-sentence attribution and methods expanding inputs to"
                " multiple steps (e.g. Integrated Gradients) are not yet supported for contrastive attribution."
            )

        args.decoder_input_ids = c_batch.target_ids
        args.decoder_input_embeds = c_batch.target_embeds
        args.decoder_attention_mask = c_batch.target_mask
    if contrast_sources:
        from ..attr.step_functions import StepFunctionEncoderDecoderArgs

        if not (is_enc_dec and isinstance(args, StepFunctionEncoderDecoderArgs)):
            raise ValueError(
                "Contrastive source inputs can only be used with encoder-decoder models. "
                "Use `contrast_targets` to set a contrastive target containing a prefix for decoder-only models."
            )
        c_enc_in = args.attribution_model.encode(contrast_sources)
        args.encoder_input_ids = c_enc_in.input_ids
        args.encoder_attention_mask = c_enc_in.attention_mask
        args.encoder_input_embeds = args.attribution_model.embed(args.encoder_input_ids, as_targets=False)
    c_batch = args.attribution_model.formatter.convert_args_to_batch(args)
    return ContrastOutput(
        forward_output=args.attribution_model.get_forward_output(c_batch, use_embeddings=is_enc_dec, **forward_kwargs),
        batch=c_batch if return_contrastive_batch else None,
        target_ids=c_tgt_ids if return_contrastive_target_ids else None,
    )


def _setup_contrast_args(
    args: "StepFunctionArgs",
    contrast_sources: Optional[FeatureAttributionInput] = None,
    contrast_targets: Optional[FeatureAttributionInput] = None,
    contrast_targets_alignments: Optional[List[List[Tuple[int, int]]]] = None,
    contrast_force_inputs: bool = False,
):
    c_output = _get_contrast_output(
        args,
        contrast_sources=contrast_sources,
        contrast_targets=contrast_targets,
        contrast_targets_alignments=contrast_targets_alignments,
        return_contrastive_target_ids=True,
        return_contrastive_batch=True,
    )
    if c_output.target_ids is not None:
        args.target_ids = c_output.target_ids
    if args.is_attributed_fn:
        if contrast_force_inputs:
            warnings.warn(
                "Forcing contrastive inputs to be used for attribution. This may result in unexpected behavior for "
                "gradient-based attribution methods.",
                stacklevel=1,
            )
            args.forward_output = c_output.forward_output
        else:
            warnings.warn(
                "Contrastive inputs do not match original inputs when using a contrastive attributed function.\n"
                "By default we force the original inputs to be used (i.e. only the contrastive predicted target is "
                "different).\nThis is a requirement for gradient-based attribution method, as contrastive inputs don't"
                " participate in gradient computation.\nFor attribution methods with less stringent requirements, "
                "set --contrast_force_inputs to True to use the contrastive inputs for attribution instead.",
                stacklevel=1,
            )
    else:
        args.forward_output = c_output.forward_output
    return args
