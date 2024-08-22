"""Implementation of the context attribution process described in `Quantifying the Plausibility of Context Reliance in
Neural Machine Translation <https://arxiv.org/abs/2310.01188>`_ for decoder-only and encoder-decoder models.

The process consists of two steps:
    - Context-sensitive Token Identification (CTI): detects which tokens in the generated output of interest are
        influenced by the presence of context.
    - Contextual Cues Imputation (CCI): attributes the generation of context-sensitive tokens identified in the first
        step to the input and output contexts.

Example usage:

```bash
inseq attribute-context \
    --model_name_or_path gpt2 \
    --input_context_text "George was sick yesterday." \
    --input_current_text "His colleagues asked him" \
    --attributed_fn contrast_prob_diff
```
"""

import json
import logging
import warnings
from copy import deepcopy

import transformers

from ... import load_model
from ...attr.step_functions import is_contrastive_step_function
from ...models import HuggingfaceModel
from ..attribute import aggregate_attribution_scores
from ..base import BaseCLICommand
from .attribute_context_args import AttributeContextArgs
from .attribute_context_helpers import (
    AttributeContextOutput,
    CCIOutput,
    concat_with_sep,
    filter_rank_tokens,
    format_template,
    get_contextless_output,
    get_filtered_tokens,
    get_source_target_cci_scores,
    prepare_outputs,
)
from .attribute_context_viz_helpers import visualize_attribute_context

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)


def attribute_context(args: AttributeContextArgs) -> AttributeContextOutput:
    """Attribute the generation of context-sensitive tokens in ``output_current_text`` to input/output contexts."""
    model: HuggingfaceModel = load_model(
        args.model_name_or_path,
        args.attribution_method,
        model_kwargs=deepcopy(args.model_kwargs),
        tokenizer_kwargs=deepcopy(args.tokenizer_kwargs),
    )
    if not isinstance(args.model_name_or_path, str):
        args.model_name_or_path = model.model_name
    return attribute_context_with_model(args, model)


def attribute_context_with_model(args: AttributeContextArgs, model: HuggingfaceModel) -> AttributeContextOutput:
    # Handle language tag for multilingual models - no need to specify it in generation kwargs
    has_lang_tag = "tgt_lang" in args.tokenizer_kwargs
    if has_lang_tag and "forced_bos_token_id" not in args.generation_kwargs:
        tgt_lang = args.tokenizer_kwargs["tgt_lang"]
        args.generation_kwargs["forced_bos_token_id"] = model.tokenizer.lang_code_to_id[tgt_lang]

    # Prepare input/outputs (generate if necessary)
    input_full_text = format_template(args.input_template, args.input_current_text, args.input_context_text)
    args.output_context_text, args.output_current_text = prepare_outputs(
        model=model,
        input_context_text=args.input_context_text,
        input_full_text=input_full_text,
        output_context_text=args.output_context_text,
        output_current_text=args.output_current_text,
        output_template=args.output_template,
        handle_output_context_strategy=args.handle_output_context_strategy,
        generation_kwargs=deepcopy(args.generation_kwargs),
        special_tokens_to_keep=args.special_tokens_to_keep,
        decoder_input_output_separator=args.decoder_input_output_separator,
    )
    output_full_text = format_template(args.output_template, args.output_current_text, args.output_context_text)

    # Tokenize inputs/outputs and compute offset
    input_context_tokens = None
    if args.input_context_text is not None:
        input_context_tokens = get_filtered_tokens(args.input_context_text, model, args.special_tokens_to_keep)
    if not model.is_encoder_decoder:
        output_full_text = concat_with_sep(input_full_text, output_full_text, args.decoder_input_output_separator)
    output_current_tokens = get_filtered_tokens(
        args.output_current_text, model, args.special_tokens_to_keep, is_target=True
    )
    output_context_tokens = None
    if args.output_context_text is not None:
        output_context_tokens = get_filtered_tokens(
            args.output_context_text, model, args.special_tokens_to_keep, is_target=True
        )
    input_full_tokens = get_filtered_tokens(input_full_text, model, args.special_tokens_to_keep)
    output_full_tokens = get_filtered_tokens(output_full_text, model, args.special_tokens_to_keep, is_target=True)
    output_current_text_offset = len(output_full_tokens) - len(output_current_tokens)
    formatted_input_current_text = args.contextless_input_current_text.format(current=args.input_current_text)
    formatted_output_current_text = args.contextless_output_current_text.format(current=args.output_current_text)
    if not model.is_encoder_decoder:
        formatted_input_current_text = concat_with_sep(
            formatted_input_current_text, "", args.decoder_input_output_separator
        )
        formatted_output_current_text = formatted_input_current_text + formatted_output_current_text

    # Part 1: Context-sensitive Token Identification (CTI)
    cti_out = model.attribute(
        formatted_input_current_text.rstrip(" "),
        formatted_output_current_text,
        attribute_target=model.is_encoder_decoder,
        step_scores=[args.context_sensitivity_metric],
        contrast_sources=input_full_text if model.is_encoder_decoder else None,
        contrast_targets=output_full_text,
        show_progress=False,
        method="dummy",
    )[0]
    if args.show_intermediate_outputs:
        cti_out.show(do_aggregation=False)

    start_pos = 1 if has_lang_tag else 0
    contextless_output_prefix = args.contextless_output_current_text.split("{current}")[0]
    contextless_output_prefix_tokens = get_filtered_tokens(
        contextless_output_prefix, model, args.special_tokens_to_keep, is_target=True
    )
    start_pos += len(contextless_output_prefix_tokens)
    cti_scores = cti_out.step_scores[args.context_sensitivity_metric][start_pos:].tolist()
    cti_tokens = [t.token for t in cti_out.target][start_pos + cti_out.attr_pos_start :]
    if model.is_encoder_decoder:
        cti_scores = cti_scores[:-1]
        cti_tokens = cti_tokens[:-1]
    cti_ranked_tokens, cti_threshold = filter_rank_tokens(
        tokens=cti_tokens,
        scores=cti_scores,
        std_threshold=args.context_sensitivity_std_threshold,
        topk=args.context_sensitivity_topk,
    )
    output = AttributeContextOutput(
        input_context=args.input_context_text,
        input_context_tokens=input_context_tokens,
        output_context=args.output_context_text,
        output_context_tokens=output_context_tokens,
        output_current=args.output_current_text,
        output_current_tokens=cti_tokens,
        cti_scores=cti_scores,
        info=args,
    )
    # Part 2: Contextual Cues Imputation (CCI)
    for cci_step_idx, (cti_idx, cti_score, cti_tok) in enumerate(cti_ranked_tokens):
        contextual_input = model.convert_tokens_to_string(input_full_tokens, skip_special_tokens=False).lstrip(" ")
        contextual_output = model.convert_tokens_to_string(
            output_full_tokens[: output_current_text_offset + cti_idx + 1], skip_special_tokens=False
        ).lstrip(" ")
        if not contextual_output:
            output_ctx_tokens = [output_full_tokens[output_current_text_offset + cti_idx]]
            if model.is_encoder_decoder:
                output_ctx_tokens.append(model.pad_token)
            contextual_output = model.convert_tokens_to_string(output_ctx_tokens, skip_special_tokens=True)
        else:
            output_ctx_tokens = model.convert_string_to_tokens(
                contextual_output, skip_special_tokens=False, as_targets=model.is_encoder_decoder
            )
        cci_kwargs = {}
        contextless_output = None
        contrast_token = None
        if args.attributed_fn is not None and is_contrastive_step_function(args.attributed_fn):
            if not model.is_encoder_decoder:
                formatted_input_current_text = concat_with_sep(
                    formatted_input_current_text, contextless_output_prefix, args.decoder_input_output_separator
                )
            contextless_output = get_contextless_output(
                model,
                formatted_input_current_text,
                output_current_tokens,
                cti_idx,
                cti_ranked_tokens,
                args.contextless_output_next_tokens,
                args.prompt_user_for_contextless_output_next_tokens,
                cci_step_idx,
                args.decoder_input_output_separator,
                args.special_tokens_to_keep,
                deepcopy(args.generation_kwargs),
            )
            cci_kwargs["contrast_sources"] = formatted_input_current_text if model.is_encoder_decoder else None
            cci_kwargs["contrast_targets"] = contextless_output
            output_ctxless_tokens = model.convert_string_to_tokens(
                contextless_output, skip_special_tokens=False, as_targets=model.is_encoder_decoder
            )
            tok_pos = -2 if model.is_encoder_decoder else -1
            contrast_token = output_ctxless_tokens[tok_pos]
            if args.attributed_fn == "kl_divergence" or output_ctx_tokens[tok_pos] == output_ctxless_tokens[tok_pos]:
                cci_kwargs["contrast_force_inputs"] = True
        bos_offset = int(
            model.is_encoder_decoder
            or (output_ctx_tokens[0] == model.bos_token and model.bos_token not in args.special_tokens_to_keep)
        )
        pos_start = output_current_text_offset + cti_idx + bos_offset + int(has_lang_tag)
        cci_attrib_out = model.attribute(
            contextual_input,
            contextual_output,
            attribute_target=model.is_encoder_decoder and args.has_output_context,
            show_progress=False,
            attr_pos_start=pos_start,
            attributed_fn=args.attributed_fn,
            method=args.attribution_method,
            **cci_kwargs,
            **args.attribution_kwargs,
        )
        cci_attrib_out = aggregate_attribution_scores(
            out=cci_attrib_out,
            selectors=args.attribution_selectors,
            aggregators=args.attribution_aggregators,
            normalize_attributions=args.normalize_attributions,
            rescale_attributions=args.rescale_attributions,
        )[0]
        if args.show_intermediate_outputs:
            cci_attrib_out.show(do_aggregation=False)
        source_scores, target_scores = get_source_target_cci_scores(
            model,
            cci_attrib_out,
            args.input_template,
            args.input_current_text,
            input_context_tokens,
            input_full_tokens,
            args.output_template,
            output_context_tokens,
            args.has_input_context,
            args.has_output_context,
            has_lang_tag,
            args.decoder_input_output_separator,
            args.special_tokens_to_keep,
        )
        cci_out = CCIOutput(
            cti_idx=cti_idx,
            cti_token=cti_tok,
            contrast_token=contrast_token,
            cti_score=cti_score,
            contextual_output=contextual_output,
            contextless_output=contextless_output,
            input_context_scores=source_scores,
            output_context_scores=target_scores,
        )
        output.cci_scores.append(cci_out)
    if args.show_viz or args.viz_path:
        visualize_attribute_context(output, model, cti_threshold, args.show_viz, args.viz_path)
    if not args.add_output_info:
        output.info = None
    if args.save_path:
        with open(args.save_path, "w") as f:
            json.dump(output.to_dict(), f, indent=4)
    return output


class AttributeContextCommand(BaseCLICommand):
    _name = "attribute-context"
    _help = "Detect context-sensitive tokens in a generated text and attribute their predictions to available context."
    _dataclasses = AttributeContextArgs

    def run(args: AttributeContextArgs):
        attribute_context(args)
