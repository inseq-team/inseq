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
import warnings
from copy import deepcopy

import transformers

from .. import load_model
from ..attr.step_functions import is_contrastive_step_function
from ..models import HuggingfaceModel
from .attribute import aggregate_attribution_scores
from .base import BaseCLICommand
from .commands_utils.attribute_context.attribute_context_args import AttributeContextArgs
from .commands_utils.attribute_context.attribute_context_helpers import (
    AttributeContextOutput,
    CCIOutput,
    filter_rank_tokens,
    format_template,
    get_contextless_prefix,
    get_filtered_tokens,
    get_source_target_cci_scores,
    prepare_outputs,
)
from .commands_utils.attribute_context.attribute_context_viz_helpers import handle_visualization

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def attribute_context(args: AttributeContextArgs):
    """Attribute the generation of context-sensitive tokens in ``output_current_text`` to input/output contexts."""
    model: HuggingfaceModel = load_model(
        args.model_name_or_path,
        args.attribution_method,
        model_kwargs=deepcopy(args.model_kwargs),
        tokenizer_kwargs=deepcopy(args.tokenizer_kwargs),
    )

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
        align_output_context_auto=args.align_output_context_auto,
        generation_kwargs=deepcopy(args.generation_kwargs),
        special_tokens_to_keep=args.special_tokens_to_keep,
    )
    output_full_text = format_template(args.output_template, args.output_current_text, args.output_context_text)

    # Tokenize inputs/outputs and compute offset
    input_context_tokens = None
    if args.input_context_text is not None:
        input_context_tokens = get_filtered_tokens(args.input_context_text, model, args.special_tokens_to_keep)
    if not model.is_encoder_decoder:
        space = " " if not output_full_text.startswith(" ") else ""
        output_full_text = input_full_text + space + output_full_text
    output_current_tokens = get_filtered_tokens(
        args.output_current_text, model, args.special_tokens_to_keep, is_target=True
    )
    output_context_tokens = None
    if args.output_context_text is not None:
        output_context_tokens = get_filtered_tokens(
            args.output_context_text, model, args.special_tokens_to_keep, is_target=True
        )
    output_full_tokens = get_filtered_tokens(output_full_text, model, args.special_tokens_to_keep, is_target=True)
    output_current_text_offset = len(output_full_tokens) - len(output_current_tokens)
    if model.is_encoder_decoder:
        prefixed_output_current_text = args.output_current_text
    else:
        space = " " if args.output_current_text is not None and not args.output_current_text.startswith(" ") else ""
        prefixed_output_current_text = args.input_current_text + space + args.output_current_text

    # Part 1: Context-sensitive Token Identification (CTI)
    cti_out = model.attribute(
        args.input_current_text,
        prefixed_output_current_text,
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
    cti_ranked_tokens, cti_threshold = filter_rank_tokens(
        tokens=[t.token for t in cti_out.target][start_pos + cti_out.attr_pos_start :],
        scores=cti_out.step_scores[args.context_sensitivity_metric][start_pos:].tolist(),
        std_threshold=args.context_sensitivity_std_threshold,
        topk=args.context_sensitivity_topk,
    )
    cti_scores = cti_out.step_scores[args.context_sensitivity_metric].tolist()
    if model.is_encoder_decoder:
        cti_scores = cti_scores[:-1]
        if has_lang_tag:
            cti_scores = cti_scores[1:]
    output = AttributeContextOutput(
        input_context=args.input_context_text,
        input_context_tokens=input_context_tokens,
        output_context=args.output_context_text,
        output_context_tokens=output_context_tokens,
        output_current=args.output_current_text,
        output_current_tokens=output_current_tokens,
        cti_scores=cti_scores,
        info=args if args.add_output_info else None,
    )
    # Part 2: Contextual Cues Imputation (CCI)
    for cti_idx, cti_score, cti_tok in cti_ranked_tokens:
        contextual_prefix = model.convert_tokens_to_string(
            output_full_tokens[: output_current_text_offset + cti_idx + 1], skip_special_tokens=False
        )
        cci_kwargs = {}
        if is_contrastive_step_function(args.attributed_fn):
            contextless_prefix = get_contextless_prefix(
                model,
                args.input_current_text,
                output_current_tokens,
                cti_idx,
                args.special_tokens_to_keep,
                deepcopy(args.generation_kwargs),
            )
            cci_kwargs["contrast_sources"] = args.input_current_text if model.is_encoder_decoder else None
            cci_kwargs["contrast_targets"] = contextless_prefix
            output_ctx_tokens = model.convert_string_to_tokens(contextual_prefix, skip_special_tokens=False)
            output_ctxless_tokens = model.convert_string_to_tokens(contextless_prefix, skip_special_tokens=False)
            tok_pos = -2 if model.is_encoder_decoder else -1
            if args.attributed_fn == "kl_divergence" or output_ctx_tokens[tok_pos] == output_ctxless_tokens[tok_pos]:
                cci_kwargs["contrast_force_inputs"] = True
        pos_start = output_current_text_offset + cti_idx + int(model.is_encoder_decoder) + int(has_lang_tag)
        cci_attrib_out = model.attribute(
            input_full_text,
            contextual_prefix,
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
        )[0]
        if args.show_intermediate_outputs:
            cci_attrib_out.show(do_aggregation=False)
        source_scores, target_scores = get_source_target_cci_scores(
            model,
            cci_attrib_out,
            args.input_template,
            input_context_tokens,
            args.output_template,
            output_context_tokens,
            args.has_input_context,
            args.has_output_context,
            has_lang_tag,
            args.special_tokens_to_keep,
        )
        cci_out = CCIOutput(
            cti_idx=cti_idx,
            cti_token=cti_tok,
            cti_score=cti_score,
            contextual_prefix=contextual_prefix,
            contextless_prefix=contextless_prefix,
            input_context_scores=source_scores,
            output_context_scores=target_scores,
        )
        output.cci_scores.append(cci_out)
    if args.save_path:
        with open(args.save_path, "w") as f:
            json.dump(output.to_dict(), f, indent=4)
    if args.show_viz or args.viz_path:
        handle_visualization(args, model, output, cti_threshold)


class AttributeContextCommand(BaseCLICommand):
    _name = "attribute-context"
    _help = "Detect context-sensitive tokens in a generated text and attribute their predictions to available context."
    _dataclasses = AttributeContextArgs

    def run(args: AttributeContextArgs):
        attribute_context(args)
