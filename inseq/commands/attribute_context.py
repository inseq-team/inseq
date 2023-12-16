import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rich import print as rprint
from rich.prompt import Confirm, Prompt

from .. import (
    AttributionModel,
    list_aggregation_functions,
    list_aggregators,
    list_feature_attribution_methods,
    list_step_functions,
    load_model,
)
from ..utils.alignment_utils import compute_word_aligns
from .base import BaseCLICommand

logger = logging.getLogger(__name__)


@dataclass
class AttributeContextArgs:
    model_name_or_path: str = field(
        metadata={"help": "The name or path of the model on which attribution is performed."},
    )
    input_current_text: str = field(
        metadata={
            "help": (
                "The input text used for generation. If the model is a decoder-only model, the input text is a "
                "prefix used for language modeling. If the model is an encoder-decoder model, the input text is the "
                "source text provided as input to the encoder."
            ),
        },
    )
    attribution_method: str = field(
        default="saliency",
        metadata={
            "help": "The attribution method used to perform feature attribution.",
            "choices": list_feature_attribution_methods(),
        },
    )
    attributed_fn: str = field(
        default=None,
        metadata={
            "help": "The function used as target for the attribution method.",
            "choices": list_step_functions(),
        },
    )
    context_sensitivity_metric: str = field(
        default="kl_divergence",
        metadata={
            "help": "The metric used to detect context-sensitive tokens in generated texts.",
            "choices": list_step_functions(),
        },
    )
    input_template: str = field(
        default="{context} {current}",
        metadata={
            "help": (
                "The template used to format model inputs. The template must contain at least the"
                " ``{current}`` placeholder, which will be replaced by the input current text. If ``{context}`` is"
                " also specified, source-side context will be used. Useful for models requiring special tokens or"
                " formatting in the input text (e.g. <brk> tags to separate context and current text)."
            ),
        },
    )
    input_context_text: Optional[str] = field(
        default=None,
        metadata={"help": "An input context for which context sensitivity should be detected."},
    )
    output_context_text: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "One or more output contexts for which context sensitivity should be detected, in case the model"
                " accepts target-side outputs (e.g. context-aware MT). If more than one context is provided, the"
                " context sensitivity metric will be computed for each context separately comparing it to the"
                " no-context baseline, and only contexts for which context-sensitivitiy is detected will be used for"
                " attribution."
            )
        },
    )
    output_current_text: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The output text generated in the contextual setting. If specified, this output is force-decoded"
                " instead of letting the model generate the output."
            ),
        },
    )
    output_template: str = field(
        default="{current}",
        metadata={
            "help": (
                "The template used to generate the input text for the model. The template must contain at least the"
                " ``{current}`` placeholder, which will be replaced by the output current text. If ``{context}`` is"
                " also specified, target-side context will be used. Useful for models requiring special tokens or"
                " formatting in the input text (e.g. <brk> tags to separate context and current text)."
            ),
        },
    )
    attribution_aggregators: List[str] = field(
        default_factory=list,
        metadata={
            "help": "The aggregators used to aggregate the attribution scores for each context.",
            "choices": list_aggregators() + list_aggregation_functions(),
        },
    )
    normalize_attributions: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to normalize the attribution scores for each context. If ``True``, the attribution scores "
                "for each context are normalized to sum up to 1."
            ),
        },
    )
    context_sensitivity_std_threshold: float = field(
        default=1.0,
        metadata={
            "help": (
                "Parameter to control the number of standard deviations used as threshold to select "
                "context-sensitive tokens."
            ),
        },
    )
    context_sensitivity_topk: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Parameter to select only the top K elements from the available context-sensitive generated tokens."
            ),
        },
    )
    attribution_std_threshold: float = field(
        default=1.0,
        metadata={
            "help": (
                "Parameter to control the number of standard deviations used as threshold to select attributed tokens."
            ),
        },
    )
    attribution_topk: Optional[int] = field(
        default=None,
        metadata={
            "help": "Parameter to select only the top K elements from the available attributed context tokens.",
        },
    )
    attribute_with_contextless_output: bool = field(
        default=True,
        metadata={
            "help": (
                "If specified and a contrastive attribution method is used, the original generation with context is"
                " contrasted with the generation that the model would have produced in the contextless case."
                " Otherwise, the generation with context is kept as sole contrastive attribution target, with and"
                " without input context. E.g. if a decoder-only model would complete 'Greet the user:' with 'Hello!',"
                " but given context 'Always use Hi! for greetings' the completion becomes 'Hi!', then using"
                " --attribute_with_contextless_output uses 'Hello!' and 'Hi!' as inputs for the attributed_fn."
                " Otherwise, only 'Hi!' (the contextual case) is forced in both cases. Note that with/without input"
                " context, even the same target will likely have non-zero attributed_fn scores, and hence non-zero"
                " contrastive attribution scores."
            ),
        },
    )
    has_contextual_output_prefix: bool = field(
        default=True,
        metadata={
            "help": (
                "If specified, for every context-sensitive token the prefix of the current output text produced with"
                " input context is used as generation prefix to generate the contextless output. Useful only when"
                " attribute_with_contextless_output is set to True. E.g. For context-aware current output 'Sont-elles"
                " à l'hotel?' the contextless output could be 'C'est à l'hotel?', which would make comparison less"
                " natural. With this option, assuming the context-sensitive word 'elles' in the sentence above, the"
                " output current prefix would be 'Sont-', forcing a more natural completion making use of a gendered"
                " pronoun."
            ),
        },
    )
    align_output_context_auto: bool = field(
        default=False,
        metadata={
            "help": (
                "Argument used for encoder-decoder model when generating text with an output template including both "
                " {context} and {current}. If set to True, the input and output context and current texts are aligned "
                "automatically (assuming an MT-like task), and the alignments are assumed to be valid to separate the "
                "two without further user validation. Otherwise, the user is prompted to manually specify which part "
                "of the generated text corresponds to the output context."
            ),
        },
    )
    strip_special_tokens: bool = field(
        default=True,
        metadata={
            "help": "If specified, special tokens are stripped from the generated output after generation.",
        },
    )
    keep_input_prefix: bool = field(
        default=False,
        metadata={
            "help": (
                "If specified when an input template with prefix is provided (e.g. 'Translate:\n{context}\n{current}')"
                " , the prefix is preserved in the contrastive contextless setting (i.e. the contextless template is"
                " 'Translate:\n{current}' in this case). Otherwise, the prefix is used only in the contextual case."
                " Can be used in combination with keep_input_suffix and keep_input_separator."
            )
        },
    )
    keep_input_separator: bool = field(
        default=False,
        metadata={
            "help": (
                "If specified when an input template with separator is given (e.g. '{context}\nTranslate:\n{current}')"
                " , the separator is preserved in the contrastive contextless setting (i.e. the contextless template"
                " is '\nTranslate:\n{current}' in this case).Otherwise, the separator is used only in the contextual"
                " case. Can be used in combination with keep_input_prefix and keep_input_suffix."
            )
        },
    )
    keep_input_suffix: bool = field(
        default=False,
        metadata={
            "help": (
                "If specified when an input template with suffix is provided (e.g. '{context}\n{current}\nTranslate"
                " the text above:\n') , the suffix is preserved in the contrastive contextless setting (i.e. the"
                " contextless template is '{current}\nTranslate the text above:\n' in this case). Otherwise, the"
                " suffix is used only in the contextual case. Can be used in combination with keep_input_prefix and"
                " keep_input_separator."
            )
        },
    )
    keep_output_prefix: bool = field(
        default=False,
        metadata={
            "help": (
                "If specified when an output template with prefix is provided (e.g. 'Translate:\n{current}')"
                " , the prefix is preserved in the contrastive contextless setting. Otherwise, the prefix is used only"
                " in the contextual case. Can be used in combination with keep_output_separator."
            )
        },
    )
    keep_output_separator: bool = field(
        default=False,
        metadata={
            "help": (
                "If specified with an output template with separator (e.g. '{context}\nTranslate:\n{current}')"
                " , the separator is preserved in the contrastive contextless setting (i.e. the contextless template"
                " is '\nTranslate:\n{current}' in this case).Otherwise, the separator is used only in the contextual"
                " case. Can be used in combination with keep_output_prefix."
            )
        },
    )
    model_src_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The source language of the model. If specified, the model is assumed to be a multilingual model using"
                " language tags. Some examples of such models are mBART, M2M100 and NLLB. The src_lang should be"
                "provided in the correct format (e.g. English is 'en_XX' for mBART, but 'eng_Latn' for NLLB)."
            ),
        },
    )
    model_tgt_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The target language of the model. If specified, the model is assumed to be a multilingual model using"
                " language tags. Some examples of such models are mBART, M2M100 and NLLB. The tgt_lang should be"
                "provided in the correct format (e.g. English is 'en_XX' for mBART, but 'eng_Latn' for NLLB)."
            ),
        },
    )

    def __post_init__(self):
        if self.input_context_text and "{context}" not in self.input_template:
            logger.warning(
                f"input_template has format {self.input_template} (no {{context}}), but --input_context_text is"
                " specified. Ignoring provided --input_context_text."
            )
            self.input_context_text = None
        if self.output_context_text and "{context}" not in self.output_template:
            logger.warning(
                f"output_template has format {self.output_template} (no {{context}}), but --output_context_text is"
                " specified. Ignoring provided --output_context_text."
            )
            self.output_context_text = None
        if not self.input_context_text and "{context}" in self.input_template:
            raise ValueError(
                f"{{context}} format placeholder is present in input_template {self.input_template},"
                " but --input_context_text is not specified."
            )
        if "{current}" not in self.input_template:
            raise ValueError(f"{{current}} format placeholder is missing from input_template {self.input_template}.")
        if not self.input_current_text:
            raise ValueError("--input_current_text must be a non-empty string.")
        if "{current}" not in self.output_template:
            raise ValueError(f"{{current}} format placeholder is missing from output_template {self.output_template}.")

        input_separator = None
        if "{context}" in self.output_template:
            input_prefix, *_ = self.output_template.partition("{context}")
            input_separator = self.output_template.split("{context}")[1].split("{current}")[0]
        else:
            input_prefix, *_ = self.output_template.partition("{current}")
        *_, input_suffix = self.output_template.partition("{current}")
        if self.keep_input_prefix and not input_prefix:
            raise ValueError(
                f"keep_input_prefix is specified but output_template {self.output_template} does not contain a prefix."
            )
        if self.keep_input_separator and not input_separator:
            raise ValueError(
                f"keep_input_separator is specified but output_template {self.output_template} does not contain a"
                " separator."
            )
        if self.keep_input_suffix and not input_suffix:
            raise ValueError(
                f"keep_input_suffix is specified but output_template {self.output_template} does not contain a suffix."
            )

        output_separator = None
        if "{context}" in self.output_template:
            output_prefix, *_ = self.output_template.partition("{context}")
            output_separator = self.output_template.split("{context}")[1].split("{current}")[0]
        else:
            output_prefix, *_ = self.output_template.partition("{current}")
        if self.keep_output_prefix and not output_prefix:
            raise ValueError(
                f"keep_output_prefix is specified but output_template {self.output_template} does not contain a"
                " prefix."
            )
        if self.keep_output_separator and not output_separator:
            raise ValueError(
                f"keep_output_separator is specified but output_template {self.output_template} does not contain a"
                " separator."
            )


def prompt_user_for_context(output: str, context_candidate: Optional[str] = None) -> str:
    while True:
        if context_candidate:
            is_correct_candidate = Confirm.ask(
                f'\n:arrow_right: The model generated the following output: "[bold]{output}[/bold]"'
                f'\n:question: Is [bold]"{context_candidate}"[/bold] the correct context you want to attribute?'
            )
        if is_correct_candidate:
            user_context = context_candidate
        else:
            user_context = Prompt.ask(
                ":writing_hand: Please enter the portion of the generated output representing the correct context"
            )
        if user_context in output and user_context.strip():
            break
        rprint(
            "[prompt.invalid]The provided context is invalid. Please provide a non-empty substring of"
            " the model output above to use as context."
        )
    return user_context


def get_output_context_from_aligned_inputs(input_context: str, output_text: str) -> str:
    aligned_context = compute_word_aligns(input_context, output_text, split_pattern=r"\s+|\b")
    max_context_id = max(pair[1] for pair in aligned_context.alignments)
    output_text_boundary_token = aligned_context.target_tokens[max_context_id]
    # Empty spans correspond to token boundaries
    spans = [m.span() for m in re.finditer(r"\s+|\b", output_text)]
    tok_start_positions = list({start if start == end else end for start, end in spans})
    output_text_context_candidate_boundary = tok_start_positions[max_context_id] + len(output_text_boundary_token)
    return output_text[:output_text_context_candidate_boundary]


def generate_model_output(
    model: AttributionModel,
    model_input: str,
    generation_kwargs: Dict[str, Any],
    strip_special_tokens: bool,
    output_template: str,
    prefix: str,
    suffix: str,
) -> str:
    # Generate outputs, strip special tokens and remove prefix/suffix
    output_gen = model.generate(model_input, skip_special_tokens=False, **generation_kwargs)[0]
    if strip_special_tokens:
        for token in model.special_tokens:
            output_gen = output_gen.strip(token).strip(" ")

    if prefix:
        if not output_gen.startswith(prefix):
            raise ValueError(
                f"Output template {output_template} contains prefix {prefix} but output '{output_gen}' does"
                " not match the prefix. Please check whether the template is correct, or force context/current"
                " outputs."
            )
        output_gen = output_gen[len(prefix) :]
    if suffix:
        if not output_gen.endswith(suffix):
            raise ValueError(
                f"Output template {output_template} contains suffix {suffix} but output '{output_gen}' does"
                " not match the suffix. Please check whether the template is correct, or force context/current"
                " outputs."
            )
        output_gen = output_gen[: -len(suffix)]
    return output_gen


def prepare_outputs(
    model: AttributionModel,
    input_context_text: Optional[str],
    input_full_text: str,
    output_context_text: Optional[str],
    output_current_text: Optional[str],
    output_template: str,
    align_output_context_auto: bool = False,
    generation_kwargs: Dict[str, Any] = {},
    strip_special_tokens: bool = True,
) -> Tuple[Optional[str], str]:
    """Handle model outputs and prepare them for attribution.
    This procedure is valid both for encoder-decoder and decoder-only models.

    | use_context | has_ctx | has_curr | setting
    |-------------|---------|----------|--------
    | (MUST) True | True    | True     | 1. Use forced context + current as output
    | False       | False   | True     | 2. Use forced current as output
    | (MUST) True | True    | False    | 3. Set inputs with forced context, generate output, use as current
    | False       | False   | False    | 4. Generate output, use it as current
    | True        | False   | False    | 5. Generate output, handle context/current splitting
    | True        | False   | True     | 6. Generate output, handle context/current splitting, force current

    NOTE: use_context must be True if has_ctx is True (checked in __post_init__)
    """
    use_context = "{context}" in output_template
    has_ctx = output_context_text is not None
    has_curr = output_current_text is not None
    model_input = input_full_text
    final_current = output_current_text
    final_context = output_context_text

    # E.g. output template "A{context}B{current}C" -> prefix = "A", suffix = "C", separator = "B"
    prefix, *_ = output_template.partition("{context}" if use_context else "{current}")
    *_, suffix = output_template.partition("{current}")
    separator = output_template.split("{context}")[1].split("{current}")[0] if use_context else None

    # Settings 1, 2
    if (has_ctx == use_context) and has_curr:
        return final_context, final_current

    if has_ctx and not has_curr:
        if model.is_encoder_decoder:
            generation_kwargs["decoder_input_ids"] = model.encode(
                prefix + output_context_text, as_targets=True, add_special_tokens=False
            ).input_ids
        else:
            model_input = input_full_text + prefix + output_context_text

    output_gen = generate_model_output(
        model, model_input, generation_kwargs, strip_special_tokens, output_template, prefix, suffix
    )

    # Settings 3, 4
    if (has_ctx == use_context) and not has_curr:
        if use_context:
            if model.is_encoder_decoder:
                final_current = output_gen[len(output_context_text + separator) :]
            else:
                final_current = output_gen[len(model_input + separator) :]
        else:
            final_current = output_gen if model.is_encoder_decoder else output_gen[len(model_input) :]
        return final_context, final_current

    # Settings 5, 6
    # Try to split the output into context and current text using the substring between {context} and {current}
    # in output_template. As we have no guarantees that this separator is unique (e.g. it could be whitespace,
    # also found between tokens in context and current) we consider the splitting successful if 2 substrings
    # are produced. If this fails, we try splitting on punctuation.
    output_context_candidate = None
    separator_split_context_current_substring = output_gen.split(separator)
    if len(separator_split_context_current_substring) == 2:
        output_context_candidate = separator_split_context_current_substring[0]
    if not output_context_candidate:
        punct_expr = re.compile(r"[\s{}]+".format(re.escape(".?!,;:)]}")))
        punctuation_split_context_current_substring = [s for s in punct_expr.split(output_gen) if s]
        if len(punctuation_split_context_current_substring) == 2:
            output_context_candidate = punctuation_split_context_current_substring[0]

    # Final resort: if the model is an encoder-decoder model, we align the full input and full output, identifying
    # which tokens correspond to context and which to current. This stems from the assumption that
    # source and target text have some relation that can be identified via alignment (e.g. source and
    # target are translations of each other). We prompt the user a yes/no question asking whether the
    # context identified is correct. If not, the user is asked to provide the correct context (check
    # if input match, otherwise ask again). If align_output_context_auto, the aligned texts are
    # assumed to be correct (no user input required, to automate the procedure)
    if not output_context_candidate and model.is_encoder_decoder and input_context_text is not None:
        output_context_candidate = get_output_context_from_aligned_inputs(input_context_text, output_gen)

    if output_context_candidate and align_output_context_auto:
        final_context = output_context_candidate
    else:
        final_context = prompt_user_for_context(output_gen, output_context_candidate)
    template_output_context = output_template.split("{current}")[0].format(context=final_context)
    final_current = output_gen[min(len(template_output_context), len(output_gen)) :]
    if not has_curr and not final_current:
        raise ValueError(
            f"The model produced an empty current output given the specified context '{final_context}'. If no"
            " context is generated naturally by the model, you can force an output context using the"
            " --output_context_text option."
        )
    if has_curr:
        logger.warning(
            f"The model produced current text '{final_current}', but the specified output_current_text"
            f" '{output_current_text}'is used instead. If you want to use the original current output text generated"
            " by the model, remove the --output_current_text option."
        )
    return final_context, final_current


def setup(args: AttributeContextArgs):
    tokenizer_kwargs = {}
    generation_kwargs = {}
    if args.model_src_lang is not None:
        tokenizer_kwargs["src_lang"] = args.model_src_lang
    if args.model_tgt_lang is not None:
        tokenizer_kwargs["tgt_lang"] = args.model_tgt_lang
    model = load_model(args.model_name_or_path, args.attribution_method, tokenizer_kwargs=tokenizer_kwargs)
    if args.model_tgt_lang is not None:
        generation_kwargs["forced_bos_token_id"] = model.tokenizer.lang_code_to_id[args.model_tgt_lang]

    in_kwargs = {"current": args.input_current_text}
    if args.input_context_text is not None:
        in_kwargs["context"] = args.input_context_text
    in_text = args.input_template.format(**in_kwargs)

    args.output_context_text, args.output_current_text = prepare_outputs(
        model=model,
        input_context_text=args.input_context_text,
        input_full_text=in_text,
        output_context_text=args.output_context_text,
        output_current_text=args.output_current_text,
        output_template=args.output_template,
        align_output_context_auto=args.align_output_context_auto,
        generation_kwargs=generation_kwargs,
        strip_special_tokens=args.strip_special_tokens,
    )

    out_kwargs = {"current": args.output_current_text}
    if args.output_context_text is not None:
        out_kwargs["context"] = args.output_context_text
    out_text = args.output_template.format(**out_kwargs)
    # out_text_no_suffix = (args.output_template.split("{current}")[0] + "{current}").format(**out_kwargs)
    return model, in_text, out_text


def attribute_context(args: AttributeContextArgs):
    model, in_text, out_text = setup(args)
    # Remove suffix from output template if present: the suffix is only used in prepare_outputs to allow the post-hoc
    # exclusion of generated tokens from the end of the current output from attribution.

    print(in_text, out_text, args.output_context_text, args.output_current_text)

    # if model.is_encoder_decoder:
    #    contextual_input = in_text
    #    contextual_output = out_text
    #    contextless_input = args.input_current_text
    #    contextless_output = args.output_current_text
    # cti_out = model.attribute(
    #    args.input_current_text,
    #    attribute_target=True,
    #    step_scores=[args.cti_metric],
    #    contrast_sources=ex.input_full,
    #    show_progress=False,
    #    method="dummy",
    # )[0]
    # Tokenize inputs and outputs
    # tokenize_inputs_outputs()
    # Apply context sensitivity metric
    # detect_context_sensitive_tokens()
    # Discretize context sensitivity scores to obtain tags
    # scores_to_tags()
    #
    # For every tagged output token:
    #   If the attribution method supports contrastive inputs, prepare contextless generation, align it with current
    #   one
    #   and identify the start position for tokens belonging to the current output
    #   prepare_contrastive_inputs()
    #   Attribute the tagged output token to input (and output) context
    #   attribute_context_sensitive_token()
    #   Aggregate output scores
    #   aggregate_attribution_scores()
    #   Discretize attribution scores to obtain tags
    #   scores_to_tags()
    # Return JSON output with detected tokens


class AttributeContextCommand(BaseCLICommand):
    _name = "attribute-context"
    _help = "Detect context-sensitive tokens in a generated text and attribute their predictions to input context."
    _dataclasses = AttributeContextArgs

    def run(args: AttributeContextArgs):
        attribute_context(args)
