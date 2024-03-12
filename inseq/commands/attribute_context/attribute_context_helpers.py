import logging
import re
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from rich import print as rprint
from rich.prompt import Confirm, Prompt
from torch import tensor

from ...data import FeatureAttributionSequenceOutput
from ...models import HuggingfaceModel
from ...utils import pretty_dict
from ...utils.alignment_utils import compute_word_aligns
from .attribute_context_args import AttributeContextArgs, HandleOutputContextSetting

logger = logging.getLogger(__name__)


@dataclass
class CCIOutput:
    """Output of the Contextual Cues Imputation (CCI) step."""

    cti_idx: int
    cti_token: str
    cti_score: float
    contextual_output: str
    contextless_output: str
    input_context_scores: Optional[list[float]] = None
    output_context_scores: Optional[list[float]] = None

    def __repr__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.__dict__)})"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__.items())


@dataclass
class AttributeContextOutput:
    """Output of the overall context attribution process."""

    input_context: Optional[str] = None
    input_context_tokens: Optional[list[str]] = None
    output_context: Optional[str] = None
    output_context_tokens: Optional[list[str]] = None
    output_current: Optional[str] = None
    output_current_tokens: Optional[list[str]] = None
    cti_scores: Optional[list[float]] = None
    cci_scores: list[CCIOutput] = field(default_factory=list)
    info: Optional[AttributeContextArgs] = None

    def __repr__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.__dict__)})"

    def to_dict(self) -> dict[str, Any]:
        out_dict = {k: v for k, v in self.__dict__.items() if k not in ["cci_scores", "info"]}
        out_dict["cci_scores"] = [cci_out.to_dict() for cci_out in self.cci_scores]
        if self.info:
            out_dict["info"] = self.info.to_dict()
        return out_dict

    @classmethod
    def from_dict(cls, out_dict: dict[str, Any]) -> "AttributeContextOutput":
        out = cls()
        for k, v in out_dict.items():
            if k not in ["cci_scores", "info", "has_input_context", "has_output_context"]:
                setattr(out, k, v)
        out.cci_scores = [CCIOutput(**cci_out) for cci_out in out_dict["cci_scores"]]
        if "info" in out_dict:
            field_names = [f.name for f in fields(AttributeContextArgs)]
            out.info = AttributeContextArgs(**{k: v for k, v in out_dict["info"].items() if k in field_names})
        return out


def concat_with_sep(s1: str, s2: str, sep: str) -> bool:
    """Adds separator between two strings if needed."""
    need_sep = not s1.endswith(sep) and not s2.startswith(sep)
    if need_sep:
        return s1 + sep + s2
    return s1 + s2


def format_template(template: str, current: str, context: Optional[str] = None) -> str:
    kwargs = {"current": current}
    if context is not None:
        kwargs["context"] = context
    return template.format(**kwargs)


def get_filtered_tokens(
    text: str,
    model: HuggingfaceModel,
    special_tokens_to_keep: list[str],
    replace_special_characters: bool = False,
    is_target: bool = False,
) -> list[str]:
    """Tokenize text and filter out special tokens, keeping only those in ``special_tokens_to_keep``."""
    as_targets = is_target and model.is_encoder_decoder
    return [
        t.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ") if replace_special_characters else t
        for t in model.convert_string_to_tokens(text, skip_special_tokens=False, as_targets=as_targets)
        if t not in model.special_tokens or t in special_tokens_to_keep
    ]


def generate_with_special_tokens(
    model: HuggingfaceModel,
    model_input: str,
    special_tokens_to_keep: list[str] = [],
    **generation_kwargs,
) -> str:
    """Generate text preserving special tokens in ``special_tokens_to_keep``."""
    # Generate outputs, strip special tokens and remove prefix/suffix
    output_gen = model.generate(model_input, skip_special_tokens=False, **generation_kwargs)[0]
    output_tokens = get_filtered_tokens(output_gen, model, special_tokens_to_keep, is_target=True)
    return model.convert_tokens_to_string(output_tokens, skip_special_tokens=False)


def generate_model_output(
    model: HuggingfaceModel,
    model_input: str,
    generation_kwargs: dict[str, Any],
    special_tokens_to_keep: list[str],
    output_template: str,
    prefix: str,
    suffix: str,
) -> str:
    """Generate the model output, validating the presence of a prefix/suffix and stripping them from the generation."""
    output_gen = generate_with_special_tokens(model, model_input, special_tokens_to_keep, **generation_kwargs)
    if prefix:
        if not output_gen.startswith(prefix):
            raise ValueError(
                f"Output template '{output_template}' contains prefix '{prefix}' but output '{output_gen}' does"
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


def prompt_user_for_context(output: str, context_candidate: Optional[str] = None) -> str:
    """Prompt the user to provide the correct context for the provided output."""
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
        if output.startswith(user_context):
            if not user_context.strip():
                use_empty_context = Confirm.ask(
                    ":question: The provided context is empty. Do you want to use an empty context?"
                )
                if use_empty_context:
                    user_context = ""
                else:
                    continue
            break
        rprint(
            "[prompt.invalid]The provided context is invalid. Please provide a non-empty substring of"
            " the model output above to use as context."
        )
    return user_context


def get_output_context_from_aligned_inputs(input_context: str, output_text: str) -> str:
    """Retrieve the output context from alignments between input context and the full output text."""
    aligned_context = compute_word_aligns(input_context, output_text, split_pattern=r"\s+|\b")
    max_context_id = max(pair[1] for pair in aligned_context.alignments)
    output_text_boundary_token = aligned_context.target_tokens[max_context_id]
    # Empty spans correspond to token boundaries
    spans = [m.span() for m in re.finditer(r"\s+|\b", output_text)]
    tok_start_positions = list({start if start == end else end for start, end in spans})
    output_text_context_candidate_boundary = tok_start_positions[max_context_id] + len(output_text_boundary_token)
    return output_text[:output_text_context_candidate_boundary]


def prepare_outputs(
    model: HuggingfaceModel,
    input_context_text: Optional[str],
    input_full_text: str,
    output_context_text: Optional[str],
    output_current_text: Optional[str],
    output_template: str,
    handle_output_context_strategy: str,
    generation_kwargs: dict[str, Any] = {},
    special_tokens_to_keep: list[str] = [],
    decoder_input_output_separator: str = " ",
) -> tuple[Optional[str], str]:
    """Handle model outputs and prepare them for attribution.
    This procedure is valid both for encoder-decoder and decoder-only models.

    | use_out_ctx | has_out_ctx | has_out_curr | setting
    |-------------|-------------|--------------|--------
    | True        | True        | True         | 1. Use forced context + current as output
    | False       | False       | True         | 2. Use forced current as output
    | True        | True        | False        | 3. Set inputs with forced context, generate output, use as current
    | False       | False       | False        | 4. Generate output, use it as current
    | True        | False       | False        | 5. Generate output, handle context/current splitting
    | True        | False       | True         | 6. Generate output, handle context/current splitting, force current

    NOTE: If ``use_out_ctx`` is True but ``has_out_ctx`` is False, the model generation is assumed to contain both
    a context and a current portion which need to be separated. ``has_out_ctx`` cannot be True if ``use_out_ctx``
    is False (pre-check in ``__post_init__``).
    """
    use_out_ctx = "{context}" in output_template
    has_out_ctx = output_context_text is not None
    has_out_curr = output_current_text is not None
    model_input = input_full_text
    final_current = output_current_text
    final_context = output_context_text

    # E.g. output template "A{context}B{current}C" -> prefix = "A", suffix = "C", separator = "B"
    prefix, _ = output_template.split("{context}" if use_out_ctx else "{current}")
    output_current_prefix_template, suffix = output_template.split("{current}")
    separator = output_template.split("{context}")[1].split("{current}")[0] if use_out_ctx else None

    # Settings 1, 2
    if (has_out_ctx == use_out_ctx) and has_out_curr:
        return final_context, final_current

    # Prepend output prefix and context, if available, if current output needs to be generated
    output_current_prefix = prefix
    if has_out_ctx and not has_out_curr:
        output_current_prefix = output_current_prefix_template.strip().format(context=output_context_text)
        if model.is_encoder_decoder:
            generation_kwargs["decoder_input_ids"] = model.encode(
                output_current_prefix, as_targets=True, add_special_tokens=False
            ).input_ids
            if "forced_bos_token_id" in generation_kwargs:
                generation_kwargs["decoder_input_ids"][0, 0] = generation_kwargs["forced_bos_token_id"]
        else:
            model_input = concat_with_sep(input_full_text, output_current_prefix, decoder_input_output_separator)
            output_current_prefix = model_input

    output_gen = generate_model_output(
        model, model_input, generation_kwargs, special_tokens_to_keep, output_template, output_current_prefix, suffix
    )

    # Settings 3, 4
    if (has_out_ctx == use_out_ctx) and not has_out_curr:
        final_current = output_gen if model.is_encoder_decoder or use_out_ctx else output_gen[len(model_input) :]
        return final_context, final_current.strip()

    # Settings 5, 6
    # Try splitting the output into context and current text using ``separator``. As we have no guarantees of its
    # uniqueness (e.g. it could be whitespace, also found between tokens in context and current) we consider the
    # splitting successful if exactly 2 substrings are produced. If this fails, we try splitting on punctuation.
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
    # which tokens correspond to context and which to current. This assumes that input and output texts are alignable
    # (e.g. translations of each other). We prompt the user a yes/no question asking whether the context identified is
    # correct. If not, the user is asked to provide the correct context. If handle_output_context_strategy = "auto", aligned
    # texts are assumed to be correct (no user input required, to automate the procedure)
    if not output_context_candidate and model.is_encoder_decoder and input_context_text is not None:
        output_context_candidate = get_output_context_from_aligned_inputs(input_context_text, output_gen)

    if output_context_candidate and handle_output_context_strategy == HandleOutputContextSetting.AUTO.value:
        final_context = output_context_candidate
    else:
        final_context = prompt_user_for_context(output_gen, output_context_candidate)
    template_output_context = output_template.split("{current}")[0].format(context=final_context)
    if not final_context:
        template_output_context = template_output_context.strip()
    final_current = output_gen[min(len(template_output_context), len(output_gen)) :]
    if not has_out_curr and not final_current:
        raise ValueError(
            f"The model produced an empty current output given the specified context '{final_context}'. If no"
            " context is generated naturally by the model, you can force an output context using the"
            " --output_context_text option."
        )
    if has_out_curr:
        logger.warning(
            f"The model produced current text '{final_current}', but the specified output_current_text"
            f" '{output_current_text}'is used instead. If you want to use the original current output text generated"
            " by the model, remove the --output_current_text option."
        )
    return final_context, final_current.strip()


def get_scores_threshold(scores: list[float], std_weight: float) -> float:
    """Compute the threshold for a given weight."""
    if std_weight is None or len(scores) == 0:
        return 0
    if std_weight == 0 or len(scores) == 1:
        return tensor(scores).mean()
    return tensor(scores).mean() + std_weight * tensor(scores).std()


def filter_rank_tokens(
    tokens: list[str],
    scores: list[float],
    std_threshold: Optional[float] = None,
    topk: Optional[int] = None,
) -> tuple[list[tuple[int, float, str]], float]:
    indices = list(range(0, len(scores)))
    token_score_tuples = sorted(zip(indices, scores, tokens), key=lambda x: abs(x[1]), reverse=True)
    threshold = get_scores_threshold(scores, std_threshold)
    token_score_tuples = [(i, s, t) for i, s, t in token_score_tuples if abs(s) >= threshold]
    if topk:
        token_score_tuples = token_score_tuples[:topk]
    return token_score_tuples, threshold


def get_contextless_output(
    model: HuggingfaceModel,
    input_current_text: str,
    output_current_tokens: list[str],
    cti_idx: int,
    cti_ranked_tokens: tuple[int, float, str],
    contextless_output_next_tokens: Optional[list[str]],
    prompt_user_for_contextless_output_next_tokens: bool,
    cci_step_idx: int,
    decoder_input_output_separator: str = " ",
    special_tokens_to_keep: list[str] = [],
    generation_kwargs: dict[str, Any] = {},
) -> tuple[str, str]:
    n_ctxless_next_tokens = len(contextless_output_next_tokens)
    next_ctxless_token = None
    if n_ctxless_next_tokens > 0:
        if n_ctxless_next_tokens != len(cti_ranked_tokens):
            raise ValueError(
                "The number of manually specified contextless output next tokens must be equal to the number "
                "of context-sensitive tokens identified by CTI."
            )
        next_ctxless_token = contextless_output_next_tokens[cci_step_idx]
    if prompt_user_for_contextless_output_next_tokens:
        next_ctxless_token = prompt_user_for_contextless_output_next_tokens(output_current_tokens, cti_idx, model)
    if isinstance(next_ctxless_token, str):
        next_ctxless_token = model.convert_string_to_tokens(
            next_ctxless_token, skip_special_tokens=False, as_targets=model.is_encoder_decoder
        )[0]
        contextless_output_tokens = output_current_tokens[:cti_idx] + [next_ctxless_token]
        contextless_output = model.convert_tokens_to_string(contextless_output_tokens, skip_special_tokens=False)
    else:
        contextless_output = generate_contextless_output(
            model,
            input_current_text,
            output_current_tokens,
            cti_idx,
            special_tokens_to_keep,
            generation_kwargs,
            decoder_input_output_separator,
        )
    return contextless_output


def generate_contextless_output(
    model: HuggingfaceModel,
    input_current_text: str,
    output_current_tokens: list[str],
    cti_idx: int,
    special_tokens_to_keep: list[str] = [],
    generation_kwargs: dict[str, Any] = {},
    decoder_input_output_separator: str = " ",
) -> tuple[str, str]:
    """Generate the contextless output for the current token identified as context-sensitive."""
    contextual_prefix_tokens = output_current_tokens[:cti_idx]
    contextual_prefix = model.convert_tokens_to_string(contextual_prefix_tokens, skip_special_tokens=False)
    if model.is_encoder_decoder:
        # One extra token for the EOS which is always forced at the end for encoder-decoders
        generation_kwargs["max_new_tokens"] = 2
        decoder_input_ids = model.encode(contextual_prefix, as_targets=True).input_ids
        if int(decoder_input_ids[0, -1]) == model.eos_token_id:
            decoder_input_ids = decoder_input_ids[0, :-1][None, ...]
        generation_kwargs["decoder_input_ids"] = decoder_input_ids
        generation_input = input_current_text
    else:
        generation_kwargs["max_new_tokens"] = 1
        generation_input = concat_with_sep(input_current_text, contextual_prefix, decoder_input_output_separator)
    contextless_output = generate_with_special_tokens(
        model,
        generation_input,
        special_tokens_to_keep,
        **generation_kwargs,
    )
    return contextless_output


def get_source_target_cci_scores(
    model: HuggingfaceModel,
    cci_attrib_out: FeatureAttributionSequenceOutput,
    input_template: str,
    input_current_text: str,
    input_context_tokens: list[str],
    input_full_tokens: list[str],
    output_template: str,
    output_context_tokens: list[str],
    has_input_context: bool,
    has_output_context: bool,
    model_has_lang_tag: bool,
    decoder_input_output_separator: str,
    special_tokens_to_keep: list[str] = [],
) -> tuple[Optional[list[float]], Optional[list[float]]]:
    """Extract attribution scores for the input and output contexts."""
    input_scores, output_scores = None, None
    if has_input_context:
        if model.is_encoder_decoder:
            input_scores = cci_attrib_out.source_attributions[:, 0].tolist()
            if model_has_lang_tag:
                input_scores = input_scores[2:]
        else:
            input_scores = cci_attrib_out.target_attributions[:, 0].tolist()
        input_prefix, *_ = input_template.partition("{context}")
        if "{current}" in input_prefix:
            input_prefix = input_prefix.format(current=input_current_text)
        input_prefix_tokens = get_filtered_tokens(input_prefix, model, special_tokens_to_keep, is_target=False)
        input_prefix_len = len(input_prefix_tokens)
        input_scores = input_scores[input_prefix_len : len(input_context_tokens) + input_prefix_len]
    if has_output_context:
        output_scores = cci_attrib_out.target_attributions[:, 0].tolist()
        if model_has_lang_tag:
            output_scores = output_scores[2:]
        output_prefix, *_ = output_template.partition("{context}")
        if not model.is_encoder_decoder and output_prefix:
            output_prefix = decoder_input_output_separator + output_prefix
        output_prefix_tokens = get_filtered_tokens(output_prefix, model, special_tokens_to_keep, is_target=True)
        prefix_len = len(output_prefix_tokens)
        if not model.is_encoder_decoder:
            prefix_len += len(input_full_tokens)
        output_scores = output_scores[prefix_len : len(output_context_tokens) + prefix_len]
    return input_scores, output_scores


def prompt_user_for_contextless_output_next_tokens(
    output_current_tokens: list[str],
    cti_idx: int,
    model: HuggingfaceModel,
    special_tokens_to_keep: list[str] = [],
) -> Optional[str]:
    """Prompt the user to provide the next tokens of the contextless output.

    Args:
        output_current_tokens (str): list of tokens of the current output
        cti_idx (int): index of the current token identified as context-sensitive

    Returns:
        str: next tokens of the contextless output specified by the user. If None, the user does not want to specify
            the contextless output.
    """
    contextual_prefix_tokens = output_current_tokens[:cti_idx]
    contextual_prefix = model.convert_tokens_to_string(contextual_prefix_tokens, skip_special_tokens=False)
    contextual_output_token = get_filtered_tokens(
        output_current_tokens[cti_idx],
        model,
        special_tokens_to_keep=special_tokens_to_keep,
        is_target=True,
        replace_special_characters=True,
    )[0]
    while True:
        force_contextless_output = Confirm.ask(
            f'\n:arrow_right: Contextual prefix: "[bold]{contextual_prefix}[/bold]"'
            f'\n:question: The token [bold]"{contextual_output_token}"[/bold] is produced in the contextual setting.'
            " Do you want to specify a word for comparison?"
        )
        if not force_contextless_output:
            return None
        provided_contextless_output = Prompt.ask(
            ":writing_hand: Please enter the word to use for comparison with the contextual output:"
        )
        if provided_contextless_output.strip():
            break
        rprint("[prompt.invalid]The provided word is empty. Please provide a non-empty word.")
    return provided_contextless_output
