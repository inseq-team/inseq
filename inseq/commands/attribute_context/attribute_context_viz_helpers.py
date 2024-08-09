from copy import deepcopy
from typing import Literal

from rich.console import Console

from ... import load_model
from ...models import HuggingfaceModel
from ...utils.viz_utils import treescope_ignore
from .attribute_context_args import AttributeContextArgs
from .attribute_context_helpers import (
    AttributeContextOutput,
    filter_rank_tokens,
    get_filtered_tokens,
    get_scores_threshold,
)


def get_formatted_procedure_details(args: AttributeContextArgs) -> str:
    def format_comment(std: float | None = None, topk: int | None = None) -> str:
        comment = []
        if std:
            comment.append(f"std λ={std:.2f}")
        if topk:
            comment.append(f"top {topk}")
        if len(comment) > 0:
            return ", ".join(comment)
        return "all"

    cti_comment = format_comment(args.context_sensitivity_std_threshold, args.context_sensitivity_topk)
    cci_comment = format_comment(args.attribution_std_threshold, args.attribution_topk)
    input_context_comment, output_context_comment = "", ""
    if args.has_input_context:
        input_context_comment = f"\n[bold]Input context:[/bold]\t{args.input_context_text}"
    if args.has_output_context:
        output_context_comment = f"\n[bold]Output context:[/bold]\t{args.output_context_text}"
    return (
        f"\nContext with [bold green]contextual cues[/bold green] ({cci_comment}) followed by output"
        f" sentence with [bold dodger_blue1]context-sensitive target spans[/bold dodger_blue1] ({cti_comment})\n"
        f'(CTI = "{args.context_sensitivity_metric}", CCI = "{args.attribution_method}" w/ "{args.attributed_fn}" '
        f"target)\n{input_context_comment}\n[bold]Input current:[/bold] {args.input_current_text}"
        f"{output_context_comment}\n[bold]Output current:[/bold]\t{args.output_current_text}"
    )


def get_formatted_attribute_context_results(
    model: HuggingfaceModel,
    args: AttributeContextArgs,
    output: AttributeContextOutput,
    cti_threshold: float,
) -> str:
    """Format the results of the context attribution process."""

    def format_context_comment(
        model: HuggingfaceModel,
        has_other_context: bool,
        special_tokens_to_keep: list[str],
        context: str,
        context_scores: list[float],
        other_context_scores: list[float] | None = None,
        is_target: bool = False,
        context_type: Literal["Input", "Output"] = "Input",
    ) -> str:
        context_tokens = get_filtered_tokens(
            context, model, special_tokens_to_keep, replace_special_characters=True, is_target=is_target
        )
        scores = context_scores
        if has_other_context:
            scores += other_context_scores
        context_ranked_tokens, threshold = filter_rank_tokens(
            tokens=context_tokens,
            scores=scores,
            std_threshold=args.attribution_std_threshold,
            topk=args.attribution_topk,
        )
        for idx, score, tok in context_ranked_tokens:
            context_tokens[idx] = f"[bold green]{tok}({score:.3f})[/bold green]"
        cci_threshold_comment = f"(CCI > {threshold:.3f})" if threshold is not None else ""
        return f"\n[bold]{context_type} context {cci_threshold_comment}:[/bold]\t{''.join(context_tokens)}"

    out_string = ""
    output_current_tokens = get_filtered_tokens(
        output.output_current, model, args.special_tokens_to_keep, replace_special_characters=True, is_target=True
    )
    cti_theshold_comment = f"(CTI > {cti_threshold:.3f})" if cti_threshold is not None else ""
    for example_idx, cci_out in enumerate(output.cci_scores, start=1):
        curr_output_tokens = output_current_tokens.copy()
        cti_idx = cci_out.cti_idx
        cti_score = cci_out.cti_score
        cti_tok = curr_output_tokens[cti_idx]
        curr_output_tokens[cti_idx] = f"[bold dodger_blue1]{cti_tok}({cti_score:.3f})[/bold dodger_blue1]"
        output_current_comment = "".join(curr_output_tokens)
        input_context_comment, output_context_comment = "", ""
        if args.has_input_context:
            input_context_comment = format_context_comment(
                model,
                args.has_output_context,
                args.special_tokens_to_keep,
                output.input_context,
                cci_out.input_context_scores,
                cci_out.output_context_scores,
            )
        if args.has_output_context:
            output_context_comment = format_context_comment(
                model,
                args.has_input_context,
                args.special_tokens_to_keep,
                output.output_context,
                cci_out.output_context_scores,
                cci_out.input_context_scores,
                is_target=True,
                context_type="Output",
            )
        out_string += (
            f"#{example_idx}."
            f"\n[bold]Generated output {cti_theshold_comment}:[/bold]\t{output_current_comment}"
            f"{input_context_comment}{output_context_comment}\n"
        )
    return out_string


@treescope_ignore
def visualize_attribute_context(
    output: AttributeContextOutput,
    model: HuggingfaceModel | str | None = None,
    cti_threshold: float | None = None,
    return_html: bool = False,
) -> str | None:
    if output.info is None:
        raise ValueError("Cannot visualize attribution results without args. Set add_output_info = True.")
    console = Console(record=True)
    if model is None:
        model = output.info.model_name_or_path
    if isinstance(model, str):
        model = load_model(
            output.info.model_name_or_path,
            output.info.attribution_method,
            model_kwargs=deepcopy(output.info.model_kwargs),
            tokenizer_kwargs=deepcopy(output.info.tokenizer_kwargs),
        )
    elif not isinstance(model, HuggingfaceModel):
        raise TypeError(f"Unsupported model type {type(model)} for visualization.")
    if cti_threshold is None and len(output.cti_scores) > 1:
        cti_threshold = get_scores_threshold(output.cti_scores, output.info.context_sensitivity_std_threshold)
    viz = get_formatted_procedure_details(output.info)
    viz += "\n\n" + get_formatted_attribute_context_results(model, output.info, output, cti_threshold)
    with console.capture() as _:
        console.print(viz, soft_wrap=False)
    if output.info.show_viz:
        console.print(viz, soft_wrap=False)
    html = console.export_html()
    if output.info.viz_path:
        with open(output.info.viz_path, "w", encoding="utf-8") as f:
            f.write(html)
    if return_html:
        return html
    return None
