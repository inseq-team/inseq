from copy import deepcopy
from typing import Literal

import treescope as ts
import treescope.figures as fg
import treescope.rendering_parts as rp
from rich.console import Console

from ... import load_model
from ...data.viz import get_single_token_heatmap_treescope, get_tokens_heatmap_treescope
from ...models import HuggingfaceModel
from ...utils.misc import isnotebook
from ...utils.viz_utils import treescope_cmap, treescope_ignore
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
def visualize_attribute_context_rich(
    output: AttributeContextOutput,
    model: HuggingfaceModel | str | None = None,
    cti_threshold: float | None = None,
    return_html: bool = False,
    show_viz: bool | None = None,
    viz_path: str | None = None,
) -> str | None:
    if output.info is None:
        raise ValueError("Cannot visualize attribution results without args. Set add_output_info = True.")
    if show_viz is None:
        show_viz = output.info.show_viz
    if viz_path is None:
        viz_path = output.info.viz_path
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
    if show_viz:
        console.print(viz, soft_wrap=False)
    html = console.export_html()
    if viz_path:
        with open(viz_path, "w", encoding="utf-8") as f:
            f.write(html)
    if return_html:
        return html
    return None


def visualize_attribute_context_treescope(
    output: AttributeContextOutput,
    return_html: bool = False,
    show_viz: bool = False,
    viz_path: str | None = None,
) -> str | rp.RenderableTreePart:
    if output.info is None:
        raise ValueError("Cannot visualize attribution results without args. Set add_output_info = True.")
    cmap_cti = treescope_cmap("greens")
    cmap_cci = treescope_cmap("blues")
    parts = [
        fg.treescope_part_from_display_object(
            fg.text_on_color("Context-sensitive tokens", value=1, colormap=cmap_cti)
        ),
        rp.text(" in the generated output can be expanded to visualize the "),
        fg.treescope_part_from_display_object(fg.text_on_color("contextual cues", value=1, colormap=cmap_cci)),
        rp.text(" motivating their prediction.\n\n"),
    ]
    if output.info.context_sensitivity_std_threshold is not None:
        cti_threshold = round(output.mean_cti + (output.std_cti * output.info.context_sensitivity_std_threshold), 4)
    parts += [
        rp.build_full_line_with_annotations(
            rp.build_custom_foldable_tree_node(
                label=rp.custom_style(
                    fg.treescope_part_from_display_object(fg.text_on_color("Parameters", value=0)),
                    css_style="font-weight: bold;",
                ),
                contents=rp.fold_condition(
                    collapsed=rp.empty_part(),
                    expanded=rp.indented_children(
                        [
                            rp.custom_style(rp.text("Model: "), css_style="font-weight: bold;"),
                            rp.indented_children([rp.text(output.info.model_name_or_path + "\n")]),
                            rp.custom_style(rp.text("Context sensitivity metric: "), css_style="font-weight: bold;"),
                            rp.indented_children([rp.text(output.info.context_sensitivity_metric + "\n")]),
                            rp.custom_style(rp.text("Attribution method: "), css_style="font-weight: bold;"),
                            rp.indented_children([rp.text(output.info.attribution_method + "\n")]),
                            rp.custom_style(rp.text("Attributed function: "), css_style="font-weight: bold;"),
                            rp.indented_children([rp.text(output.info.attributed_fn + "\n")]),
                            rp.custom_style(
                                rp.text("Context sensitivity selection: "), css_style="font-weight: bold;"
                            ),
                            rp.indented_children(
                                [
                                    rp.text(
                                        f"|x| ≥ {cti_threshold} (Mean ± {output.info.context_sensitivity_std_threshold} standard deviation)\n"
                                        if output.info.context_sensitivity_std_threshold is not None
                                        else f"Top {output.info.context_sensitivity_topk} scores\n"
                                        if output.info.context_sensitivity_topk is not None
                                        else "All scores\n"
                                    )
                                ]
                            ),
                        ]
                    ),
                ),
                expand_state=rp.ExpandState.COLLAPSED,
            )
        ),
        rp.text("\n\n"),
    ]
    if output.input_context is not None:
        if len(output.input_context) > 1000 or "\n" in output.input_context:
            parts += [
                rp.build_full_line_with_annotations(
                    rp.build_custom_foldable_tree_node(
                        label=rp.custom_style(rp.text("Input context: "), css_style="font-weight: bold;"),
                        contents=rp.fold_condition(
                            collapsed=rp.custom_style(
                                rp.text(
                                    output.input_context[:100].replace("\n", " ")
                                    + ("..." if len(output.input_context) > 100 else "")
                                ),
                                css_style="font-style: italic; color: #888888;",
                            ),
                            expanded=rp.indented_children([rp.text(output.input_context)]),
                        ),
                        expand_state=rp.ExpandState.COLLAPSED,
                    )
                ),
                rp.text("\n"),
            ]
        else:
            parts += [
                rp.custom_style(rp.text(" Input context: "), css_style="font-weight: bold;"),
                rp.text(output.input_context + "\n"),
            ]
    parts += [
        rp.custom_style(rp.text(" Input current: "), css_style="font-weight: bold;"),
        rp.text(output.info.input_current_text + "\n"),
    ]
    if output.output_context is not None:
        if len(output.output_context) > 1000 or "\n" in output.output_context:
            parts += [
                rp.build_full_line_with_annotations(
                    rp.build_custom_foldable_tree_node(
                        label=rp.custom_style(rp.text("Output context: "), css_style="font-weight: bold;"),
                        contents=rp.fold_condition(
                            collapsed=rp.custom_style(
                                rp.text(
                                    output.output_context[:100].replace("\n", " ")
                                    + ("..." if len(output.output_context) > 100 else "")
                                ),
                                css_style="font-style: italic; color: #888888;",
                            ),
                            expanded=rp.indented_children([rp.text(output.output_context)]),
                        ),
                        expand_state=rp.ExpandState.COLLAPSED,
                    )
                ),
                rp.text("\n"),
            ]
        else:
            parts += [
                rp.custom_style(rp.text(" Output context: "), css_style="font-weight: bold;"),
                rp.text(output.output_context + "\n"),
            ]
    parts += [rp.custom_style(rp.text("\n Output current: "), css_style="font-weight: bold;")]
    replace_chars = {"Ġ": " ", "Ċ": "\n", "▁": " "}
    cci_idx_map = {cci.cti_idx: cci for cci in output.cci_scores} if output.cci_scores is not None else {}
    for curr_tok_idx, curr_tok in enumerate(output.output_current_tokens):
        curr_tok_parts, highlighted_idx = get_single_token_heatmap_treescope(
            curr_tok,
            score=output.cti_scores[curr_tok_idx],
            max_val=output.max_cti,
            colormap=cmap_cti,
            strip_chars=replace_chars,
            show_empty_tokens=True,
            return_highlighted_idx=True,
        )
        if curr_tok_idx in cci_idx_map:
            cci_parts = [rp.text("\n")]
            cci = cci_idx_map[curr_tok_idx]
            if cci.input_context_scores is not None:
                cci_parts.append(
                    get_tokens_heatmap_treescope(
                        tokens=output.input_context_tokens,
                        scores=cci.input_context_scores,
                        title=f'Input context CCI scores for "{cci.cti_token}"',
                        title_style="font-style: italic; color: #888888;",
                        min_val=output.min_cci,
                        max_val=output.max_cci,
                        rounding=10,
                        colormap=cmap_cci,
                        strip_chars=replace_chars,
                    )
                )
                cci_parts.append(rp.text("\n\n"))
            if cci.output_context_scores is not None:
                cci_parts.append(
                    get_tokens_heatmap_treescope(
                        tokens=output.output_context_tokens,
                        scores=cci.output_context_scores,
                        title=f'Output context CCI scores for "{cci.cti_token}"',
                        title_style="font-style: italic; color: #888888;",
                        min_val=output.min_cci,
                        max_val=output.max_cci,
                        rounding=10,
                        colormap=cmap_cci,
                        strip_chars=replace_chars,
                    )
                )
                cci_parts.append(rp.text("\n\n"))
            curr_tok_parts[highlighted_idx] = rp.custom_style(
                rp.build_full_line_with_annotations(
                    rp.build_custom_foldable_tree_node(
                        label=curr_tok_parts[highlighted_idx],
                        contents=rp.fold_condition(
                            collapsed=rp.empty_part(),
                            expanded=rp.indented_children([rp.siblings(*cci_parts)]),
                        ),
                    )
                ),
                css_style="margin-left: 0.7em;",
            )
        parts += curr_tok_parts
    out_tree = rp.custom_style(rp.siblings(*parts), css_style="white-space: pre-wrap")
    with ts.active_autovisualizer.set_scoped(ts.ArrayAutovisualizer()):
        fig = fg.figure_from_treescope_rendering_part(out_tree)
        if show_viz:
            import IPython

            IPython.display.display(fig)
        html = ts.lowering.render_to_html_as_root(out_tree)
        if viz_path:
            with open(viz_path, "w", encoding="utf-8") as f:
                f.write(html)
    if return_html:
        return html
    return out_tree


def visualize_attribute_context(
    output: AttributeContextOutput,
    model: HuggingfaceModel | str | None = None,
    cti_threshold: float | None = None,
    show_viz: bool = True,
    viz_path: str | None = None,
    return_html: bool = False,
) -> str | None:
    if isnotebook() or not show_viz:
        return visualize_attribute_context_treescope(output, return_html, show_viz=show_viz, viz_path=viz_path)
    return visualize_attribute_context_rich(output, model, cti_threshold, return_html, show_viz, viz_path)
