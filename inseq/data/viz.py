# Adapted from https://github.com/slundberg/shap/blob/v0.39.0/shap/plots/_text.py, licensed MIT:
# Copyright © 2021 Scott Lundberg. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the “Software”), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import random
import string
from typing import TYPE_CHECKING, Literal

import numpy as np
import treescope as ts
import treescope.figures as fg
import treescope.rendering_parts as rp
from matplotlib.colors import Colormap
from rich import box
from rich.color import Color
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text
from tqdm.std import tqdm

from ..utils import isnotebook
from ..utils.misc import clean_tokens
from ..utils.typing import TextSequences
from ..utils.viz_utils import (
    final_plot_html,
    get_colors,
    get_instance_html,
    maybe_add_linebreak,
    red_transparent_blue_colormap,
    saliency_heatmap_html,
    saliency_heatmap_table_header,
    sanitize_html,
    test_dim,
    treescope_cmap,
)

if TYPE_CHECKING:
    from .attribution import FeatureAttributionSequenceOutput

if isnotebook():
    cmap = treescope_cmap()
    ts.basic_interactive_setup(autovisualize_arrays=True)
    ts.default_diverging_colormap.set_globally(cmap)
    ts.default_sequential_colormap.set_globally(cmap)


def show_attributions(
    attributions: "FeatureAttributionSequenceOutput",
    min_val: int | None = None,
    max_val: int | None = None,
    display: bool = True,
    return_html: bool | None = False,
) -> str | None:
    """Core function allowing for visualization of feature attribution maps in console/HTML format.

    Args:
        attributions (:class:`~inseq.data.attribution.FeatureAttributionSequenceOutput`):
            Sequence attributions to be visualized.
        min_val (:obj:`Optional[int]`, *optional*, defaults to None):
            Lower attribution score threshold for color map.
        max_val (`Optional[int]`, *optional*, defaults to None):
            Upper attribution score threshold for color map.
        display (`bool`, *optional*, defaults to True):
            Whether to show the output of the visualization function.
        return_html (`Optional[bool]`, *optional*, defaults to False):
            If true, returns the HTML corresponding to the notebook visualization of the attributions in string format,
            for saving purposes.

    Returns:
        `Optional[str]`: Returns the HTML output if `return_html=True`
    """
    from inseq.data.attribution import FeatureAttributionSequenceOutput

    if isinstance(attributions, FeatureAttributionSequenceOutput):
        attributions = [attributions]
    html_out = ""
    html_colors = get_attribution_colors(attributions, min_val, max_val, cmap=red_transparent_blue_colormap())
    if not isnotebook():
        colors = get_attribution_colors(attributions, min_val, max_val, return_alpha=False, return_strings=False)
    idx = 0
    for ex_id, attribution in enumerate(attributions):
        instance_html = get_instance_html(ex_id)
        curr_html = ""
        curr_html_color = None
        if attribution.source_attributions is not None:
            curr_html_color = html_colors[idx]
            curr_html += instance_html
            curr_html += get_heatmap_type(attribution, curr_html_color, "Source", use_html=True)
            if attribution.target_attributions is not None:
                curr_html_color = html_colors[idx + 1]
        display_scores = attribution.source_attributions is None and attribution.step_scores
        if attribution.target_attributions is not None or display_scores:
            if curr_html_color is None and html_colors:
                curr_html_color = html_colors[idx]
            curr_html += instance_html
            curr_html += get_heatmap_type(attribution, curr_html_color, "Target", use_html=True)
        if display and isnotebook():
            from IPython.core.display import HTML, display

            display(HTML(curr_html))
        html_out += curr_html
        if not isnotebook():
            console = Console()
            curr_color = None
            if attribution.source_attributions is not None:
                curr_color = colors[idx]
                if display:
                    print("\n\n")
                    console.print(
                        get_heatmap_type(attribution, curr_color, "Source", use_html=False), overflow="ignore"
                    )
                if attribution.target_attributions is not None:
                    curr_color = colors[idx + 1]
            display_scores = attribution.source_attributions is None and attribution.step_scores
            if (attribution.target_attributions is not None or display_scores) and display:
                if curr_color is None and colors:
                    curr_color = colors[idx]
                print("\n\n")
                console.print(get_heatmap_type(attribution, curr_color, "Target", use_html=False), overflow="ignore")
        if any(x is None for x in [attribution.source_attributions, attribution.target_attributions]):
            idx += 1
        else:
            idx += 2
    if return_html:
        return html_out


def show_granular_attributions(
    attributions: "FeatureAttributionSequenceOutput",
    max_show_size: int = 20,
    min_val: int | None = None,
    max_val: int | None = None,
    show_dim: int | str | None = None,
    slice_dims: dict[int | str, tuple[int, int]] | None = None,
    display: bool = True,
    return_html: bool | None = False,
    return_figure: bool = False,
) -> str | None:
    """Visualizes granular attribution heatmaps in HTML format.

    Args:
        attributions (:class:`~inseq.data.attribution.FeatureAttributionSequenceOutput`):
            Sequence attributions to be visualized. Does not require pre-aggregation.
        min_val (:obj:`int`, *optional*, defaults to None):
            Lower attribution score threshold for color map.
        max_val (:obj:`int`, *optional*, defaults to None):
            Upper attribution score threshold for color map.
        max_show_size (:obj:`int`, *optional*, defaults to None):
            Maximum dimension size for additional dimensions to be visualized. Default: 20.
        show_dim (:obj:`int` or :obj:`str`, *optional*, defaults to None):
            Dimension to be visualized along with the source and target tokens. Can be either the dimension index or
            the dimension name. Works only if the dimension size is less than or equal to `max_show_size`.
        slice_dims (:obj:`dict[int or str, tuple[int, int]]`, *optional*, defaults to None):
            Dimensions to be sliced and visualized along with the source and target tokens. The dictionary should
            contain the dimension index or name as the key and the slice range as the value.
        display (:obj:`bool`, *optional*, defaults to True):
            Whether to show the output of the visualization function.
        return_html (:obj:`bool`, *optional*, defaults to False):
            If true, returns the HTML corresponding to the notebook visualization of the attributions in
            string format, for saving purposes.
        return_figure (:obj:`bool`, *optional*, defaults to False):
            If true, returns the Treescope figure object for further manipulation.

    Returns:
        `str`: Returns the HTML output if `return_html=True`
    """
    from inseq.data.attribution import FeatureAttributionSequenceOutput

    if isinstance(attributions, FeatureAttributionSequenceOutput):
        attributions: list["FeatureAttributionSequenceOutput"] = [attributions]
    if not isnotebook() and display:
        raise ValueError(
            "Granular attribution heatmaps visualization is  only supported in Jupyter notebooks. "
            "Please set `display=False` and `return_html=True` to avoid this error."
        )
    if return_html and return_figure:
        raise ValueError("Only one of `return_html` and `return_figure` can be set to True.")
    items_to_render = []
    for attribution in attributions:
        if attribution.source_attributions is not None:
            items_to_render += [
                fg.bolded("Source Saliency Heatmap"),
                get_saliency_heatmap_treescope(
                    attribution.source_attributions.numpy(),
                    [t.token for t in attribution.target[attribution.attr_pos_start : attribution.attr_pos_end]],
                    [t.token for t in attribution.source],
                    attribution._attribution_dim_names["source_attributions"],
                    max_show_size=max_show_size,
                    max_val=max_val,
                    min_val=min_val,
                    show_dim=show_dim,
                    slice_dims=slice_dims,
                ),
            ]
        if attribution.target_attributions is not None:
            items_to_render += [
                fg.bolded("Target Saliency Heatmap"),
                get_saliency_heatmap_treescope(
                    attribution.target_attributions.numpy(),
                    [t.token for t in attribution.target[attribution.attr_pos_start : attribution.attr_pos_end]],
                    [t.token for t in attribution.target],
                    attribution._attribution_dim_names["target_attributions"],
                    max_show_size=max_show_size,
                    max_val=max_val,
                    min_val=min_val,
                    show_dim=show_dim,
                    slice_dims=slice_dims,
                ),
            ]
        items_to_render.append("")
    fig = fg.inline(*items_to_render)
    if return_figure:
        return fig
    if display:
        ts.show(fig)
    if return_html:
        return ts.render_to_html(fig)


def show_token_attributions(
    attributions: "FeatureAttributionSequenceOutput",
    min_val: int | None = None,
    max_val: int | None = None,
    display: bool = True,
    return_html: bool | None = False,
    return_figure: bool = False,
    replace_char: dict[str, str] | None = None,
    wrap_after: int | str | list[str] | tuple[str] | None = None,
    step_score_highlight: str | None = None,
):
    """Visualizes token-level attributions in HTML format.

    Args:
        attributions (:class:`~inseq.data.attribution.FeatureAttributionSequenceOutput`):
            Sequence attributions to be visualized.
        min_val (:obj:`Optional[int]`, *optional*, defaults to None):
            Lower attribution score threshold for color map.
        max_val (`Optional[int]`, *optional*, defaults to None):
            Upper attribution score threshold for color map.
        display (`bool`, *optional*, defaults to True):
            Whether to show the output of the visualization function.
        return_html (`Optional[bool]`, *optional*, defaults to False):
            If true, returns the HTML corresponding to the notebook visualization of the attributions in string format,
            for saving purposes.
        return_figure (`Optional[bool]`, *optional*, defaults to False):
            If true, returns the Treescope figure object for further manipulation.
        replace_char (`Optional[dict[str, str]]`, *optional*, defaults to None):
            Dictionary mapping strings to be replaced to replacement options, used for cleaning special characters.
            Default: {}.
        wrap_after (`Optional[int | str | list[str] | tuple[str]]`, *optional*, defaults to None):
            Token indices or tokens after which to wrap lines. E.g. 10 = wrap after every 10 tokens, "hi" = wrap after
            word hi occurs, ["." "!", "?"] or ".!?" = wrap after every sentence-ending punctuation.
        step_score_highlight (`Optional[str]`, *optional*, defaults to None):
            Name of the step score to use to highlight generated tokens in the visualization. If None, no highlights are
            shown. Default: None.
    """
    from inseq.data.attribution import FeatureAttributionSequenceOutput

    if isinstance(attributions, FeatureAttributionSequenceOutput):
        attributions: list["FeatureAttributionSequenceOutput"] = [attributions]
    if not isnotebook() and display:
        raise ValueError(
            "Token attribution visualization is only supported in Jupyter notebooks. "
            "Please set `display=False` and `return_html=True` to avoid this error."
        )
    if return_html and return_figure:
        raise ValueError("Only one of `return_html` and `return_figure` can be set to True.")
    if replace_char is None:
        replace_char = {}
    if max_val is None:
        max_val = max(attribution.maximum for attribution in attributions)
    if step_score_highlight is not None and (
        attributions[0].step_scores is None or step_score_highlight not in attributions[0].step_scores
    ):
        raise ValueError(
            f'The requested step score "{step_score_highlight}" is not available for highlights in the provided '
            "attribution object. Please set `step_score_highlight=None` or recompute `model.attribute` by passing "
            f'`step_scores=["{step_score_highlight}"].'
        )
    generated_token_parts = []
    for attr in attributions:
        cleaned_generated_tokens = clean_tokens(
            [t.token for t in attr.target[attr.attr_pos_start : attr.attr_pos_end]], replace_chars=replace_char
        )
        cleaned_input_tokens = clean_tokens([t.token for t in attr.source], replace_chars=replace_char)
        cleaned_target_tokens = clean_tokens([t.token for t in attr.target], replace_chars=replace_char)
        step_scores = None
        title = "Generated text:\n\n"
        if step_score_highlight is not None:
            step_scores = attr.step_scores[step_score_highlight]
            scores_vmax = step_scores.max().item()
            # Use different cmap to differentiate from attribution scores
            scores_cmap = (
                treescope_cmap("greens") if all(x >= 0 for x in step_scores) else treescope_cmap("brown_to_green")
            )
            title = f"Generated text with {step_score_highlight} highlights:\n\n"
        generated_token_parts.append(rp.custom_style(rp.text(title), css_style="font-weight: bold;"))
        for gen_idx, curr_gen_tok in enumerate(cleaned_generated_tokens):
            attributed_token_parts = [rp.text("\n")]
            if attr.source_attributions is not None:
                attributed_token_parts.append(
                    get_tokens_heatmap_treescope(
                        tokens=cleaned_input_tokens,
                        scores=attr.source_attributions[:, gen_idx].numpy(),
                        title=f'Source attributions for "{curr_gen_tok}"',
                        title_style="font-style: italic; color: #888888;",
                        min_val=min_val,
                        max_val=max_val,
                        wrap_after=wrap_after,
                    )
                )
                attributed_token_parts.append(rp.text("\n\n"))
            if attr.target_attributions is not None:
                attributed_token_parts.append(
                    get_tokens_heatmap_treescope(
                        tokens=cleaned_target_tokens[: attr.attr_pos_start + gen_idx],
                        scores=attr.target_attributions[:, gen_idx].numpy(),
                        title=f'Target attributions for "{curr_gen_tok}"',
                        title_style="font-style: italic; color: #888888;",
                        min_val=min_val,
                        max_val=max_val,
                        wrap_after=wrap_after,
                    )
                )
                attributed_token_parts.append(rp.text("\n\n"))
            if step_scores is not None:
                gen_tok_label = get_single_token_heatmap_treescope(
                    curr_gen_tok,
                    step_scores[gen_idx].item(),
                    max_val=scores_vmax,
                    colormap=scores_cmap,
                    show_empty_tokens=True,
                )[0]
            else:
                gen_tok_label = rp.text(curr_gen_tok)
            generated_token_parts.append(
                rp.build_full_line_with_annotations(
                    rp.build_custom_foldable_tree_node(
                        label=gen_tok_label,
                        contents=rp.fold_condition(
                            collapsed=rp.text(" "),
                            expanded=rp.indented_children([rp.siblings(*attributed_token_parts)]),
                        ),
                    )
                )
            )
    fig = fg.figure_from_treescope_rendering_part(
        rp.custom_style(rp.siblings(*generated_token_parts), css_style="white-space: pre-wrap")
    )
    if return_figure:
        return fig
    if display:
        ts.show(fig)
    if return_html:
        return ts.render_to_html(fig)


def get_attribution_colors(
    attributions: list["FeatureAttributionSequenceOutput"],
    min_val: int | None = None,
    max_val: int | None = None,
    cmap: str | Colormap | None = None,
    return_alpha: bool = True,
    return_strings: bool = True,
) -> list[list[list[str | tuple[float, float, float]]]]:
    """A list (one element = one sentence) of lists (one element = attributions for one token)
    of lists (one element = one attribution) of colors. Colors are either strings or RGB(A) tuples.
    """
    if max_val is None:
        max_val = max(attribution.maximum for attribution in attributions)
    if min_val is None:
        min_val = -max_val
    colors = []
    for attribution in attributions:
        if attribution.source_attributions is not None:
            colors.append(
                get_colors(
                    attribution.source_attributions.numpy(), min_val, max_val, cmap, return_alpha, return_strings
                )
            )
        if attribution.target_attributions is not None:
            colors.append(
                get_colors(
                    attribution.target_attributions.numpy(), min_val, max_val, cmap, return_alpha, return_strings
                )
            )
    return colors


def get_heatmap_type(
    attribution: "FeatureAttributionSequenceOutput",
    colors,
    heatmap_type: Literal["Source", "Target"] = "Source",
    use_html: bool = False,
) -> str:
    heatmap_func = get_saliency_heatmap_html if use_html else get_saliency_heatmap_rich
    step_scores = None
    if attribution.step_scores is not None:
        step_scores = {k: v.numpy() for k, v in attribution.step_scores.items()}
    if heatmap_type == "Source":
        return heatmap_func(
            attribution.source_attributions.numpy(),
            [t.token for t in attribution.target[attribution.attr_pos_start : attribution.attr_pos_end]],  # noqa
            [t.token for t in attribution.source],
            colors,
            step_scores,
            label="Source",
        )
    elif heatmap_type == "Target":
        if attribution.target_attributions is not None:
            target_attributions = attribution.target_attributions.numpy()
        else:
            target_attributions = None
        return heatmap_func(
            target_attributions,
            [t.token for t in attribution.target[attribution.attr_pos_start : attribution.attr_pos_end]],  # noqa
            [t.token for t in attribution.target],
            colors,
            step_scores,
            label="Target",
        )
    else:
        raise ValueError(f"Type {heatmap_type} is not supported.")


def get_saliency_heatmap_html(
    scores: np.ndarray | None,
    column_labels: list[str],
    row_labels: list[str],
    input_colors: list[list[str]],
    step_scores: dict[str, np.ndarray] | None = None,
    label: str = "",
    step_scores_threshold: float | dict[str, float] = 0.5,
):
    # unique ID added to HTML elements and function to avoid collision of differnent instances
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    out = saliency_heatmap_table_header
    # add top row containing target tokens
    out += "<tr><th></th><th></th>"
    for column_idx in range(len(column_labels)):
        out += f"<th>{column_idx}</th>"
    out += "</tr><tr><th></th><th></th>"
    for column_label in column_labels:
        out += f"<th>{sanitize_html(column_label)}</th>"
    out += "</tr>"
    if scores is not None:
        for row_index in range(scores.shape[0]):
            out += f"<tr><th>{row_index}</th><th>{sanitize_html(row_labels[row_index])}</th>"
            for col_index in range(scores.shape[1]):
                score = ""
                if not np.isnan(scores[row_index, col_index]):
                    score = round(float(scores[row_index][col_index]), 3)
                out += f'<th style="background:{input_colors[row_index][col_index]}">{score}</th>'
            out += "</tr>"
    if step_scores is not None:
        for step_score_name, step_score_values in step_scores.items():
            out += f'<tr style="outline: thin solid"><th></th><th><b>{step_score_name}</b></th>'
            if isinstance(step_scores_threshold, float):
                threshold = step_scores_threshold
            else:
                threshold = step_scores_threshold.get(step_score_name, 0.5)
            style = lambda val, limit: abs(val) >= limit and isinstance(val, float)
            for col_index in range(len(column_labels)):
                if isinstance(step_score_values[col_index].item(), float):
                    score = round(step_score_values[col_index].item(), 3)
                else:
                    score = step_score_values[col_index].item()
                is_bold = style(score, threshold)
                out += f'<th>{"<b>" if is_bold else ""}{score}{"</b>" if is_bold else ""}</th>'
    out += "</table>"
    saliency_heatmap_markup = saliency_heatmap_html.format(uuid=uuid, content=out, label=label)
    plot_uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    return final_plot_html.format(
        uuid=plot_uuid,
        saliency_plot_markup=saliency_heatmap_markup,
    )


def get_saliency_heatmap_rich(
    scores: np.ndarray | None,
    column_labels: list[str],
    row_labels: list[str],
    input_colors: list[list[str]],
    step_scores: dict[str, np.ndarray] | None = None,
    label: str = "",
    step_scores_threshold: float | dict[str, float] = 0.5,
):
    columns = [
        Column(header="", justify="right", overflow="fold"),
        Column(header="", justify="right", overflow="fold"),
    ]
    for idx, column_label in enumerate(column_labels):
        columns.append(Column(header=f"{idx}\n{escape(column_label)}", justify="center", overflow="fold"))
    table = Table(
        *columns,
        title=f"{label + ' ' if label else ''}Saliency Heatmap",
        caption="→ : Generated tokens, ↓ : Attributed tokens",
        padding=(0, 1, 0, 1),
        show_lines=False,
        box=box.HEAVY_HEAD,
    )
    if scores is not None:
        for row_index in range(scores.shape[0]):
            row = [Text(f"{row_index}", style="bold"), Text(escape(row_labels[row_index]), style="bold")]
            for col_index in range(scores.shape[1]):
                color = Color.from_rgb(*input_colors[row_index][col_index])
                score = ""
                if not np.isnan(scores[row_index][col_index]):
                    score = round(float(scores[row_index][col_index]), 2)
                row.append(Text(f"{score}", justify="center", style=Style(color=color)))
            table.add_row(*row, end_section=row_index == scores.shape[0] - 1)
    if step_scores is not None:
        for step_score_name, step_score_values in step_scores.items():
            if isinstance(step_scores_threshold, float):
                threshold = step_scores_threshold
            else:
                threshold = step_scores_threshold.get(step_score_name, 0.5)
            style = lambda val, limit: "bold" if abs(val) >= limit and isinstance(val, float) else ""
            score_row = [Text(""), Text(escape(step_score_name), style="bold")]
            for score in step_score_values:
                curr_score = round(score.item(), 2) if isinstance(score, float) else score.item()
                score_row.append(Text(f"{score:.2f}", justify="center", style=style(curr_score, threshold)))
            table.add_row(*score_row, end_section=True)
    return table


def get_saliency_heatmap_treescope(
    scores: np.ndarray | None,
    column_labels: list[str],
    row_labels: list[str],
    dim_names: dict[int, str] | None = None,
    max_show_size: int | None = None,
    max_val: float | None = None,
    min_val: float | None = None,
    show_dim: int | str | None = None,
    slice_dims: dict[int | str, tuple[int, int]] | None = None,
):
    if max_show_size is None:
        max_show_size = 20
    if dim_names is None:
        dim_names = {}
    item_labels_dict = {0: row_labels, 1: column_labels}
    rev_dim_names = {v: k for k, v in dim_names.items()}
    col_dims = [1]
    slider_dims = []
    if slice_dims is not None:
        slices = [slice(None)] * scores.ndim
        for dim_name, slice_idxs in slice_dims.items():
            dim_idx = test_dim(dim_name, dim_names, rev_dim_names, scores)
            slices[dim_idx] = slice(slice_idxs[0], slice_idxs[1])
        scores = scores[tuple(slices)]
    if show_dim is not None:
        show_dim_idx = test_dim(show_dim, dim_names, rev_dim_names, scores)
        if scores.shape[show_dim_idx] > max_show_size:
            raise ValueError(
                f"Dimension {show_dim_idx} has size {scores.shape[show_dim_idx]} which is greater than the maximum "
                f"show size {max_show_size}. Please choose a different dimension or slice the tensor before "
                "visualizing it using SliceAggregator."
            )
        col_dims.append(show_dim_idx)
    for dim_idx, dim_name in dim_names.items():
        if dim_idx > 1:
            if scores.shape[dim_idx] <= max_show_size and len(col_dims) < 2:
                col_dims.append(dim_idx)
            else:
                slider_dims.append(dim_idx)
            item_labels_dict[dim_idx] = [f"{dim_name} #{i}" for i in range(scores.shape[dim_idx])]
    return ts.render_array(
        scores,
        rows=[0],
        columns=col_dims,
        sliders=slider_dims,
        axis_labels={k: f"{v}: {scores.shape[k]}" for k, v in dim_names.items()},
        axis_item_labels=item_labels_dict,
        vmax=max_val,
        vmin=min_val,
    )


def get_single_token_heatmap_treescope(
    token: str,
    score: float,
    min_val: float | None = None,
    max_val: float = 1,
    rounding: int = 4,
    colormap: list[tuple[int, int, int]] | None = None,
    strip_chars: dict[str, str] = {},
    show_empty_tokens: bool = False,
    return_highlighted_idx: bool = False,
) -> list[rp.RenderableTreePart] | tuple[list[rp.RenderableTreePart], int]:
    parts = [None]
    idx_highlight = 0
    curr_tok = token
    for char, repl in strip_chars.items():
        if curr_tok.startswith(char):
            curr_tok = curr_tok.lstrip(char)
            parts = [rp.text(repl)] + parts
            idx_highlight += 1
        if curr_tok.endswith(char):
            curr_tok = curr_tok.rstrip(char)
            parts.append(rp.text(repl))
    if (show_empty_tokens and token != "") or curr_tok != "":
        show_token = token if show_empty_tokens and curr_tok == "" else curr_tok
        highlighted_text = fg.treescope_part_from_display_object(
            fg.text_on_color(show_token, value=round(score, rounding), vmin=min_val, vmax=max_val, colormap=colormap)
        )
        parts[idx_highlight] = highlighted_text
    else:
        parts.pop(idx_highlight)
    if return_highlighted_idx:
        return parts, idx_highlight, show_token
    return parts


def get_tokens_heatmap_treescope(
    tokens: list[str],
    scores: np.ndarray,
    title: str | None = None,
    title_style: str | None = None,
    min_val: float | None = None,
    max_val: float = 1,
    rounding: int = 4,
    wrap_after: int | str | list[str] | tuple[str] | None = None,
    colormap: str | list[tuple[int, int, int]] = "blue_to_red",
    strip_chars: dict[str, str] = {},
    show_empty_tokens: bool = True,
):
    parts = []
    if title is not None:
        parts.append(rp.custom_style(rp.text(title + ":\n"), css_style=title_style))
    if isinstance(colormap, str):
        colormap = treescope_cmap(colormap)
    if not isinstance(colormap, list):
        raise ValueError("If specified, colormap must be a string or a list of RGB tuples.")
    for tok_idx, tok in enumerate(tokens):
        parts += get_single_token_heatmap_treescope(
            tok,
            scores[tok_idx],
            min_val=min_val,
            max_val=max_val,
            rounding=rounding,
            colormap=colormap,
            strip_chars=strip_chars,
            show_empty_tokens=show_empty_tokens,
        )
        parts += maybe_add_linebreak(tok, tok_idx, wrap_after)
    return rp.siblings(*parts)


# Progress bar utilities


def get_progress_bar(
    sequences: TextSequences,
    target_lengths: list[int],
    method_name: str,
    show: bool,
    pretty: bool,
    attr_pos_start: int,
    attr_pos_end: int,
) -> tqdm | tuple[Progress, Live] | None:
    if not show:
        return None
    elif show and not pretty:
        return tqdm(
            total=attr_pos_end,
            desc=f"Attributing with {method_name}...",
            initial=attr_pos_start,
        )
    elif show and pretty:
        job_progress = Progress(
            TextColumn("{task.description}", table_column=Column(ratio=3, no_wrap=False)),
            BarColumn(table_column=Column(ratio=1)),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        for idx, (tgt, tgt_len) in enumerate(zip(sequences.targets, target_lengths, strict=False)):
            clean_tgt = escape(tgt.replace("\n", "\\n"))
            job_progress.add_task(f"{idx}. {clean_tgt}", total=tgt_len)
        progress_table = Table.grid()
        row_contents = [
            Panel.fit(
                job_progress,
                title=f"[b]Attributing with {escape(method_name)}",
                border_style="green",
                padding=(1, 2),
            )
        ]
        if sequences.sources is not None:
            sources = []
            for idx, src in enumerate(sequences.sources):
                clean_src = escape(src.replace("\n", "\\n"))
                sources.append(f"{idx}. {clean_src}")
            row_contents = [
                Panel.fit(
                    "\n".join(sources),
                    title="Source sentences",
                    border_style="red",
                    padding=(1, 2),
                )
            ] + row_contents
        progress_table.add_row(*row_contents)
        live = Live(Padding(progress_table, (1, 0, 1, 0)), refresh_per_second=10)
        live.start(refresh=live._renderable is not None)
        return job_progress, live


def update_progress_bar(
    pbar: tqdm | tuple[Progress, Live] | None,
    skipped_prefixes: list[str] | None = None,
    attributed_sentences: list[str] | None = None,
    unattributed_suffixes: list[str] | None = None,
    skipped_suffixes: list[str] | None = None,
    whitespace_indexes: list[list[int]] = None,
    show: bool = False,
    pretty: bool = False,
) -> None:
    if not show:
        return
    elif show and not pretty:
        pbar.update(1)
    else:
        split_targets = (skipped_prefixes, attributed_sentences, unattributed_suffixes, skipped_suffixes)
        for job in pbar[0].tasks:
            if not job.finished:
                pbar[0].advance(job.id)
                formatted_desc = f"{job.id}. "
                past_length = 0
                for split, color in zip(split_targets, ["grey58", "green", "orange1", "grey58"], strict=False):
                    if split[job.id]:
                        formatted_desc += f"[{color}]" + escape(split[job.id].replace("\n", "\\n")) + "[/]"
                        past_length += len(split[job.id])
                        if past_length in whitespace_indexes[job.id]:
                            formatted_desc += " "
                            past_length += 1
                pbar[0].update(job.id, description=formatted_desc, refresh=True)


def close_progress_bar(pbar: tqdm | tuple[Progress, Live] | None, show: bool, pretty: bool) -> None:
    if not show:
        return
    elif show and not pretty:
        pbar.close()
    else:
        _, live = pbar
        live.stop()
