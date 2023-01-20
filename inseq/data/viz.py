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
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from matplotlib.colors import Colormap
from rich import box
from rich import print as rprint
from rich.color import Color
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text
from tqdm.std import tqdm

from ..utils import isnotebook
from ..utils.typing import TextSequences
from ..utils.viz_utils import (
    final_plot_html,
    get_colors,
    get_instance_html,
    red_transparent_blue_colormap,
    saliency_heatmap_html,
    saliency_heatmap_table_header,
    sanitize_html,
)
from .attribution import FeatureAttributionSequenceOutput


def show_attributions(
    attributions: FeatureAttributionSequenceOutput,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    display: bool = True,
    return_html: Optional[bool] = False,
) -> Optional[str]:
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
        curr_html_color = html_colors[idx]
        if attribution.source_attributions is not None:
            curr_html += instance_html
            curr_html += get_heatmap_type(attribution, curr_html_color, "Source", use_html=True)
            if attribution.target_attributions is not None:
                curr_html_color = html_colors[idx + 1]
        if attribution.target_attributions is not None:
            curr_html += instance_html
            curr_html += get_heatmap_type(attribution, curr_html_color, "Target", use_html=True)
        if display and isnotebook():
            from IPython.core.display import HTML, display

            display(HTML(curr_html))
        html_out += curr_html
        if not isnotebook():
            curr_color = colors[idx]
            if attribution.source_attributions is not None:
                if display:
                    print("\n\n")
                    rprint(get_heatmap_type(attribution, curr_color, "Source", use_html=False))
                if attribution.target_attributions is not None:
                    curr_color = colors[idx + 1]
            if attribution.target_attributions is not None and display:
                print("\n\n")
                rprint(get_heatmap_type(attribution, curr_color, "Target", use_html=False))
        if any(x is None for x in [attribution.source_attributions, attribution.target_attributions]):
            idx += 1
        else:
            idx += 2
    if return_html:
        return html_out


def get_attribution_colors(
    attributions: List[FeatureAttributionSequenceOutput],
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    cmap: Union[str, Colormap, None] = None,
    return_alpha: bool = True,
    return_strings: bool = True,
) -> List[List[List[Union[str, Tuple[float, float, float]]]]]:
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
    attribution: FeatureAttributionSequenceOutput,
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
        mask = np.ones_like(attribution.target_attributions.numpy()) * float("nan")
        mask = np.tril(mask, k=-attribution.attr_pos_start)
        return heatmap_func(
            attribution.target_attributions.numpy() + mask,
            [t.token for t in attribution.target[attribution.attr_pos_start : attribution.attr_pos_end]],  # noqa
            [t.token for t in attribution.target],
            colors,
            step_scores,
            label="Target",
        )
    else:
        raise ValueError(f"Type {heatmap_type} is not supported.")


def get_saliency_heatmap_html(
    scores: np.ndarray,
    column_labels: List[str],
    row_labels: List[str],
    input_colors: List[List[str]],
    step_scores: Optional[Dict[str, np.ndarray]] = None,
    label: str = "",
    step_scores_threshold: Union[float, Dict[str, float]] = 0.5,
):
    # unique ID added to HTML elements and function to avoid collision of differnent instances
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    out = saliency_heatmap_table_header
    # add top row containing target tokens
    for head_id in range(scores.shape[1]):
        out += f"<th>{sanitize_html(column_labels[head_id])}</th>"
    out += "</tr>"
    for row_index in range(scores.shape[0]):
        out += f"<tr><th>{sanitize_html(row_labels[row_index])}</th>"
        for col_index in range(scores.shape[1]):
            score = ""
            if not np.isnan(scores[row_index, col_index]):
                score = round(float(scores[row_index][col_index]), 3)
            out += f'<th style="background:{input_colors[row_index][col_index]}">{score}</th>'
        out += "</tr>"
    if step_scores is not None:
        for step_score_name, step_score_values in step_scores.items():
            out += f'<tr style="outline: thin solid"><th><b>{step_score_name}</b></th>'
            if isinstance(step_scores_threshold, float):
                threshold = step_scores_threshold
            else:
                threshold = step_scores_threshold.get(step_score_name, 0.5)
            style = lambda val, limit: abs(val) >= limit
            for col_index in range(scores.shape[1]):
                score = round(float(step_score_values[col_index]), 3)
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
    scores: np.ndarray,
    column_labels: List[str],
    row_labels: List[str],
    input_colors: List[List[str]],
    step_scores: Optional[Dict[str, np.ndarray]] = None,
    label: str = "",
    step_scores_threshold: Union[float, Dict[str, float]] = 0.5,
):
    columns = [Column(header="", justify="right")]
    for head_id in range(scores.shape[1]):
        columns.append(Column(header=column_labels[head_id], justify="center"))
    table = Table(
        *columns,
        title=f"{label + ' ' if label else ''}Saliency Heatmap",
        caption="x: Generated tokens, y: Attributed tokens",
        padding=(0, 1, 0, 1),
        show_lines=False,
        box=box.HEAVY_HEAD,
    )
    for row_index in range(scores.shape[0]):
        row = [Text(row_labels[row_index], style="bold")]
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
            style = lambda val, limit: "bold" if abs(val) >= limit else ""
            score_row = [Text(step_score_name, style="bold")]
            for score in step_score_values:
                score_row.append(
                    Text(f"{score:.2f}", justify="center", style=style(round(float(score), 2), threshold))
                )
            table.add_row(*score_row, end_section=True)
    return table


# Progress bar utilities


def get_progress_bar(
    sequences: TextSequences,
    target_lengths: List[int],
    method_name: str,
    show: bool,
    pretty: bool,
    attr_pos_start: int,
    attr_pos_end: int,
) -> Union[tqdm, Tuple[Progress, Live], None]:
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
        for idx, (tgt, tgt_len) in enumerate(zip(sequences.targets, target_lengths)):
            clean_tgt = tgt.replace("\n", "\\n")
            job_progress.add_task(f"{idx}. {clean_tgt}", total=tgt_len)
        progress_table = Table.grid()
        row_contents = [
            Panel.fit(
                job_progress,
                title=f"[b]Attributing with {method_name}",
                border_style="green",
                padding=(1, 2),
            )
        ]
        if sequences.sources is not None:
            sources = []
            for idx, src in enumerate(sequences.sources):
                clean_src = src.replace("\n", "\\n")
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
    pbar: Union[tqdm, Tuple[Progress, Live], None],
    skipped_prefixes: Optional[List[str]] = None,
    attributed_sentences: Optional[List[str]] = None,
    unattributed_suffixes: Optional[List[str]] = None,
    skipped_suffixes: Optional[List[str]] = None,
    whitespace_indexes: List[List[int]] = None,
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
                for split, color in zip(split_targets, ["grey58", "green", "orange1", "grey58"]):
                    if split[job.id]:
                        formatted_desc += f"[{color}]" + split[job.id].replace("\n", "\\n") + "[/]"
                        past_length += len(split[job.id])
                        if past_length in whitespace_indexes[job.id]:
                            formatted_desc += " "
                            past_length += 1
                pbar[0].update(job.id, description=formatted_desc, refresh=True)


def close_progress_bar(pbar: Union[tqdm, Tuple[Progress, Live], None], show: bool, pretty: bool) -> None:
    if not show:
        return
    elif show and not pretty:
        pbar.close()
    else:
        _, live = pbar
        live.stop()
