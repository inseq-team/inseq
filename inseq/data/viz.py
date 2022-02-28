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

from typing import List, Optional, Tuple, Union

import random
import string

import numpy as np
from matplotlib.colors import Colormap
from rich import box
from rich import print as rprint
from rich.color import Color
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text
from tqdm.std import tqdm

from ..utils import isnotebook
from ..utils.viz_utils import (
    final_plot_html,
    get_colors,
    get_instance_html,
    red_transparent_blue_colormap,
    saliency_heatmap_html,
    saliency_heatmap_table_header,
    sanitize_html,
)
from .attribution import FeatureAttributionSequenceOutput, OneOrMoreFeatureAttributionSequenceOutputs


def show_attributions(
    attributions: OneOrMoreFeatureAttributionSequenceOutputs,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    display: bool = True,
    return_html: Optional[bool] = False,
) -> Optional[str]:
    if return_html and not isnotebook():
        raise AttributeError("return_html=True is can be used only inside an IPython environment.")
    if isinstance(attributions, FeatureAttributionSequenceOutput):
        attributions = [attributions]
    html_out = ""
    if isnotebook():
        colors = get_attribution_colors(attributions, min_val, max_val, cmap=red_transparent_blue_colormap())
    else:
        colors = get_attribution_colors(attributions, min_val, max_val, return_alpha=False, return_strings=False)
    idx = 0
    for ex_id, attribution in enumerate(attributions):
        if isnotebook():
            from IPython.core.display import HTML, display

            instance_html = get_instance_html(ex_id)
            curr_html = ""
            curr_html += instance_html
            curr_html += get_saliency_heatmap_html(
                attribution.source_scores,
                attribution.target_tokens,
                attribution.source_tokens,
                colors[idx],
                label="Source",
            )
            idx += 1
            if attribution.target_scores is not None:
                curr_html += instance_html
                curr_html += get_saliency_heatmap_html(
                    attribution.target_scores,
                    attribution.target_tokens,
                    attribution.target_tokens,
                    colors[idx],
                    label="Target",
                )
                idx += 1
            if display:
                display(HTML(curr_html))
            html_out += curr_html
        else:
            if not display:
                raise AttributeError("display=False is not supported outside of an IPython environment.")
            rprint(
                get_saliency_heatmap_rich(
                    attribution.source_scores,
                    attribution.target_tokens,
                    attribution.source_tokens,
                    colors[idx],
                    attribution.deltas,
                    label="Source",
                )
            )
            idx += 1
            if attribution.target_scores is not None:
                rprint(
                    get_saliency_heatmap_rich(
                        attribution.target_scores,
                        attribution.target_tokens,
                        attribution.target_tokens,
                        colors[idx],
                        attribution.deltas,
                        label="Target",
                    )
                )
                idx += 1
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
    if min_val is None:
        min_val = min(attribution.minimum for attribution in attributions)
    if max_val is None:
        max_val = max(attribution.maximum for attribution in attributions)
    colors = []
    for attribution in attributions:
        colors.append(get_colors(attribution.source_scores, min_val, max_val, cmap, return_alpha, return_strings))
        if attribution.target_scores is not None:
            colors.append(get_colors(attribution.target_scores, min_val, max_val, cmap, return_alpha, return_strings))
    return colors


def get_saliency_heatmap_html(
    scores: np.ndarray,
    row_labels: List[str],
    column_labels: List[str],
    input_colors: List[List[str]],
    label: str = "",
):
    # unique ID added to HTML elements and function to avoid collision of differnent instances
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    out = saliency_heatmap_table_header
    # add top row containing target tokens
    for head_id in range(scores.shape[1]):
        out += f"<th>{sanitize_html(row_labels[head_id])}</th>"
    out += "</tr>"
    for row_index in range(scores.shape[0]):
        out += f"<tr><th>{sanitize_html(column_labels[row_index])}</th>"
        for col_index in range(scores.shape[1]):
            score = ""
            if not np.isnan(scores[row_index, col_index]):
                score = str(round(scores[row_index][col_index], 3))
            out += f'<th style="background:{input_colors[row_index][col_index]}">{score}</th>'
        out += "</tr>"
    out += "</table>"
    saliency_heatmap_markup = saliency_heatmap_html.format(uuid=uuid, content=out, label=label)
    plot_uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    return final_plot_html.format(
        uuid=plot_uuid,
        saliency_plot_markup=saliency_heatmap_markup,
    )


def get_saliency_heatmap_rich(
    scores: np.ndarray,
    row_labels: List[str],
    column_labels: List[str],
    input_colors: List[List[Tuple[float, float, float]]],
    deltas: Optional[List[float]] = None,
    label: str = "",
):
    table = Table(
        *[Column(header="", justify="right", footer="δ")]
        + [
            Column(
                header=row_labels[head_id],
                justify="center",
                footer=f"{deltas[head_id]:.2f}" if deltas and len(deltas) > head_id else "",
            )
            for head_id in range(scores.shape[1])
        ],
        title=f"{label + ' ' if label else ''}Saliency Heatmap",
        caption="x: Generated tokens, y: Attributed tokens",
        padding=(0, 1, 0, 1),
        show_lines=True,
        box=box.SIMPLE_HEAVY,
        show_footer=True,
    )
    for row_index in range(scores.shape[0]):
        table.add_row(
            *[Text(column_labels[row_index], style="bold")]
            + [
                Text(
                    f"{round(scores[row_index][col_index], 2) if not np.isnan(scores[row_index][col_index]) else ''}",
                    justify="center",
                    style=Style(color=Color.from_rgb(*input_colors[row_index][col_index])),
                )
                for col_index in range(scores.shape[1])
            ]
        )
    return table


# Progress bar utilities


def get_progress_bar(
    all_sentences: Tuple[List[str], List[str], List[int]],
    method_name: str,
    show: bool,
    pretty: bool,
) -> Union[tqdm, Tuple[Progress, Live], None]:
    sources, targets, lengths = all_sentences
    if not show:
        return None
    elif show and not pretty:
        return tqdm(
            total=max(tgt_len for tgt_len in lengths),
            desc=f"Attributing with {method_name}...",
        )
    elif show and pretty:
        job_progress = Progress(
            "{task.description}",
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        for idx, (tgt, tgt_len) in enumerate(zip(targets, lengths)):
            job_progress.add_task(f"{idx}. {tgt}", total=tgt_len)
        progress_table = Table.grid()
        progress_table.add_row(
            Panel.fit(
                "\n".join([f"{idx}. {src}" for idx, src in enumerate(sources)]),
                title="Source sentences",
                border_style="red",
                padding=(1, 2),
            ),
            Panel.fit(
                job_progress,
                title=f"[b]Attributing with {method_name}",
                border_style="green",
                padding=(1, 2),
            ),
        )
        live = Live(Padding(progress_table, (1, 0, 1, 0)), refresh_per_second=10)
        live.start(refresh=live._renderable is not None)
        return job_progress, live


def update_progress_bar(
    pbar: Union[tqdm, Tuple[Progress, Live], None],
    split_targets: Tuple[List[str], List[str], List[str], List[str]] = None,
    whitespace_indexes: List[List[int]] = None,
    show: bool = False,
    pretty: bool = False,
) -> None:
    if not show:
        return
    elif show and not pretty:
        pbar.update(1)
    else:
        for job in pbar[0].tasks:
            if not job.finished:
                pbar[0].advance(job.id)
                formatted_desc = f"{job.id}. "
                past_length = 0
                for split, color in zip(split_targets, ["grey58", "green", "orange1", "grey58"]):
                    if split[job.id]:
                        formatted_desc += f"[{color}]" + split[job.id] + "[/]"
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
