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

from rich import print as rprint
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.spinner import Spinner
from rich.table import Table
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
    display_html: bool = True,
    return_html: Optional[bool] = False,
) -> Optional[str]:
    if display_html:
        try:
            from IPython.core.display import HTML, display
        except ImportError:
            raise ImportError("IPython should be installed to visualize attributions.")
    if isinstance(attributions, FeatureAttributionSequenceOutput):
        attributions = [attributions]
    if min_val is None:
        min_val = min(attribution.minimum for attribution in attributions)
    if max_val is None:
        max_val = max(attribution.maximum for attribution in attributions)
    html_out = ""
    for i, attribution in enumerate(attributions):
        curr_html = ""
        curr_html += get_instance_html(i)
        curr_html += seq2seq_plots(attribution, min_val, max_val)
        if display_html:
            display(HTML(curr_html))
        html_out += curr_html
    if return_html:
        return html_out


def seq2seq_plots(
    attribution: FeatureAttributionSequenceOutput,
    min_score: int,
    max_score: int,
):
    # unique ID added to HTML elements and function to avoid collision of differnent instances
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    saliency_heatmap_markup = saliency_heatmap(attribution, min_score, max_score)
    html = final_plot_html.format(
        uuid=uuid,
        saliency_plot_markup=saliency_heatmap_markup,
    )
    return html


def saliency_heatmap(
    attribution: FeatureAttributionSequenceOutput,
    min_score: int,
    max_score: int,
):
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    cmap = red_transparent_blue_colormap()
    input_colors = get_colors(min_score, max_score, attribution.scores, cmap)
    out = saliency_heatmap_table_header
    # add top row containing target tokens
    for head_id in range(attribution.scores.shape[1]):
        out += f"<th>{sanitize_html(attribution.target_tokens[head_id])}</th>"
    out += "</tr>"
    for row_index in range(attribution.scores.shape[0]):
        out += f"<tr><th>{sanitize_html(attribution.source_tokens[row_index])}</th>"
        for col_index in range(attribution.scores.shape[1]):
            score = str(round(attribution.scores[row_index][col_index], 3))
            out += f'<th style="background:{input_colors[row_index][col_index]}">{score}</th>'
        out += "</tr>"
    out += "</table>"
    return saliency_heatmap_html.format(uuid=uuid, content=out)


# Progress bar utilities


def get_progress_bar(
    target_sentences: List[Tuple[str, str, int]],
    method_name: str,
    show: bool,
    pretty: bool,
) -> Union[tqdm, Tuple[Progress, Live], None]:
    if not show:
        return None
    elif show and not pretty:
        return tqdm(
            total=max(tgt_len for _, _, tgt_len in target_sentences),
            desc=f"Attributing with {method_name}...",
        )
    elif show and pretty:
        job_progress = Progress(
            "{task.description}",
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        _ = [job_progress.add_task(tgt, total=tgt_len) for _, tgt, tgt_len in target_sentences]
        progress_table = Table.grid()
        progress_table.add_row(
            Panel.fit(
                "\n".join([src for src, _, _ in target_sentences]),
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
    skipped_prefixes: List[str] = None,
    attributed_sentences: List[str] = None,
    unattributed_suffixes: List[str] = None,
    skipped_suffixes: List[str] = None,
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
                formatted_desc = ""
                past_length = 0
                splits = [
                    skipped_prefixes,
                    attributed_sentences,
                    unattributed_suffixes,
                    skipped_suffixes,
                ]
                for split, color in zip(splits, ["grey58", "green", "orange1", "grey58"]):
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


class LoadingMessage:
    def __init__(self, msg, spinner="dots", style="green", padding=(1, 0, 1, 0), verbose=True):
        self.msg = msg
        self.spinner = spinner
        self.style = style
        self.padding = padding
        self.live = None
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            if isnotebook():
                rprint(Padding(Text(self.msg, style=self.style), self.padding))
            else:
                self.live = Live(
                    Padding(
                        Spinner("dots", text=Text(self.msg, style=self.style)),
                        self.padding,
                    )
                )
                self.live.start(refresh=self.live._renderable is not None)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not isnotebook() and self.verbose and self.live is not None:
            self.live.stop()
