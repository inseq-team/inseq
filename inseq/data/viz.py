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

from typing import Optional

import random
import string

from ..data import (
    FeatureAttributionSequenceOutput,
    OneOrMoreFeatureAttributionSequenceOutputs,
)
from ..utils.viz_utils import (
    final_plot_html,
    get_colors,
    get_instance_html,
    red_transparent_blue_colormap,
    saliency_heatmap_html,
    saliency_heatmap_table_header,
    sanitize_html,
)


def show_attributions(
    attributions: OneOrMoreFeatureAttributionSequenceOutputs,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    return_html: Optional[bool] = False,
) -> Optional[str]:
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
        if not return_html:
            display(HTML(get_instance_html(i)))
            display(HTML(seq2seq_plots(attribution, min_val, max_val)))
        else:
            html_out += get_instance_html(i)
            html_out += seq2seq_plots(attribution, min_val, max_val)
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
