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

import json
import random
import string

from ..data import (
    FeatureAttributionSequenceOutput,
    OneOrMoreFeatureAttributionSequenceOutputs,
)
from ..utils.viz_utils import (
    final_plot_html,
    final_plot_javascript,
    get_color,
    get_colors,
    get_instance_html,
    heatmap_color_score_maps,
    heatmap_html,
    heatmap_javascript,
    heatmap_token_html,
    red_transparent_blue_colormap,
    saliency_plot_html,
    saliency_table_header,
)


def show_attributions(
    attributions: OneOrMoreFeatureAttributionSequenceOutputs,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    display: Optional[bool] = True,
):
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

    for i, attribution in enumerate(attributions):
        display(HTML(get_instance_html(i)))
        display(HTML(seq2seq_plots(attribution, min_val, max_val)))


def seq2seq_plots(
    attribution: FeatureAttributionSequenceOutput,
    min_score: int,
    max_score: int,
):
    # unique ID added to HTML elements and function to avoid collision of differnent instances
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    saliency_plot_markup = saliency_plot(attribution, min_score, max_score)
    heatmap_markup = heatmap(attribution, min_score, max_score)
    html = final_plot_html.format(
        uuid=uuid,
        saliency_plot_markup=saliency_plot_markup,
        heatmap_markup=heatmap_markup,
    )
    javascript = final_plot_javascript.format(uuid=uuid)
    return javascript + html


def saliency_plot(
    attribution: FeatureAttributionSequenceOutput,
    min_score: int,
    max_score: int,
):
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    cmap = red_transparent_blue_colormap()
    input_colors = get_colors(min_score, max_score, attribution.scores, cmap)
    out = saliency_table_header
    # add top row containing input tokens
    for j in range(attribution.scores.shape[0]):
        out += f"<th>{attribution.source_tokens[j]}</th>"
    out += "</tr>"
    for row_index in range(attribution.scores.shape[1]):
        out += f"<tr><th>{attribution.target_tokens[row_index]}</th>"
        for col_index in range(attribution.scores.shape[0]):
            score = str(round(attribution.scores[col_index][row_index], 3))
            out += f'<th style="background:{input_colors[col_index][row_index]}">{score}</th>'
        out += "</tr>"
    out += "</table>"
    return saliency_plot_html.format(uuid=uuid, content=out)


def heatmap(
    attribution: FeatureAttributionSequenceOutput,
    min_score: int,
    max_score: int,
    src_id: str = "source",
    tgt_id: str = "target",
    tok_id: str = "token",
    val_id: str = "value_label",
):
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    # generate dictionary containing precomputed background colors
    # and shap values which are addressable by html token ids
    colors_dict = {}
    scores_dict = {}
    cmax = max(min_score, max_score)
    cmap = red_transparent_blue_colormap()

    # source tokens -> target tokens color and label value mapping
    for row_index in range(len(attribution.source_tokens)):
        color_values = {}
        scores_list = {}
        for col_index in range(len(attribution.target_tokens)):
            tok_label = f"{uuid}_{tgt_id}_{tok_id}_{row_index}"
            val_label = f"{uuid}_{tgt_id}_{val_id}_{row_index}"
            score = attribution.scores[row_index][col_index]
            color_values[tok_label] = get_color(score, cmax, cmap)
            scores_list[val_label] = str(round(score, 3))
        colors_dict[f"{uuid}_{src_id}_{tok_id}_{row_index}"] = color_values
        scores_dict[f"{uuid}_{src_id}_{tok_id}_{row_index}"] = scores_list

    # target tokens -> source tokens color and label value mapping
    for col_index in range(len(attribution.target_tokens)):
        color_values = {}
        scores_list = {}
        for row_index in range(len(attribution.source_tokens)):
            tok_label = f"{uuid}_{src_id}_{tok_id}_{col_index}"
            val_label = f"{uuid}_{src_id}_{val_id}_{col_index}"
            score = attribution.scores[row_index][col_index]
            color_values[tok_label] = get_color(score, cmax, cmap)
            scores_list[val_label] = str(round(score, 3))
        colors_dict[f"{uuid}_{tgt_id}_{tok_id}_{row_index}"] = color_values
        scores_dict[f"{uuid}_{tgt_id}_{tok_id}_{row_index}"] = scores_list

    # convert python dictionary into json to be inserted into the runtime javascript environment
    javascript_values = heatmap_color_score_maps.format(
        uuid=uuid,
        colors_json=json.dumps(colors_dict),
        scores_json=json.dumps(scores_dict),
    )
    input_text_html = ""
    output_text_html = ""
    for i, token in enumerate(attribution.source_tokens):
        input_text_html += heatmap_token_html.format(
            i=i,
            uuid=uuid,
            token=token,
            name=src_id,
            value_id=val_id,
            token_id=tok_id,
        )
    for i, token in enumerate(attribution.target_tokens):
        output_text_html += heatmap_token_html.format(
            i=i,
            uuid=uuid,
            token=token,
            name=tgt_id,
            value_id=val_id,
            token_id=tok_id,
        )

    return (
        heatmap_html.format(
            uuid=uuid,
            input_text_html=input_text_html,
            output_text_html=output_text_html,
        )
        + heatmap_javascript.format(
            uuid=uuid,
        )
        + javascript_values
    )
