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

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap
from numpy.typing import NDArray

from .misc import ordinal_str
from .typing import TokenWithId


def get_instance_html(i: int):
    return "<br/><b>" + ordinal_str(i) + " instance:</b><br/>"


def red_transparent_blue_colormap():
    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, l))
    for l in np.linspace(0, 1, 100):
        colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, l))
    return LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def get_color(
    score: float,
    min_value: Union[float, int],
    max_value: Union[float, int],
    cmap: Colormap,
    return_alpha: bool = True,
    return_string: bool = True,
):
    # Normalize between 0-1 for the color scale
    scaled_value = (score - min_value) / (max_value - min_value)
    color = cmap(scaled_value)
    if return_alpha:
        color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
        if return_string:
            color = "rgba" + str(color)
    else:
        color = (color[0] * 255, color[1] * 255, color[2] * 255)
        if return_string:
            color = "rgba" + str(color)
    return color


def sanitize_html(txt: Union[str, TokenWithId]) -> str:
    if isinstance(txt, TokenWithId):
        txt = txt.token
    return txt.replace("<", "&lt;").replace(">", "&gt;")


def get_colors(
    scores: NDArray,
    min_value: Union[float, int],
    max_value: Union[float, int],
    cmap: Union[str, Colormap, None] = None,
    return_alpha: bool = True,
    return_strings: bool = True,
):
    if isinstance(cmap, Colormap):
        out_cmap = cmap
    else:
        out_cmap: Colormap = plt.get_cmap(cmap if isinstance(cmap, str) else "coolwarm", 200)
    input_colors = []
    for row_idx in range(scores.shape[0]):
        input_colors_row = []
        for col_idx in range(scores.shape[1]):
            color = get_color(scores[row_idx, col_idx], min_value, max_value, out_cmap, return_alpha, return_strings)
            input_colors_row.append(color)
        input_colors.append(input_colors_row)
    return input_colors


# Full plot

final_plot_html = """
<html>
<div id="{uuid}_viz_container">
    <div id="{uuid}_content" style="padding:15px;border-style:solid;margin:5px;">
        <div id = "{uuid}_saliency_plot_container" class="{uuid}_viz_container" style="display:block">
            {saliency_plot_markup}
        </div>
    </div>
</div>
</html>
"""

# Saliency plot

saliency_heatmap_table_header = """
<table border="1" cellpadding="5" cellspacing="5"
    style="overflow-x:scroll;display:block;">
    <tr><th></th>
"""

saliency_heatmap_html = """
<div id="{uuid}_saliency_plot" class="{uuid}_viz_content">
    <div style="margin:5px;font-family:sans-serif;font-weight:bold;">
        <span style="font-size: 20px;">{label} Saliency Heatmap</span>
        <br>
        x: Generated tokens, y: Attributed tokens
    </div>
    {content}
</div>
"""
