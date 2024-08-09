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


from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import treescope as ts
from matplotlib.colors import Colormap, LinearSegmentedColormap
from numpy.typing import NDArray

from .misc import isnotebook, ordinal_str
from .typing import TokenWithId

red = (178, 24, 43)
beige = (247, 252, 253)
blue = (33, 102, 172)
green = (0, 109, 44)
brown = (140, 81, 10)


def get_instance_html(i: int):
    return "<br/><b>" + ordinal_str(i) + " instance:</b><br/>"


def interpolate_color(color1, color2, t):
    return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2, strict=False))


def generate_colormap(start_color, end_color, num_colors):
    return [interpolate_color(start_color, end_color, t) for t in np.linspace(0, 1, num_colors)]


def red_transparent_blue_colormap():
    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((*(float(c) / 255 for c in blue), l))
    for l in np.linspace(0, 1, 100):
        colors.append((*(float(c) / 255 for c in red), l))
    return LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def treescope_cmap(colors: Literal["blue_to_red", "brown_to_green", "greens", "blues"] = "blue_to_red", n: int = 200):
    match colors:
        case "blue_to_red":
            first_half = generate_colormap(blue, beige, n // 2)
            second_half = generate_colormap(beige, red, n - len(first_half))
            cmap = first_half + second_half
        case "brown_to_green":
            first_half = generate_colormap(brown, beige, n // 2)
            second_half = generate_colormap(beige, green, n - len(first_half))
            cmap = first_half + second_half
        case "greens":
            cmap = generate_colormap(beige, green, n)
        case "blues":
            cmap = generate_colormap(beige, blue, n)
        case _:
            raise ValueError(f"Invalid color scheme {colors}: valid options are 'blue_to_red', 'greens', 'blues'")
    return cmap


def get_color(
    score: float,
    min_value: float | int,
    max_value: float | int,
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


def sanitize_html(txt: str | TokenWithId) -> str:
    if isinstance(txt, TokenWithId):
        txt = txt.token
    return txt.replace("<", "&lt;").replace(">", "&gt;")


def get_colors(
    scores: NDArray,
    min_value: float | int,
    max_value: float | int,
    cmap: str | Colormap | None = None,
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


def test_dim(dim: int | str, dim_names: dict[int, str], rev_dim_names: dict[str, int], scores: np.ndarray) -> int:
    if isinstance(dim, str):
        if dim not in rev_dim_names:
            raise ValueError(f"Invalid dimension name {dim}: valid names are {list(rev_dim_names.keys())}")
        dim_idx = rev_dim_names[dim]
    else:
        dim_idx = dim
    if dim_idx <= 1 or dim_idx > scores.ndim or dim_idx not in dim_names:
        raise ValueError(f"Invalid dimension {dim_idx}: valid indices are {list(range(2, scores.ndim))}")
    return dim_idx


def maybe_add_linebreak(tok: str, i: int, wrap_after: int | str | list[str] | tuple[str]) -> list[str]:
    if isinstance(wrap_after, str) and tok == wrap_after:
        return [ts.rendering_parts.text("\n")]
    elif isinstance(wrap_after, list | tuple) and tok in wrap_after:
        return [ts.rendering_parts.text("\n")]
    elif isinstance(wrap_after, int) and i % wrap_after == 0:
        return [ts.rendering_parts.text("\n")]
    else:
        return []


def treescope_ignore(f: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(f)
    def treescope_unhooked_wrapper(self, *args, **kwargs):
        if isnotebook():
            # Unhook the treescope visualization to allow `rich.jupyter.JupyterRenderable` to render correctly
            import IPython

            del IPython.get_ipython().display_formatter.formatters["text/html"].type_printers[object]
        out = f(self, *args, **kwargs)
        if isnotebook():
            # Re-hook the treescope visualization
            ts.register_as_default()
        return out

    return treescope_unhooked_wrapper


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
"""

saliency_heatmap_html = """
<div id="{uuid}_saliency_plot" class="{uuid}_viz_content">
    <div style="margin:5px;font-family:sans-serif;font-weight:bold;">
        <span style="font-size: 20px;">{label} Saliency Heatmap</span>
        <br>
        → : Generated tokens, ↓ : Attributed tokens
    </div>
    {content}
</div>
"""
