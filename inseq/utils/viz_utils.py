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

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .misc import ordinal_str


def get_instance_html(i: int):
    return "<br/><b>" + ordinal_str(i) + " instance:</b><br/>"


def red_transparent_blue_colormap():
    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, l))
    for l in np.linspace(0, 1, 100):
        colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, l))
    return LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def get_color(score, cmax, cmap):
    scaled_value = 0.5 + 0.5 * score / cmax
    color = cmap(scaled_value)
    color = "rgba" + str((color[0] * 255, color[1] * 255, color[2] * 255, color[3]))
    return color


def get_colors(
    min_value,
    max_value,
    scores,
    cmap,
):
    input_colors = []
    cmax = max(abs(min_value), abs(max_value))
    for row_index in range(scores.shape[0]):
        input_colors_row = []
        for col_index in range(scores.shape[1]):
            color = get_color(scores[row_index, col_index], cmax, cmap)
            input_colors_row.append(color)
        input_colors.append(input_colors_row)
    return input_colors


# Full plot

final_plot_html = """
<html>
<div id="{uuid}_viz_container">
    <div id="{uuid}_viz_header"
        style="padding:15px;border-style:solid;margin:5px;font-family:sans-serif;font-weight:bold;">
    Visualization Type:
    <select name="viz_type" id="{uuid}_viz_type" onchange="selectVizType_{uuid}(this)">
        <option value="heatmap" selected="selected">Input/Output - Heatmap</option>
        <option value="saliency-plot">Saliency Plot</option>
    </select>
    </div>
    <div id="{uuid}_content" style="padding:15px;border-style:solid;margin:5px;">
        <div id = "{uuid}_saliency_plot_container" class="{uuid}_viz_container" style="display:none">
            {saliency_plot_markup}
        </div>

        <div id = "{uuid}_heatmap_container" class="{uuid}_viz_container">
            {heatmap_markup}
        </div>
    </div>
</div>
</html>
"""

final_plot_javascript = """
<script>
    function selectVizType_{uuid}(selectObject) {{
        /* Hide all viz */
        var elements = document.getElementsByClassName("{uuid}_viz_container")
        for (var i = 0; i < elements.length; i++){{
            elements[i].style.display = 'none';
        }}
        var value = selectObject.value;
        if ( value === "saliency-plot" ){{
            document.getElementById('{uuid}_saliency_plot_container').style.display  = "block";
        }}
        else if ( value === "heatmap" ) {{
            document.getElementById('{uuid}_heatmap_container').style.display  = "block";
        }}
    }}
</script>
"""

# Saliency plot

saliency_table_header = """
<table border="1" cellpadding="5" cellspacing="5"
    style="overflow-x:scroll;display:block;">
    <tr><th></th>
"""

saliency_plot_html = """
<div id="{uuid}_saliency_plot" class="{uuid}_viz_content">
    <div style="margin:5px;font-family:sans-serif;font-weight:bold;">
        <span style="font-size: 20px;"> Saliency Plot </span>
        <br>
        x-axis: Input Text
        <br>
        y-axis: Output Text
    </div>
    {content}
</div>
"""

# Heatmap plot

heatmap_token_html = """
<div style="display:inline; text-align:center;">
    <div id="{uuid}_{name}_{value_id}_{i}"
        style="display:none;color: #999; padding-top: 0px; font-size:12px;">
    </div>
    <div id="{uuid}_{name}_{token_id}_{i}"
        style="display: inline; background:transparent; border-radius: 3px; padding: 0px;cursor: default;cursor: pointer;"
        onmouseover="onMouseHoverFlat_{uuid}(this.id)"
        onmouseout="onMouseOutFlat_{uuid}(this.id)"
        onclick="onMouseClickFlat_{uuid}(this.id)">
        {token} \
    </div>
</div>
"""  # noqa

heatmap_color_score_maps = """
<script>
    colors_{uuid} = {colors_json}
    scores_{uuid} = {scores_json}
</script>
"""


heatmap_html = """
<div id="{uuid}_heatmap" class="{uuid}_viz_content">
    <div id="{uuid}_heatmap_header" style="padding:15px;margin:5px;font-family:sans-serif;font-weight:bold;">
    <div style="display:inline">
        <span style="font-size: 20px;"> Input/Output - Heatmap </span>
    </div>
    <div style="display:inline;float:right">
        Layout :
        <select name="alignment" id="{uuid}_alignment" onchange="selectAlignment_{uuid}(this)">
        <option value="left-right" selected="selected">Left/Right</option>
        <option value="top-bottom">Top/Bottom</option>
        </select>
    </div>
    </div>
    <div id="{uuid}_heatmap_content" style="display:flex;">
    <div id="{uuid}_input_container" style="padding:15px;border-style:solid;margin:5px;flex:1;">
        <div id="{uuid}_input_header" style="margin:5px;font-weight:bold;font-family:sans-serif;margin-bottom:10px">
        Input Text
        </div>
        <div id="{uuid}_input_content" style="margin:5px;font-family:sans-serif;">
            {input_text_html}
        </div>
    </div>
    <div id="{uuid}_output_container" style="padding:15px;border-style:solid;margin:5px;flex:1;">
        <div id="{uuid}_output_header" style="margin:5px;font-weight:bold;font-family:sans-serif;margin-bottom:10px">
        Output Text
        </div>
        <div id="{uuid}_output_content" style="margin:5px;font-family:sans-serif;">
            {output_text_html}
        </div>
    </div>
    </div>
</div>
"""

heatmap_javascript = """
<script>
    function selectAlignment_{uuid}(selectObject) {{
        var value = selectObject.value;
        if ( value === "left-right" ){{
            document.getElementById('{uuid}_heatmap_content').style.display  = "flex";
        }}
        else if ( value === "top-bottom" ) {{
            document.getElementById('{uuid}_heatmap_content').style.display  = "inline";
        }}
    }}

    var {uuid}_heatmap_flat_state = null;

    function onMouseHoverFlat_{uuid}(id) {{
        if ({uuid}_heatmap_flat_state === null) {{
            setBackgroundColors_{uuid}(id);
            document.getElementById(id).style.backgroundColor  = "grey";
        }}
    }}

    function onMouseOutFlat_{uuid}(id) {{
        if ({uuid}_heatmap_flat_state === null) {{
            cleanValuesAndColors_{uuid}(id);
            document.getElementById(id).style.backgroundColor  = "transparent";
        }}
    }}

    function onMouseClickFlat_{uuid}(id) {{
        if ({uuid}_heatmap_flat_state === id) {{

            // If the clicked token was already selected

            document.getElementById(id).style.backgroundColor  = "transparent";
            cleanValuesAndColors_{uuid}(id);
            {uuid}_heatmap_flat_state = null;
        }}
        else {{
            if ({uuid}_heatmap_flat_state === null) {{

                // No token previously selected, new token clicked on

                cleanValuesAndColors_{uuid}(id)
                {uuid}_heatmap_flat_state = id;
                document.getElementById(id).style.backgroundColor  = "grey";
                setLabelValues_{uuid}(id);
                setBackgroundColors_{uuid}(id);
            }}
            else {{
                if (getIdSide_{uuid}({uuid}_heatmap_flat_state) === getIdSide_{uuid}(id)) {{

                    // User clicked a token on the same side as the currently selected token

                    cleanValuesAndColors_{uuid}({uuid}_heatmap_flat_state)
                    document.getElementById({uuid}_heatmap_flat_state).style.backgroundColor  = "transparent";
                    {uuid}_heatmap_flat_state = id;
                    document.getElementById(id).style.backgroundColor  = "grey";
                    setLabelValues_{uuid}(id);
                    setBackgroundColors_{uuid}(id);
                }}
                else{{

                    if (document.getElementById(id).previousElementSibling.style.display == 'none') {{
                        document.getElementById(id).previousElementSibling.style.display = 'block';
                        document.getElementById(id).parentNode.style.display = 'inline-block';
                        }}
                    else {{
                        document.getElementById(id).previousElementSibling.style.display = 'none';
                        document.getElementById(id).parentNode.style.display = 'inline';
                        }}

                }}
            }}

        }}
    }}
    function setLabelValues_{uuid}(id) {{
        for(const token in scores_{uuid}[id]){{
            document.getElementById(token).innerHTML = scores_{uuid}[id][token];
            document.getElementById(token).nextElementSibling.title = 'Attribution Value : ' + scores_{uuid}[id][token];
        }}
    }}
    function setBackgroundColors_{uuid}(id) {{
        for(const token in colors_{uuid}[id]){{
            document.getElementById(token).style.backgroundColor  = colors_{uuid}[id][token];
        }}
    }}
    function cleanValuesAndColors_{uuid}(id) {{
        for(const token in scores_{uuid}[id]){{
            document.getElementById(token).innerHTML = "";
            document.getElementById(token).nextElementSibling.title = "";
        }}
            for(const token in colors_{uuid}[id]){{
            document.getElementById(token).style.backgroundColor  = "transparent";
            document.getElementById(token).previousElementSibling.style.display = 'none';
            document.getElementById(token).parentNode.style.display = 'inline';
            document.getElementById(token).style.textShadow  = "inherit";
        }}
    }}

    function getIdSide_{uuid}(id) {{
        if (id === null) {{
            return 'null'
        }}
        return id.split("_")[1];
    }}
</script>
"""  # noqa
