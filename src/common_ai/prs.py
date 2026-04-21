import numpy as np
import random
import string
from shap.plots import colors
from shap.plots._text import (
    unpack_shap_explanation_contents,
    _ipython_display_html,
    process_shap_values,
)


def my_text(
    shap_values,
    num_starting_labels=0,
    grouping_threshold=0.01,
    separator="",
    xmin=None,
    xmax=None,
    cmax=None,
    display=True,
):
    """Plots an explanation of a string of text using coloring and interactive labels.

    The output is interactive HTML and you can click on any token to toggle the display of the
    SHAP value assigned to that token.

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap values for a string (#input_tokens x output_tokens).

    num_starting_labels : int
        Number of tokens (sorted in descending order by corresponding SHAP values)
        that are uncovered in the initial view.
        When set to 0, all tokens are covered.

    grouping_threshold : float
        If the component substring effects are less than a ``grouping_threshold``
        fraction of an unlowered interaction effect, then we visualize the entire group
        as a single chunk. This is primarily used for explanations that were computed
        with fixed_context set to 1 or 0 when using the :class:`.explainers.Partition`
        explainer, since this causes interaction effects to be left on internal nodes
        rather than lowered.

    separator : string
        The string separator that joins tokens grouped by interaction effects and
        unbroken string spans. Defaults to the empty string ``""``.

    xmin : float
        Minimum shap value bound.

    xmax : float
        Maximum shap value bound.

    cmax : float
        Maximum absolute shap value for sample. Used for scaling colors for input tokens.

    display: bool
        Whether to display or return html to further manipulate or embed. Default: ``True``

    Examples
    --------
    See `text plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html>`_.

    """

    def values_min_max(values, base_values):
        """Used to pick our axis limits."""
        fx = base_values + values.sum()
        xmin = fx - values[values > 0].sum()
        xmax = fx - values[values < 0].sum()
        cmax = max(abs(values.min()), abs(values.max()))
        d = xmax - xmin
        xmin -= 0.1 * d
        xmax += 0.1 * d

        return xmin, xmax, cmax

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    # loop when we get multi-row inputs
    if len(shap_values.shape) == 2 and (
        shap_values.output_names is None or isinstance(shap_values.output_names, str)
    ):
        xmin = 0
        xmax = 0
        cmax = 0

        for i, v in enumerate(shap_values):
            values, clustering = unpack_shap_explanation_contents(v)
            tokens, values, group_sizes = process_shap_values(
                v.data, values, grouping_threshold, separator, clustering
            )

            if i == 0:
                xmin, xmax, cmax = values_min_max(values, v.base_values)
                continue

            xmin_i, xmax_i, cmax_i = values_min_max(values, v.base_values)
            if xmin_i < xmin:
                xmin = xmin_i
            if xmax_i > xmax:
                xmax = xmax_i
            if cmax_i > cmax:
                cmax = cmax_i
        out = ""
        for i, v in enumerate(shap_values):
            out += f"""
    <br>
    <hr style="height: 1px; background-color: #fff; border: none; margin-top: 18px; margin-bottom: 18px; border-top: 1px dashed #ccc;"">
    <div align="center" style="margin-top: -35px;"><div style="display: inline-block; background: #fff; padding: 5px; color: #999; font-family: monospace">[{i}]</div>
    </div>
                """
            out += my_text(
                v,
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )
        if display:
            _ipython_display_html(out)
            return
        else:
            return out

    if len(shap_values.shape) == 2 and shap_values.output_names is not None:
        xmin_computed = None
        xmax_computed = None
        cmax_computed = None

        for i in range(shap_values.shape[-1]):
            values, clustering = unpack_shap_explanation_contents(shap_values[:, i])
            tokens, values, group_sizes = process_shap_values(
                shap_values[:, i].data,
                values,
                grouping_threshold,
                separator,
                clustering,
            )

            # if i == 0:
            #     xmin, xmax, cmax = values_min_max(values, shap_values[:,i].base_values)
            #     continue

            xmin_i, xmax_i, cmax_i = values_min_max(
                values, shap_values[:, i].base_values
            )
            if xmin_computed is None or xmin_i < xmin_computed:
                xmin_computed = xmin_i
            if xmax_computed is None or xmax_i > xmax_computed:
                xmax_computed = xmax_i
            if cmax_computed is None or cmax_i > cmax_computed:
                cmax_computed = cmax_i

        if xmin is None:
            xmin = xmin_computed
        if xmax is None:
            xmax = xmax_computed
        if cmax is None:
            cmax = cmax_computed

        out = f"""<div align='center'>
<script>
    document._hover_{uuid} = '_tp_{uuid}_output_0';
    document._zoom_{uuid} = undefined;
    function _output_onclick_{uuid}(i) {{
        var next_id = undefined;

        if (document._zoom_{uuid} !== undefined) {{
            document.getElementById(document._zoom_{uuid}+ '_zoom').style.display = 'none';

            if (document._zoom_{uuid} === '_tp_{uuid}_output_' + i) {{
                document.getElementById(document._zoom_{uuid}).style.display = 'block';
                document.getElementById(document._zoom_{uuid}+'_name').style.borderBottom = '3px solid #000000';
            }} else {{
                document.getElementById(document._zoom_{uuid}).style.display = 'none';
                document.getElementById(document._zoom_{uuid}+'_name').style.borderBottom = 'none';
            }}
        }}
        if (document._zoom_{uuid} !== '_tp_{uuid}_output_' + i) {{
            next_id = '_tp_{uuid}_output_' + i;
            document.getElementById(next_id).style.display = 'none';
            document.getElementById(next_id + '_zoom').style.display = 'block';
            document.getElementById(next_id+'_name').style.borderBottom = '3px solid #000000';
        }}
        document._zoom_{uuid} = next_id;
    }}
    function _output_onmouseover_{uuid}(i, el) {{
        if (document._zoom_{uuid} !== undefined) {{ return; }}
        if (document._hover_{uuid} !== undefined) {{
            document.getElementById(document._hover_{uuid} + '_name').style.borderBottom = 'none';
            document.getElementById(document._hover_{uuid}).style.display = 'none';
        }}
        document.getElementById('_tp_{uuid}_output_' + i).style.display = 'block';
        el.style.borderBottom = '3px solid #000000';
        document._hover_{uuid} = '_tp_{uuid}_output_' + i;
    }}
</script>
<div style=\"color: rgb(120,120,120); font-size: 12px;\">outputs</div>"""
        output_values = shap_values.values.sum(0) + shap_values.base_values
        output_max = np.max(np.abs(output_values))
        for i, name in enumerate(shap_values.output_names):
            scaled_value = 0.5 + 0.5 * output_values[i] / (output_max + 1e-8)
            color = colors.red_transparent_blue(scaled_value)
            color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
            # '#dddddd' if i == 0 else '#ffffff' border-bottom: {'3px solid #000000' if i == 0 else 'none'};
            out += f"""
<div style="display: inline; border-bottom: {"3px solid #000000" if i == 0 else "none"}; background: rgba{color}; border-radius: 3px; padding: 0px" id="_tp_{uuid}_output_{i}_name"
    onclick="_output_onclick_{uuid}({i})"
    onmouseover="_output_onmouseover_{uuid}({i}, this);">{name}</div>"""
        out += "<br><br>"
        for i, name in enumerate(shap_values.output_names):
            out += f"<div id='_tp_{uuid}_output_{i}' style='display: {'block' if i == 0 else 'none'}';>"
            out += my_text(
                shap_values[:, i],
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )
            out += "</div>"
            out += f"<div id='_tp_{uuid}_output_{i}_zoom' style='display: none;'>"
            out += my_text(
                shap_values[:, i],
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                display=False,
            )
            out += "</div>"
        out += "</div>"
        if display:
            _ipython_display_html(out)
            return
        else:
            return out
        # text_to_text(shap_values)
        # return

    if len(shap_values.shape) == 3:
        xmin_computed = None
        xmax_computed = None
        cmax_computed = None

        for i in range(shap_values.shape[-1]):
            for j in range(shap_values.shape[0]):
                values, clustering = unpack_shap_explanation_contents(
                    shap_values[j, :, i]
                )
                tokens, values, group_sizes = process_shap_values(
                    shap_values[j, :, i].data,
                    values,
                    grouping_threshold,
                    separator,
                    clustering,
                )

                xmin_i, xmax_i, cmax_i = values_min_max(
                    values, shap_values[j, :, i].base_values
                )
                if xmin_computed is None or xmin_i < xmin_computed:
                    xmin_computed = xmin_i
                if xmax_computed is None or xmax_i > xmax_computed:
                    xmax_computed = xmax_i
                if cmax_computed is None or cmax_i > cmax_computed:
                    cmax_computed = cmax_i

        if xmin is None:
            xmin = xmin_computed
        if xmax is None:
            xmax = xmax_computed
        if cmax is None:
            cmax = cmax_computed

        out = ""
        for i, v in enumerate(shap_values):
            out += f"""
<br>
<hr style="height: 1px; background-color: #fff; border: none; margin-top: 18px; margin-bottom: 18px; border-top: 1px dashed #ccc;"">
<div align="center" style="margin-top: -35px;"><div style="display: inline-block; background: #fff; padding: 5px; color: #999; font-family: monospace">[{i}]</div>
</div>
            """
            out += my_text(
                v,
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )
        if display:
            _ipython_display_html(out)
            return
        else:
            return out

    # set any unset bounds
    xmin_new, xmax_new, cmax_new = values_min_max(
        shap_values.values, shap_values.base_values
    )
    if xmin is None:
        xmin = xmin_new
    if xmax is None:
        xmax = xmax_new
    if cmax is None:
        cmax = cmax_new

    values, clustering = unpack_shap_explanation_contents(shap_values)
    tokens, values, group_sizes = process_shap_values(
        shap_values.data, values, grouping_threshold, separator, clustering
    )

    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(values))[:num_starting_labels]
    out = ""
    # ev_str = str(shap_values.base_values)
    # vsum_str = str(values.sum())
    # fx_str = str(shap_values.base_values + values.sum())

    # uuid = ''.join(random.choices(string.ascii_lowercase, k=20))
    encoded_tokens = [
        t.replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "") for t in tokens
    ]
    output_name = (
        shap_values.output_names if isinstance(shap_values.output_names, str) else ""
    )
    out += my_svg_force_plot(
        values,
        shap_values.base_values,
        shap_values.base_values + values.sum(),
        encoded_tokens,
        uuid,
        xmin,
        xmax,
        output_name,
    )
    out += "<div align='center'><div style=\"color: rgb(120,120,120); font-size: 12px; margin-top: -15px;\">inputs</div>"
    for i, token in enumerate(tokens):
        scaled_value = 0.5 + 0.5 * values[i] / (cmax + 1e-8)
        color = colors.red_transparent_blue(scaled_value)
        color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])

        # display the labels for the most important words
        label_display = "none"
        wrapper_display = "inline"
        if i in top_inds:
            label_display = "block"
            wrapper_display = "inline-block"

        # create the value_label string
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(values[i].round(3))
        else:
            value_label = str(values[i].round(3)) + " / " + str(group_sizes[i])

        # the HTML for this token
        out += f"""<div style='display: {wrapper_display}; text-align: center;'
    ><div style='display: {label_display}; color: #999; padding-top: 0px; font-size: 12px;'>{value_label}</div
        ><div id='_tp_{uuid}_ind_{i}'
            style='display: inline; background: rgba{color}; border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {{
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            }} else {{
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }}"
            onmouseover="document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 1; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 0; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 0;"
        >{token.replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "")}</div></div>"""
    out += "</div>"

    if display:
        _ipython_display_html(out)
        return
    else:
        return out


def my_svg_force_plot(values, base_values, fx, tokens, uuid, xmin, xmax, output_name):
    def xpos(xval):
        return 100 * (xval - xmin) / (xmax - xmin + 1e-8)

    s = ""
    s += '<svg width="100%" height="80px">'

    ### x-axis marks ###

    # draw x axis line
    s += '<line x1="0" y1="33" x2="100%" y2="33" style="stroke:rgb(150,150,150);stroke-width:1" />'

    # draw base value
    def draw_tick_mark(xval, label=None, bold=False, backing=False):
        s = ""
        s += f'<line x1="{xpos(xval)}%" y1="33" x2="{xpos(xval)}%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" />'
        if not bold:
            if backing:
                s += f'<text x="{xpos(xval)}%" y="27" font-size="13px" style="stroke:#ffffff;stroke-width:8px;" fill="rgb(255,255,255)" dominant-baseline="bottom" text-anchor="middle">{xval:g}</text>'
            s += f'<text x="{xpos(xval)}%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">{xval:g}</text>'
        else:
            if backing:
                s += f'<text x="{xpos(xval)}%" y="27" font-size="13px" style="stroke:#ffffff;stroke-width:8px;" font-weight="bold" fill="rgb(255,255,255)" dominant-baseline="bottom" text-anchor="middle">{xval:g}</text>'
            s += f'<text x="{xpos(xval)}%" y="27" font-size="13px" font-weight="bold" fill="rgb(0,0,0)" dominant-baseline="bottom" text-anchor="middle">{xval:g}</text>'
        if label is not None:
            s += f'<text x="{xpos(xval)}%" y="10" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">{label}</text>'
        return s

    xcenter = round((xmax + xmin) / 2, int(round(1 - np.log10(xmax - xmin + 1e-8))))
    s += draw_tick_mark(xcenter)
    #    np.log10(xmax - xmin)

    tick_interval = round(
        (xmax - xmin) / 7, int(round(1 - np.log10(xmax - xmin + 1e-8)))
    )

    # tick_interval = (xmax - xmin) / 7
    side_buffer = (xmax - xmin) / 14
    for i in range(1, 10):
        pos = xcenter - i * tick_interval
        if pos < xmin + side_buffer:
            break
        s += draw_tick_mark(pos)
    for i in range(1, 10):
        pos = xcenter + i * tick_interval
        if pos > xmax - side_buffer:
            break
        s += draw_tick_mark(pos)
    s += draw_tick_mark(base_values, label="base value", backing=True)
    s += draw_tick_mark(
        fx,
        bold=True,
        label=f'f<tspan baseline-shift="sub" font-size="8px">{output_name}</tspan>(inputs)',
        backing=True,
    )

    ### Positive value marks ###
    red = tuple((colors.red_rgb * 255).tolist())
    light_red = (255, 195, 213)

    # draw base red bar
    x = fx - values[values > 0].sum()
    w = 100 * values[values > 0].sum() / (xmax - xmin + 1e-8)
    s += f'<rect x="{xpos(x)}%" width="{w}%" y="40" height="18" style="fill:rgb{red}; stroke-width:0; stroke:rgb(0,0,0)" />'

    # draw underline marks and the text labels
    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] > 0]
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # a line under the bar to animate
        s += f'<line x1="{xpos(pos)}%" x2="{xpos(last_pos)}%" y1="60" y2="60" id="_fb_{uuid}_ind_{ind}" style="stroke:rgb{red};stroke-width:2; opacity: 0"/>'

        # the text label cropped and centered
        s += f'<text x="{(xpos(last_pos) + xpos(pos)) / 2}%" y="71" font-size="12px" id="_fs_{uuid}_ind_{ind}" fill="rgb{red}" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">{values[ind].round(3)}</text>'

        # the text label cropped and centered
        s += f'<svg x="{xpos(pos)}%" y="40" height="20" width="{xpos(last_pos) - xpos(pos)}%">'
        s += '  <svg x="0" y="0" width="100%" height="100%">'
        s += f'    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">{tokens[ind].strip()}</text>'
        s += "  </svg>"
        s += "</svg>"

        last_pos = pos

    # draw the divider padding (which covers the text near the dividers)
    pos = fx
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        if i != 0:
            for j in range(4):
                s += f'<g transform="translate({2 * j - 8},0)">'
                s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{red};stroke-width:2" />'
                s += "  </svg>"
                s += "</g>"

        if i + 1 != len(inds):
            for j in range(4):
                s += f'<g transform="translate({2 * j - 0},0)">'
                s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{red};stroke-width:2" />'
                s += "  </svg>"
                s += "</g>"

        last_pos = pos

    # center padding
    s += f'<rect transform="translate(-8,0)" x="{xpos(fx)}%" y="40" width="8" height="18" style="fill:rgb{red}"/>'

    # cover up a notch at the end of the red bar
    pos = fx - values[values > 0].sum()
    s += '<g transform="translate(-11.5,0)">'
    s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
    s += '    <path d="M 10 -9 l 6 18 L 10 25 L 0 25 L 0 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />'
    s += "  </svg>"
    s += "</g>"

    # draw the light red divider lines and a rect to handle mouseover events
    pos = fx
    last_pos = pos
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # divider line
        if i + 1 != len(inds):
            s += '<g transform="translate(-1.5,0)">'
            s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
            s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{light_red};stroke-width:2" />'
            s += "  </svg>"
            s += "</g>"

        # mouse over rectangle
        s += f'<rect x="{xpos(pos)}%" y="40" height="20" width="{xpos(last_pos) - xpos(pos)}%"'
        s += '      onmouseover="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'underline';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 1;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 1;"
        s += '"'
        s += '      onmouseout="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'none';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 0;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 0;"
        s += '" style="fill:rgb(0,0,0,0)" />'

        last_pos = pos

    ### Negative value marks ###

    blue = tuple((colors.blue_rgb * 255).tolist())
    light_blue = (208, 230, 250)

    # draw base blue bar
    w = 100 * -values[values < 0].sum() / (xmax - xmin + 1e-8)
    s += f'<rect x="{xpos(fx)}%" width="{w}%" y="40" height="18" style="fill:rgb{blue}; stroke-width:0; stroke:rgb(0,0,0)" />'

    # draw underline marks and the text labels
    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] < 0]
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # a line under the bar to animate
        s += f'<line x1="{xpos(last_pos)}%" x2="{xpos(pos)}%" y1="60" y2="60" id="_fb_{uuid}_ind_{ind}" style="stroke:rgb{blue};stroke-width:2; opacity: 0"/>'

        # the value text
        s += f'<text x="{(xpos(last_pos) + xpos(pos)) / 2}%" y="71" font-size="12px" fill="rgb{blue}" id="_fs_{uuid}_ind_{ind}" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">{values[ind].round(3)}</text>'

        # the text label cropped and centered
        s += f'<svg x="{xpos(last_pos)}%" y="40" height="20" width="{xpos(pos) - xpos(last_pos)}%">'
        s += '  <svg x="0" y="0" width="100%" height="100%">'
        s += f'    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">{tokens[ind].strip()}</text>'
        s += "  </svg>"
        s += "</svg>"

        last_pos = pos

    # draw the divider padding (which covers the text near the dividers)
    pos = fx
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        if i != 0:
            for j in range(4):
                s += f'<g transform="translate({-2 * j + 2},0)">'
                s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{blue};stroke-width:2" />'
                s += "  </svg>"
                s += "</g>"

        if i + 1 != len(inds):
            for j in range(4):
                s += f'<g transform="translate(-{2 * j + 8},0)">'
                s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{blue};stroke-width:2" />'
                s += "  </svg>"
                s += "</g>"

        last_pos = pos

    # center padding
    s += f'<rect transform="translate(0,0)" x="{xpos(fx)}%" y="40" width="8" height="18" style="fill:rgb{blue}"/>'

    # cover up a notch at the end of the blue bar
    pos = fx - values[values < 0].sum()
    s += '<g transform="translate(-6.0,0)">'
    s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
    s += '    <path d="M 8 -9 l -6 18 L 8 25 L 20 25 L 20 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />'
    s += "  </svg>"
    s += "</g>"

    # draw the light blue divider lines and a rect to handle mouseover events
    pos = fx
    last_pos = pos
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # divider line
        if i + 1 != len(inds):
            s += '<g transform="translate(-6.0,0)">'
            s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
            s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{light_blue};stroke-width:2" />'
            s += "  </svg>"
            s += "</g>"

        # mouse over rectangle
        s += f'<rect x="{xpos(last_pos)}%" y="40" height="20" width="{xpos(pos) - xpos(last_pos)}%"'
        s += '      onmouseover="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'underline';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 1;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 1;"
        s += '"'
        s += '      onmouseout="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'none';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 0;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 0;"
        s += '" style="fill:rgb(0,0,0,0)" />'

        last_pos = pos

    s += "</svg>"

    return s
