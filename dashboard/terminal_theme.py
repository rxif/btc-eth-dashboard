"""
terminal_theme.py — "Financial terminal" Plotly theme (black background, amber orange, monospace)
=================================================================================================

Meant to be used with the `style.css` file from the same repo (placed in `assets/`).
The CSS styles the page (background, titles, cards, tabs, dropdowns); this module styles
the CHARTS, which CSS cannot reach.

Usage
-----
    import terminal_theme                      # <- that's it: every go.Figure() is styled
    from terminal_theme import C, FONT         # colors / font when you need them explicitly

    fig = go.Figure()
    fig.add_scatter(x=..., y=...)              # 1st trace orange, 2nd blue, etc.

If the file lives inside a package (e.g. `dashboard/`), import it as
`import dashboard.terminal_theme` and `from dashboard.terminal_theme import C`.

What the import does automatically
----------------------------------
- registers a Plotly template named "terminal" and sets it as the default;
- transparent background (the black comes from the CSS), dark grey grid, IBM Plex Mono font;
- trace color order: orange, blue, yellow, green, red, amber, grey;
- black hover labels with an orange border, horizontal legend on top;
- margins that grow automatically so axis labels are never cut off;
- orange gradient by default for 3D surfaces, heatmaps and contours;
- 3D axes (scene) styled the same way.

Sizes and colors to remember
----------------------------
    Chart font ............ IBM Plex Mono, 12 px, #e6e6e6
    Chart title ........... 14 px, orange #ff9900
    Orange (primary) ...... #ff9900     Blue ...... #3d8bff
    Yellow (highlight) .... #ffd400     Amber ..... #ffb84d
    Up .................... #00d26a     Down ...... #ff3b3b
    Grid / borders ........ #2a2a2a     Secondary text #8a8a8a

Dash side: keep `external_stylesheets=[dbc.themes.DARKLY]` as the dark Bootstrap base;
the CSS in assets/ overrides it.

To turn the theme off for one chart: fig.update_layout(template="plotly_dark")
"""

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
class C:
    ORANGE = "#ff9900"
    BLUE = "#3d8bff"
    YELLOW = "#ffd400"
    AMBER = "#ffb84d"
    UP = "#00d26a"
    DOWN = "#ff3b3b"
    TEXT = "#e6e6e6"
    MUTED = "#8a8a8a"
    GRID = "#2a2a2a"
    CARD = "#0f0f0f"
    BLACK = "#000000"


COLORWAY = [C.ORANGE, C.BLUE, C.YELLOW, C.UP, C.DOWN, C.AMBER, C.MUTED]

SCALE = [[0.0, "#3a1a00"], [0.35, "#8a3c00"], [0.65, "#ff6a00"],
         [0.85, C.ORANGE], [1.0, C.YELLOW]]

# Diverging scale (down -> neutral -> up), useful for changes / P&L
DIVERGING = [[0.0, C.DOWN], [0.5, "#1a1a1a"], [1.0, C.UP]]

FONT_FAMILY = "IBM Plex Mono, Consolas, Courier New, monospace"
FONT = dict(family=FONT_FAMILY, size=12, color=C.TEXT)

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
_axis = dict(
    gridcolor=C.GRID, linecolor=C.GRID, zeroline=False, automargin=True,
    tickfont=dict(color=C.MUTED), title=dict(font=dict(color=C.MUTED), standoff=12),
)
_axis3d = dict(
    gridcolor=C.GRID, zerolinecolor=C.GRID, showbackground=False,
    backgroundcolor="rgba(0,0,0,0)", tickfont=dict(color=C.MUTED),
    title=dict(font=dict(color=C.MUTED)),
)
_colorbar = dict(thickness=12, len=0.6, outlinewidth=0, tickfont=FONT,
                 title=dict(font=dict(color=C.MUTED)))

terminal = go.layout.Template(
    layout=dict(
        font=FONT,
        title=dict(font=dict(size=14, color=C.ORANGE), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        colorscale=dict(sequential=SCALE, diverging=DIVERGING),
        margin=dict(l=60, r=20, t=40, b=50),
        xaxis=_axis, yaxis=_axis,
        scene=dict(bgcolor="rgba(0,0,0,0)", xaxis=_axis3d, yaxis=_axis3d, zaxis=_axis3d),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C.TEXT)),
        hoverlabel=dict(bgcolor=C.BLACK, bordercolor=C.ORANGE, font=FONT),
        hovermode="closest",
        coloraxis=dict(colorbar=_colorbar),
        shapedefaults=dict(line=dict(color=C.ORANGE)),
        annotationdefaults=dict(font=dict(color=C.MUTED), arrowcolor=C.ORANGE),
    ),
    data=dict(
        scatter=[go.Scatter(line=dict(width=2), marker=dict(size=7))],
        bar=[go.Bar(marker=dict(line=dict(width=0)))],
        surface=[go.Surface(colorscale=SCALE, colorbar=_colorbar)],
        heatmap=[go.Heatmap(colorscale=SCALE, colorbar=_colorbar)],
        contour=[go.Contour(colorscale=SCALE, colorbar=_colorbar)],
        candlestick=[go.Candlestick(
            increasing=dict(line=dict(color=C.UP), fillcolor=C.UP),
            decreasing=dict(line=dict(color=C.DOWN), fillcolor=C.DOWN),
        )],
    ),
)

pio.templates["terminal"] = terminal
pio.templates.default = "terminal"

# ---------------------------------------------------------------------------
# Table style (for dash_table.DataTable)
# ---------------------------------------------------------------------------
TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_header={
        "backgroundColor": C.CARD, "color": C.ORANGE, "fontWeight": "bold",
        "border": f"1px solid {C.GRID}",
    },
    style_cell={
        "backgroundColor": "transparent", "color": C.TEXT,
        "border": f"1px solid {C.GRID}", "textAlign": "center",
        "fontFamily": FONT_FAMILY, "fontSize": 13, "padding": "8px",
    },
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def empty_figure(msg="No data", height=480):
    """Empty figure with a centered message, styled with the theme."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=msg, showarrow=False, font=dict(size=16, color=C.MUTED))],
    )
    return fig