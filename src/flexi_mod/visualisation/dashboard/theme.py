# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Colours, the Plotly chart template and the page CSS used by the dashboards.

Categorical hues are assigned to *entities* (a market, a fuel, a storage) in a fixed order,
never by rank, so an entity keeps its colour on every chart and when filters change. The
palette is the validated eight-hue reference set; light and dark values are stepped
separately for their own surface rather than inverted.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# Categorical slots 1-8 in their validated order.
CATEGORICAL_LIGHT = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

# One-hue sequential ramp (steps 100 -> 700), light to dark.
SEQUENTIAL_LIGHT = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
# On the dark surface the lightest end is the high-magnitude end.
SEQUENTIAL_DARK = [
    "#1f2d3f",
    "#184f95",
    "#256abf",
    "#3987e5",
    "#6da7ec",
    "#9ec5f4",
    "#cde2fb",
]

# Entity -> categorical slot (index into the palettes above).
ENTITY_SLOT = {
    "day_ahead": 0,
    "intraday": 1,
    "afrr_energy": 2,
    "gas": 3,
    "electric_boiler": 4,
    "storage": 5,
    "afrr_capacity": 6,
    "emissions": 7,
    "electricity": 0,
    "hydrogen": 2,
    "cost": 0,
    "revenue": 2,
    "pv": 3,
    "export": 2,
    "import": 0,
    "ev": 5,
}


@dataclass(frozen=True)
class Theme:
    """Resolved colour tokens for one display mode."""

    mode: str
    page: str
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    axis: str
    border: str
    accent_wash: float
    neutral_series: str
    neutral_fill: str
    categorical: list[str]
    sequential: list[str]

    def entity(self, name: str) -> str:
        """Return the fixed colour of an entity."""

        return self.categorical[ENTITY_SLOT[name]]

    @property
    def accent(self) -> str:
        return self.categorical[0]

    def colorscale(self) -> list[list[float | str]]:
        steps = len(self.sequential) - 1
        return [[index / steps, color] for index, color in enumerate(self.sequential)]


LIGHT = Theme(
    mode="light",
    page="#f9f9f7",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    border="rgba(11,11,11,0.10)",
    accent_wash=0.10,
    neutral_series="#52514e",
    neutral_fill="#c3c2b7",
    categorical=CATEGORICAL_LIGHT,
    sequential=SEQUENTIAL_LIGHT,
)

DARK = Theme(
    mode="dark",
    page="#0d0d0d",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    border="rgba(255,255,255,0.10)",
    accent_wash=0.16,
    neutral_series="#c3c2b7",
    neutral_fill="#383835",
    categorical=CATEGORICAL_DARK,
    sequential=SEQUENTIAL_DARK,
)


def get_theme(mode: str) -> Theme:
    return DARK if mode == "dark" else LIGHT


def color_swap_map(source: Theme, target: Theme) -> dict[str, str]:
    """Return a ``{hex: hex}`` map that re-colours figures built for ``source``."""

    pairs = {
        source.surface: target.surface,
        source.text_primary: target.text_primary,
        source.text_secondary: target.text_secondary,
        source.grid: target.grid,
        source.axis: target.axis,
        source.neutral_series: target.neutral_series,
        source.neutral_fill: target.neutral_fill,
    }
    pairs.update(zip(source.categorical, target.categorical, strict=True))
    pairs.update(zip(source.sequential, target.sequential, strict=True))
    return {key.lower(): value.lower() for key, value in pairs.items() if key != value}


def with_alpha(color: str, alpha: float) -> str:
    """Return ``color`` (``#rrggbb``) as an ``rgba(r,g,b,a)`` string without spaces."""

    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha:g})"


def apply_layout(fig: go.Figure, theme: Theme, height: int = 360) -> None:
    """Apply the shared chart chrome: recessive grid, thin axes, horizontal legend."""

    axis_style = {
        "showgrid": True,
        "gridcolor": theme.grid,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": theme.axis,
        "tickcolor": theme.axis,
        "tickfont": {"color": theme.text_muted, "size": 11},
        "title": {"font": {"color": theme.text_secondary, "size": 12}},
        "automargin": True,
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    # Room for the legend (about six entries per row) and any subplot titles beneath it.
    legend_rows = -(-sum(1 for trace in fig.data if trace.showlegend is not False) // 6)
    top_margin = 14 + 22 * max(legend_rows, 1) + (26 if len(fig.layout.annotations) else 0)
    fig.update_layout(
        height=height,
        font={"family": FONT_FAMILY, "color": theme.text_secondary, "size": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 56, "r": 16, "t": top_margin, "b": 40},
        legend={
            "orientation": "h",
            "xref": "container",
            "yref": "container",
            "yanchor": "top",
            "y": 1.0,
            "x": 0,
            "font": {"size": 11, "color": theme.text_secondary},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "bgcolor": theme.surface,
            "bordercolor": theme.axis,
            "font": {"family": FONT_FAMILY, "color": theme.text_primary, "size": 12},
        },
        hovermode="x unified",
        bargap=0.15,
    )
    for annotation in fig.layout.annotations:
        annotation.font = {"color": theme.text_secondary, "size": 12}
        if annotation.xref == "paper" and annotation.xanchor in (None, "center"):
            # Subplot titles: left-align with the plot area.
            annotation.x = 0
            annotation.xanchor = "left"


# ---------------------------------------------------------------------------
# Page styling shared by the static HTML dashboard and the Dash app
# ---------------------------------------------------------------------------


def css_tokens(theme: Theme) -> str:
    """Return the page colours of ``theme`` as CSS custom properties.

    The page and the charts read the same values, so a colour changes in one place.
    """

    return f"""
  color-scheme: {theme.mode};
  --page: {theme.page};
  --surface: {theme.surface};
  --text-primary: {theme.text_primary};
  --text-secondary: {theme.text_secondary};
  --text-muted: {theme.text_muted};
  --grid: {theme.grid};
  --axis: {theme.axis};
  --border: {theme.border};
  --accent: {theme.accent};
  --accent-wash: {with_alpha(theme.accent, theme.accent_wash)};
"""


# Light is the default; dark applies for an OS dark preference unless the viewer chose light,
# and unconditionally when ``data-theme="dark"``.
BASE_CSS = (
    ":root {"
    + css_tokens(LIGHT)
    + '}\n@media (prefers-color-scheme: dark) {\n  :root:not([data-theme="light"]) {'
    + css_tokens(DARK)
    + '}\n}\n:root[data-theme="dark"] {'
    + css_tokens(DARK)
    + "}\n"
    + """
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 14px;
  line-height: 1.45;
}
.app { max-width: 1480px; margin: 0 auto; padding: 20px 16px 48px; }
header.top {
  display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: flex-end;
  justify-content: space-between; margin-bottom: 16px;
}
header.top h1 { margin: 0; font-size: 22px; font-weight: 650; letter-spacing: -0.01em; }
header.top .meta { color: var(--text-secondary); margin-top: 4px; display: flex; flex-wrap: wrap; gap: 4px 14px; }
header.top .meta span + span::before { content: "·"; margin-right: 14px; color: var(--text-muted); }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.controls label { color: var(--text-secondary); font-size: 12px; display: flex; flex-direction: column; gap: 3px; }
select, input[type="search"], input[type="text"], .btn {
  font: inherit; color: var(--text-primary); background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px; min-height: 34px;
}
select { max-width: 320px; }
.btn { cursor: pointer; }
.btn:hover, select:hover { background: var(--accent-wash); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
nav.tabs {
  display: flex; gap: 2px; overflow-x: auto; border-bottom: 1px solid var(--border);
  margin-bottom: 16px; scrollbar-width: thin;
}
nav.tabs button {
  font: inherit; color: var(--text-secondary); background: none; border: 0;
  border-bottom: 2px solid transparent; padding: 10px 14px; cursor: pointer; white-space: nowrap;
}
nav.tabs button:hover { color: var(--text-primary); background: var(--accent-wash); }
nav.tabs button[aria-selected="true"] {
  color: var(--text-primary); border-bottom-color: var(--accent); font-weight: 600;
}
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
.kpi .label { color: var(--text-secondary); font-size: 12px; }
.kpi .value { font-size: 24px; font-weight: 650; letter-spacing: -0.01em; margin-top: 2px; }
.kpi .detail { color: var(--text-muted); font-size: 12px; margin-top: 2px; min-height: 17px; overflow-wrap: anywhere; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px 14px 8px; min-width: 0; }
.card.wide { grid-column: 1 / -1; }
.card h2 { font-size: 15px; font-weight: 650; margin: 0; display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.card h2 .caption { font-size: 12px; font-weight: 500; color: var(--text-secondary); }
.card p.desc { color: var(--text-secondary); margin: 3px 0 6px; font-size: 12.5px; max-width: 90ch; }
.chart { width: 100%; min-height: 120px; }
.table-wrap { overflow: auto; max-height: 520px; border: 1px solid var(--border); border-radius: 6px; margin-top: 8px; }
table.data { border-collapse: collapse; width: 100%; font-size: 12.5px; }
table.data th, table.data td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--grid); white-space: nowrap; }
table.data th { position: sticky; top: 0; background: var(--surface); color: var(--text-secondary); font-weight: 600; }
table.data td.num { text-align: right; font-variant-numeric: tabular-nums; }
.empty { color: var(--text-secondary); padding: 40px 0; text-align: center; }
.note { color: var(--text-secondary); font-size: 12.5px; margin-top: 18px; }
.hidden, [hidden] { display: none !important; }
@media (max-width: 960px) {
  .grid { grid-template-columns: minmax(0, 1fr); }
  .card.wide { grid-column: auto; }
}
"""
)
