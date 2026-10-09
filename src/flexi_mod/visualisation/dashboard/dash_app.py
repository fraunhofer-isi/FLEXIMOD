# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Live Dash front end: browse case outputs, filter by plant and period, compare cases.

The charts and tabs come from the same builders as the static ``dashboard.html`` so both
front ends always show the same content. Start it with ``fleximod-dashboard``.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import pandas as pd

from flexi_mod.visualisation.analytics import RESULT_FILES, resolve_table_path
from flexi_mod.visualisation.dashboard.charts import ChartContext
from flexi_mod.visualisation.dashboard.comparison import (
    METRIC_LABELS,
    available_metrics,
    comparison_kpis,
    comparison_table,
    group_columns,
    load_case_summaries,
    ranking_chart,
    tradeoff_chart,
)
from flexi_mod.visualisation.dashboard.data import (
    ALL_PLANTS,
    CaseData,
    discover_cases,
    load_case,
)
from flexi_mod.visualisation.dashboard.sections import Block, Tab, build_tabs, tab_ids
from flexi_mod.visualisation.dashboard.theme import BASE_CSS, get_theme

RESOLUTION_OPTIONS = [
    {"label": "Auto resolution", "value": "auto"},
    {"label": "Native steps", "value": "native"},
    {"label": "Hourly", "value": "1h"},
    {"label": "6-hourly", "value": "6h"},
    {"label": "Daily", "value": "1D"},
    {"label": "Weekly", "value": "1W"},
]

# Map Dash 4 widget variables (dropdowns, date picker) onto the dashboard colour tokens.
_DASH_THEME_CSS = """
:root, :root[data-theme] {
  --Dash-Fill-Inverse-Strong: var(--surface);
  --Dash-Text-Strong: var(--text-primary);
  --Dash-Text-Primary: var(--text-primary);
  --Dash-Text-Weak: var(--text-secondary);
  --Dash-Text-Disabled: var(--text-muted);
  --Dash-Stroke-Strong: var(--axis);
  --Dash-Stroke-Weak: var(--grid);
  --Dash-Fill-Interactive-Strong: var(--accent);
  --Dash-Fill-Interactive-Weak: var(--accent-wash);
  --Dash-Fill-Primary-Hover: var(--accent-wash);
  --Dash-Fill-Primary-Active: var(--accent-wash);
  --Dash-Fill-Disabled: var(--grid);
  --Dash-Tooltip-Background-Color: var(--surface);
  --Dash-Tooltip-Border-Color: var(--axis);
}
.field {
  display: flex; flex-direction: column; gap: 3px; font-size: 12px; color: var(--text-secondary);
}
.field > .dash-dropdown, .field > div { min-width: 190px; }
.field.wide > div { min-width: 320px; }
.field input[type="text"], .field input[type="number"] { min-width: 260px; }
.tabs-radio {
  display: flex; gap: 2px; overflow-x: auto; margin-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.tabs-radio label {
  color: var(--text-secondary); border-bottom: 2px solid transparent; padding: 10px 14px;
  cursor: pointer; white-space: nowrap; display: block;
}
.tabs-radio label:hover { color: var(--text-primary); background: var(--accent-wash); }
.tabs-radio input { position: absolute; opacity: 0; pointer-events: none; }
.tabs-radio label:has(input:checked) {
  color: var(--text-primary); border-bottom-color: var(--accent); font-weight: 600;
}
.tabs-radio.top { margin-bottom: 12px; }
.panel { margin-top: 4px; }
.dash-spreadsheet-container .dash-spreadsheet-inner *, .dash-spreadsheet-container input {
  font-family: inherit !important;
}
.loading-note { color: var(--text-secondary); padding: 24px 0; }
"""


def _require_dash():
    try:
        import dash  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "The live dashboard needs Dash. Install it with: pip install 'dash>=2.18' "
            "(or pip install -e .[dashboard])."
        ) from error


@lru_cache(maxsize=3)
def _cached_case(path: str, mtime: float) -> CaseData:
    return load_case(path)


def get_case(path: str) -> CaseData:
    """Load (and memoise) a case; the cache is keyed on the dispatch file's mtime."""

    table = resolve_table_path(Path(path) / RESULT_FILES["dispatch_results"])
    return _cached_case(path, table.stat().st_mtime if table.exists() else 0.0)


@lru_cache(maxsize=4)
def _cached_summaries(roots: tuple[str, ...]) -> pd.DataFrame:
    return load_case_summaries(roots)


def create_app(root: str | Path = "data/output", initial_case: str | Path | None = None):
    """Build the Dash app. ``root`` is scanned for case output folders."""

    _require_dash()
    from dash import Dash, dcc, html

    app = Dash(__name__, title="FLEXIMOD dashboard", suppress_callback_exceptions=True)
    app.index_string = (
        '<!DOCTYPE html><html lang="en"><head>{%metas%}<title>{%title%}</title>{%favicon%}'
        "{%css%}<style>"
        + BASE_CSS
        + _DASH_THEME_CSS
        + "</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}"
        "</footer></body></html>"
    )
    root = Path(root).resolve()
    initial = str(Path(initial_case).resolve()) if initial_case else None

    app.layout = html.Div(
        className="app",
        children=[
            dcc.Store(id="theme", storage_type="local"),
            html.Header(
                className="top",
                children=[
                    html.Div(
                        [
                            html.H1("FLEXIMOD dashboard"),
                            html.Div(id="case-meta", className="meta"),
                        ]
                    ),
                    html.Div(
                        className="controls",
                        children=[
                            html.Button(
                                "Dark mode", id="theme-toggle", n_clicks=0, className="btn"
                            ),
                        ],
                    ),
                ],
            ),
            dcc.RadioItems(
                id="main-tabs",
                className="tabs-radio top",
                options=[
                    {"label": "Case dashboard", "value": "case"},
                    {"label": "Compare cases", "value": "compare"},
                ],
                value="case",
                inline=True,
            ),
            html.Div(id="case-view", children=_case_controls(str(root))),
            html.Div(
                id="compare-view", style={"display": "none"}, children=_compare_controls(str(root))
            ),
        ],
    )
    _register_callbacks(app, root, initial)
    return app


def _case_controls(root: str):
    from dash import dcc, html

    return [
        html.Div(
            className="controls",
            style={"marginBottom": "14px"},
            children=[
                _field(
                    "Output folder to scan",
                    dcc.Input(id="root-input", type="text", value=root, debounce=True),
                    "wide",
                ),
                _field("Case", dcc.Dropdown(id="case-dd", clearable=False, options=[]), "wide"),
                _field("Plant", dcc.Dropdown(id="plant-dd", clearable=False, options=[])),
                _field(
                    "Period",
                    dcc.DatePickerRange(
                        id="date-range", display_format="DD MMM YYYY", clearable=True
                    ),
                ),
                _field(
                    "Resolution",
                    dcc.Dropdown(
                        id="resolution-dd",
                        clearable=False,
                        options=RESOLUTION_OPTIONS,
                        value="auto",
                    ),
                ),
            ],
        ),
        dcc.RadioItems(id="tab-radio", className="tabs-radio", options=[], inline=True),
        dcc.Loading(html.Div(id="case-content", className="panel"), type="dot"),
    ]


def _compare_controls(root: str):
    from dash import dcc, html

    return [
        html.Div(
            className="controls",
            style={"marginBottom": "14px"},
            children=[
                _field(
                    "Folders to compare (comma separated)",
                    dcc.Input(id="cmp-roots", type="text", value=root, debounce=True),
                    "wide",
                ),
                _field("Rank by", dcc.Dropdown(id="cmp-metric", clearable=False, options=[])),
                _field("Colour by", dcc.Dropdown(id="cmp-group", clearable=False, options=[])),
                _field(
                    "Show best",
                    dcc.Input(
                        id="cmp-top", type="number", value=30, min=5, max=200, step=5, debounce=True
                    ),
                ),
                _field("Scatter x", dcc.Dropdown(id="cmp-x", clearable=False, options=[])),
                _field("Scatter y", dcc.Dropdown(id="cmp-y", clearable=False, options=[])),
            ],
        ),
        dcc.Loading(html.Div(id="cmp-content", className="panel"), type="dot"),
    ]


def _field(label: str, component, extra: str = ""):
    from dash import html

    return html.Label(className=f"field {extra}".strip(), children=[label, component])


def _graph(figure, height: int):
    from dash import dcc

    return dcc.Graph(
        figure=figure,
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        },
        style={"height": f"{height}px"},
    )


def _table(frame: pd.DataFrame, page_size: int = 15):
    from dash import dash_table
    from dash.dash_table.Format import Format, Scheme

    columns = []
    for name in frame.columns:
        column = {"name": str(name), "id": str(name)}
        if pd.api.types.is_float_dtype(frame[name]):
            column.update(
                type="numeric", format=Format(precision=4, scheme=Scheme.decimal_si_prefix)
            )
        elif pd.api.types.is_integer_dtype(frame[name]):
            column["type"] = "numeric"
        columns.append(column)
    cell = {
        "backgroundColor": "var(--surface)",
        "color": "var(--text-primary)",
        "borderBottom": "1px solid var(--grid)",
        "borderTop": "0",
        "borderLeft": "0",
        "borderRight": "0",
        "padding": "6px 10px",
        "fontFamily": "inherit",
        "fontSize": "12.5px",
        "textAlign": "left",
        "minWidth": "90px",
        "maxWidth": "420px",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    }
    return dash_table.DataTable(
        data=frame.astype(object).where(frame.notna(), None).to_dict("records"),
        columns=columns,
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell=cell,
        style_header={**cell, "color": "var(--text-secondary)", "fontWeight": 600},
        style_filter={**cell, "color": "var(--text-primary)"},
        style_data_conditional=[],
    )


def _card(block: Block):
    from dash import html

    head = [
        html.H2(
            [block.title, html.Span(block.caption, className="caption")]
            if block.caption
            else block.title
        )
    ]
    if block.description:
        head.append(html.P(block.description, className="desc"))
    if block.figure is not None:
        height = int(block.figure.layout.height or 360)
        body = _graph(block.figure, height)
    elif block.table is not None:
        body = _table(block.table)
    else:
        body = html.Div()
    wide = " wide" if block.wide or block.table is not None else ""
    return html.Article(className=f"card{wide}", children=[*head, body])


def _kpi_row(tab: Tab):
    from dash import html

    return html.Div(
        className="kpis",
        children=[
            html.Div(
                className="kpi",
                children=[
                    html.Div(kpi.label, className="label"),
                    html.Div(kpi.value, className="value"),
                    html.Div(kpi.detail, className="detail"),
                ],
            )
            for kpi in tab.kpis
        ],
    )


def _register_callbacks(app, root: Path, initial: str | None) -> None:
    from dash import Input, Output, State, html, no_update

    app.clientside_callback(
        """
        function (clicks, stored) {
          const media = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
          let mode = stored || (media && media.matches ? 'dark' : 'light');
          if (clicks) { mode = mode === 'dark' ? 'light' : 'dark'; }
          document.documentElement.setAttribute('data-theme', mode);
          return mode;
        }
        """,
        Output("theme", "data"),
        Input("theme-toggle", "n_clicks"),
        State("theme", "data"),
    )
    app.clientside_callback(
        """
        function (mode) { return mode === 'dark' ? 'Light mode' : 'Dark mode'; }
        """,
        Output("theme-toggle", "children"),
        Input("theme", "data"),
    )

    @app.callback(
        Output("case-view", "style"),
        Output("compare-view", "style"),
        Input("main-tabs", "value"),
    )
    def switch_view(view):
        show, hide = {"display": "block"}, {"display": "none"}
        return (show, hide) if view == "case" else (hide, show)

    # ---- case dashboard ----------------------------------------------------------
    @app.callback(
        Output("case-dd", "options"),
        Output("case-dd", "value"),
        Input("root-input", "value"),
    )
    def scan(folder):
        base = Path(folder or root)
        cases = discover_cases(base)
        options = [
            {
                "label": str(case.relative_to(base)) if case != base else case.name,
                "value": str(case),
            }
            for case in cases
        ]
        values = [option["value"] for option in options]
        chosen = initial if initial in values else (values[0] if values else None)
        if initial and initial not in values and Path(initial).exists():
            options.insert(0, {"label": Path(initial).name, "value": initial})
            chosen = initial
        return options, chosen

    @app.callback(
        Output("plant-dd", "options"),
        Output("plant-dd", "value"),
        Output("date-range", "min_date_allowed"),
        Output("date-range", "max_date_allowed"),
        Output("date-range", "start_date"),
        Output("date-range", "end_date"),
        Output("tab-radio", "options"),
        Output("tab-radio", "value"),
        Output("case-meta", "children"),
        Input("case-dd", "value"),
    )
    def case_changed(path):
        if not path:
            return [], None, None, None, None, None, [], None, "No case output found"
        case = get_case(path)
        first, last = case.period
        ctx = ChartContext(case)
        tabs = tab_ids(ctx)
        meta = [
            html.Span(case.name),
            html.Span(case.family_label),
            html.Span(f"{first:%d %b %Y} – {last:%d %b %Y}"),
            html.Span(f"{len(case.plants)} plant{'s' if len(case.plants) != 1 else ''}"),
            html.Span(f"{case.step_hours * 60:g}-min data"),
        ]
        return (
            [{"label": label, "value": value} for value, label in case.plant_options()],
            ALL_PLANTS,
            first.date(),
            last.date(),
            first.date(),
            last.date(),
            [{"label": label, "value": tab_id} for tab_id, label in tabs],
            tabs[0][0],
            meta,
        )

    @app.callback(
        Output("case-content", "children"),
        Input("case-dd", "value"),
        Input("plant-dd", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("resolution-dd", "value"),
        Input("tab-radio", "value"),
        Input("theme", "data"),
    )
    def render_case(path, plant, start, end, resolution, tab, mode):
        if not path:
            return html.P(
                "No case folders found below the selected output folder.", className="empty"
            )
        case = get_case(path)
        ctx = ChartContext(
            case, plant or ALL_PLANTS, get_theme(mode or "light"), start, end, resolution or "auto"
        )
        valid = [tab_id for tab_id, _ in tab_ids(ctx)]
        selected = tab if tab in valid else valid[0]
        tabs = build_tabs(ctx, only=selected)
        if not tabs:
            return html.P("Nothing to show for this selection.", className="empty")
        content = tabs[0]
        children = []
        if content.kpis:
            children.append(_kpi_row(content))
        children.append(
            html.Div(className="grid", children=[_card(block) for block in content.blocks])
        )
        return children

    # ---- comparison -----------------------------------------------------------------
    def _roots(text: str | None) -> tuple[str, ...]:
        return tuple(part.strip() for part in (text or str(root)).split(",") if part.strip())

    @app.callback(
        Output("cmp-metric", "options"),
        Output("cmp-metric", "value"),
        Output("cmp-group", "options"),
        Output("cmp-group", "value"),
        Output("cmp-x", "options"),
        Output("cmp-x", "value"),
        Output("cmp-y", "options"),
        Output("cmp-y", "value"),
        Input("cmp-roots", "value"),
        Input("main-tabs", "value"),
    )
    def compare_options(text, view):
        if view != "compare":
            return (no_update,) * 8
        frame = _cached_summaries(_roots(text))
        if frame.empty:
            return [], None, [], None, [], None, [], None
        metrics = [{"label": label, "value": key} for key, label in available_metrics(frame)]
        keys = [option["value"] for option in metrics]
        groups = [{"label": label, "value": key} for key, label in group_columns(frame)]
        pick = lambda name, fallback: name if name in keys else fallback  # noqa: E731
        return (
            metrics,
            pick("net_cost", keys[0]),
            groups,
            groups[0]["value"] if groups else None,
            metrics,
            pick("net_cost", keys[0]),
            metrics,
            pick("co2", keys[min(1, len(keys) - 1)]),
        )

    @app.callback(
        Output("cmp-content", "children"),
        Input("cmp-roots", "value"),
        Input("cmp-metric", "value"),
        Input("cmp-group", "value"),
        Input("cmp-top", "value"),
        Input("cmp-x", "value"),
        Input("cmp-y", "value"),
        Input("theme", "data"),
        Input("main-tabs", "value"),
    )
    def render_compare(text, metric, group, top, x, y, mode, view):
        if view != "compare":
            return no_update
        frame = _cached_summaries(_roots(text))
        if frame.empty:
            return html.P(
                "No summary_indicators tables found in the selected folders.", className="empty"
            )
        theme = get_theme(mode or "light")
        top = int(top or 30)
        children = [
            html.Div(
                className="kpis",
                children=[
                    html.Div(
                        className="kpi",
                        children=[
                            html.Div(label, className="label"),
                            html.Div(value, className="value"),
                            html.Div(detail, className="detail"),
                        ],
                    )
                    for label, value, detail in comparison_kpis(frame)
                ],
            )
        ]
        cards = []
        ranking = ranking_chart(frame, metric, group, top, theme) if metric else None
        if ranking is not None:
            cards.append(
                _card(
                    Block(
                        "ranking",
                        "Ranking",
                        caption=f"{METRIC_LABELS.get(metric, metric)} — lowest first",
                        figure=ranking,
                    )
                )
            )
        scatter = tradeoff_chart(frame, x, y, group, theme) if x and y else None
        if scatter is not None:
            cards.append(
                _card(
                    Block(
                        "tradeoff",
                        "Trade-off",
                        caption=f"{METRIC_LABELS.get(x, x)} vs {METRIC_LABELS.get(y, y)}",
                        figure=scatter,
                        wide=True,
                    )
                )
            )
        cards.append(
            _card(
                Block(
                    "table",
                    "All cases",
                    "Sort or filter the columns; open a case under 'Case dashboard'.",
                    table=comparison_table(frame),
                )
            )
        )
        children.append(html.Div(className="grid", children=cards))
        return children


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the live FLEXIMOD Dash dashboard.")
    parser.add_argument("--root", default="data/output", help="Folder scanned for case outputs.")
    parser.add_argument("--case", help="Output folder of a case to open first.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    args = parser.parse_args()
    app = create_app(args.root, args.case)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
