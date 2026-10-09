# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare many case output folders: summary table, ranking bars and trade-off scatter."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from flexi_mod.visualisation.analytics import RESULT_FILES, resolve_table_path
from flexi_mod.visualisation.dashboard.charts import money, number
from flexi_mod.visualisation.dashboard.data import column_kind, discover_cases
from flexi_mod.visualisation.dashboard.static_html import (
    _render_table,
    page_shell,
    plotly_script,
)
from flexi_mod.visualisation.dashboard.theme import (
    LIGHT,
    Theme,
    apply_layout,
)

COMPARISON_FILENAME = "comparison.html"

# metric key -> (label, unit, candidate summary columns, scale)
_METRIC_SOURCES: dict[str, tuple[str, str, tuple[str, ...], float]] = {
    "net_cost": (
        "Net operating cost",
        "EUR",
        (
            "net_operating_cost_incl_grid_fees_EUR",
            "net_operating_cost_EUR",
            "total_net_operating_cost_EUR",
            "total_cost",
        ),
        1.0,
    ),
    "electricity": (
        "Electricity use",
        "MWh",
        (
            "total_electricity_consumption_MWh",
            "total_actual_electricity_consumption_MWh",
            "total_grid_import_MWh",
        ),
        1.0,
    ),
    "co2": (
        "CO₂ emissions",
        "t",
        ("total_co2_emissions_t", "total_emissions_kg"),
        1.0,
    ),
    "production": (
        "Production",
        "t",
        ("total_clinker_production_t", "total_steel_production_t"),
        1.0,
    ),
    "heat": ("Heat demand", "MWh th", ("total_heat_demand_MWh",), 1.0),
    "afrr_capacity": (
        "aFRR capacity value",
        "EUR",
        ("total_afrr_capacity_net_value_EUR",),
        1.0,
    ),
    "grid_fees": ("Grid fees", "EUR", ("grid_fee_total_EUR",), 1.0),
}

METRIC_LABELS = {
    "net_cost": "Net operating cost (EUR)",
    "electricity": "Electricity use (MWh)",
    "co2": "CO₂ emissions (t)",
    "production": "Production (t)",
    "heat": "Heat demand (MWh th)",
    "afrr_capacity": "aFRR capacity value (EUR)",
    "grid_fees": "Grid fees (EUR)",
    "specific_cost": "Specific cost (EUR per t or MWh th)",
    "specific_co2": "Specific CO₂ (t per t product)",
    "specific_electricity": "Specific electricity (MWh per t)",
}

_YEAR = re.compile(r"(?<![0-9])(20[2-9][0-9])(?![0-9])")
_FAMILY_MARKERS = {
    "total_clinker_production_t": "cement",
    "total_steel_production_t": "steel",
    "total_heat_demand_MWh": "steam",
    "total_bus_charge_MWh": "building",
}


def load_case_summaries(
    roots: Iterable[str | Path],
    max_depth: int = 3,
) -> pd.DataFrame:
    """Return one row per case with comparable indicators.

    ``roots`` may be case output folders or folders that contain many of them.
    """

    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        folders = (
            [root]
            if resolve_table_path(root / RESULT_FILES["summary_indicators"]).exists()
            else discover_cases(root, max_depth, table="summary_indicators")
        )
        for folder in folders:
            if folder in seen:
                continue
            seen.add(folder)
            row = _summarise_case(folder)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def _summarise_case(folder: Path) -> dict[str, object] | None:
    path = resolve_table_path(folder / RESULT_FILES["summary_indicators"])
    if not path.exists():
        return None
    summary = pd.read_csv(path)
    if summary.empty:
        return None
    numeric = summary.select_dtypes("number")
    totals = {
        column: float(
            numeric[column].mean() if column_kind(column) == "price" else numeric[column].sum()
        )
        for column in numeric.columns
    }

    row: dict[str, object] = {"case": folder.name, "path": str(folder), "plants": len(summary)}
    row["family"] = next(
        (family for column, family in _FAMILY_MARKERS.items() if column in summary.columns),
        "other",
    )
    row.update(_name_parts(folder.name))
    for key, (_, _, candidates, scale) in _METRIC_SOURCES.items():
        column = next((c for c in candidates if c in totals), None)
        if column is None:
            continue
        value = totals[column] * scale
        row[key] = value / 1000.0 if column.endswith("_kg") else value

    production = row.get("production") or row.get("heat")
    if production and "net_cost" in row:
        row["specific_cost"] = float(row["net_cost"]) / float(production)  # type: ignore[arg-type]
    if row.get("production"):
        if "co2" in row:
            row["specific_co2"] = float(row["co2"]) / float(row["production"])  # type: ignore[arg-type]
        if "electricity" in row:
            row["specific_electricity"] = float(row["electricity"]) / float(row["production"])  # type: ignore[arg-type]
    return row


def _name_parts(name: str) -> dict[str, object]:
    """Split names such as ``scenario_2030_route...`` into scenario, year and variant."""

    match = _YEAR.search(name)
    if not match:
        return {"scenario": name, "year": None, "variant": ""}
    scenario = name[: match.start()].strip("_") or name
    variant = name[match.end() :].strip("_")
    return {"scenario": scenario, "year": int(match.group(1)), "variant": variant}


def available_metrics(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Return ``(key, label)`` for metrics that have at least one value."""

    return [
        (key, label)
        for key, label in METRIC_LABELS.items()
        if key in frame.columns and frame[key].notna().any()
    ]


def group_columns(frame: pd.DataFrame) -> list[tuple[str, str]]:
    options = [("family", "Plant family"), ("scenario", "Scenario"), ("year", "Year")]
    return [(key, label) for key, label in options if key in frame and frame[key].nunique() > 1]


def _group_labels(frame: pd.DataFrame, group: str | None, limit: int = 7) -> pd.Series:
    """Label every row with a group; rarer groups beyond ``limit`` fold into 'Other'."""

    if not group or group not in frame:
        return pd.Series("Cases", index=frame.index)
    labels = frame[group].astype(str)
    keep = labels.value_counts().index[:limit]
    return labels.where(labels.isin(keep), "Other")


def _group_colors(labels: pd.Series, theme: Theme) -> dict[str, str]:
    """Fixed colour per group in order of appearance; 'Other' is neutral."""

    ordered = [g for g in sorted(labels.unique(), key=_natural) if g != "Other"]
    colors = {group: theme.categorical[index % 8] for index, group in enumerate(ordered)}
    if "Other" in labels.values:
        colors["Other"] = theme.neutral_fill
    return colors


def _natural(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def ranking_chart(
    frame: pd.DataFrame,
    metric: str,
    group: str | None = None,
    top: int = 30,
    theme: Theme = LIGHT,
    ascending: bool = True,
) -> go.Figure | None:
    """Horizontal bars of one metric, best (lowest) first, coloured by group."""

    if metric not in frame or not frame[metric].notna().any():
        return None
    data = frame.dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(top)
    labels = _group_labels(frame, group).loc[data.index]
    colors = _group_colors(_group_labels(frame, group), theme)
    fig = go.Figure()
    shown = list(dict.fromkeys(labels))
    for name in sorted(shown, key=_natural):
        part = data[labels == name]
        fig.add_trace(
            go.Bar(
                y=part["case"],
                x=part[metric].to_numpy(dtype="float64"),
                orientation="h",
                name=name,
                marker={"color": colors[name], "line": {"width": 0}},
                hovertemplate="%{x:,.2f}<extra>%{y}</extra>",
            )
        )
    fig.update_yaxes(autorange="reversed", automargin=True, tickfont={"size": 10})
    fig.update_xaxes(title_text=METRIC_LABELS.get(metric, metric), tickformat=".3s")
    apply_layout(fig, theme, height=max(320, 120 + 22 * len(data)))
    fig.update_layout(hovermode="closest", showlegend=len(shown) > 1, margin={"l": 8})
    return fig


def tradeoff_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    group: str | None = None,
    theme: Theme = LIGHT,
) -> go.Figure | None:
    """Scatter of two metrics, one trace per group (capped at seven plus 'Other')."""

    if x not in frame or y not in frame or x == y:
        return None
    data = frame.dropna(subset=[x, y])
    if len(data) < 2:
        return None
    all_labels = _group_labels(frame, group)
    labels = all_labels.loc[data.index]
    colors = _group_colors(all_labels, theme)
    fig = go.Figure()
    for name in sorted(dict.fromkeys(labels), key=_natural):
        part = data[labels == name]
        fig.add_trace(
            go.Scatter(
                x=part[x].to_numpy(dtype="float64"),
                y=part[y].to_numpy(dtype="float64"),
                mode="markers",
                name=name,
                text=part["case"],
                # Markers keep a surface-coloured ring so overlapping points stay separable.
                marker={
                    "size": 10,
                    "color": colors[name],
                    "line": {"width": 2, "color": theme.surface},
                },
                hovertemplate="%{text}<br>x: %{x:,.2f}<br>y: %{y:,.2f}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text=METRIC_LABELS.get(x, x), tickformat=".3s")
    fig.update_yaxes(title_text=METRIC_LABELS.get(y, y), tickformat=".3s")
    apply_layout(fig, theme, height=480)
    fig.update_layout(hovermode="closest", showlegend=labels.nunique() > 1)
    return fig


def comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["case", "family", "scenario", "year", "variant", "plants"]
    columns += [key for key in METRIC_LABELS if key in frame.columns]
    table = frame[[c for c in columns if c in frame.columns]].copy()
    if "year" in table:
        table["year"] = table["year"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    return table.rename(columns={key: label for key, label in METRIC_LABELS.items()})


def comparison_kpis(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Headline tiles: case count, cheapest case and lowest-emission case."""

    tiles = [("Cases compared", f"{len(frame)}", ", ".join(sorted(frame["family"].unique())))]
    for key, label, formatter in (
        ("net_cost", "Lowest net cost", money),
        ("co2", "Lowest CO₂", lambda value: number(value, "t")),
    ):
        if key in frame and frame[key].notna().any():
            best = frame.loc[frame[key].idxmin()]
            tiles.append((label, formatter(float(best[key])), str(best["case"])))
    return tiles


def write_comparison_dashboard(
    roots: Iterable[str | Path],
    path: str | Path,
    plotly_js: str = "embed",
    top: int = 30,
    case_href: str | None = None,
) -> Path:
    """Write a static ``comparison.html`` for the cases found below ``roots``."""

    frame = load_case_summaries(roots)
    if frame.empty:
        raise ValueError("No case summaries (summary_indicators tables) found to compare")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_comparison_html(frame, plotly_js=plotly_js, top=top, case_href=case_href),
        encoding="utf-8",
    )
    return path


def render_comparison_html(
    frame: pd.DataFrame,
    plotly_js: str = "embed",
    top: int = 30,
    case_href: str | None = None,
) -> str:
    metrics = available_metrics(frame)
    groups = group_columns(frame)
    group = groups[0][0] if groups else None
    cards = []
    for index, (key, label) in enumerate(metrics):
        fig = ranking_chart(frame, key, group, top)
        if fig is None:
            continue
        hidden = "" if index == 0 else " hidden"
        cards.append(
            f'<div class="view" data-metric="{key}"{hidden}>'
            + _card(
                "Ranking",
                f"{label} — lowest first, top {min(top, len(frame))} of {len(frame)}",
                fig,
                wide=True,
            )
            + "</div>"
        )
    scatter_pairs = [("net_cost", "co2"), ("specific_cost", "specific_co2")]
    scatter_cards = []
    for x, y in scatter_pairs:
        fig = tradeoff_chart(frame, x, y, group)
        if fig is not None:
            scatter_cards.append(
                _card(
                    "Trade-off",
                    f"{METRIC_LABELS[x]} vs {METRIC_LABELS[y]} — bottom-left is better",
                    fig,
                    wide=False,
                )
            )
    kpis = "".join(
        '<div class="kpi">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="detail">{html.escape(detail)}</div></div>'
        for label, value, detail in comparison_kpis(frame)
    )
    options = "".join(
        f'<option value="{key}">{html.escape(label)}</option>' for key, label in metrics
    )
    table = (
        '<article class="card wide"><h2>All cases</h2>'
        '<p class="desc">Summed over the plants of each case. Specific values divide by '
        "production (or heat demand).</p>"
        f"{_render_table(comparison_table(frame))}</article>"
    )
    body = (
        f'<div class="kpis">{kpis}</div>'
        f'<div id="ranking">{"".join(cards)}</div>'
        f'<div class="grid" style="margin-top:16px">{"".join(scatter_cards)}{table}</div>'
    )
    meta = [f"{len(frame)} cases", ", ".join(sorted(frame["family"].unique()))]
    link = f'<a class="btn" href="{html.escape(case_href)}">Case dashboard</a>' if case_href else ""
    selector = f'<label>Rank by<select id="metric-select">{options}</select></label>'
    return page_shell(
        title="Case comparison",
        meta=meta,
        controls=selector + link,
        nav="",
        body=body,
        script=plotly_script(plotly_js),
        page_js=_COMPARISON_JS,
    )


def _card(title: str, caption: str, fig: go.Figure, wide: bool) -> str:
    height = fig.layout.height or 360
    figure_json = fig.to_json().replace("</", "<\\/")
    return (
        f'<article class="card{" wide" if wide else ""}">'
        f'<h2>{html.escape(title)}<span class="caption">{html.escape(caption)}</span></h2>'
        f'<div class="chart" style="height:{height}px"></div>'
        f'<script type="application/json" class="fig">{figure_json}</script></article>'
    )


_COMPARISON_JS = r"""
(function () {
  const select = document.getElementById('metric-select');
  const views = Array.from(document.querySelectorAll('#ranking .view'));
  function show() {
    views.forEach(view => { view.hidden = view.dataset.metric !== select.value; });
    const view = views.find(v => !v.hidden);
    if (view) FM.renderPanel(view);
  }
  if (select) select.addEventListener('change', show);
  document.querySelectorAll('.grid > .card .chart').forEach(el => FM.renderPanel(el.parentElement));
  show();
})();
"""
