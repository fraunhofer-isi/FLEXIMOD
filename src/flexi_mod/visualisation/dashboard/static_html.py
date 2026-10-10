# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render a case as one self-contained, interactive HTML file (no server needed)."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs, get_plotlyjs_version

from flexi_mod.visualisation.dashboard.charts import ChartContext
from flexi_mod.visualisation.dashboard.data import (
    ALL_PLANTS,
    CaseData,
    load_case,
)
from flexi_mod.visualisation.dashboard.sections import Block, Tab, build_tabs, tab_ids
from flexi_mod.visualisation.dashboard.theme import BASE_CSS, DARK, LIGHT, color_swap_map

DASHBOARD_FILENAME = "dashboard.html"

_PLOT_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def write_case_dashboard(
    case: CaseData | str | Path,
    path: str | Path | None = None,
    max_plants: int = 12,
    plotly_js: str = "embed",
    start: str | None = None,
    end: str | None = None,
    resolution: str = "auto",
    comparison_href: str | None = None,
) -> Path:
    """Write ``dashboard.html`` for a case (an output folder or a loaded case)."""

    if not isinstance(case, CaseData):
        case = load_case(case)
    if path is None:
        if case.output_dir is None:
            raise ValueError("A target path is required when the case has no output folder")
        path = case.output_dir / DASHBOARD_FILENAME
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_case_html(
            case,
            max_plants=max_plants,
            plotly_js=plotly_js,
            start=start,
            end=end,
            resolution=resolution,
            comparison_href=comparison_href,
        ),
        encoding="utf-8",
    )
    return path


def render_case_html(
    case: CaseData,
    max_plants: int = 12,
    plotly_js: str = "embed",
    start: str | None = None,
    end: str | None = None,
    resolution: str = "auto",
    comparison_href: str | None = None,
) -> str:
    """Return the dashboard HTML. ``plotly_js`` is ``"embed"`` (offline) or ``"cdn"``."""

    views = _select_views(case, max_plants)
    rendered: list[tuple[str, str, list[Tab]]] = []
    for value, label in views:
        ctx = ChartContext(case, value, LIGHT, start, end, resolution)
        rendered.append((value, label, build_tabs(ctx)))

    overview_ctx = ChartContext(case, ALL_PLANTS, LIGHT, start, end, resolution)
    tabs = tab_ids(overview_ctx)
    first, last = overview_ctx.raw.index.min(), overview_ctx.raw.index.max()
    meta = [
        case.family_label,
        f"{first:%d %b %Y} – {last:%d %b %Y}",
        f"{len(case.plants)} plant{'s' if len(case.plants) != 1 else ''}",
        f"{case.step_hours * 60:g}-min data shown per {overview_ctx.step_label}"
        if not overview_ctx.native
        else f"{case.step_hours * 60:g}-min data",
    ]
    if len(views) - 1 < len(case.plants) and len(case.plants) > 1:
        meta.append(f"top {len(views) - 1} of {len(case.plants)} plants by electricity use")

    options = "".join(
        f'<option value="{html.escape(value)}">{html.escape(label)}</option>'
        for value, label, _ in rendered
    )
    plant_control = (
        f'<label>Plant<select id="plant-select">{options}</select></label>'
        if len(rendered) > 1
        else ""
    )
    compare_link = (
        f'<a class="btn" href="{html.escape(comparison_href)}">Compare cases</a>'
        if comparison_href
        else ""
    )
    nav = "".join(
        f'<button role="tab" data-tab="{tab_id}" aria-selected="false">{html.escape(label)}</button>'
        for tab_id, label in tabs
    )
    body = "".join(_render_view(value, rendered_tabs, tabs) for value, _, rendered_tabs in rendered)

    script = plotly_script(plotly_js)
    return page_shell(
        title=case.name,
        meta=meta,
        controls=f"{plant_control}{compare_link}",
        nav=nav,
        body=body,
        script=script,
        page_js=_CASE_JS,
    )


def page_shell(
    title: str,
    meta: list[str],
    controls: str,
    nav: str,
    body: str,
    script: str,
    page_js: str,
) -> str:
    """Wrap page content in the shared header, styling and chart/theme JavaScript."""

    swap = {"toDark": color_swap_map(LIGHT, DARK)}
    safe_title = html.escape(title)
    meta_html = "".join(f"<span>{html.escape(item)}</span>" for item in meta)
    nav_html = f'<nav class="tabs" role="tablist">{nav}</nav>' if nav else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title} - FLEXIMOD</title>
<style>{BASE_CSS}</style>
{script}
</head>
<body>
<div class="app">
  <header class="top">
    <div>
      <h1>{safe_title}</h1>
      <div class="meta">{meta_html}</div>
    </div>
    <div class="controls">
      {controls}
      <button class="btn" id="theme-toggle" type="button" aria-label="Toggle dark mode">Dark mode</button>
    </div>
  </header>
  {nav_html}
  <main>{body}</main>
  <p class="note">Generated by FLEXIMOD. Drag to zoom, double-click to reset, hover for values;
  use the camera icon on a chart to save it as an image.</p>
</div>
<script>
const SWAP = {json.dumps(swap)};
{_COMMON_JS}
{page_js}
</script>
</body>
</html>
"""


def plotly_script(plotly_js: str) -> str:
    if plotly_js == "cdn":
        return f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>'
    return f"<script>{get_plotlyjs()}</script>"


def _select_views(case: CaseData, max_plants: int) -> list[tuple[str, str]]:
    options = case.plant_options()
    if len(options) <= 1:
        return options
    ranked = case.dispatch.groupby("plant_name")[_ranking_column(case)].sum().sort_values()
    keep = set(ranked.index[::-1][:max_plants])
    return [options[0], *[(value, label) for value, label in options[1:] if value in keep]]


def _ranking_column(case: CaseData) -> str:
    for column in (
        "actual_electricity_consumption_MWh",
        "total_electricity_consumption_MWh",
        "electricity_consumption_MWh",
        "grid_import_MWh",
    ):
        if column in case.dispatch:
            return column
    return case.dispatch.select_dtypes("number").columns[0]


def _render_view(value: str, tabs: list[Tab], nav: list[tuple[str, str]]) -> str:
    by_id = {tab.id: tab for tab in tabs}
    panels = []
    for tab_id, _ in nav:
        tab = by_id.get(tab_id)
        if tab is None:
            panels.append(
                f'<section class="tabpanel" data-tab="{tab_id}" hidden>'
                '<p class="empty">Nothing to show for this selection.</p></section>'
            )
            continue
        kpis = "".join(
            '<div class="kpi">'
            f'<div class="label">{html.escape(kpi.label)}</div>'
            f'<div class="value">{html.escape(kpi.value)}</div>'
            f'<div class="detail">{html.escape(kpi.detail)}</div></div>'
            for kpi in tab.kpis
        )
        cards = "".join(_render_block(block) for block in tab.blocks)
        panels.append(
            f'<section class="tabpanel" data-tab="{tab_id}" hidden>'
            + (f'<div class="kpis">{kpis}</div>' if kpis else "")
            + f'<div class="grid">{cards}</div></section>'
        )
    return f'<div class="view" data-plant="{html.escape(value)}" hidden>{"".join(panels)}</div>'


def _render_block(block: Block) -> str:
    wide = " wide" if block.wide or block.table is not None else ""
    caption = f'<span class="caption">{html.escape(block.caption)}</span>' if block.caption else ""
    description = (
        f'<p class="desc">{html.escape(block.description)}</p>' if block.description else ""
    )
    head = f"<h2>{html.escape(block.title)}{caption}</h2>{description}"
    if block.figure is not None:
        figure_json = block.figure.to_json().replace("</", "<\\/")
        height = block.figure.layout.height or 360
        return (
            f'<article class="card{wide}">{head}'
            f'<div class="chart" style="height:{height}px"></div>'
            f'<script type="application/json" class="fig">{figure_json}</script></article>'
        )
    if block.table is not None:
        return f'<article class="card{wide}">{head}{_render_table(block.table)}</article>'
    return ""


def _render_table(table: pd.DataFrame) -> str:
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in table.columns)
    rows = []
    for record in table.itertuples(index=False):
        cells = []
        for value in record:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if pd.isna(value):
                    cells.append('<td class="num"></td>')
                else:
                    text = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.3f}".rstrip("0")
                    cells.append(f'<td class="num">{text.rstrip(".")}</td>')
            else:
                cells.append(f"<td>{html.escape('' if pd.isna(value) else str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<input type="search" class="table-filter" placeholder="Filter rows" '
        'aria-label="Filter table rows">'
        f'<div class="table-wrap"><table class="data"><thead><tr>{header}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


_COMMON_JS = r"""
const FM = (function () {
  const root = document.documentElement;
  const themeButton = document.getElementById('theme-toggle');

  function storedTheme() {
    try { return localStorage.getItem('fleximod-theme'); } catch (e) { return null; }
  }
  function effectiveTheme() {
    const set = root.getAttribute('data-theme');
    if (set) return set;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark' : 'light';
  }

  // Re-colour drawn figures with a palette swap (the dark steps are selected, not inverted).
  const SKIP = new Set(['x', 'y', 'z', 'customdata', 'text', 'hovertemplate', 'name', 'ids',
                        'meta', 'legendgroup', 'stackgroup', 'orientation', 'type', 'template']);
  function mapColor(value, map) {
    if (typeof value === 'string') {
      const key = value.toLowerCase();
      if (map[key]) return map[key];
      const m = /^rgba\((\d+),(\d+),(\d+),([\d.]+)\)$/.exec(key);
      if (m) {
        const hex = '#' + [m[1], m[2], m[3]].map(n => (+n).toString(16).padStart(2, '0')).join('');
        const target = map[hex];
        if (target) {
          const c = [1, 3, 5].map(i => parseInt(target.slice(i, i + 2), 16));
          return 'rgba(' + c.join(',') + ',' + m[4] + ')';
        }
      }
      return value;
    }
    if (Array.isArray(value)) return value.map(v => mapColor(v, map));
    return value;
  }
  // Walks an object and records every colour that the map changes: new values in `out`,
  // the original values in `old` (so the light state is restored exactly, not inverted).
  function collect(obj, path, map, out, old) {
    Object.keys(obj).forEach(key => {
      if (SKIP.has(key) || key.startsWith('_')) return;
      const value = obj[key];
      const p = path ? path + '.' + key : key;
      if (Array.isArray(value) && value.length && value[0] && typeof value[0] === 'object'
          && !Array.isArray(value[0])) {
        value.forEach((item, i) => collect(item, p + '[' + i + ']', map, out, old));
      } else if (value && typeof value === 'object' && !Array.isArray(value)) {
        collect(value, p, map, out, old);
      } else {
        const next = mapColor(value, map);
        if (JSON.stringify(next) !== JSON.stringify(value)) { out[p] = next; old[p] = value; }
      }
    });
  }
  // Called once, right after the first (light) draw.
  function prepareThemes(el) {
    const dark = { layout: {}, traces: [] };
    const light = { layout: {}, traces: [] };
    collect(el.layout, '', SWAP.toDark, dark.layout, light.layout);
    el.data.forEach(trace => {
      const next = {}, old = {};
      collect(trace, '', SWAP.toDark, next, old);
      dark.traces.push(next);
      light.traces.push(old);
    });
    el._themes = { dark: dark, light: light };
  }
  function recolor(el, target) {
    if (!el.dataset.rendered || el.dataset.figTheme === target) return;
    const theme = el._themes[target];
    Plotly.relayout(el, theme.layout);
    theme.traces.forEach((update, i) => {
      const wrapped = {};
      Object.keys(update).forEach(k => { wrapped[k] = [update[k]]; });
      if (Object.keys(wrapped).length) Plotly.restyle(el, wrapped, [i]);
    });
    el.dataset.figTheme = target;
  }

  // Draw the charts of a visible container; charts are drawn lazily when first shown.
  function renderPanel(panel) {
    const theme = effectiveTheme();
    panel.querySelectorAll('.chart').forEach(el => {
      if (el.dataset.rendered) { Plotly.Plots.resize(el); recolor(el, theme); return; }
      const spec = JSON.parse(el.nextElementSibling.textContent);
      Plotly.newPlot(el, spec.data, spec.layout, __PLOT_CONFIG__).then(() => {
        prepareThemes(el);
        el.dataset.rendered = '1';
        el.dataset.figTheme = 'light';
        recolor(el, theme);
      });
    });
  }

  function applyTheme(mode, persist) {
    root.setAttribute('data-theme', mode);
    themeButton.textContent = mode === 'dark' ? 'Light mode' : 'Dark mode';
    if (persist) { try { localStorage.setItem('fleximod-theme', mode); } catch (e) {} }
    document.querySelectorAll('.chart[data-rendered]').forEach(el => {
      if (el.offsetParent !== null) recolor(el, mode);
    });
  }
  themeButton.addEventListener('click', () => {
    applyTheme(effectiveTheme() === 'dark' ? 'light' : 'dark', true);
  });
  const saved = storedTheme();
  if (saved === 'dark' || saved === 'light') root.setAttribute('data-theme', saved);
  themeButton.textContent = effectiveTheme() === 'dark' ? 'Light mode' : 'Dark mode';

  function wireTableFilters() {
    document.querySelectorAll('.table-filter').forEach(input => {
      input.addEventListener('input', () => {
        const needle = input.value.toLowerCase();
        input.nextElementSibling.querySelectorAll('tbody tr').forEach(row => {
          row.hidden = needle && !row.textContent.toLowerCase().includes(needle);
        });
      });
    });
  }
  wireTableFilters();
  return { renderPanel, effectiveTheme };
})();
""".replace("__PLOT_CONFIG__", json.dumps(_PLOT_CONFIG))

_CASE_JS = r"""
(function () {
  const views = Array.from(document.querySelectorAll('.view'));
  const tabButtons = Array.from(document.querySelectorAll('nav.tabs button'));
  const plantSelect = document.getElementById('plant-select');
  const state = {
    plant: views.length ? views[0].dataset.plant : null,
    tab: tabButtons.length ? tabButtons[0].dataset.tab : null,
  };

  function show() {
    views.forEach(view => { view.hidden = view.dataset.plant !== state.plant; });
    tabButtons.forEach(button => {
      button.setAttribute('aria-selected', String(button.dataset.tab === state.tab));
    });
    const view = views.find(v => v.dataset.plant === state.plant);
    if (!view) return;
    view.querySelectorAll('.tabpanel').forEach(panel => {
      panel.hidden = panel.dataset.tab !== state.tab;
      if (!panel.hidden) FM.renderPanel(panel);
    });
  }

  tabButtons.forEach(button => button.addEventListener('click', () => {
    state.tab = button.dataset.tab;
    show();
  }));
  if (plantSelect) plantSelect.addEventListener('change', () => {
    state.plant = plantSelect.value;
    show();
  });
  show();
})();
"""
