<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Interactive Dashboards

FLEXIMOD turns the result tables of a case into interactive [Plotly](https://plotly.com/python/)
dashboards. There are two ways to look at the same charts:

| | **Static HTML** | **Live Dash app** |
|---|---|---|
| What it is | One self-contained `dashboard.html` file | A small web app that runs on your machine |
| Start it | Written after every run, or `plot_case.py` | `fleximod-dashboard` |
| Needs | Nothing: open the file in a browser | `pip install -e .[dashboard]` |
| Works offline | Yes (plotly.js is embedded) | Yes (served locally) |
| Best for | Looking at or sharing one finished case; archiving | Browsing many cases; changing plant, period or resolution without rebuilding |
| Can be emailed | Yes | No |

Both read the same files and use the same chart code, so a chart looks the same in either one.
Use the static file by default. Use the Dash app when you want to explore many results.

- [Quick start](#quick-start)
- [Static HTML dashboard](#static-html-dashboard)
- [Live Dash app](#live-dash-app)
- [Working on a remote server](#working-on-a-remote-server)
- [Comparing cases](#comparing-cases)
- [What the dashboard shows](#what-the-dashboard-shows)
- [How the data is prepared](#how-the-data-is-prepared)
- [Sharing a dashboard](#sharing-a-dashboard)
- [Using the charts from Python](#using-the-charts-from-python)
- [Extending the dashboard](#extending-the-dashboard)
- [Troubleshooting](#troubleshooting)

## Quick start

From the project root with the virtual environment active:

```bash
# 1. Run a case. This writes dashboard.html next to the result tables.
python src/flexi_mod/simulation/run_case.py --example hybrid_ETES_DA

# 2. Rebuild the dashboard of any existing result folder.
python src/flexi_mod/simulation/plot_case.py --output-dir "data/output/<case_folder>" --open

# 3. Or start the live app and open http://127.0.0.1:8050
fleximod-dashboard --root data/output
```

`plot_case.py` also works with the editor's **play button**. Run without arguments, it builds the
dashboard of the most recently written result folder under `data/output`. To always use a fixed
folder, set `DEFAULT_OUTPUT_DIR` near the top of
[`plot_case.py`](../src/flexi_mod/simulation/plot_case.py).

## Static HTML dashboard

### Where the file is written

| How you start it | File |
|---|---|
| `run_case.py` (default) | `data/output/<case_name>_<strategy_name>/dashboard.html` |
| `plot_case.py --output-dir <folder>` | `<folder>/dashboard.html` |
| `plot_case.py --case <input> --study-case <name>` or `--example <name>` | `data/output/<case_name>_<strategy_name>/dashboard.html` |
| `plot_case.py --compare <folders...>` | `comparison.html` in the first folder, or in `--output-dir` |

`plot_case.py` prints the path as `Dashboard created: <path>` when it finishes.

`run_case.py --no-plots` skips the dashboard. A dashboard that fails to build only prints a
warning; it never discards a finished simulation.

### `plot_case.py` options

| Option | Meaning |
|---|---|
| `--output-dir FOLDER` | Case output folder that contains `dispatch_results.csv` (or `.csv.zst`). |
| `--case FOLDER`, `--study-case NAME` | Locate the output folder from a case input folder and its `config.yaml`. |
| `--example NAME` | Same, for a named example of `run_case.py`. |
| `--compare FOLDER ...` | Build a comparison page for the cases found below these folders instead. |
| `--start DATE`, `--end DATE` | Show only this period, for example `--start 2025-01-01 --end 2025-01-31`. |
| `--resolution` | `auto` (default), `native`, `1h`, `6h`, `1D` or `1W`. See [resolution](#resolution). |
| `--max-plants N` | Plants shown individually (default 12). The aggregate is always included. |
| `--plotly-js embed\|cdn` | `embed` (default) works offline; `cdn` is about 5 MB smaller but needs internet when opened. |
| `--open` | Open the file in your browser (needs a desktop). |
| `--verbose` | Print extra detail. |

With no arguments the script uses the latest result folder and opens it (see the quick start).

### Using the page

- **Tabs** along the top group the charts. Which tabs and charts appear depends on the plant type.
- **Plant** (top right, shown when the case has several plants) switches between the aggregate and
  one plant. Only the `--max-plants` largest electricity users are included individually.
- **Charts:** drag to zoom, double-click to reset, hover for values (one tooltip lists every series
  at that time), click a legend entry to hide a series. The camera icon saves the chart as a PNG.
- **Dark mode** (top right) switches theme. The choice is remembered in your browser.
- **Data tab:** the summary indicators and other tables, with a filter box above each.

## Live Dash app

```bash
fleximod-dashboard --root data/output
```

Then open <http://127.0.0.1:8050> in a browser. Stop it with `Ctrl+C`.

| Option | Meaning |
|---|---|
| `--root FOLDER` | Folder scanned for case outputs (default `data/output`). Plain `.csv` and `.csv.zst` tables both work. |
| `--case FOLDER` | Case output folder to open first. |
| `--host`, `--port` | Address to listen on (default `127.0.0.1:8050`). Use another port if 8050 is taken. |
| `--debug` | Dash debug mode (reloads on code changes). |

Without the installed command you can run
`python src/flexi_mod/visualisation/dashboard/dash_app.py --root data/output`, or press play on
that file. If Dash is missing, install it with `pip install -e .[dashboard]`.

### Case dashboard tab

1. Pick the **Case** from the list (everything found under the scanned folder).
2. Optionally choose a **Plant**, a **Period** (date range) and a **Resolution**.
3. Click through the tabs. The charts redraw whenever a control changes.

Changing the **Output folder to scan** and pressing Enter rescans for cases. The first load of a very
large case takes about 10 seconds; it is then cached, so switching plant or tab is quick.

### Compare cases tab

See [Comparing cases](#comparing-cases).

## Working on a remote server

A server has no screen, so `--open` does nothing there. Instead:

- **Static file:** download `dashboard.html` (for example, right-click it in the VS Code Explorer and
  choose Download, or use `scp`) and open it on your own computer.
- **Dash app:** forward the port to your computer. In VS Code with Remote-SSH, open the **Ports**
  tab, choose **Forward a Port** and enter `8050`. With plain SSH:

  ```bash
  ssh -L 8050:localhost:8050 user@server
  ```

  Then run `fleximod-dashboard` on the server and open <http://localhost:8050> on your computer.

## Comparing cases

The comparison reads only the small `summary_indicators` table of each case, so it stays fast even
for hundreds of cases. It works on case folders, or folders that contain many case folders.

```bash
python src/flexi_mod/simulation/plot_case.py --compare data/output/steel_results data/output/cement_results
```

The page shows:

- headline tiles: number of cases, lowest net cost, lowest CO₂;
- a **ranking** bar chart of any metric, lowest first, coloured by plant family, scenario or year;
- **trade-off** scatter plots, such as net cost against CO₂ (bottom-left is better);
- a table of all cases that you can filter.

Metrics are summed over the plants of a case: net operating cost, electricity use, CO₂, production,
heat demand, aFRR capacity value and grid fees. The page also shows specific values (cost, CO₂ and
electricity per tonne of product, or cost per MWh of heat). Case names such as
`scenario_2030_route` are split into scenario, year and variant for grouping. In the Dash app,
the same page lets you change the metric, grouping, number of cases and scatter axes live.

## What the dashboard shows

The plant type is detected from the dispatch table (`heat_demand_MWh` → steam, `building_demand_MWh`
→ building, `clinker_output_t` → cement, `steel_output_t` → steel). A chart that needs columns the
case does not have, such as IDC or aFRR series in a day-ahead-only run, is simply left out.

### Steam, ETES and boiler plants

| Tab | Content |
|---|---|
| Overview | Tiles: net operating cost, cost of heat, heat demand, share of gas replaced, electricity used, emissions, aFRR capacity value. Heat supply and storage content. Prices and electricity procurement by market. |
| Markets | aFRR capacity reserved and its price. Gas heat remaining after each market stage. Stored heat by the market it was procured on. |
| Operation | A three-day window at full resolution around the most active day. |
| Costs & emissions | Cost breakdown, cumulative cost, grid fees, emissions. |
| Patterns | Load by hour of day, load response to price, load duration curve. |
| Data | Summary indicators, aFRR capacity blocks, grid fee summary. |

### Cement and steel plants

| Tab | Content |
|---|---|
| Overview | Tiles: net operating cost, production, specific cost, electricity used, CO₂, aFRR capacity value. Production and remaining demand. Prices and electricity procurement. |
| Markets | aFRR capacity reserved and its price. |
| Operation | Energy by carrier, electricity by process unit, share of each day a unit runs, representative window. |
| Costs & emissions, Patterns, Data | As for steam plants. |

### Buildings with EV fleets and PV

| Tab | Content |
|---|---|
| Overview | Tiles: total cost, grid import and export, peak import, PV used on site, unmet trip energy. Grid exchange with PV, tariff against the billing peak. |
| Grid & PV | Regional grid stress against building import, where the PV generation goes. |
| Operation | Fleet state of charge and charge/discharge schedule, representative window. |
| Costs | Cost breakdown (energy and demand charge), cumulative cost. |
| Patterns | Net grid import by hour of day, load duration curve. |

### Reading the main charts

- **Prices and electricity procurement.** Top: market prices. Bottom: stacked bars show where each
  MWh was bought (day-ahead, intraday, aFRR activation). Bars below zero are intraday volumes sold
  back. The black line is the actual consumption, which equals the sum of the bars.
- **Cost breakdown.** Costs are blue, credits (revenue) are green. The components add up to the net
  operating cost including the grid fee correction, which is shown in the heading.
- **Load response to price.** Average load in each price decile. A flexible plant leans towards the
  cheap deciles on the left.
- **Process unit utilisation.** Share of each day in which a unit is running. Light gaps show when
  the plant used its flexibility.

## How the data is prepared

### Files read

| File | Used for |
|---|---|
| `dispatch_results.csv` | All time-series charts (required) |
| `summary_indicators.csv` | Summary table, comparison page, grid fee correction |
| `storage_cost_ledger.csv` | Stored heat by procurement market (steam plants) |
| `grid_fee_summary.csv` | Grid fee chart and table |
| `afrr_capacity_block_summary.csv` | aFRR capacity block table |

Each file may be plain `.csv` or zstd-compressed `.csv.zst`. Timestamps are read as local
wall-clock time; the UTC offset in the files is ignored so daylight-saving changes do not matter.

### Aggregation

Columns are treated according to their meaning:

| Kind | Examples | Over time | Over plants |
|---|---|---|---|
| Flow | MWh, EUR, tonnes, kg | summed | summed |
| Level | storage content, power in MW | averaged | summed |
| Price or share | EUR/MWh, fractions, on/off status | averaged | averaged |

### Resolution

With `auto`, each chart uses the finest step (native, hourly, 6-hourly, daily, weekly) that keeps it to
about 2,200 points, so a full year at 15 minutes is drawn 6-hourly. The "representative window" chart
always uses the native step. To see every step of a long run, build a shorter period with `--start`
and `--end` and `--resolution native`.

### Period filter and totals

Tiles and charts follow the selected period and plant. Items that exist only for the whole run, such
as the grid fee correction and the grid fee chart, are hidden when a period is selected. The
Summary indicators table always covers the whole run.

### Size and speed

A dashboard builds in seconds. The largest case tested (cement, 1.1 million rows, 32 plants) built in
about 9 seconds, used about 1.8 GB of memory and produced an 18 MB file (13 views, plotly.js
embedded). To make the file smaller, lower `--max-plants` or use `--plotly-js cdn`.

## Sharing a dashboard

The static file can be sent as it is. Keep in mind:

- It works offline when built with the default `--plotly-js embed`. With `cdn` the viewer needs
  internet.
- It contains the numbers behind every chart (prices, costs, per-plant results), not just pictures.
  Do not send it outside your organisation if the data is confidential.
- It is a snapshot. After a new run, send the new file.
- The Dash app cannot be shared as a file. It runs on one machine.

## Using the charts from Python

```python
from flexi_mod.visualisation.dashboard.data import load_case
from flexi_mod.visualisation.dashboard.charts import BUILDERS, ChartContext
from flexi_mod.visualisation.dashboard.static_html import write_case_dashboard

case = load_case("data/output/<case_folder>")
write_case_dashboard(case, "my_dashboard.html", max_plants=6, resolution="1D")

# One chart in a notebook:
ctx = ChartContext(case, plant="__all__", start="2025-01-01", end="2025-01-31")
BUILDERS["cost_breakdown"](ctx).show()
```

`BUILDERS` lists every chart by name. A builder returns `None` when the case lacks the data it needs.

## Extending the dashboard

The code is in `src/flexi_mod/visualisation/dashboard/`:

| File | Job |
|---|---|
| `data.py` | Load a case folder, detect the plant type, aggregate and resample |
| `charts.py` | The Plotly figures, one function per chart |
| `sections.py` | Which charts go on which tab, per plant type, and the headline tiles |
| `theme.py` | Palette, chart template and page CSS |
| `static_html.py` | Writes the single HTML file |
| `dash_app.py` | The live app |
| `comparison.py` | Case comparison, used by both front ends |

**Add a chart.** Write a function `my_chart(ctx: ChartContext) -> go.Figure | None` in `charts.py`,
add it to `BUILDERS`, and list it under the right plant type and tab in `_LAYOUT` in `sections.py`.
It then appears in the static file and the Dash app. Add a test in `tests/test_dashboard.py`.

**Add a plant type.** Extend `detect_family` in `data.py`, add an entry to `_LAYOUT` and a KPI
function in `sections.py`.

**Design rules the charts follow:**

- Colours belong to *entities* (day-ahead, intraday, aFRR, gas, storage, and so on) and keep the same
  colour on every chart. Add new entities to `ENTITY_SLOT` in `theme.py`.
- No chart has two y-axes. Measures with different units go on stacked rows that share the time axis.
- Titles live in the page, not inside the figure, so the legend never collides with them.
- Dark mode uses its own selected steps of the same hues, not an inversion. Build figures with the
  light theme; the static page recolours them in the browser.
- Charts must tolerate missing columns and return `None`.

## Troubleshooting

| Problem | What to do |
|---|---|
| "No case results found under data/output" | Run a case first, or pass `--output-dir`. |
| `dashboard.html` is not where you expect | Look for the `Dashboard created:` line the script printed, or search for `dashboard.html` below `data/output`. |
| Dash: "needs Dash" | `pip install -e .[dashboard]`. |
| Dash page does not open on a server | Forward the port, see [remote server](#working-on-a-remote-server). |
| Port already in use | Start with `--port 8060`. |
| A chart or tab is missing | The case lacks the columns for it (for example no aFRR in a day-ahead-only run). This is intended. |
| Charts look too coarse | Use `--resolution native` with a shorter `--start`/`--end` period, or the Resolution control in Dash. |
| File is very large | Lower `--max-plants`, or use `--plotly-js cdn`. |
| `fleximod-dashboard` not found | Run `pip install -e .` again, or run the `dash_app.py` file directly. |
| No dashboard after a run | Check that the run did not use `--no-plots`, and read the warnings printed at the end. |
