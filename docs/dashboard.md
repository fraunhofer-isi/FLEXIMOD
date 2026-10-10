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
- [Step-by-step tutorial](#step-by-step-tutorial)
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

## Step-by-step tutorial

Every step below shows the command to run. Run all commands from the project root.

### A. Static dashboard (one file)

**1. Open a terminal in the project and activate the environment.**

```bash
cd <path-to-your-FLEXIMOD-checkout>
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**2. Get a case output folder.** Either run a case, which also creates the dashboard...

```bash
python src/flexi_mod/simulation/run_case.py --example hybrid_ETES_DA
```

...or use result folders you already have. Any folder that contains `dispatch_results.csv` (or
`dispatch_results.csv.zst`) works. To list them:

```bash
find data/output -name "dispatch_results.csv*"          # Windows PowerShell: Get-ChildItem data\output -Recurse -Filter "dispatch_results.csv*"
```

**3. Build the dashboard of that folder.**

```bash
python src/flexi_mod/simulation/plot_case.py --output-dir "data/output/<case_folder>"
```

For example:

```bash
python src/flexi_mod/simulation/plot_case.py --output-dir data/output/building_use_case_analysis/i07_reference
```

It prints the location of the file:

```text
Dashboard created: <path-to-your-FLEXIMOD-checkout>/data/output/building_use_case_analysis/i07_reference/dashboard.html
```

**4. Open the file.** On your own computer, double-click `dashboard.html`, or run:

```bash
python src/flexi_mod/simulation/plot_case.py --output-dir "data/output/<case_folder>" --open
```

On a server without a screen, download the file first, then open it on your computer. From your
computer, with `scp`:

```bash
scp user@server:<path-to-your-FLEXIMOD-checkout>/data/output/<case_folder>/dashboard.html .
```

(In VS Code you can instead right-click the file in the Explorer and choose Download.)

**5. Optional: change what is shown.** Add options to the step 3 command:

```bash
# one month only, every timestep
python src/flexi_mod/simulation/plot_case.py --output-dir "data/output/<case_folder>" --start 2025-01-01 --end 2025-01-31 --resolution native

# smaller file for a case with many plants
python src/flexi_mod/simulation/plot_case.py --output-dir "data/output/<case_folder>" --max-plants 4
```

**6. Shortcut.** Running the script with no arguments (or pressing play on `plot_case.py`) builds
the dashboard of the newest result folder:

```bash
python src/flexi_mod/simulation/plot_case.py
```

### B. Comparison page (many cases, one file)

```bash
python src/flexi_mod/simulation/plot_case.py --compare data/output/steel_results data/output/cement_results
```

It prints `Comparison dashboard created: <path>`. The file is `comparison.html` in the first
folder you listed. Open it like the dashboard in step A4.

### C. Live Dash app

**1. Activate the environment and install Dash (once).**

```bash
cd <path-to-your-FLEXIMOD-checkout>
source .venv/bin/activate
pip install -e .[dashboard]
```

**2. Start the app.**

```bash
fleximod-dashboard --root data/output
```

If the command is not found, run the file directly (or press play on `dash_app.py`):

```bash
python src/flexi_mod/visualisation/dashboard/dash_app.py --root data/output
```

The terminal prints `Running on http://127.0.0.1:8050`. Leave it running.

**3. Open it in a browser.** On your own computer, go to <http://127.0.0.1:8050>.

On a remote server, forward the port first. Run this on your computer, then open
<http://localhost:8050>, or use the Ports tab in VS Code (Forward a Port, `8050`):

```bash
ssh -L 8050:localhost:8050 user@server
```

**4. Use it.** Pick a **Case**, then optionally a **Plant**, **Period** and **Resolution**, and
click through the tabs. The **Compare cases** tab at the top ranks many cases.

**5. Optional: other start options.**

```bash
fleximod-dashboard --root data/output --case data/output/<case_folder>   # open a case first
fleximod-dashboard --root data/output --port 8060                        # if 8050 is taken
```

**6. Stop it.** Press `Ctrl+C` in the terminal.

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
| `--input-dir FOLDER` | Case input folder (`plants.csv`, `additional_charges.csv`) for the System setup tab and CAPEX/OPEX. Found automatically from `--case`, `--example` or the output folder name when omitted. |
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
- for steam cases, a **cost components** chart (gas, day-ahead, intraday, aFRR, charges, CO₂, grid
  fee correction, with the net cost as a diamond) and a **trade activity** chart (traded volumes
  and reserved aFRR capacity);
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

These plants carry the analyses of the report notebook `notebooks/fleximod_report_analysis.ipynb`
(see [Relation to the report notebook](#relation-to-the-report-notebook)).

| Tab | Content |
|---|---|
| Overview | Tiles: net operating cost, gas-only benchmark, savings against it, net savings after CAPEX and OPEX, cost of heat, heat demand, electrification rate, electricity used, emissions, CAPEX and OPEX, aFRR capacity value. Heat supply and storage content. Prices and electricity procurement by market. |
| Heat & storage | Heat demand coverage (direct electric supply, storage discharge, gas boiler), monthly coverage and electrified share, gas heat remaining after each market stage, stored heat by procurement market, storage charging and discharging with state of charge. |
| Markets | Monthly electricity procurement by market, monthly market value against the gas-based electricity benchmark, aFRR capacity reserved and its price, monthly day-ahead price, tables of the electricity balance by market channel and of price statistics. |
| Operation | Sequential market profile of the most active week: day-ahead baseline, intraday adjustment, aFRR and final electricity, gas boiler heat, storage content. |
| Costs & financials | Cashflow waterfall from the gas-only benchmark to the net cost, cost breakdown, cumulative cost, emissions. |
| Grid fees | Tiles for the grid fee components and the full-load-hour tier, fee breakdown, the peak basis of the capacity charge, and the average weekday profile around the German high-load windows. Only for cases with a grid fee settlement. |
| Patterns | Heat demand by hour of day, by weekday and hour, averages, heat load duration curve, electric load by hour, load response to price, electric load duration. |
| System setup | Thermal profile type, installed e-heater power, storage size and efficiencies, plant configuration, additional charges, results per installed MW of e-heater. Only when the case input folder is found. |
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
- **Cashflow waterfall.** Starts at what the heat would cost with the gas boiler alone. Blue steps
  add cost, green steps save money or earn revenue, and the last bar is the net cost (including
  CAPEX and OPEX when the input folder is available). The caption gives the saving against the
  benchmark.
- **Heat demand coverage.** "Direct supply" is heat produced while the storage charges at the same
  time, so the electricity passes straight through. "Storage discharge" is heat released from stored
  energy alone.
- **High-load windows.** Shaded bands are the DSO high-load windows of the German grid fee rules.
  Grid draw falls to zero inside them while the gas boiler and stored heat supply the heat, which
  keeps the billed capacity peak low.
- **Process unit utilisation.** Share of each day in which a unit is running. Light gaps show when
  the plant used its flexibility.

## Relation to the report notebook

`notebooks/fleximod_report_analysis.ipynb` builds a report for one steam case. Its analyses are
part of both dashboards now, so you no longer need to run the notebook for them:

| Notebook section | Dashboard |
|---|---|
| Executive summary | Overview tiles |
| System setup | System setup tab |
| Input analysis (demand, prices, charges) | Patterns tab, price statistics and additional charges tables |
| Operations: demand coverage, electricity balance, sequential profile, ETES operation, aFRR | Heat & storage, Markets and Operation tabs |
| Financials: waterfall, monthly market value | Costs & financials and Markets tabs |
| Grid fees and atypical grid use | Grid fees tab |
| Scenario comparison | Comparison page (cost components, trade activity) |
| Per-MW e-heater results | System setup tab |

Differences to be aware of:

- The dashboards add **plant selection, period filter, dark mode and hover** and work for building,
  cement and steel cases as well.
- Savings against the gas-only benchmark use the net cost **including the ex-post grid fee
  correction**, so they are slightly more conservative than the notebook's.
- Charts that used a second y-axis in the notebook (monthly coverage with the electrified share,
  aFRR price with reserved capacity, trade volumes with capacity) are drawn on **stacked rows**
  instead, so no chart has two scales.
- The boiler efficiency for the benchmark comes from the dispatch (heat out divided by gas in),
  falling back to `plants.csv`, then 90%.

### Input folder and investment costs

The System setup tab and the CAPEX/OPEX figures need the case **input** folder. The dashboard finds
`data/input/<case>` when the output folder is called `<case>` or `<case>_<strategy>` (the default
naming), and the runner and `plot_case.py --case/--example` pass it directly. Use `--input-dir` for
anything else. Without it those parts are simply left out.

CAPEX and OPEX use **placeholder assumptions** carried over from the notebook, not project data:
20,000 EUR per MWh of thermal storage, 200,000 EUR per MW of e-heater, OPEX 2% of CAPEX per year,
15 years and 8% WACC. Change them in `INVESTMENT_ASSUMPTIONS` in
`src/flexi_mod/visualisation/dashboard/data.py`. For runs shorter than a year the annual costs are
pro-rated to the simulated period.

## How the data is prepared

### Files read

| File | Used for |
|---|---|
| `dispatch_results.csv` | All time-series charts (required) |
| `summary_indicators.csv` | Summary table, comparison page, grid fee correction |
| `storage_cost_ledger.csv` | Stored heat by procurement market (steam plants) |
| `grid_fee_summary.csv` | Grid fee chart and table |
| `afrr_capacity_block_summary.csv` | aFRR capacity block table |
| `plants.csv`, `additional_charges.csv` (case **input** folder, optional) | System setup tab, CAPEX and OPEX |

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
| `charts.py` | The generic Plotly figures, one function per chart, grouped by plant type |
| `steam_charts.py` | The steam/ETES analyses of the report notebook (benchmark, waterfall, grid fees, tables) |
| `sections.py` | One dashboard **class** per plant type: tabs, charts and headline tiles |
| `theme.py` | Palette, chart template and page CSS |
| `static_html.py` | Writes the single HTML file |
| `dash_app.py` | The live app |
| `comparison.py` | Case comparison, used by both front ends |

### One class per plant type

A plant type is a class that inherits from the generic `PlantDashboard` and overrides only what
differs:

```text
PlantDashboard                  generic: markets, costs, patterns
|-- SteamDashboard              steam / ETES / boiler plants
|-- IndustrialDashboard         production-based plants
|   |-- CementDashboard
|   `-- SteelDashboard
`-- BuildingDashboard           buildings with EV fleet and PV
```

| A class defines | How |
|---|---|
| Its chart builders | `builders = {...}`. They are **merged with the parents' builders**, so a subclass keeps every parent chart and adds its own. Tables go in `tables = {...}`. |
| The tabs | Override `overview_tab()`, `costs_tab()`, `patterns_tab()`, or `layout()` to add and reorder tabs. Each tab is a `TabSpec` listing `ChartSpec(key, title, description, wide)`. |
| The headline tiles | Override `overview_kpis(ctx)` (and `tab_kpis` for other tabs). |
| Which tabs appear | Override `tab_available(ctx, tab_id)`, for example to hide a tab whose inputs are missing. |
| The plant type | Set `family = "..."`. Setting it registers the class for that detected type. |

Both front ends call the same class, so a change shows up in the static file and in Dash.

**Add a chart to an existing plant type.** Write `my_chart(ctx: ChartContext) -> go.Figure | None` in
`charts.py` (or `steam_charts.py`), add it to the matching `builders` group, and add a
`ChartSpec("my_chart", "Title", "What to read here")` to a tab of that class. Add a test in
`tests/test_dashboard.py`.

**Add a plant type.** Subclass `PlantDashboard` (or a closer relative) and extend `detect_family` in
`data.py` so the new type is recognised:

```python
from flexi_mod.visualisation.dashboard.sections import ChartSpec, PlantDashboard, TabSpec


def heat_pump_cop(ctx):  # returns a Plotly figure, or None when the data is missing
    ...


class HeatPumpDashboard(PlantDashboard):
    family = "heat_pump"
    builders = {"heat_pump_cop": heat_pump_cop}

    def overview_tab(self) -> TabSpec:
        return TabSpec("overview", "Overview", [ChartSpec("heat_pump_cop", "Coefficient of performance")])
```

The costs and patterns tabs, the Data tab, grid fees and the load charts come from the parent
unchanged. A test checks that every chart named in any class exists, so typos are caught.

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
