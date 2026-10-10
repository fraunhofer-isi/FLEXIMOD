<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# FLEXIMOD

FLEXIMOD stands for **FLEXibility Integration of Demand-side Technologies for
Market Opportunities and Dispatch**.

It is a multi-sector modelling framework for assessing the flexibility,
physically feasible dispatch, and electricity-market opportunities of
demand-side technologies. Its scope includes industrial energy systems,
buildings, electric-vehicle fleets, charging depots, storage-backed demand,
and future flexible energy assets. The framework represents connected
technologies and their market-oriented dispatch decisions in a modular and
extensible way.

The current market-simulation MVP uses a hybrid ETES + gas boiler steam plant
as its first case study. It implements Germany-oriented day-ahead, intraday
continuous, and proxy aFRR down energy stages with a rule-based market strategy
and a Pyomo rolling-horizon plant dispatch model. FLEXIMOD also includes an
electric-bus-depot building model with charging stations and V2G capability.
Future cases can extend the same structure to other sectors, technologies,
countries, and market designs.

The architecture is intentionally modular:

- `config.yaml` contains one or more study cases under a top-level `cases:` mapping.
- `flexi_mod.simulation.run_case` contains example selection, input paths, output paths and output switches.
- `plants.csv` defines one energy asset by grouping connected technology rows.
- `forecasts_df.csv` contains all time series.

## Quick Start For Beginners

FLEXIMOD targets Python 3.13 or newer. From a fresh checkout, open PowerShell in
the repository folder and run:

```powershell
git clone https://github.com/fraunhofer-isi/FLEXIMOD.git
cd FLEXIMOD
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS users can activate the same `.venv` layout with:

```bash
source .venv/bin/activate
```

Run the first case:

```powershell
python src\flexi_mod\simulation\run_case.py --case data\input\hybrid_ETES_ID_buy --study-case hybrid_ETES_ID_buy
```

The runner also writes an interactive dashboard (`dashboard.html`) next to the results.
To rebuild it from existing output:

```powershell
python src\flexi_mod\simulation\plot_case.py --case data\input\hybrid_ETES_ID_buy --study-case hybrid_ETES_ID_buy --open
```

Check the output folder:

```text
data/output/hybrid_ETES_ID_buy_hybrid_etes_gas/
|-- dispatch_results.csv
|-- market_ledger.csv
|-- storage_cost_ledger.csv
|-- summary_indicators.csv
`-- dashboard.html
```

## Input Structure

The first case is stored in:

```text
data/input/hybrid_ETES_ID_buy/
|-- config.yaml
|-- plants.csv
|-- forecasts_df.csv
`-- additional_charges.csv  # optional, only used when the selected case enables it
```

`plants.csv` groups technologies by asset name. For example, two rows with
`name=plant_1` define the ETES storage and gas boiler attached to the same
industrial plant; a bus depot groups its electric-vehicle and charging-station
rows in the same way.

Inside each study case, `strategy`, `market_sequence`, and `markets` define the
operator strategy and market configuration for that case.

`forecasts_df.csv` contains all time series. The current hybrid case uses:

```text
datetime
plant_1_heat_demand
DE_DA_price
DE_ID3_price
natural_gas_price
aFRR_energy_down_price
aFRR_energy_down_quantity
aFRR_capacity_down_price
aFRR_capacity_down_quantity
```

CO2 cost is currently disabled in the active MVP objective and benchmark, so
`co2_price` is optional for now.

Heat demand is interpreted as average MW_th over the time step and is converted internally to MWh_th using `case.timestep_minutes`.

### Electricity Network Charges (Netzentgelte)

Industrial electricity use faces network charges and levies on top of the market
energy price. When the selected `cases.<case_name>` entry sets
`additional_charges: true`, `additional_charges.csv` is interpreted by the
network-tariff **regulation** selected from `case.country`
(`src/flexi_mod/regulations.py`). For Germany (`country: DE`) the regulation
models:

- flat levies and taxes (CHP, offshore grid levy, concession, electricity tax);
- the grid **energy** charge, tiered by annual full-load hours (`>=`/`< 2500 h/a`);
- the grid **capacity** charge (`EUR/MW.a x peak`), tiered the same way;
- the **special network use** group A/B split (group A on the first 1 GWh); and
- §19(2) StromNEV **atypical grid use**: the capacity charge is billed on the
  maximum load inside the DSO high-load windows, not the annual peak.

Provide the rates in `additional_charges.csv` (one row per component, units
`EUR/MWh` or `EUR/MW.a`):

```text
component,unit,plant_1
Grid energy charge >=2500 h/a,EUR/MWh,36.9
Grid energy charge <2500 h/a,EUR/MWh,45.6
Grid capacity charge >=2500 h/a,EUR/MW.a,66570
Grid capacity charge <2500 h/a,EUR/MW.a,44850
CHP surcharge,EUR/MWh,2.77
Surcharge for special network use (group A),EUR/MWh,15.58
Surcharge for special network use (group B),EUR/MWh,0.5
```

During dispatch the strategy adds a **marginal per-MWh charge**
(`levies + grid energy[assumed tier] + special-use group B`) to DA, IDC, and aFRR
energy prices. The capacity charge, the group-A premium, and any full-load-hour
tier true-up are settled **ex-post** into `grid_fee_summary.csv`, and
`net_operating_cost_incl_grid_fees_EUR` is added to the summary indicators.

**Atypical grid use** is driven by a `high_load_window` (`0/1`) column in
`forecasts_df.csv`: where it is `1`, grid charging is blocked (and aFRR-down
capacity is not reserved in overlapping blocks), so the billed window peak — and
the capacity charge — go to zero. Generate the column from the regulation's
`compute_high_load_window(...)` helper.

The assumed full-load-hour tier defaults to `>=2500 h/a`; override with
`--assumed-grid-tier {high,low}`. If the realized tier differs, the bill is
corrected ex-post and a warning suggests re-running. Adding another country is
one `GridFeeRegulation` subclass in `regulations.py`.

These charges apply only to consumed electricity. aFRR capacity *revenue* is
unaffected.

## Run The First Case

The beginner setup above shows the complete installation and first run. Once the
environment is active, you can run the registered example:

```bash
python src/flexi_mod/simulation/run_case.py --example hybrid_ETES_ID_buy
```

You can also run a case directory directly:

```bash
python src/flexi_mod/simulation/run_case.py --case data/input/hybrid_ETES_ID_buy --study-case hybrid_ETES_ID_buy
```

Outputs are written by the runner to `data/output/<case_name>_<strategy_name>/` by default.
For the first case this is `data/output/hybrid_ETES_ID_buy_hybrid_etes_gas/`:

```text
dispatch_results.csv
market_ledger.csv
storage_cost_ledger.csv
summary_indicators.csv
afrr_energy_data_quality_summary.csv
dashboard.html
```

### Interactive dashboards

Results are explored with Plotly dashboards. There are two front ends that show the same
charts, so use whichever fits. The full guide, with every option and a description of each tab,
is in [docs/dashboard.md](docs/dashboard.md).

| | Static HTML | Live Dash app |
|---|---|---|
| Start | written by every run, or `plot_case.py` | `fleximod-dashboard` |
| Needs | nothing (open the file) | `pip install -e .[dashboard]` |
| Works offline | yes | yes |
| Best for | one finished case, sharing, archiving | browsing many cases, filtering a period |

**Static dashboard.** `dashboard.html` is one self-contained file with tabs (Overview,
Markets, Operation, Costs, Patterns, Data), KPI tiles, a plant selector, hover values,
zoom, and a light/dark switch. It adapts to the plant family detected in the dispatch table
(steam/ETES and boiler plants, buildings with EV fleets and PV, cement, steel). Every run writes
it into the output folder (skip it with `--no-plots`). To rebuild it, or to build a comparison
page across many cases:

```bash
python src/flexi_mod/simulation/plot_case.py --output-dir data/output/<case_folder> --open
python src/flexi_mod/simulation/plot_case.py --case data/input/<case> --start 2025-01-01 --end 2025-01-31
python src/flexi_mod/simulation/plot_case.py --compare data/output/steel_results data/output/cement_results
```

Step by step, from a terminal in the project folder:

```bash
source .venv/bin/activate                                   # 1. activate the environment
find data/output -name "dispatch_results.csv*"              # 2. find a result folder
python src/flexi_mod/simulation/plot_case.py --output-dir "data/output/<case_folder>"   # 3. build
# 4. open the printed dashboard.html in a browser (download it first on a remote server)
```

Running `plot_case.py` with no arguments (for example with the editor's play button) builds the
dashboard of the most recently written result folder under `data/output`. The script prints
`Dashboard created: <path>` when it is done.

Long runs are shown at an automatic resolution (hourly, 6-hourly or daily) so each chart stays
readable; `--resolution native` shows every step. `--plotly-js cdn` makes the file about 5 MB
smaller but needs internet access when it is opened.

**Live dashboard.** Scans a folder for case outputs (plain `.csv` or `.csv.zst` tables) and
lets you pick the case, plant, period and resolution, and compare many cases:

```bash
fleximod-dashboard --root data/output            # then open http://127.0.0.1:8050
fleximod-dashboard --case data/output/<case_folder>
```

Without the installed command: `python src/flexi_mod/visualisation/dashboard/dash_app.py`.

**On a remote server** there is no browser: download `dashboard.html` and open it on your own
computer, or forward the Dash port (`ssh -L 8050:localhost:8050 user@server`, or the Ports tab in
VS Code) and open <http://localhost:8050> locally.

The dashboards include the analyses of `notebooks/fleximod_report_analysis.ipynb` for steam/ETES
cases (savings against the gas-only benchmark, cashflow waterfall, demand coverage, market value,
grid fees and atypical grid use, system setup). The system setup and the CAPEX/OPEX figures need
the case input folder, which is found automatically for the default folder naming or given with
`--input-dir`.

The dashboard code lives in `src/flexi_mod/visualisation/dashboard/`. Charts that need
columns a case does not have (for example IDC or aFRR in a day-ahead-only run) are skipped
rather than failing.

## Troubleshooting

- If `pre-commit` is not recognized, run `python -m pip install -r requirements.txt`.
- If solver errors mention HiGHS or `highspy`, confirm installation with `python -m pip show highspy`.
- If input data are missing, check that `data/input/hybrid_ETES_ID_buy/` contains `config.yaml`, `plants.csv`, and `forecasts_df.csv`.

## Pre-Commit Hooks

Install the development tools and enable pre-commit hooks with:

```bash
pip install -r requirements.txt
pre-commit install
```

Run all hooks manually with:

```bash
pre-commit run --all-files
```

The configured hooks run REUSE SPDX annotation, Ruff linting and formatting, basic file hygiene checks, YAML/TOML checks, and codespell.

## Documentation

Additional documentation is available in:

- [Modeling Philosophy And Architecture](docs/modeling_philosophy_and_architecture.md)
- [Plant Input Reference](docs/plant_inputs.md)
- [Interactive Dashboards](docs/dashboard.md)
- [Strategy Documentation](docs/strategies.md)

## Market Layer

Market classes live in `src/flexi_mod/markets/`. They describe market design:
the traded product, product resolution, gate-open/gate-close metadata,
configured signal columns and product-rule parameters. They prepare market data
for the strategy, but they do not decide the industrial operator's buy, sell or
bid behaviour.

The current market classes cover aFRR down capacity, day-ahead energy,
intraday continuous energy adjustments, and aFRR down energy.

## Configuration Philosophy

`config.yaml` does not contain file paths, output switches or detailed strategy rules. Those are owned by the runner and strategy classes.

The current config keeps only case and model assumptions. Each top-level
`cases.<case_name>` entry owns:

- simulation period and resolution;
- strategy name and Pyomo rolling-horizon dispatch settings;
- solver choice;
- market sequence;
- market enable flags, product rules, timing metadata and signal column mappings.

Intraday continuous can also define `allowed_actions.buy` and
`allowed_actions.sell`. This lets a modeller run buy-only, sell-only, both
directions, or observe-only IDC studies without changing the strategy code. The
gas-based benchmark and later IDC/aFRR bidding rules are embedded in
`HybridETESGasStrategy`, so the config stays compact and close to the market
setup.

## Market-Calendar Simulation

Markets are evaluated in the order given by `market_sequence`, inside each
rolling decision window. The intended industrial sequence is:

```text
aFRR down capacity reservation
-> day-ahead electricity procurement
-> intraday continuous adjustment
-> aFRR down energy activation
-> final physical dispatch/accounting
```

With the current German case settings:

```yaml
cases:
  hybrid_ETES_ID_buy:
    strategy:
      dispatch:
        rolling_horizon_enabled: true
        dispatch_horizon_hours: 24
        rolling_step_hours: 24
```

FLEXIMOD runs one delivery day at a time:

```text
Day 1: configured market stages -> final dispatch/accounting
Day 2: configured market stages -> final dispatch/accounting
...
```

If `dispatch_horizon_hours` and `rolling_step_hours` are both changed to `48`,
the model makes two-day decision windows instead. Disabled markets are skipped
cleanly, and missing intermediate markets use zero positions or reserves where
that is physically meaningful.

aFRR down capacity, when enabled, reserves ETES charging headroom before the
day-ahead stage. Day-ahead then creates a fixed electricity baseline. Intraday
continuous can adjust that baseline through buy/sell volumes. aFRR down energy
adds proxy activated electricity consumption on top of the final planned
position.

The market timing metadata in `config.yaml`, such as `gate_open` and
`gate_close`, is read and reported by the runner. For the German case, this
keeps day-ahead, intraday, aFRR capacity, and aFRR energy timing assumptions in
the case configuration rather than in the strategy code.

The aFRR down activation signal is system-level/proxy activation, not plant-specific activation. Results should be interpreted as a scenario based on the available system activation proxy unless plant-specific bid acceptance and activation data are available.

The aFRR down price sign convention is:

- positive price: the plant pays for activated electricity;
- zero price: activated electricity is settled at zero price;
- negative price: the plant is effectively paid to consume electricity.

If source data use another convention, preprocess it before putting it into `forecasts_df.csv`.

## Decision Windows And Pyomo Dispatch

The plant dispatch is solved with Pyomo inside each market stage and decision
window. For the current case:

- `dispatch_horizon_hours` defines the market decision window;
- `rolling_step_hours` defines how far the window advances;
- all market stages in one decision window start from the same physical ETES
  state of charge;
- only the final enabled stage in that window updates ETES state of charge for
  the next window.

The Pyomo model enforces technology limits, storage state of charge, and strict
useful heat dispatch. For the current ETES + gas boiler case:

```text
gas boiler heat + ETES useful discharge = heat demand
```

There is no artificial unmet-heat or heat-dump variable. If fixed market
positions cannot be physically absorbed and converted into useful heat, the
solve is intentionally infeasible so the modeller sees the inconsistency.

The plant model follows a component/plant split similar to the reference ASSUME-style scripts:

- `plants/technologies.py` defines technology attributes, variables, parameters, and component constraints.
- `plants/steam_generation_plant.py` connects technologies on the plant heat/electricity buses and owns the rolling-horizon solve.
- `plants/building.py` connects a building, electric vehicles, and charging stations for timetable-aware charging and bidirectional V2G dispatch. See [the building model guide](docs/building.md).

## Market Data Warning

Do not push licensed EPEX/EEX or other proprietary market data to GitHub. The `.gitignore` excludes `data/input/**/forecasts_df.csv` and generated outputs by default. Keep only small non-confidential examples and templates under version control.
