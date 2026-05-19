<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# FLEXIMOD

FlexIMOD stands for **Flexible Industrial Market-Oriented Dispatch Model**.

It is a modelling framework for industrial energy systems that participate in electricity and flexibility markets. The goal is to represent industrial plants, their connected technologies, and their market-oriented dispatch decisions in a modular and extensible way.

The current MVP uses a hybrid ETES + gas boiler steam plant as the first case study. This first case implements Germany-oriented day-ahead, intraday continuous, and proxy aFRR down energy stages with a rule-based market strategy and a Pyomo rolling-horizon plant dispatch model. Future cases can extend the same structure to other industrial processes, technologies, countries, and market designs.

The architecture is intentionally modular:

- `config.yaml` contains modelling settings, active markets, market product rules, market timing and market signal mappings.
- `flexi_mod.simulation.run_case` contains example selection, input paths, output paths and output switches.
- `plants.csv` defines one plant by grouping connected technology rows.
- `forecasts_df.csv` contains all time series.

## Quick Start For Beginners

FLEXIMOD targets Python 3.13 or newer. From a fresh checkout, open PowerShell in
the repository folder and run:

```powershell
git clone https://github.com/Manish-Khanra/FLEXIMOD.git
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
python src\flexi_mod\simulation\run_case.py --case data\input\hybrid_ETES_DE
```

Create plots from the generated output:

```powershell
python src\flexi_mod\simulation\plot_case.py --case data\input\hybrid_ETES_DE
```

Check the output folder:

```text
data/output/hybrid_ETES_DE/
|-- dispatch_results.csv
|-- market_ledger.csv
|-- storage_cost_ledger.csv
|-- summary_indicators.csv
`-- plots/
```

## Input Structure

The first case is stored in:

```text
data/input/hybrid_ETES_DE/
|-- config.yaml
|-- plants.csv
|-- forecasts_df.csv
`-- additional_charges.csv  # optional, only used when case.additional_charges is true
```

`plants.csv` groups technologies by plant name. For example, two rows with `name=plant_1` define the ETES storage and gas boiler attached to the same industrial plant.

`forecasts_df.csv` contains all time series. The current hybrid case uses:

```text
datetime
plant_1_heat_demand
DE_DA_price
DE_ID3_price
natural_gas_price
aFRR_energy_down_price
aFRR_energy_down_quantity
```

CO2 cost is currently disabled in the active MVP objective and benchmark, so
`co2_price` is optional for now.

Heat demand is interpreted as average MW_th over the time step and is converted internally to MWh_th using `case.timestep_minutes`.

### Optional Electricity Consumption Charges

Industrial electricity use may face additional energy charges on top of the
market energy price, for example network consumption charges, metering and
operation, concession fees, surcharges, levies, and electricity tax.

Enable them in `config.yaml` with:

```yaml
case:
  additional_charges: true
```

Then provide `additional_charges.csv` in the case input folder:

```text
component,unit,plant_1
Network consumption price,EUR/MWh,6.9
Metering and operation,EUR/MWh,2.0
```

FLEXIMOD sums the component rows per plant and adds the result to DA, IDC, and
aFRR energy prices before the strategy compares electricity against the
gas-based benchmark. The same adder enters the plant dispatch objective and
storage cost ledger as an electricity consumption charge.

These charges apply only to consumed energy. They do not apply to aFRR capacity
reservation or capacity revenue.

## Run The First Case

The beginner setup above shows the complete installation and first run. Once the
environment is active, you can run the registered example:

```bash
python src/flexi_mod/simulation/run_case.py --example hybrid_etes_de
```

You can also run a case directory directly:

```bash
python src/flexi_mod/simulation/run_case.py --case data/input/hybrid_ETES_DE
```

Create plots from existing output files:

```bash
python src/flexi_mod/simulation/plot_case.py --case data/input/hybrid_ETES_DE --format png
```

Outputs are written by the runner to `data/output/hybrid_ETES_DE/` by default:

```text
dispatch_results.csv
market_ledger.csv
storage_cost_ledger.csv
summary_indicators.csv
afrr_energy_data_quality_summary.csv
plots/
```

The plotting command recalculates analytics from the output CSV files, refreshes
`summary_indicators.csv`, and writes report-ready figures to
`data/output/<case_name>/plots/`. The first plotting suite includes combined
plant operation and storage dynamics, market prices and benchmark, electricity
procurement, storage content by source market, a sample-day explanation figure,
cost breakdown, heat supply share, electricity market share, and price-response
plots.

## Troubleshooting

- If `pre-commit` is not recognized, run `python -m pip install -r requirements.txt`.
- If solver errors mention HiGHS or `highspy`, confirm installation with `python -m pip show highspy`.
- If input data are missing, check that `data/input/hybrid_ETES_DE/` contains `config.yaml`, `plants.csv`, and `forecasts_df.csv`.

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

The current config keeps only case and model assumptions:

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

## Market Data Warning

Do not push licensed EPEX/EEX or other proprietary market data to GitHub. The `.gitignore` excludes `data/input/**/forecasts_df.csv` and generated outputs by default. Keep only small non-confidential examples and templates under version control.
