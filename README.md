<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# FLEXIMOD

FLEXIMOD models an industrial hybrid ETES + gas boiler plant participating sequentially in electricity markets. The MVP implements a Germany day-ahead case with a rule-based market strategy and a Pyomo rolling-horizon steam plant dispatch model.

The architecture is intentionally modular:

- `config.yaml` contains modelling settings, active markets, market timing and market signal mappings.
- `scripts/run_case.py` contains example selection, input paths, output paths and output switches.
- `plants.csv` defines one plant by grouping connected technology rows.
- `forecasts_df.csv` contains all time series.

## Input Structure

The first case is stored in:

```text
data/input/hybrid_ETES_DE/
|-- config.yaml
|-- plants.csv
`-- forecasts_df.csv
```

`plants.csv` groups technologies by plant name. For example, two rows with `name=plant_1` define the ETES storage and gas boiler attached to the same plant.

`forecasts_df.csv` contains all time series. The DA-only MVP needs:

```text
datetime
plant_1_heat_demand
DE_DA_price
natural_gas_price
co2_price
```

Heat demand is interpreted as average MW_th over the time step and is converted internally to MWh_th using `case.timestep_minutes`.

## Run The DA-Only Case

FLEXIMOD targets Python 3.13 or newer.

Run the registered example:

```bash
python scripts/run_case.py --example hybrid_etes_de
```

You can also run a case directory directly:

```bash
python scripts/run_case.py --case data/input/hybrid_ETES_DE
```

Create plots from existing output files:

```bash
python scripts/plot_case.py --example hybrid_etes_de
```

Outputs are written by the runner to `data/output/hybrid_ETES_DE/` by default:

```text
dispatch_results.csv
market_ledger.csv
storage_cost_ledger.csv
summary_indicators.csv
plots/
```

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

## Configuration Philosophy

`config.yaml` does not contain file paths, output switches or detailed strategy rules. Those are owned by the runner and strategy classes.

The current config keeps only case and model assumptions:

- simulation period and resolution;
- strategy name and Pyomo rolling-horizon dispatch settings;
- solver choice;
- market sequence;
- market enable flags, timing metadata and signal column mappings.

The gas-based benchmark and later IDC/aFRR bidding rules are embedded in `HybridETESGasStrategy`, so the config stays compact and close to the market setup.

## Sequential Simulation

Markets are evaluated in the order given by `market_sequence`. The MVP enables only `day_ahead`; `intraday_continuous`, `afrr_energy`, and `afrr_capacity` are present as disabled placeholders. Later stages should respect fixed earlier decisions, so IDC may adjust but not overwrite DA positions, and aFRR energy must use only remaining ETES charging headroom.

To activate later stages, set the relevant market block to `enabled: true` and provide the configured signal columns in `forecasts_df.csv`. The implementation currently raises a clear `NotImplementedError` if those future stages are enabled.

## Pyomo Rolling Horizon

The plant dispatch is deterministic rolling horizon:

- solve a 48-hour Pyomo dispatch horizon;
- implement the first 24 hours;
- carry the ETES state of charge into the next solve;
- repeat until the simulation period ends.

The Pyomo model enforces heat balance, ETES state of charge, charge/discharge limits, gas boiler heat limits, and emergency unmet-heat slack with a high penalty.

The plant model follows a component/plant split similar to the reference ASSUME-style scripts:

- `plants/technologies.py` defines technology attributes, variables, parameters, and component constraints.
- `plants/steam_generation_plant.py` connects technologies on the plant heat/electricity buses and owns the rolling-horizon solve.

## Market Data Warning

Do not push licensed EPEX/EEX or other proprietary market data to GitHub. The `.gitignore` excludes `data/input/**/forecasts_df.csv` and generated outputs by default. Keep only small non-confidential examples and templates under version control.
