<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Modeling Philosophy And Architecture

FlexIMOD stands for **Flexible Industrial Market-Oriented Dispatch Model**. It is
designed to model industrial energy systems that participate in sequential
electricity and flexibility markets.

The first implemented case is a hybrid ETES + gas boiler steam plant in Germany,
but the architecture is meant to support other industrial processes, technologies,
countries, and market designs.

## Core Philosophy

FlexIMOD separates three questions that are often mixed in monolithic models:

1. Which market stages exist, and in which order are they evaluated?
2. Which market actions are attractive or allowed according to a strategy?
3. Which plant operation is physically feasible and cost-minimal?

The answer to the first question belongs to the case configuration and runner.
The answer to the second question belongs to strategy classes. The answer to the
third question belongs to the plant and technology model.

This gives the project its central modelling principle:

```text
Rule-based market strategy + Pyomo-based plant dispatch and feasibility
```

The strategy should not hard-code plant physics. The plant model should not
hard-code market rules. The runner coordinates the order in which both are used.

## Sequential Market Logic

Markets are evaluated in the order defined by `market_sequence` in `config.yaml`.
For the first MVP only day-ahead is enabled:

```text
day_ahead -> intraday_continuous -> afrr_energy -> afrr_capacity
```

The later markets are present as placeholders and should be activated one by one.
The intended rule is that earlier market decisions become fixed when later markets
are evaluated.

Examples:

- intraday continuous may adjust a day-ahead position, but should not overwrite it;
- aFRR energy must use only the remaining ETES charging headroom after earlier
  markets;
- aFRR capacity should reserve headroom before day-ahead and intraday decisions.

## Rolling-Horizon Plant Dispatch

The current plant dispatch is deterministic rolling horizon:

- solve a 48-hour Pyomo horizon;
- implement the first 24 hours;
- carry the implemented final state of charge into the next horizon;
- continue until the simulation period ends.

This rolling horizon is located in `SteamGenerationPlant.solve_rolling`.

The market simulation is sequential. The plant dispatch inside each market stage
is rolling horizon. These are related but distinct concepts.

## Main Packages

The current Python import package is still `etes_market_model`, while new
generic visualisation utilities live under `flexi_mod`. Over time, the codebase
can be migrated further toward the generic FlexIMOD namespace.

Important modules:

```text
src/etes_market_model/config/case_config.py
src/etes_market_model/data/data_loader.py
src/etes_market_model/plants/technologies.py
src/etes_market_model/plants/steam_generation_plant.py
src/etes_market_model/strategies/base_strategy.py
src/etes_market_model/strategies/hybrid_etes_gas_strategy.py
src/etes_market_model/ledgers/market_ledger.py
src/etes_market_model/ledgers/storage_cost_ledger.py
src/etes_market_model/simulation/simulation_runner.py
src/flexi_mod/visualisation/analytics.py
src/flexi_mod/visualisation/plots.py
```

## Configuration Layer

The case configuration describes modelling assumptions and market setup:

- case name, country, time range, and time resolution;
- active strategy name;
- Pyomo dispatch horizon and rolling step;
- solver choice;
- market sequence;
- market enable flags;
- market timing metadata;
- mapping from market signals to columns in `forecasts_df.csv`.

`config.yaml` intentionally does not contain output paths, output switches, or
detailed strategy rules. Those belong to the runner and strategy classes.

## Input Data Layer

Each case input folder contains:

```text
config.yaml
plants.csv
forecasts_df.csv
```

`plants.csv` defines industrial plants and their connected technologies. Rows
with the same `name` belong to one plant. Different `technology` values define
connected components.

`forecasts_df.csv` contains all time series. For the current day-ahead MVP the
minimum required time-series columns are:

```text
datetime
plant_1_heat_demand
DE_DA_price
natural_gas_price
```

CO2 is currently disabled in the active objective and benchmark. A `co2_price`
column may still exist in input files for later use, but it is not required for
the current MVP.

## Plant And Technology Layer

The plant model follows a reference-style split:

- `technologies.py` defines technology classes, attributes, Pyomo variables,
  parameters, and component-level constraints.
- `steam_generation_plant.py` connects those technologies into one plant-level
  Pyomo model.

For the first case, the plant contains:

- `ThermalStorage`, representing ETES storage;
- `GasBoiler`, representing natural-gas heat supply.

The plant-level model connects both technologies through a heat bus:

```text
storage discharge + gas boiler heat + unmet heat >= heat demand
```

Electricity consumption is currently equal to ETES electric charging:

```text
electricity consumption = electric charge to storage
```

## Objective Function

The current MVP minimizes:

```text
electricity procurement cost
+ gas fuel cost
+ unmet heat penalty
```

CO2 cost is kept as a zero-valued output column for compatibility, but it is not
included in the active objective for now.

Unmet heat is an emergency slack variable with a very high penalty. It exists so
the model can return a diagnostic result if the configured plant cannot meet heat
demand.

## Output Layer

The main output files are:

```text
dispatch_results.csv
market_ledger.csv
storage_cost_ledger.csv
summary_indicators.csv
plots/
```

`dispatch_results.csv` contains physical plant operation and costs.

`market_ledger.csv` contains market positions and electricity consumption by
market stage.

`storage_cost_ledger.csv` tracks the economic cost of stored heat and now also
contains source-market inventory columns for future day-ahead, intraday, and aFRR
storage attribution.

`summary_indicators.csv` is calculated from the outputs by the analytics module.

## Visualisation And Analytics Layer

The generic plotting and analytics code lives in:

```text
src/flexi_mod/visualisation/analytics.py
src/flexi_mod/visualisation/plots.py
```

The plotting script reads the case config, locates the output folder using the
case name, refreshes analytics, and writes report-ready figures to:

```text
data/output/<case_name>/plots/
```

Plots are designed to handle missing future-market columns gracefully. For
example, if IDC or aFRR columns are absent in a day-ahead-only simulation, the
plotting module warns and skips only those series.
