<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Strategy Documentation

This document describes how FlexIMOD strategies are intended to work and
documents the first implemented strategy.

## Strategy Role

A strategy represents market-facing decision logic. It decides which market
actions are economically attractive or allowed, but it should not duplicate the
plant physics.

In FlexIMOD, the intended split is:

```text
strategy: market logic, eligibility, benchmarks, bid rules
plant model: feasible dispatch, heat balance, storage balance, costs
runner: sequencing, input/output coordination
```

The current base strategy interface is defined in:

```text
src/etes_market_model/strategies/base_strategy.py
```

It exposes stage-level methods:

```text
decide_day_ahead(...)
decide_intraday_continuous(...)
decide_afrr_energy(...)
decide_afrr_capacity(...)
```

Only `decide_day_ahead` is implemented for the first MVP. The other stages are
placeholders and should later respect fixed decisions from earlier markets.

## First Implemented Strategy

The first implemented strategy is:

```text
HybridETESGasStrategy
```

It lives in:

```text
src/etes_market_model/strategies/hybrid_etes_gas_strategy.py
```

It is designed for the first case study:

```text
hybrid ETES + gas boiler steam plant
Germany
day-ahead market only
15-minute dispatch resolution
```

## Day-Ahead Strategy Logic

The strategy is a price-taking day-ahead procurement strategy for ETES charging.

The idea is:

1. Calculate the cost of producing one MWh of useful heat with the gas boiler.
2. Calculate the cost of producing one MWh of useful heat through electric
   charging and later storage discharge.
3. Allow ETES charging only when electric heat is cheaper than gas heat.
4. Let the Pyomo plant model decide the feasible amount of charging, discharging,
   gas heat, and state of charge.

The strategy does not directly choose the exact charging volume. It creates a
time-dependent gate:

```text
charge_allowed[t] = True or False
```

The plant model then enforces:

```text
ETES charge[t] <= ETES max charge[t] * charge_allowed[t]
```

So if the strategy says charging is not allowed, the Pyomo model cannot buy
day-ahead electricity for ETES charging in that time step.

## Gas-Based Heat Benchmark

For the current MVP, CO2 is disabled. The gas benchmark is therefore:

```text
gas_based_heat_cost =
    natural_gas_price / gas_boiler_efficiency
```

Example:

```text
natural_gas_price = 50 EUR/MWh_fuel
gas_boiler_efficiency = 0.90

gas_based_heat_cost = 50 / 0.90 = 55.56 EUR/MWh_th
```

This benchmark is stored in the dispatch output as:

```text
gas_based_heat_benchmark_EUR_per_MWh_th
```

## Electric Heat Cost

The ETES route loses energy during charging and discharging. Therefore, the
effective electric heat cost is calculated as:

```text
electric_heat_cost =
    day_ahead_price
    / (ETES charge efficiency * ETES discharge efficiency)
```

If charge efficiency is `0.92` and discharge efficiency is `0.92`, then one MWh
of electricity produces:

```text
0.92 * 0.92 = 0.8464 MWh_th
```

So a day-ahead price of `40 EUR/MWh_el` corresponds to:

```text
40 / 0.8464 = 47.26 EUR/MWh_th
```

## Charging Rule

The charging rule is:

```text
electric_heat_cost <= gas_based_heat_cost - safety_margin
```

The current safety margin is:

```text
0.0 EUR/MWh
```

So the current rule is simply:

```text
charge ETES only when electric heat is cheaper than gas heat
```

The safety margin is currently embedded in the strategy code:

```text
ELECTRICITY_PRICE_SAFETY_MARGIN_EUR_PER_MWH = 0.0
```

Later, this can be made configurable if we want to test more conservative or
more aggressive bidding behaviour.

## Day-Ahead Position

For the MVP, the day-ahead electricity position is the optimized ETES electricity
consumption:

```text
DA_position_MWh = electricity_consumption_MWh
```

Because the current MVP has no intraday or aFRR stage active, all electricity
used for ETES charging is assigned to the day-ahead market.

## What Pyomo Decides

After the strategy computes the benchmark and charging gate, the plant model
decides the feasible dispatch by minimizing:

```text
electricity cost
+ gas fuel cost
+ unmet heat penalty
```

The plant model decides:

- ETES charging;
- ETES discharging;
- ETES state of charge;
- gas boiler heat output;
- unmet heat, only as emergency slack;
- electricity consumption.

The strategy only decides when charging is economically allowed. The plant model
decides how much charging is useful and feasible.

## Current Simplifications

The day-ahead strategy is deliberately simple:

- the plant is treated as a price taker;
- there is no explicit bid curve;
- there is no market clearing uncertainty;
- there is no sell-back or arbitrage trading yet;
- day-ahead positions are not fixed before the plant optimization;
- CO2 cost is disabled for the active MVP objective and benchmark;
- day-ahead prices are treated as deterministic input data.

These simplifications are acceptable for the first MVP because the objective is
to test the plant physics, ledgers, and output workflow before adding more
realistic market mechanisms.

## Planned Strategy Extensions

The next strategy stages should be added in this order:

1. Intraday continuous using an ID3 price signal.
2. Negative aFRR energy using exogenous activation.
3. Negative aFRR capacity with reserved charging headroom.

The key rule for all future stages:

```text
later markets must respect earlier fixed positions
```

For example:

- IDC may buy additional electricity when intraday prices are attractive;
- IDC may sell or reduce a day-ahead position only if heat can still be supplied;
- negative aFRR energy may increase electricity consumption only if charging
  headroom remains;
- aFRR capacity should reserve headroom before day-ahead and intraday dispatch.

## Strategy Design Principles

New strategies should follow these principles:

- use market signal column names from `config.yaml` when they are market-specific;
- keep general strategy constants in the strategy class or a dedicated strategy
  configuration object;
- do not hard-code plant names;
- do not duplicate Pyomo plant constraints inside the strategy;
- pass eligibility signals and economic signals to the plant model;
- keep disabled market stages safe and explicit;
- record market decisions through ledgers.
