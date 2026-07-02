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
markets: product rules, timing, configured signal preparation
strategy: operator decisions, eligibility, benchmarks, bid rules
plant model: feasible dispatch, heat balance, storage balance, costs
runner: sequencing, input/output coordination
```

The runner applies these strategy methods inside each configured decision
window. For example, with a 24-hour decision window the sequence is:

```text
day 1: decide_afrr_capacity -> decide_day_ahead -> decide_intraday_continuous -> decide_afrr_energy
day 2: decide_afrr_capacity -> decide_day_ahead -> decide_intraday_continuous -> decide_afrr_energy
...
```

If a market is disabled, its strategy method is skipped and the downstream
positions default to zero where this is physically meaningful.

The current base strategy interface is defined in:

```text
src/flexi_mod/strategies/base_strategy.py
```

It exposes stage-level methods:

```text
decide_day_ahead(...)
decide_intraday_continuous(...)
decide_afrr_energy(...)
decide_afrr_capacity(...)
```

`decide_afrr_capacity`, `decide_day_ahead`, `decide_intraday_continuous`, and
the first aFRR down energy implementation are available. aFRR down capacity is
evaluated before day-ahead and reserves charging headroom that later market
stages must respect.

## First Implemented Strategy

The first implemented strategy is:

```text
HybridETESGasStrategy
```

It lives in:

```text
src/flexi_mod/strategies/hybrid_etes_gas_strategy.py
```

It is designed for the first case study:

```text
hybrid ETES + gas boiler steam plant
Germany
day-ahead market with optional intraday continuous and aFRR down energy stages
15-minute dispatch resolution
```

<p align="center">
  <img src="assets/hybrid_etes_strategy_flowchart_9x16.svg" alt="Hybrid ETES and gas boiler sequential market strategy flowchart" width="420">
</p>

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
    delivered_day_ahead_price
    / (ETES charge efficiency * ETES discharge efficiency)
```

If charge efficiency is `0.92` and discharge efficiency is `0.92`, then one MWh
of electricity produces:

```text
0.92 * 0.92 = 0.8464 MWh_th
```

So a delivered day-ahead electricity price of `40 EUR/MWh_el` corresponds to:

```text
40 / 0.8464 = 47.26 EUR/MWh_th
```

The delivered electricity price is the market energy price plus the marginal
per-MWh network charge from the case's grid-fee regulation:

```text
delivered_price = market_price + marginal_network_charge
```

The marginal charge is loaded from `additional_charges.csv` (interpreted per
`case.country`; see `regulations.py`) only when `additional_charges: true`. For
Germany it is `flat levies + grid energy charge (assumed full-load-hour tier) +
special network use group B`, applied to consumed electricity in the DA, IDC, and
aFRR energy stages. The tiered capacity charge, group-A premium, and any tier
true-up are settled ex-post, not in the per-step price. None of these apply to
aFRR capacity reservation or capacity revenue.

Under §19(2) StromNEV atypical grid use, the strategy additionally blocks grid
charging during DSO high-load windows (the `high_load_window` column in
`forecasts_df.csv`) and skips aFRR-down capacity reservation in blocks that
overlap a window, driving the billed capacity peak — and the capacity charge — to
zero.

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

When IDC is disabled, all electricity used for ETES charging is assigned to the
day-ahead market. When IDC is enabled, the DA position remains fixed and the
final ETES charging position is adjusted through IDC buy or sell/reduction
volumes.

## Intraday Continuous Strategy Logic

The first IDC implementation is an index-based adjustment model. It uses the
configured intraday price signal, for example an ID3 column in a German case,
through the intraday continuous market class and the configured price signal.
In `config.yaml`, that signal lives under:

```text
markets -> intraday_continuous -> signals -> price
```

IDC is not modelled as an order book. There are no repeated trading loops,
liquidity limits, bid depth assumptions, or individual transactions. The IDC
volume signal may exist in the input data, but it is optional and unused in this
first implementation.

The day-ahead result is fixed before IDC is evaluated. IDC can only adjust the
fixed DA position:

```text
final_planned_electricity_MWh =
    DA_position_MWh + IDC_buy_MWh - IDC_sell_MWh
```

For the current hybrid ETES + gas plant, this final planned electricity is the
ETES charging electricity:

```text
etes_charge_MWh = final_planned_electricity_MWh
```

This mapping is plant-specific and will need to be generalized for future
industrial plants with several electric processes.

## IDC Benchmark And Rules

The gas benchmark is converted to an electricity-side ETES benchmark:

```text
electricity_trading_benchmark =
    gas_based_heat_cost
    * ETES charge efficiency
    * ETES discharge efficiency
```

The current IDC margin is embedded in the strategy code:

```text
IDC_MARGIN_EUR_PER_MWH = 0.0
```

The rules are:

```text
if delivered_IDC_price < electricity_trading_benchmark - margin:
    allow IDC buy

if delivered_IDC_price > electricity_trading_benchmark + margin:
    allow IDC sell/reduction

otherwise:
    no IDC action
```

The configured IDC action switch is applied after these price rules:

```text
markets -> intraday_continuous -> allowed_actions -> buy
markets -> intraday_continuous -> allowed_actions -> sell
```

If `buy` is false, IDC buy bounds are zero even when the IDC price is cheap. If
`sell` is false, IDC sell/reduction bounds are zero even when the IDC price is
expensive. Setting both to false is an observe-only mode: IDC prices are loaded
and reported, but no intraday trade is created.

The strategy creates upper bounds. Pyomo decides the feasible volume:

```text
IDC_buy_upper_bound =
    max(0, ETES max charge per timestep - DA_position_MWh)

IDC_sell_upper_bound =
    DA_position_MWh
```

If individual IDC price values are missing, the strategy issues a warning and
sets both IDC buy and sell bounds to zero for those timesteps. Missing prices
are not silently filled for trading logic.

## Negative aFRR Down Energy Strategy Logic

The first aFRR energy implementation uses direction-specific down columns:

```text
markets -> afrr_energy -> signals -> price
markets -> afrr_energy -> signals -> system_activation
```

The input quantity is treated as the activation request for the representative
plant. The model has no separate system-wide allocation or merit-order award
step. Positive quantities are used directly. Negative quantities are converted
to absolute magnitude and flagged in the data-quality summary. Missing prices
always block bidding, even if activation quantity is zero. Price values of zero
are valid.

The strategy calculates feasible bid potential before Pyomo. The bid potential
uses:

```text
charge-power headroom after DA + IDC
storage-capacity headroom
minimum bid eligibility in MW
bid increments in MW
```

Bid potential is rounded down to the configured increment and set to zero below
the configured minimum. These rules apply to bids, not realised activation. A
bid that passes the price, technical and product-rule checks is treated as
accepted. Actual activation is calculated before the plant solve:

```text
afrr_energy_activated_MWh =
    min(afrr_energy_bid_upper_bound_MWh,
        afrr_system_activation_MWh)
```

The aFRR down price rule uses an explicit benchmark bid price. The benchmark is
the electricity-side ETES value of replacing gas heat:

```text
afrr_energy_bid_price_EUR_per_MWh =
    electricity_trading_benchmark_EUR_per_MWh_el
    + AFRR_ENERGY_BID_MARGIN_EUR_PER_MWH
```

The current margin is embedded in the strategy code and set to zero:

```text
AFRR_ENERGY_BID_MARGIN_EUR_PER_MWH = 0.0
```

The market-side pay-as-cleared spread is reported as:

```text
afrr_energy_market_spread =
    afrr_energy_bid_price - afrr_energy_down_price
```

The plant only places free aFRR down energy bids when the deal is profitable
after industrial consumption charges:

```text
afrr_energy_down_price + additional_electricity_charge
    <= afrr_energy_bid_price
```

So the market settlement stays:

```text
afrr_energy_cost_EUR =
    afrr_energy_activated_MWh * afrr_energy_down_price
```

If the aFRR energy price is negative, this settlement becomes a credit. The
net plant value after charges is reported separately from this settlement.

Activated energy may be used immediately or stored in ETES for later gas
replacement. It is limited by charge-power and storage-capacity headroom over
the complete optimisation horizon, including space needed by already-contracted
future DA and IDC charging. It is not limited by same-timestep gas consumption.

Pyomo receives the activated volume as a fixed parameter. It does not decide TSO
activation. For the current hybrid ETES + gas plant:

```text
actual_electricity_consumption_MWh =
    final_planned_electricity_MWh + afrr_energy_activated_MWh

etes_charge_MWh =
    actual_electricity_consumption_MWh
```

This ETES mapping is plant-specific and must be generalized for future
industrial plants with multiple electric processes.

## aFRR Variable and Parameter Glossary

The suffixes identify both the physical quantity and its unit:

```text
MW      = power or capacity
MWh_el  = electrical energy during one timestep
MWh_th  = stored or delivered thermal energy
[t]     = value for one timestep
```

Python `Series` values contain one value for every timestep. A "scalar at `t`"
is the single value selected from such a series inside the timestep loop.

### Configuration and Plant Parameters

| Code name | Kind | Unit | Definition and use |
|---|---|---:|---|
| `timestep_hours` | Configuration-derived scalar | h | Timestep duration, for example `0.25` for 15 minutes. It converts between MW and MWh. |
| `min_bid_mw` | Market configuration | MW | Smallest permitted aFRR bid. A smaller feasible offer becomes zero. |
| `bid_increment_mw` | Market configuration | MW | Permitted bid step. Feasible bids are rounded down to a multiple of this value. |
| `max_charge_mwh` | Derived plant parameter | MWh_el | Maximum ETES electricity intake in one timestep: `max_power_charge_mw * timestep_hours`. |
| `max_discharge_mwh` | Derived plant parameter | MWh_th | Maximum useful heat ETES can discharge in one timestep: `max_power_discharge_mw * timestep_hours`. |
| `efficiency_charge` | Plant parameter | MWh_th/MWh_el | Fraction of charging electricity entering thermal storage. |
| `efficiency_discharge` | Plant parameter | MWh_th delivered/MWh_th stored | Fraction of withdrawn storage energy delivered as useful heat. |
| `storage_loss_rate` | Plant parameter | fraction/timestep | Fraction of stored thermal energy lost between consecutive timesteps. |
| `max_capacity_mwh` | Plant parameter | MWh_th | Maximum ETES thermal state of charge. |

### Fixed Market and Baseline Quantities

| Code name | Kind | Unit | Definition and use |
|---|---|---:|---|
| `da_position` | Series | MWh_el | Electricity already procured in the day-ahead market. |
| `idc_buy` | Series | MWh_el | Additional electricity bought in intraday continuous trading. |
| `idc_sell` | Series | MWh_el | DA electricity sold back or reduced through IDC. |
| `final_planned` | Series | MWh_el | Fixed electricity schedule before aFRR: `DA + IDC buy - IDC sell`. |
| `reserved_capacity` | Series | MWh_el | Capacity-backed aFRR-down energy headroom reserved for one timestep: `reserved_capacity_MW * timestep_hours`. Reservation itself is not electricity consumption. |
| `baseline_storage_soc` | Series | MWh_th | ETES state of charge resulting from the fixed DA and IDC dispatch before aFRR activation. |
| `baseline_storage_discharge` | Series | MWh_th | Useful heat discharged by ETES in the baseline dispatch. |
| `baseline_gas_heat` | Series | MWh_th | Useful heat supplied by gas in the baseline dispatch. This is the maximum heat source that aFRR-charged storage could potentially replace. |
| `afrr_energy_bid_price` | Series | EUR/MWh_el | Maximum economically acceptable delivered aFRR price, derived from the gas-replacement benchmark and margin. |
| `delivered_afrr_price` | Series | EUR/MWh_el | aFRR energy price plus marginal electricity-consumption charges. |
| `price_allowed` | Boolean Series | — | True only when price data exist, delivered aFRR energy is below the benchmark, and grid charging is not blocked. |
| `system_activation_for_bid` | Series | MWh_el | Representative-plant activation request after blocked, missing-price, and uneconomic timesteps have been set to zero. |

### Physical Headroom, Bids, and Activation

| Code name | Kind | Unit | Definition and use |
|---|---|---:|---|
| `storage_capacity_headroom` | Series | MWh_el | Electricity that could enter the currently unused ETES capacity after charging efficiency. |
| `charge_power_headroom_after_reserve` | Series | MWh_el | Remaining one-timestep charging-power capability after the fixed schedule and capacity reservation. |
| `storage_headroom_after_reserve` | Series | MWh_el | Current storage-capacity headroom after the capacity-backed reservation. |
| `free_bid_potential` | Series | MWh_el | Smaller of charge-power and storage-capacity headroom, before market-increment rounding. |
| `free_bid_upper_bound` | Series | MWh_el | Price-qualified free bid after applying minimum bid and bid-increment rules. It may be reduced further by horizon feasibility. |
| `planned_charge` | Scalar at `t` | MWh_el | Fixed DA plus IDC electricity charging ETES in the current timestep. |
| `baseline_soc` | Scalar at `t` | MWh_th | Baseline ETES state of charge at the end of the current timestep. |
| `replaceable_gas_heat` | Series | MWh_th | Gas heat that additional ETES discharge can physically replace: the smaller of baseline gas heat and remaining ETES discharge-power headroom. |
| `replaceable_heat` | Scalar at `t` | MWh_th | Current-timestep value selected from `replaceable_gas_heat`. |
| `additional_soc_mwh` | Running scalar | MWh_th | Thermal inventory attributable to earlier aFRR activation, after storage losses and gas replacement. |
| `storage_capacity_offer` | Scalar at `t` | MWh_el | Immediate electrical activation that fits in storage, including current replaceable gas heat as a valid outlet. |
| `power_offer` | Scalar at `t` | MWh_el | Immediate electrical activation permitted by charging power: `max_charge_mwh - planned_charge`. |
| `physical_activation_cap` | Scalar at `t` | MWh_el | Immediate activation limit: `min(power_offer, storage_capacity_offer)`. It does not yet account for later contracted charging. |
| `future_storage_input_headroom` | Series | MWh_th | Backward-calculated extra thermal inventory permitted before each timestep's heat outlet while preserving room for all later fixed charging. |
| `future_storage_cap` | Scalar at `t` | MWh_el | Electrical activation possible now after subtracting thermal inventory carried from earlier activation from future headroom. |
| `horizon_activation_cap` | Scalar at `t` | MWh_el | Final physical limit: `min(physical_activation_cap, future_storage_cap)`. This prevents activation now from overfilling ETES later. |
| `capacity_bid` | Scalar at `t` | MWh_el | Capacity-backed aFRR energy bid. It receives priority over the optional free bid. |
| `free_room_after_capacity` | Scalar at `t` | MWh_el | Horizon-feasible headroom remaining after the capacity-backed bid: `max(0, horizon_activation_cap - capacity_bid)`. |
| `free_bid` | Scalar at `t` | MWh_el | Optional, profitable aFRR energy bid after physical limitation and a second market-increment rounding. |
| `total_bid` | Scalar at `t` | MWh_el | Accepted-bid proxy: `capacity_bid + free_bid`. |
| `system_activation` | Scalar at `t` | MWh_el | Current representative-plant activation request. It is not rounded to bid increments. |
| `proxy_activation` | Scalar at `t` | MWh_el | Requested activation covered by the bid: `min(total_bid, system_activation)`. |
| `feasible_activation` | Scalar at `t` | MWh_el | Requested activation after the final horizon headroom check. With a correctly limited free bid, it equals `proxy_activation`. |
| `capacity_activated` | Scalar at `t` | MWh_el | Activated volume allocated to the capacity-backed bid first. |
| `free_activated` | Scalar at `t` | MWh_el | Remaining activated volume allocated to the optional free bid. |
| `total_activated` | Scalar at `t` | MWh_el | Fixed plant instruction: `capacity_activated + free_activated`. |
| `curtailment` | Scalar at `t` | MWh_el | Requested bid-covered activation that physical headroom could not support. It is reported explicitly through `afrr_curtailment_MWh`. |

### Pyomo aFRR Quantities

| Code name | Pyomo kind | Unit | Definition and use |
|---|---|---:|---|
| `m.final_planned_electricity_mwh[t]` | Fixed `Param` | MWh_el | DA plus IDC schedule passed unchanged to the aFRR plant solve. |
| `m.afrr_energy_bid_mwh[t]` | Fixed `Param` | MWh_el | Submitted/accepted-bid proxy used for reporting and market accounting. |
| `m.afrr_energy_activated_mwh[t]` | Fixed `Param` | MWh_el | Activation instruction calculated by the strategy. Pyomo cannot choose or reduce it. |
| `m.actual_electricity_consumption_mwh[t]` | `Expression` | MWh_el | `final planned electricity + activated aFRR energy`. |
| `storage.electric_charge_to_storage[t]` | Decision `Var` constrained to the expression | MWh_el | Physical ETES electricity intake; for this plant it must equal actual electricity consumption. |
| `storage.discharge_heat[t]` | Decision `Var` | MWh_th | Useful heat discharged from ETES. |
| `storage.soc[t]` | Decision `Var` | MWh_th | ETES state of charge after loss, charging, and discharge in timestep `t`. |
| `boiler.heat_out[t]` | Decision `Var` | MWh_th | Gas-boiler useful heat, chosen with storage discharge to meet heat demand exactly. |
| `m.electricity_consumption[t]` | Decision `Var` tied to ETES charge | MWh_el | Metered physical electricity consumption used for consumption charges and result extraction. |

The core identities are therefore:

```text
final planned electricity = DA + IDC buy - IDC sell
actual electricity = final planned electricity + aFRR activation
ETES charge = actual electricity
useful heat demand = ETES discharge + gas heat
```

## What Pyomo Decides

After the strategy computes the benchmark and charging gate, the plant model
decides the feasible dispatch by minimizing:

```text
electricity market cost
+ additional electricity consumption charges
+ gas fuel cost
```

The plant model decides:

- ETES charging;
- ETES discharging;
- ETES state of charge;
- gas boiler heat output;
- electricity consumption.

For day-ahead, the strategy decides when charging is economically allowed. For
IDC it creates feasible buy and sell bounds, and for aFRR it creates compliant
bids and a fixed activation instruction. In every stage, the plant model still
decides the physically feasible ETES discharge, state of charge, gas production,
and exact useful-heat dispatch.

## aFRR Down Capacity Strategy Logic

When enabled, aFRR down capacity is evaluated before day-ahead. The strategy
uses 4-hour capacity blocks prepared by the market class and applies a
conservative rule:

```text
reserve capacity only if:
    positive activation is forecast in the block
    capacity revenue covers estimated opportunity cost
    possible activation energy is economically safe versus the gas benchmark
    ETES charge-power and storage-capacity headroom are sufficient
```

Activation forecast magnitude does not size the capacity bid. Capacity is sized
from technical capability and rounded down to the configured market increment.
For example, a 0.5 MW activation forecast can support a 1 MW capacity bid when
the plant can physically provide the full 1 MW product.

Capacity reservation itself does not add energy to storage. It only reserves
charging headroom and earns capacity revenue. If aFRR energy is also enabled,
later aFRR down activation first uses the capacity-backed energy bid. The
strategy can now also place optional free aFRR energy bids above the reserved
capacity if the additional bid volume is profitable after charges and physically
feasible. If aFRR capacity is enabled but aFRR energy is disabled, the runner
logs this clearly: capacity revenue can be modelled, but activation energy is
not modelled.

The capacity-price safety margin is currently embedded in the strategy code and
set to zero:

```text
AFRR_CAPACITY_MARGIN_EUR_PER_MW_H = 0.0
```

## Current Simplifications

The current DA + IDC + aFRR down strategy is deliberately simple:

- the plant is treated as a price taker;
- there is no explicit multi-step bid curve;
- there is no market clearing uncertainty;
- IDC is an index-based adjustment, not an order-book model;
- aFRR down activation is treated as a representative-plant request;
- day-ahead positions are fixed before IDC adjustments;
- DA and IDC positions are fixed before aFRR down activation;
- CO2 cost is disabled for the active MVP objective and benchmark;
- day-ahead, IDC, and aFRR down prices are treated as deterministic input data.

These simplifications are acceptable for the first MVP because the objective is
to test the plant physics, ledgers, and output workflow before adding more
realistic market mechanisms.

## Planned Strategy Extensions

The next strategy extensions should improve market realism around the existing
stages:

1. Bid acceptance probability or merit-order award modelling.
2. Forecast-based or stochastic price expectations.

The key rule for all future stages:

```text
later markets must respect earlier fixed positions
```

For example:

- IDC may buy additional electricity when intraday prices are attractive;
- IDC may sell or reduce a day-ahead position only if heat can still be supplied;
- negative aFRR energy may increase electricity consumption only if charging
  headroom remains;
- aFRR capacity reserves headroom before day-ahead and intraday dispatch when it
  is enabled.

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
