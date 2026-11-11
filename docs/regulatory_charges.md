<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Regulatory Charges by Country

This page lists the electricity network charges (grid fees), levies and taxes
that FLEXIMOD models per country, **where you provide each one**, and whether it
is **static** or **dynamic**. Use it when preparing input data for a case.

The regulation is selected **by name** with `grid_fee_regulation:` in the case
`config.yaml`. Each name matches a regulation registered in
[`src/flexi_mod/regulations/grid_fees.py`](../src/flexi_mod/regulations/grid_fees.py):

| `grid_fee_regulation` | Models |
|---|---|
| `de_atypical_grid_use` | Germany: full-load-hour tiers, special network use A/B, §19(2) StromNEV atypical grid use |
| `de_agnes_enflex` | Germany, prospective AgNes design: capacity charge on an ordered capacity, energy charge within / above the order |
| `es_peajes_6xtd` | Spain: 6.xTD access tariff (peajes) + IEE + capacity charge |
| `fr_turpe` | France: TURPE + accise + fixed annual charges |

If `grid_fee_regulation` is omitted, `country:` (`DE` / `ES` / `FR`) picks that
country's regulation, so existing cases keep working. Naming the regulation lets
one country carry several designs, for example today's rules and a prospective
reform, as separate study cases in one `config.yaml`:

```yaml
cases:
  de_baseline:
    country: DE
    grid_fee_regulation: de_atypical_grid_use
    additional_charges: true
  de_reform_test:
    country: DE
    grid_fee_regulation: de_atypical_grid_use   # same rules, different numbers
    additional_charges_file: additional_charges_reform.csv
    additional_charges: true
```

- **Same rules, different values** (a new levy rate, a higher capacity charge):
  keep the regulation name and point `additional_charges_file` at another CSV in
  the case folder (default `additional_charges.csv`).
- **Different rules** (a new tier scheme, another peak basis): add a
  `GridFeeRegulation` subclass and one registry entry, then name it here.

## The two input files

| File | Holds | Example |
|---|---|---|
| `additional_charges.csv` | **Static** charges — one fixed value each | `grid_capacity_charge,EUR/MW.a,19629` |
| `forecasts_df.csv` | **Dynamic** charges — one value per 15-min time slot | `grid_energy_charge` column |

Only three units are accepted in `additional_charges.csv`: **`EUR/MWh`** (per-energy),
**`EUR/MW.a`** (per-power, per year) and **`MW`** (a power value, for example the ordered
grid capacity of a capacity-order design).

## Static vs. dynamic — the quick rule

- **Static** → a single fixed number in `additional_charges.csv`. Its component
  name **must match** the regulation's charge name exactly (an unknown name
  raises an error rather than being silently ignored).
- **Dynamic** → a time-varying series in `forecasts_df.csv`, always the column
  **`grid_energy_charge`** (EUR/MWh). Only Spain and France use it; Germany's
  grid fees are flat.
- **Tax** → a multiplicative rate defined in code (only Spain's IEE), not a data
  input.

## How each charge is applied

| Unit / type | When it hits the model | Effect |
|---|---|---|
| `EUR/MWh` (static or dynamic) | **During dispatch** (per consumed MWh) | added to the delivered electricity price the plant optimizes against |
| `EUR/MW.a` (capacity) | **Ex-post only** (after the run) | `rate × annual peak MW`; **no** dispatch feedback |
| Tax rate (multiplicative) | On the electricity bill | `× (1 + rate)` on (market price + per-MWh charges) |

---

## 🇩🇪 Germany (`DE`)

All charges are **static**; Germany has **no** dynamic `grid_energy_charge` column.

| Charge | File | Component name | Unit | Type | Notes |
|---|---|---|---|---|---|
| Grid energy charge (≥ 2500 h/a) | `additional_charges.csv` | `grid_energy_charge_high` | EUR/MWh | Static | rate used when annual full-load hours ≥ 2500 |
| Grid energy charge (< 2500 h/a) | `additional_charges.csv` | `grid_energy_charge_low` | EUR/MWh | Static | rate used when < 2500 |
| Grid capacity charge (≥ 2500 h/a) | `additional_charges.csv` | `grid_capacity_charge_high` | EUR/MW.a | Static | × peak, ex-post |
| Grid capacity charge (< 2500 h/a) | `additional_charges.csv` | `grid_capacity_charge_low` | EUR/MW.a | Static | × peak, ex-post |
| Special network use — group A | `additional_charges.csv` | `special_network_use_a` | EUR/MWh | Static | applies to the first 1 GWh/a |
| Special network use — group B | `additional_charges.csv` | `special_network_use_b` | EUR/MWh | Static | energy beyond 1 GWh/a |
| CHP surcharge (KWKG) | `additional_charges.csv` | `chp_surcharge` | EUR/MWh | Static | levy |
| Offshore grid levy | `additional_charges.csv` | `offshore_grid_levy` | EUR/MWh | Static | levy |
| Concession fee (Konzessionsabgabe) | `additional_charges.csv` | `concession_fee` | EUR/MWh | Static | levy |
| Electricity tax (Stromsteuer) | `additional_charges.csv` | `electricity_tax` | EUR/MWh | Static | flat per-MWh levy |

**Tiers:** the grid energy/capacity rate is chosen from the `high`/`low` pair by
the plant's realized annual full-load hours (threshold 2500 h/a). Both halves of
a tiered pair must be provided.

**Levies are not grid fees.** The last four rows (CHP surcharge, offshore grid
levy, concession fee, electricity tax) are fixed amounts per MWh set by law. They
are the same under every German design, so they are not part of a grid-fee class:
they are modelled once in
[`src/flexi_mod/regulations/levies.py`](../src/flexi_mod/regulations/levies.py)
(`ElectricityLevies`) and shared by `de_atypical_grid_use` and `de_agnes_enflex`.
They stay in `additional_charges.csv`, under the same names as before. Every MWh
drawn from the grid pays their sum, in the electricity price the plant sees and in
the settlement after the run.

**§19(2) atypical grid use:** if `capacity_peak_basis = high_load_window`, the
capacity charge is billed on the peak **within the high-load windows** — so
avoiding consumption in those windows lowers the bill.

---

## 🇩🇪 Germany, prospective AgNes design (`de_agnes_enflex`)

Selected with `grid_fee_regulation: de_agnes_enflex`. The consumer orders a grid
capacity and pays for it whether or not it is used; energy drawn above the order
is charged at a higher rate. All charges are **static** (no dynamic column), and
all are required except the levies.

| Charge | Name | Unit | Applied | Notes |
|---|---|---|---|---|
| Ordered capacity (P_ordered) | `grid_ordered_capacity` | MW | scenario input | fixed for the whole run; a sweep is a set of cases that differ only here |
| Capacity charge (KP) | `grid_ordered_capacity_charge` | EUR/MW.a | ex-post | `charge x ordered capacity`, independent of the realised load; annual, not prorated. 100 EUR/kW.a = 100000 EUR/MW.a |
| Energy charge within the order (AP1) | `grid_energy_charge_within_order` | EUR/MWh | dispatch | marginal adder on every MWh |
| Energy charge above the order (AP2) | `grid_energy_charge_above_order` | EUR/MWh | dispatch | given directly (not derived from a factor); must not be below AP1. The plant pays `AP2 - AP1` on top of AP1 for every MWh above the order |
| Levies | `chp_surcharge`, `offshore_grid_levy`, `concession_fee`, `electricity_tax` | EUR/MWh | dispatch | as for `de_atypical_grid_use` |

The order is judged **per timestep**: a step of `dt` hours is within the order up
to `ordered capacity x dt` MWh and above it beyond that. There is no
high-load-window block, no full-load-hour tier and no special network use.
A charge given in a different unit than listed, a missing required charge or an
unknown name raises an error.

```csv
component,unit,plant_1
grid_ordered_capacity,MW,5
grid_ordered_capacity_charge,EUR/MW.a,100000
grid_energy_charge_within_order,EUR/MWh,20
grid_energy_charge_above_order,EUR/MWh,60
chp_surcharge,EUR/MWh,2.77
offshore_grid_levy,EUR/MWh,8.16
concession_fee,EUR/MWh,1.10
electricity_tax,EUR/MWh,0.50
```

The summary additionally reports `grid_ordered_capacity_MW` and
`grid_excess_over_order_MWh`. The settlement reproduces the cost curves of the
study's capacity-order Excel (tested).

The dashboard (tab "Grid fees") shows the ordered capacity on the capacity-charge card,
a card "Energy above the order" with its share of the grid energy, and a chart "Load
against the ordered capacity": the load duration curve with the order as a dashed line and
the energy above it shaded.

**How the dispatch sees it.** In every timestep the plant draws some energy from the
grid. The ordered capacity allows `ordered capacity x timestep length` MWh per
timestep. The plant objective charges the surcharge `AP2 - AP1` for each MWh above that
limit (a linear term, so the model stays a linear programme), and every MWh pays
`AP1 + levies`. The plant therefore weighs the cost of drawing more than ordered
against its alternatives, such as gas. The capacity charge does not depend on the load,
so it is settled after the run. The dispatch column
`additional_electricity_charges_cost_EUR` contains `AP1 + levies` on all energy plus the
surcharge on the energy above the order.

**How the strategies handle the order.** The study uses day-ahead, aFRR energy and aFRR
capacity (intraday is switched off).

| Market | Rule |
|---|---|
| Day-ahead | The gate is a yes/no test on the price within the order (`AP1 + levies`). The plant optimisation then decides how much to buy, paying the surcharge above the order. |
| aFRR energy | The charging headroom is split in two. The part below the order is offered if the aFRR price passes the normal price test. The part between the order and the maximum charging power is offered only if the price also passes the test with the surcharge added. |
| aFRR capacity | A reservation cannot exceed the ordered capacity. In the plant model, wherever capacity is reserved, the day-ahead load plus the reservation must stay below the order, so an activation of reserved capacity never lands above it. |

**Current limits.** The aFRR capacity rule is a fixed cap, not a price test: the plant
never reserves capacity that would not fit below the order, even if the above-order
price would still pay. The capacity price test does not include the effect of the order on
day-ahead charging. This is an assumption of the study: once the capacity is ordered, an
industrial operator keeps its load below the order most of the time. The intraday rules
still use only the charging power, so they are not order-aware (intraday is not part of the
study).

---

## 🇪🇸 Spain (`ES`)

| Charge | File | Name | Unit | Type | Notes |
|---|---|---|---|---|---|
| Grid capacity charge (potencia contratada) | `additional_charges.csv` | `grid_capacity_charge` | EUR/MW.a | Static | × peak, ex-post |
| Access tariff / peajes (6.2TD) | `forecasts_df.csv` | `grid_energy_charge` | EUR/MWh | **Dynamic** | per-MWh; varies by time-of-use period (P1–P6) |
| Electricity tax — IEE | *code constant* | `ELECTRICITY_TAX_RATE` | % | Rate | multiplicative on (market + peajes) |

> **Note:** all of Spain's per-MWh network charges are contained in the dynamic
> `grid_energy_charge` column, so `additional_charges.csv` holds only the
> capacity charge. The IEE rate is a code constant (currently `0.007669044`);
> the statutory IEE is `5.11269632%` — **this rate is under review**.

---

## 🇫🇷 France (`FR`)

| Charge | File | Name | Unit | Type | Notes |
|---|---|---|---|---|---|
| Capacity Obligation (Obligation de Capacité) | `additional_charges.csv` | `capacity_obligation` | EUR/MW.a | Static | × peak, ex-post |
| TURPE — management (composante de gestion) | `additional_charges.csv` | `turpe_management` | EUR/MW.a | Static | × peak, ex-post |
| TURPE — metering (composante de comptage) | `additional_charges.csv` | `turpe_metering` | EUR/MW.a | Static | × peak, ex-post |
| TURPE — fixed withdrawal charge (soutirage) | `additional_charges.csv` | `turpe_fix` | EUR/MW.a | Static | × peak, ex-post |
| TURPE energy + accise | `forecasts_df.csv` | `grid_energy_charge` | EUR/MWh | **Dynamic** | per-MWh; pre-summed per time slot |

> **Note:** France has no multiplicative tax. All per-MWh charges (TURPE energy +
> accise) are pre-summed into the dynamic `grid_energy_charge` column, so
> `additional_charges.csv` holds only the fixed `EUR/MW.a` charges, which are
> summed and settled on the annual peak.

---

## Adding or changing a static charge

Each regulation's constructor **is** the list of its supported grid-fee charges —
the component names above are exactly the constructor parameters. The German levies
are the exception: they are listed in `levies.py` (`ElectricityLevies`) and taken
out of the CSV first. The loader maps the other `additional_charges.csv` rows to the
constructor parameters and **rejects an unknown or misspelled name** with a clear
error, e.g.:

```
GridFeeConfigError: Unknown grid-fee charge 'grid_capcity_charge_high' for
GermanGridFeeRegulation. Supported charges: chp_surcharge, concession_fee, ...
```

To add a genuinely new charge for a country, add a named parameter to that
regulation's `__init__` (with its unit in a comment) and a matching row in the
case's `additional_charges.csv`. This keeps the data, the code, and this table in
step.
