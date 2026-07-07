<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Regulatory Charges by Country

This page lists the electricity network charges (grid fees), levies and taxes
that FLEXIMOD models per country, **where you provide each one**, and whether it
is **static** or **dynamic**. Use it when preparing input data for a case.

The country is selected by `country:` in the case `config.yaml`
(`DE` / `ES` / `FR`), which picks the matching regulation in
[`src/flexi_mod/regulations.py`](../src/flexi_mod/regulations.py).

## The two input files

| File | Holds | Example |
|---|---|---|
| `additional_charges.csv` | **Static** charges — one fixed value each | `grid_capacity_charge,EUR/MW.a,19629` |
| `forecasts_df.csv` | **Dynamic** charges — one value per 15-min time slot | `grid_energy_charge` column |

Only two units are accepted in `additional_charges.csv`: **`EUR/MWh`** (per-energy)
and **`EUR/MW.a`** (per-peak-power, per year).

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

**§19(2) atypical grid use:** if `capacity_peak_basis = high_load_window`, the
capacity charge is billed on the peak **within the high-load windows** — so
avoiding consumption in those windows lowers the bill.

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

Each regulation's constructor **is** the list of supported static charges — the
component names above are exactly the constructor parameters. The loader maps
`additional_charges.csv` rows to those parameters and **rejects an unknown or
misspelled name** with a clear error, e.g.:

```
GridFeeConfigError: Unknown grid-fee charge 'grid_capcity_charge_high' for
GermanGridFeeRegulation. Supported charges: chp_surcharge, concession_fee, ...
```

To add a genuinely new charge for a country, add a named parameter to that
regulation's `__init__` (with its unit in a comment) and a matching row in the
case's `additional_charges.csv`. This keeps the data, the code, and this table in
step.
