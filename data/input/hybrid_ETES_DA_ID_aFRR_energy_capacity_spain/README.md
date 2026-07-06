<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Spanish Pay-As-Cleared aFRR Capacity Example

This is a deliberately small, synthetic two-hour example for inspecting the
Spanish-style pay-as-cleared aFRR-capacity workflow. The values are illustrative
and are not historical Spanish market observations.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe .\src\flexi_mod\simulation\run_case.py `
  --example hybrid_ETES_DA_ID_aFRR_energy_capacity_spain `
  --no-plots
```

The important configuration choices are:

```yaml
strategy:
  name: hybrid_etes_gas

markets:
  afrr_capacity:
    clearing_mechanism: pay_as_cleared
    product_length: 15min
    price_unit: EUR_per_MW_per_product
```

`ES_aFRR_capacity_down_marginal_price` is therefore interpreted as an exogenous
marginal clearing price in `EUR/MW` for each 15-minute product. FLEXIMOD
normalises it internally to `EUR/MW/h`: a raw value of `25 EUR/MW` becomes
`100 EUR/MW/h`, and settlement multiplies by `0.25 h`, recovering `25 EUR/MW`.

Inspect these output columns in `market_ledger.csv`:

```text
afrr_capacity_pricing_rule
afrr_capacity_bid_price_EUR_per_MW_h
afrr_capacity_clearing_price_EUR_per_MW_h
afrr_capacity_settlement_price_EUR_per_MW_h
afrr_capacity_revenue_EUR
afrr_capacity_opportunity_cost_EUR
afrr_capacity_market_surplus_EUR
afrr_capacity_net_value_EUR
```
