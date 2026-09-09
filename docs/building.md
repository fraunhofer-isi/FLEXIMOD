<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Building Model

`Building` is the electrical boundary that connects technologies to the grid.
The current project uses it to represent an electrified bus depot, but the
class itself represents a building rather than a special bus-depot unit.

The current building contains:

- one `electric_vehicle` row for every bus;
- one `charging_station` row for every charging point;
- optional `pv_plant` rows;
- an inflexible building electricity-demand profile;
- grid import and, when V2G is enabled, grid export.

## How the model is organized

The code in `plants/building.py` follows the physical model in this order:

1. `from_rows()` reads the connected technologies from `plants.csv`.
2. `define_parameters()` adds demand, timestep, and electricity prices.
3. `initialize_components()` adds vehicles, charging stations, and optional PV.
4. `define_variables()` adds building grid import and export.
5. `define_constraints()` connects the technologies through the electricity
   balance.
6. `define_objective()` minimizes net electricity cost.

The vehicle, charging-station, and PV equations are in `plants/technologies.py`.

## Units

- Vehicle battery capacity: MWh per vehicle.
- Vehicle and charger power: MW per vehicle or charging point.
- Building demand: MW.
- Available PV generation: MW.
- Trip distance input: km per bus and timestep.
- Mileage: MWh/km, used to convert trip distance into battery energy.
- Availability: `1` when that bus is at the building and can charge or provide
  V2G; `0` when it is away. Other values are rejected.

FLEXIMOD converts MW to MWh using `case.timestep_minutes`. It does not perform
implicit kW-to-MW or kWh-to-MWh conversions.

## Fleet, charger, and terminal SOC inputs

`terminal_soc` is the minimum bus SOC required at the final timestep of an
optimization horizon. A value of `0.8` means the optimizer must finish the
horizon with at least 80% SOC. In a rolling-horizon run, this condition is
checked at the end of every look-ahead window. If `terminal_soc` is omitted,
the window must finish with at least the SOC it had at the start of that
window.

Repeat the `electric_vehicle` row to add buses and repeat the
`charging_station` row to add chargers. Every row needs a unique
`component_name`. Each bus keeps its own capacity, availability, trip distance,
charging, discharging, and SOC. The building converts each distance to driving
energy using that bus's `mileage_mwh_per_km`. It connects the individual buses to
the available charging-station pool and limits their combined power to what
the individual charger rows can supply.

## Example `plants.csv`

The two rows have the same building name:

```csv
name,unit_type,technology,component_name,node,demand,battery_capacity_mwh,mileage_mwh_per_km,max_power_charge,max_power_discharge,min_soc,max_soc,initial_soc,terminal_soc,efficiency_charge,efficiency_discharge,power_flow_directionality,availability_column,trip_distance_column
building_1,building,electric_vehicle,bus_1,grid_node,building_1_electricity_demand,0.3,0.00125,0.15,0.15,0.0,0.9,0.8,0.8,0.95,0.95,bidirectional,bus_1_availability,bus_1_trip_distance_km
building_1,building,charging_station,charger_1,grid_node,building_1_electricity_demand,,,0.15,0.15,,,,,,,bidirectional,,
```

Use `unidirectional` on both technology rows for hardware that cannot
discharge. Use `bidirectional` for V2B/V2G-capable hardware. A case can still
operate bidirectional hardware as unidirectional by setting
`vehicle_discharge_enabled: false`. The two hardware directionality settings
must match.

If bus profile columns are omitted, their names default to:

- `<building_name>_electricity_demand`
- `<component_name>_availability`
- `<component_name>_trip_distance_km`

Charging-station availability is optional and defaults to one. Configure the
station's `availability_column` only when outages or operating restrictions
must be represented.

## Optional rooftop PV

Add a `pv_plant` row only in scenarios that include PV:

```csv
name,unit_type,technology,component_name,node,demand,max_power,generation_column
building_1,building,pv_plant,pv_1,grid_node,building_1_electricity_demand,0.25,pv_1_generation_mw
```

`max_power` is installed PV power in MW. The forecast column contains the
available PV generation in MW at every timestep and defaults to
`<component_name>_generation_mw` when `generation_column` is omitted. Forecast
values must be between zero and `max_power`.

The optimizer can use available PV for the building demand, bus charging, or
grid export. It may curtail PV when the energy cannot be used economically.
Outputs report available generation, used generation, and curtailment for each
PV row and for the whole building. When no `pv_plant` row is present, all PV
totals are zero and the building behaves as before.

## Import tariff, demand charge, and export cases

The building examples assume a connection below 12 kV. Prices are stored in THB/MWh so
they match the model's MWh energy variables. Its 15-minute forecast contains:

- a 4.3297 THB/kWh on-peak and 2.6369 THB/kWh off-peak base energy charge;
- a uniform 0.1623 THB/kWh fuel-adjustment charge;
- resulting import prices of 4492.0 and 2799.2 THB/MWh, excluding VAT;
- the corresponding VAT-inclusive values as reference columns;
- the current 2200 THB/MWh export payment and sensitivity columns in
  500 THB/MWh increments.

`tou_period` is `on_peak` from 09:00 through 21:45 on Monday to Friday. It is
`off_peak` overnight, at weekends, and on specified public holidays. The
example day is a non-holiday weekday. For longer studies, the forecast must
mark the applicable holidays as off-peak.

Each example configuration controls its physical operating boundary:

```yaml
currency: THB
vehicle_discharge_enabled: true
grid_export_limit_mw: 0.005
demand_charge_per_kw_month: 210
```

The use cases are separate runnable examples:

| Example | Vehicle discharge | Grid export | Purpose |
| --- | --- | --- | --- |
| `building_v1g_baseline` | disabled | disabled | unidirectional reference |
| `building_v2b_cost_annual` | enabled | capped at 0.005 MW | I02 least-cost V2G charging and regulated export |
| `building_v2b_no_export` | enabled | disabled | building support and peak shaving |
| `building_v2g_example` | enabled | capped at 0.005 MW | current payment and export-price sweep |

The V1G example also uses physically unidirectional bus and charger rows. The
I02 least-cost example uses bidirectional equipment and the current
2.20 THB/kWh export payment with a 5 kW export limit. Because this study has
no non-traction building load, its economic service is charging and discharging
to the grid; it is not a behind-the-meter building-support case. The maximum
grid-support and renewable-shifting examples use their own export-limit
sensitivities.

For each calendar month, the optimizer adds the incremental cost of the
highest 15-minute grid import:

```text
total cost = sum(import energy * TOU import price
                 - export energy * export payment)
             + monthly peak import in kW * 210 THB/kW-month
```

The rolling-horizon runner carries forward both each bus's implemented SOC
and the highest grid import already observed in the month. This prevents a new
window from forgetting a demand peak that has already been billed. Outputs
separate `energy_cost`, `demand_charge_cost`, and `total_cost` and state their
currency.

The export-payment sweep is an operational break-even test. The first case
with positive `total_grid_export_MWh` gives the upper end of the break-even
interval; the preceding zero-export case gives the lower end. This includes
purchase cost, charging/discharging losses, mobility constraints, and the
demand-charge interaction.

## Regional grid load

Every building example forecast includes a 2024-derived regional load profile
for grid-support analysis. `regional_grid_load_mw` is the external regional
demand and is not part of the depot electricity balance.

`grid_congestion_weight` is zero at or below the cleaned 2024 P90 load and one
at or above P99, with linear scaling between them. Dispatch and summary outputs
report regional load, congestion-weighted grid import, import during regional
high-load intervals, and bus discharge during those intervals. The source,
cleaning decisions, thresholds, and reproduction command are documented in
`docs/sources/regional_grid_load_2024.md`.

## Full-year examples

The annual examples use every 15-minute interval from 1 January through
31 December 2024 (35,136 rows). The regional load follows the cleaned 2024
chronology. The route timetable is the representative daily schedule repeated
for all 366 days; replace it when measured annual vehicle-block data becomes
available. The 2026 tariff values are modelling assumptions applied to the
2024 operating calendar and are identified by `tariff_assumption_year`.

| Example | Grid export | Optimization purpose |
| --- | --- | --- |
| `building_v1g_annual` | no | unidirectional annual reference |
| `building_v2b_cost_annual` | 5 kW | least-cost V2G charging and regulated grid export at the current feed-in price |
| `building_v2b_grid_support_annual` | configurable | maximize congestion relief with 0, 5, or 450 kW export |
| `building_v2b_pv_self_consumption_annual` | no | use measured-roof PV for depot demand and bus charging |
| `building_v2b_renewable_alignment_annual` | configurable | maximize renewable-equivalent energy shifting with 0, 5, or 450 kW export |
| `building_v2g_current_tariff_annual` | 5 kW | export at 2.20 THB/kWh |
| `building_v2g_tariff_sweep_annual` | 5 kW | export-payment break-even sweep |

The grid-support file contains a behind-the-meter case and export cases with
5 kW policy and 450 kW technical limits. Each rolling horizon first maximizes
the continuous congestion-weighted reduction in net grid load. It then removes
unnecessary battery cycling and finally minimizes import cost without reducing
the maximum service. Trip energy, availability, charger power, SOC bounds, and
terminal SOC remain hard constraints.

Annual summaries split grid import and bus discharge into three readable
states while optimization keeps the exact continuous weight:

- `normal`: weight = 0;
- `elevated`: 0 < weight < 0.8;
- `stressed`: 0.8 <= weight <= 1.

The 0.8 threshold is stored in `grid_stress_threshold`. It changes reporting
only; the optimizer does not round the weight or turn it into a binary signal.

Run the default case in each annual example with:

```powershell
python -m flexi_mod.simulation.run_case --example building_v1g_annual --no-plots
python -m flexi_mod.simulation.run_case --example building_v2b_cost_annual --no-plots
python -m flexi_mod.simulation.run_case --example building_v2b_grid_support_annual --no-plots
python -m flexi_mod.simulation.run_case --example building_v2b_pv_self_consumption_annual --no-plots
python -m flexi_mod.simulation.run_case --example building_v2b_renewable_alignment_annual --no-plots
python -m flexi_mod.simulation.run_case --example building_v2g_current_tariff_annual --no-plots
python -m flexi_mod.simulation.run_case --example building_v2g_tariff_sweep_annual --no-plots
```

Select another grid-support export limit or export-payment sensitivity with
`--study-case` and a case name from the corresponding `config.yaml`.

The PV self-consumption example uses seven digitised gross roof sections,
3,923.0 m2 gross roof area, a 65% usable fraction, 0.510 MWp DC, and a
0.425 MW AC limit. Grid export is disabled, so PV can serve building demand or
bus charging and any remaining energy is curtailed. The roof geometry,
uncertainty cases, imagery attribution, and structural-survey limitation are
documented in `docs/sources/depot_rooftop_pv_assessment.md`.

The renewable-shifting example contains the same 0, 5, and 450 kW export
limits. Each bus starts with zero renewable-tagged energy. Charging adds a
renewable-equivalent share according to the continuous availability signal;
only previously tagged stored energy can count as later renewable discharge.
The service objective uses net tagged delivery in renewable-deficit intervals,
so simultaneous bus charging is counted as rebound rather than as support.
The service objective counts net tagged delivery in renewable-deficit periods,
so charging another bus in the same interval is treated as rebound consumption,
not as renewable support. Renewable-tagged SOC is carried between rolling
horizons alongside physical SOC.
The result is a technical shifting proxy, not measured national curtailment or
renewable generation. The continuous signal, PV/wind conversion, capacity
weights, national demand connection, limitations, and reproduction steps are documented in
`docs/sources/renewable_availability_2024.md`.

## Running the building

The repository includes three complete synthetic examples in `data/input`:
`building_v1g_baseline`, `building_v2b_no_export`, and
`building_v2g_example`. Each directory has its own `config.yaml`, `plants.csv`,
and `forecasts_df.csv`. They contain three electric-bus rows, three
charging-station rows, and one day of demand, binary availability,
trip-distance, and electricity-price profiles. Every bus has a 300 kWh
battery, every charger is rated at 150 kW, and the minimum bus SOC is zero.

The example uses the registered `building_v2g` strategy:

```yaml
strategy:
  name: building_v2g
  dispatch:
    dispatch_method: pyomo
    rolling_horizon_enabled: true
    dispatch_horizon_hours: 24
    rolling_step_hours: 6
```

Run each use case through the normal FLEXIMOD command-line interface:

```powershell
python -m flexi_mod.simulation.run_case --example building_v1g_baseline --no-plots
python -m flexi_mod.simulation.run_case --example building_v2b_no_export --no-plots
python -m flexi_mod.simulation.run_case --example building_v2g_example --no-plots
```

The runner writes `dispatch_results.csv`, `market_ledger.csv`,
`storage_cost_ledger.csv`, and `summary_indicators.csv`, using building-specific
grid and bus-battery columns.

## Diesel benchmark and use-case comparison

`notebooks/building_use_case_analysis.ipynb` calculates the diesel reference
directly from the same individual bus-distance columns used by the electric
examples. It is a bus-traction-only benchmark; depot building electricity and
demand charges are out of scope. Diesel is therefore an analysis benchmark
rather than another optimization technology.

The notebook keeps fuel consumption, fuel price, tailpipe CO2 factor, grid CO2
factor, and the demand-charge rate together in one editable assumptions cell.
It reports diesel fuel use and cost, depot building electricity cost, demand
charge, operational emissions, cost per kilometre, and emissions per
kilometre. It then loads completed V1G, V2B, and V2G outputs for comparison.
The notebook includes annual KPI plots, a representative operating week, grid
import duration curves, the rooftop-PV energy balance, indicative geographic
route corridors, and the digitised depot roofs with three PV-sizing scenarios.
The D0 diesel values use documented 2024 operational assumptions: 31.89 THB/L
retail diesel, 0.3922 L/km diesel-bus consumption, B7 fossil Scope 1 emissions
of 2.5504 kgCO2e/L, and no separate vehicle carbon-tax charge. The B7
biogenic CO2 component is reported separately. See
`docs/sources/diesel_reference_2024.md` for sources and scope.

The annual examples use a 48-hour look-ahead horizon, 15-minute timesteps, and
implement 24 hours after each optimization. They contain one independently
scheduled bus and one charging station for each of three services:

- `bus_route_7k`: route 7ก, six representative round trips and 240 km/day;
- `bus_route_79`: route 79, eight representative round trips and 280 km/day;
- `bus_route_101`: route 101, six representative round trips and 288 km/day.

Each route has its own binary availability and distance columns. A value of
`0` covers the complete period away from the building; the `1` intervals
between round trips are explicit depot breaks in which charging or V2G is
possible. The distances are transparent representative assumptions selected
for this test, not measured vehicle-block data. The service timing inputs are
based on the published timetables for [route 7ก](https://moovitapp.com/index/th/%E0%B8%A3%E0%B8%B0%E0%B8%9A%E0%B8%9A%E0%B8%82%E0%B8%99%E0%B8%AA%E0%B9%88%E0%B8%87%E0%B8%AA%E0%B8%B2%E0%B8%98%E0%B8%B2%E0%B8%A3%E0%B8%93%E0%B8%B0-time-7%E0%B8%81-Bangkok-2401-1363476-4195059-3773465-0),
[route 79](https://moovitapp.com/index/en/public_transit-time-79_%E0%B8%9B%E0%B8%AD_ac-Bangkok-2401-1363476-5901910-3833005-0),
and [route 101](https://moovitapp.com/index/en/public_transit-time-101-Bangkok-2401-1363476-4195066-3773632-0).

The next rolling window starts with each bus's implemented SOC and re-optimizes
against the newest horizon. `SimulationRunner` owns these windows and passes
each price horizon and the individual initial SOC values to `BuildingStrategy`.
The strategy then asks `Building` to solve the physical schedule.

## Main equations

The building electricity balance is:

```text
grid import + vehicle discharge + PV generation
    = building demand + vehicle charge + grid export
```

The vehicle battery balance is:

```text
SOC[t] = retained SOC[t-1]
         + charge[t] * charging efficiency
         - discharge[t] / discharging efficiency
         - trip energy[t]
```

where `trip energy[t] = trip distance in km[t] * mileage_mwh_per_km`.

Vehicles exchange power only while available. Vehicle and charging-station
power limits are both enforced. Charging and discharging cannot happen at the
same time. Minimum, maximum, initial, and terminal SOC remain hard constraints,
so an impossible operating schedule is reported as infeasible.
