<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Building Model

`Building` is the electrical boundary that connects technologies to the grid.
The current project uses it to represent an electrified bus depot, but the
class itself represents a building rather than a special bus-depot unit.

The current building contains:

- one `electric_vehicle` row representing one vehicle or an aggregated fleet;
- one `charging_station` row representing one charger or a group of identical
  charging points;
- an inflexible building electricity-demand profile;
- grid import and, when V2G is enabled, grid export.

## How the model is organized

The code in `plants/building.py` follows the physical model in this order:

1. `from_rows()` reads the connected technologies from `plants.csv`.
2. `define_parameters()` adds demand, timestep, and electricity prices.
3. `initialize_components()` adds vehicles and charging stations.
4. `define_variables()` adds building grid import and export.
5. `define_constraints()` connects the technologies through the electricity
   balance.
6. `define_objective()` minimizes electricity and battery-cycling costs.

The vehicle and charging-station equations are in `plants/technologies.py`.

## Units

- Vehicle battery capacity: MWh per vehicle.
- Vehicle and charger power: MW per vehicle or charging point.
- Building demand: MW.
- Trip energy: aggregate fleet MWh per timestep.
- Availability: connected share between 0 and 1.
- Trip distance, when used: aggregate fleet-km per timestep.
- Mileage: MWh/km.

FLEXIMOD converts MW to MWh using `case.timestep_minutes`. It does not perform
implicit kW-to-MW or kWh-to-MWh conversions.

## Example `plants.csv`

The two rows have the same building name:

```csv
name,unit_type,technology,node,demand,battery_capacity_mwh,fleet_size,charger_count,max_power_charge,max_power_discharge,min_soc,max_soc,initial_soc,terminal_soc,efficiency_charge,efficiency_discharge,power_flow_directionality,availability_column,trip_energy_column,degradation_cost_eur_per_mwh
building_1,building,electric_vehicle,grid_node,building_1_electricity_demand,0.35,100,,0.15,0.15,0.2,1.0,0.8,0.8,0.95,0.95,bidirectional,building_1_vehicle_availability,building_1_vehicle_trip_energy,15
building_1,building,charging_station,grid_node,building_1_electricity_demand,,,60,0.15,0.15,,,,,,,bidirectional,,,
```

Use `unidirectional` on both technology rows when vehicles cannot provide V2G.
Use `bidirectional` on both rows when grid export is allowed. The two
directionality settings must match.

If profile columns are omitted, their names default to:

- `<building_name>_electricity_demand`
- `<building_name>_vehicle_availability`
- `<building_name>_vehicle_trip_energy`

Charging-station availability is optional and defaults to one. Configure the
station's `availability_column` only when outages or operating restrictions
must be represented.

## Running the building

The repository includes a complete synthetic example in
`data/input/building_v2g_example`. It contains the case configuration, the two
building technology rows, and one day of demand, availability, trip-energy,
and electricity-price profiles.

The example uses the registered `building_v2g` strategy:

```yaml
strategy:
  name: building_v2g
  dispatch:
    dispatch_method: pyomo
    rolling_horizon_enabled: true
    dispatch_horizon_hours: 12
    rolling_step_hours: 6
```

Run it through the normal FLEXIMOD command-line interface:

```powershell
python -m flexi_mod.simulation.run_case --example building_v2g_example --no-plots
```

The runner writes `dispatch_results.csv`, `market_ledger.csv`,
`storage_cost_ledger.csv`, and `summary_indicators.csv`, using building-specific
grid and bus-battery columns.

The example uses a 12-hour look-ahead horizon and implements six
hours after each optimization. The next window starts with the implemented bus
SOC and re-optimizes against the newest horizon. `SimulationRunner` owns these
windows and passes each price horizon and initial SOC to `BuildingStrategy`.
The strategy then asks `Building` to solve the physical schedule.

## Main equations

The building electricity balance is:

```text
grid import + vehicle discharge
    = building demand + vehicle charge + grid export
```

The vehicle battery balance is:

```text
SOC[t] = retained SOC[t-1]
         + charge[t] * charging efficiency
         - discharge[t] / discharging efficiency
         - trip energy[t]
```

Vehicles exchange power only while available. Vehicle and charging-station
power limits are both enforced. Charging and discharging cannot happen at the
same time. Minimum, maximum, initial, and terminal SOC remain hard constraints,
so an impossible operating schedule is reported as infeasible.
