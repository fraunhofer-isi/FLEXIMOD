<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Plant Input Reference

`plants.csv` is deliberately modeller-facing: each row represents one
technology and all rows with the same `name` belong to one physical plant.
`unit_type` selects the plant topology and `technology` selects a component
model. Keep shared plant fields (`name`, `unit_type`, `node`, `objective`, and
the demand-column name) identical on every row belonging to that plant.

The generic `DataLoader` checks the CSV shape, prepares each plant's parameters
and components, and checks the datetime grid and requested forecast columns.
Each plant class then checks its own topology and component values when the
plant factory creates it. Calling
`build_model()` or `solve_horizon()` checks the plant's forecast contract before
creating Pyomo components. Errors therefore identify the plant, technology, or
missing `forecasts_df.csv` column that must be corrected.

For an explicit pre-flight check, call `validate_inputs()` on the physical
plant: building, steel, and cement accept their forecasts plus price-column
arguments; steam also accepts its market stage and signal object. `build_model()`
and `solve_horizon()` invoke the same validation automatically.

## Shared conventions

| Item | Convention |
| --- | --- |
| Power and heat capacities | MW |
| Electricity, fuel, and heat inside a timestep | MWh |
| Steel, DRI, clinker, and raw meal profiles | tonnes per timestep |
| Energy and commodity prices | EUR/MWh or EUR/t, consistent with the associated specific consumption |
| CO2 price and factors | EUR/tCO2 and tCO2/MWh |
| Demand profile | A plant-specific `<plant>_<signal>` column is preferred; a global `<signal>` column is allowed where the plant model supports it. |

All required numeric plant parameters and forecast values must be finite. The
steel and cement component contracts below additionally check power limits,
efficiencies, and fractions for physically valid ranges.

## Steam plant

Use `unit_type=steam_plant`. The supported routes are:

- `thermal_storage` + `boiler` (ETES backed by gas); or
- `electric_boiler` + `boiler` (direct electric heat backed by gas).

Both routes require exactly one `boiler` row and exactly one electric-heat
route. The conventional heat-demand column is `<plant>_heat_demand`; it is an
average MW_th profile and is converted to MWh_th internally using
`timestep_minutes`.

Minimal ETES route:

```csv
name,unit_type,technology,demand,max_capacity,min_capacity,max_power_charge,max_power_discharge,initial_soc,efficiency_charge,efficiency_discharge,storage_loss_rate,max_power,min_power,efficiency,fuel_type
steam_1,steam_plant,thermal_storage,steam_1_heat_demand,80,0,16,16,20,0.95,1.0,0.00025,,,,
steam_1,steam_plant,boiler,steam_1_heat_demand,,,,,,,,,20,0,0.9,natural_gas
```

Steam uses product-specific market inputs: `DayAheadPosition`,
`IntradayAdjustment`, and `BalancingEnergyActivation`. A strategy wraps the
selected product in an `ElectricityMarketRequest`; the shared electricity
settlement model adds its market position, while the steam plant checks heat
and technology feasibility. An awarded aFRR-capacity product is represented
separately by `BalancingCapacityAward`.

For the day-ahead stage, forecasts normally include the configured
electricity-price column, `natural_gas_price`, and the heat-demand column.
`co2_price` is only needed when the selected steam operating inputs use it.

An empty intraday-price value is the existing explicit exception: it is handled
as a no-action interval and the steam strategy emits a data-quality warning.

## Steel plant

Use `unit_type=steel_plant`. A steel plant requires one `dri_plant` row and one
`eaf` row; an `electrolyser` row is optional. DRI accepts
`fuel_type=hydrogen`, `natural_gas`, or `both`.

```csv
name,unit_type,technology,steel_demand,fuel_type,max_power,specific_electricity_consumption,specific_hydrogen_consumption,specific_natural_gas_consumption,specific_iron_ore_consumption,natural_gas_co2_factor,specific_dri_demand,specific_lime_demand
steel_1,steel_plant,dri_plant,steel_1_steel_demand,hydrogen,10,0.20,2.0,0.0,1.4,0.0,,
steel_1,steel_plant,eaf,steel_1_steel_demand,,20,0.40,,,,,1.0,0.05
```

For an optional electrolyser row, provide `max_power`, optional `min_power`,
and `efficiency`. Its hydrogen output is connected directly to the DRI plant.

The steel physical model requires:

```text
steel_1_steel_demand  (or the column named by steel_demand/demand)
electricity price column passed to build_model()/solve_horizon()
iron_ore_price
lime_price
co2_price
natural_gas_price  (for natural_gas or both DRI)
hydrogen_price     (for hydrogen or both DRI without an electrolyser)
```

## Cement plant

Use `unit_type=cement_plant`. A cement kiln line requires exactly one each of
`preheater`, `calciner`, and `kiln`. `simple_calciner` and `simple_kiln` are
accepted aliases for existing input files. A thermal stage accepts
`fuel_type=electricity`, `fossil`, `both`, or `hydrogen`.

```csv
name,unit_type,technology,clinker_demand,fuel_type,max_heat_out,specific_heat_demand,specific_electricity_aux,eta_electric,eta_fossil,fossil_ng_share,ng_co2_factor,coal_co2_factor,calcination_emission_factor,raw_meal_to_clinker_ratio
cement_1,cement_plant,preheater,cement_1_clinker_demand,fossil,10,1.0,0.02,0.95,0.90,1.0,0.20,0.30,,1.55
cement_1,cement_plant,calciner,cement_1_clinker_demand,fossil,10,1.0,0.02,0.95,0.90,1.0,0.20,0.30,0.525,1.55
cement_1,cement_plant,kiln,cement_1_clinker_demand,fossil,10,1.0,0.02,0.95,0.90,1.0,0.20,0.30,,1.55
```

The cement physical model requires the clinker-demand column, the electricity
price column passed to `build_model()` or `solve_horizon()`, and
`natural_gas_price`, `coal_price`, `hydrogen_price`, and `co2_price`. Price
columns are required even if the selected fuel route sets a carrier flow to
zero, which keeps the shared technology-cost interface consistent.

## Running cement and steel cases

Cement and steel use the same file-based workflow as the steam plant:
`DataLoader` reads and validates `plants.csv` and `forecasts_df.csv`, the plant
factory creates the physical model from `unit_type`, the strategy selects the
market instruction, and the plant solves its own Pyomo formulation.

For the current first industrial market route, configure:

```yaml
strategy:
  name: industrial_day_ahead_cost_minimisation
  dispatch:
    dispatch_method: pyomo
market_sequence:
  - day_ahead
markets:
  day_ahead:
    enabled: true
    signals:
      price: DE_DA_price
```

This strategy is intentionally day-ahead only. It records each feasible
plant's electricity use as its day-ahead procurement position and writes the
same dispatch, market-ledger, storage-ledger, and summary files as other
runner cases. Cement and steel do not have a thermal-storage ledger; their dashboard is built from
the dispatch table, and their storage ledger is an empty, schema-consistent
table.

## Building / bus depot

Use `unit_type=building`, `bus_depot`, or `electric_bus_depot`, with one
`electric_vehicle` row and one `charging_station` row. See
[Building model inputs](building.md) for its full vehicle, charger, trip, and
availability profile contract.

## Use the generic loader and plant factory

The generic loader intentionally does not guess a plant's physical topology.
It prepares plant parameters and components in an ASSUME-style grouped form;
the plant model remains the authority for technology-specific validation.

```python
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.plants.factory import build_plants

loader = DataLoader(config, input_dir=case_dir)
plant_inputs = loader.load_plant_inputs()
plant = build_plants(plant_inputs)[0]

forecasts = loader.load_case_inputs(plant_inputs=plant_inputs).forecasts
plant.validate_inputs(forecasts, electricity_price_column="DE_DA_price")
model = plant.build_model(config, forecasts, electricity_price_column="DE_DA_price")
```

`SimulationRunner` follows the same path for every supported plant family:
it loads grouped plant inputs, delegates construction to `build_plants`, then
loads one validated forecast frame through `load_case_inputs`.
