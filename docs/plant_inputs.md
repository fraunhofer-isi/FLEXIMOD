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

The generic `DataLoader` checks the CSV shape, plant grouping, datetime grid,
and requested forecast columns. Each plant class then checks its own topology
and component values when it is constructed with `from_rows()`. Calling
`build_model()` or `solve_horizon()` checks the plant's forecast contract before
creating Pyomo components. Errors therefore identify the plant, technology, or
missing `forecasts_df.csv` column that must be corrected.

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

The existing market-stage methods take their price and position signals through
`DispatchSignals`, `IDCAdjustmentSignals`, or `AFRRDownSignals`. For the
day-ahead stage, forecasts normally include the configured electricity-price
column, `natural_gas_price`, and the heat-demand column. `co2_price` is only
needed when it is selected in the stage signals.

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

## Building / bus depot

Use `unit_type=building`, `bus_depot`, or `electric_bus_depot`, with one
`electric_vehicle` row and one `charging_station` row. See
[Building model inputs](building.md) for its full vehicle, charger, trip, and
availability profile contract.

## Use the generic loader and a physical plant model

The generic loader intentionally does not guess a plant's physical topology.
It returns plant/component definitions in an ASSUME-style grouped form; the
plant model remains the authority for technology-specific validation.

```python
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.plants.steel_plant import SteelPlant

loader = DataLoader(config, input_dir=case_dir)
plants = loader.load_plants()
steel_rows = plants.loc[plants["name"] == "steel_1"]
plant = SteelPlant.from_rows("steel_1", steel_rows)

forecasts = loader.load_forecasts()
plant.validate_inputs(forecasts, electricity_price_column="DE_DA_price")
model = plant.build_model(config, forecasts, electricity_price_column="DE_DA_price")
```

Steel and cement currently expose their physical `build_model()` and
`solve_horizon()` APIs directly. They are not yet registered in the sequential
market runner; that integration requires a sector-specific market strategy and
is intentionally separate from the physical model and input contract.
