<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Cement plant model

This document is the engineering and modelling reference for FLEXIMOD's cement
clinker plant. It describes the implemented model, not a complete process-design
model of a cement works. The model dispatches a kiln line against time-varying
commodity and electricity prices while meeting a per-timestep clinker demand.

The implementation lives in:

- `src/flexi_mod/plants/technologies.py` — equipment blocks and their equations;
- `src/flexi_mod/plants/cement_plant.py` — route assembly, inter-stage balances,
  prices, optimisation and reporting; and
- `src/flexi_mod/data/data_loader.py` — conditional forecast-column discovery.

Grinding, cement blending, clinker silos, raw-material purchasing and transport/
permanent storage of captured CO2 are outside the present boundary.

## Scope, units and time indexing

Every technology is a Pyomo block indexed by dispatch timestep `t`.

| Quantity | Unit | Meaning |
| --- | --- | --- |
| Heat, fuel, electricity and hydrogen flows | MWh per timestep | Input/output energy during the timestep, not average power. |
| Heat and electrical ratings | MW | Converted to MWh per timestep using the configured timestep duration. |
| Clinker and raw meal | t per timestep | Material throughput. |
| CO2 | tCO2 per timestep | Physical or accounting emissions. |
| Prices | EUR/MWh or EUR/tCO2 | The unit matching the commodity. |

The objective is variable-cost minimisation. Each timestep must meet its clinker
demand; production is not an annual or cumulative target. Consequently, flexibility
comes from fuel switching, thermal storage and electricity-consuming equipment, not
from deliberately moving clinker production between timesteps.

## Plant boundary and route assembly

A `cement_plant` in `plants.csv` is defined by rows sharing the same `name`. At
most one technology may occupy each physical role.

```text
raw meal
  │
  ├─ preheater ───────────────────────────────────────────┐
  │                                                        │
  ├─ calciner (simple, LEILAC or oxyfuel) ── clinker ─ kiln ── plant clinker output
  │                                                        │
  └─ optional thermal storage discharges heat to calciner ┘

optional electrolyser ── hydrogen to hydrogen-fired stages
                       └─ oxygen coproduct to oxyfuel stages

eligible flue-gas CO2 ── optional CCS block
```

Supported kiln-line technologies are:

| Role | Technology key | Python class |
| --- | --- | --- |
| Preheater | `preheater` | `CementPreheater` |
| Conventional calciner | `simple_calciner` | `SimpleCementCalciner` |
| Indirect-separation calciner | `leilac_calciner` | `LEILACCementCalciner` |
| Oxyfuel calciner | `oxyfuel_calciner` | `OxyfuelCementCalciner` |
| Conventional kiln | `simple_kiln` | `SimpleCementKiln` |
| Oxyfuel kiln | `oxyfuel_kiln` | `OxyfuelCementKiln` |
| Post-combustion CCS | `amine_ccs` | `AmineCCS` |
| Cryogenic CCS | `cryogenic_ccs` | `CryogenicCCS` |
| Oxyfuel CO2 recovery | `oxyfuel_ccs` | `OxyfuelCCS` |
| Thermal store | `thermal_storage` | `ThermalStorage` |
| Electrolyser | `electrolyser` | `Electrolyser` |
| Hydrogen store | `hydrogen_buffer_storage` | `HydrogenBufferStorage` |

Only one calciner variant, one kiln variant and one CCS variant may be configured.
`cement_mill` and `grinding_mill` are deliberately rejected. A thermal store requires
a calciner; a hydrogen buffer requires an electrolyser.

The resolved `cement_route` is the configured line-stage keys in flow order, for
example `preheater_simple_calciner_simple_kiln`. Calciner reporting always uses the
`simple_calciner_*` result prefix even when a LEILAC or oxyfuel calciner fills that
role. This keeps result schemas stable across variants.

### Inter-stage material and heat links

The terminal stage is the plant output: the kiln if present, otherwise the calciner.

- A preheater produces raw meal. It feeds the calciner where one exists; otherwise
  it feeds the kiln directly.
- A calciner and kiln, when both are present, produce the same clinker throughput in
  each timestep.
- Thermal storage adds discharged heat to the calciner's `effective_heat_in`. It does
  not directly make clinker.
- Waste heat from the kiln can feed the preheater through
  `waste_heat_per_t_clinker_mwh` and `waste_heat_utilization_efficiency`.
- An electrolyser's hydrogen output must cover all hydrogen-fired stage demand when
  installed. Without an electrolyser, hydrogen is externally purchased.

## Common kiln-line stage model

Preheater, calciner and kiln variants use the same commitment, fuel and cost
framework. Their process-specific output and CO2 equations differ.

### Heat balance and fuel modes

For a stage `s`:

```text
heat_out = generated_heat + external_heat
```

The allowed `fuel_type` values are:

| `fuel_type` | Generated heat |
| --- | --- |
| `electricity` | `power_in × eta_electric` |
| `fossil` | `combustion_in × eta_fossil` |
| `hybrid_electricity_fossil` | `power_in × eta_electric + combustion_in × eta_fossil` |
| `hydrogen` | `hydrogen_in × eta_fossil` |

Fuel modes are exclusive except the explicitly hybrid mode. For example, a fossil
stage has zero primary electrical and hydrogen input, while a hydrogen stage has zero
electric, natural-gas, coal, biomass, RDF and fossil-combustion input. Auxiliary
electricity is separate and remains possible for all modes.

### Capacity, commitment and ramping

Each stage has an on/off binary `operational_status`:

```text
min_heat_out × operational_status <= heat_out
heat_out <= max_heat_out × operational_status
```

It also has heat ramp-up/down constraints, optional minimum operating and down times,
and start-up/shut-down variables. The rolling horizon carries the committed final
status, consecutive status duration and heat output into the next window.

Auxiliary electricity is proportional to stage throughput:

```text
aux_power_in = output × specific_electricity_aux
```

### Conventional fuel, separately procured biomass and mixed RDF

For fossil or hybrid-fired stages, total combustion energy is split as:

```text
combustion_in = biomass_in + rdf_in + fossil_in
fossil_in = natural_gas_in + coal_in

biomass_in = biomass_share × combustion_in
rdf_in     = rdf_share × combustion_in

natural_gas_in = fossil_ng_share × fossil_in
coal_in        = (1 - fossil_ng_share) × fossil_in
```

`biomass_in` represents separately procured biomass. `rdf_in` represents total
mixed RDF energy. Its internal biogenic material is an emissions characteristic, not
an additional energy share. Therefore, do not use an `rdf_excl_biomass_share` column;
the supported input is `rdf_share`.

#### Default fuel shares

The defaults are technology-class attributes on `CementKilnLineStage`.

| Default profile | Coal | Natural gas | Separate biomass | Mixed RDF |
| --- | ---: | ---: | ---: | ---: |
| Conventional fossil profile | 96.6% | 3.4% | 0% | 0% |
| R2 alternative-fuel profile | 25.7% | 0.9% | 24.5% | 48.9% |

The conventional profile applies when share columns are absent. R2 is selected by
`cement_route_id`, `route_id` or `route` set to `R2` (or `2`) on a fossil or hybrid
stage. Explicit CSV values for `fossil_ng_share`, `biomass_share` and `rdf_share`
override the corresponding default individually. R1, R3–R7, R9–R10 and R13–R16
therefore use the conventional split unless the CSV overrides it.

The R2 fossil natural-gas share is defined inside the fossil remainder:

```text
fossil_ng_share = 0.009 / (0.009 + 0.257)
```

It produces 0.9% natural gas and 25.7% coal of total combustion energy.

## Stage technologies

### Preheater

The preheater converts its own heat output to raw-meal throughput:

```text
raw_meal_out = heat_out / specific_heat_demand
co2_process = 0
```

It may use any common fuel mode, although a route normally uses it as a heat-recovery
or electrically heated stage. `external_heat_in` is the kiln-waste-heat contribution.

### Simple calciner

The simple calciner converts effective heat to clinker and creates calcination CO2:

```text
clinker_out = effective_heat_in / specific_heat_demand
co2_process = clinker_out × calcination_emission_factor
```

Without thermal storage, `effective_heat_in = heat_out`. With storage, it equals the
sum of stage heat and storage discharge. The model assigns calcination-process CO2 to
the calciner, not the kiln.

### Simple kiln

The simple kiln converts heat directly to clinker:

```text
clinker_out = heat_out / specific_heat_demand
co2_process = 0
```

This is intentional in the current route architecture: when a calciner exists it owns
the process CO2. A standalone kiln remains a valid operational route, but the present
model does not add a separate kiln calcination-emission factor.

### LEILAC calciner

LEILAC inherits the simple-calciner heat, fuel, throughput and combustion equations.
It directly separates a fraction of calcination CO2:

```text
co2_separated        = direct_separation_efficiency × co2_process
co2_process_residual = co2_process - co2_separated
```

Only `co2_process_residual` enters the stage physical and priced flue-gas balance.
Combustion CO2 remains in the flue gas. The separated stream has no storage,
compression or transport model in the present boundary.

### Oxyfuel calciner and kiln

Oxyfuel stages inherit the relevant simple-stage equations and add combustion oxygen
demand:

```text
oxygen_demand = natural_gas_in × natural_gas_oxygen_demand
              + coal_in        × coal_oxygen_demand
              + biomass_in     × biomass_oxygen_demand
              + rdf_in         × rdf_oxygen_demand
              + hydrogen_in    × hydrogen_oxygen_demand

oxygen_generated = oxygen_demand - oxygen_from_electrolyser
electricity_consumption = oxygen_generated × specific_oxygen_electricity_consumption
```

`oxygen_from_electrolyser` is limited by demand and, at plant level, by actual
electrolyser oxygen output. Remaining oxygen is generated internally and its
electricity is included in stage operating cost. The oxyfuel CCS block does **not**
recalculate oxygen or air-separation electricity.

## CO2 model and carbon accounting

The model intentionally keeps physical flue-gas CO2 separate from CO2 that is charged
under the selected accounting convention.

### Fuel CO2 components

For each non-LEILAC stage:

```text
co2_fossil = natural_gas_in × natural_gas_co2_factor
           + coal_in        × coal_co2_factor
           + rdf_in         × rdf_mixed_fossil_co2_factor

co2_biomass     = biomass_in × biomass_co2_factor
co2_rdf_fossil  = rdf_in × rdf_mixed_fossil_co2_factor
co2_rdf_biogenic = rdf_in × rdf_mixed_biogenic_co2_factor

co2_biogenic = co2_biomass + co2_rdf_biogenic
co2_energy   = co2_fossil + co2_biogenic
co2_physical = co2_process + co2_energy
```

For LEILAC, replace `co2_process` in the final equation with
`co2_process_residual`. `co2_emission` is an alias for `co2_physical` and is the
stream used by plant-level physical CO2 connections.

#### Mixed RDF factors

`rdf_mixed_fossil_co2_factor` is defined per MWh of **total mixed RDF energy**. Its
technology default is:

```text
rdf_mixed_fossil_co2_factor = 0.243 tCO2/MWh_th
```

Therefore:

```text
co2_rdf_fossil = rdf_in × 0.243
```

It is not multiplied again by a fossil or biogenic fraction. Doing so would undercount
fossil RDF CO2. Physical biogenic RDF CO2 is represented separately by the required
`rdf_mixed_biogenic_co2_factor` whenever `rdf_share > 0`.

If an RDF specification provides fossil and biogenic CO2/carbon fractions, convert
them to explicit factors before passing them to the model. For example, if 35.6% is
biogenic and 64.4% fossil on a CO2 or carbon basis, then:

```text
rdf_mixed_biogenic_co2_factor = 0.243 × (0.356 / 0.644)
                                  = 0.13433 tCO2/MWh_th
```

This conversion is not valid for an energy-basis or wet-mass fraction.

### Priced CO2

Carbon-price exposure is:

```text
co2_priced = co2_process
           + co2_fossil
           + biomass_co2_accounting_share × co2_biomass
           + rdf_biogenic_co2_accounting_share × co2_rdf_biogenic
```

Again, LEILAC uses `co2_process_residual`. Accounting shares lie between zero and
one. A zero share models zero-rated qualifying biomass/RDF biogenic CO2; a non-zero
share allows a scenario to represent another regulatory treatment. The physical factor
must remain positive even when its accounting share is zero so that the real flue-gas
mass remains available to CCS.

Stage operating cost includes fuel/electricity expenditure plus:

```text
co2_priced × co2_price
```

## CCS technologies and accounting

CCS has no CO2 inventory, transport, storage, compressor-train dynamics or
refrigeration-cycle model. Capture represents an eligible downstream route outside
the plant boundary; applying a carbon-price credit assumes that captured priced CO2
is eligible for that credit.

All CCS blocks receive two inlet balances:

```text
co2_in        = physical eligible flue-gas CO2
co2_priced_in = priced eligible CO2
co2_unpriced_in = co2_in - co2_priced_in
```

They retain a physical mass balance and split capture/residuals by accounting status:

```text
co2_captured = co2_priced_captured + co2_unpriced_captured
co2_residual = co2_priced_residual + co2_unpriced_residual
```

The configured capture/recovery minimum and maximum fractions apply independently to
the priced and unpriced streams, while `max_capture_rate` limits total physical capture.
This is a linear accounting formulation. If the configured minimum equals the maximum
efficiency/recovery fraction, each component is captured at exactly that fraction. If
capture is allowed to vary between the bounds, the cost optimisation may preferentially
allocate the optional capture to priced CO2; this is the intentional linear
approximation rather than a molecular-species separation model.

Only priced captured CO2 receives a carbon-price credit:

```text
CCS operating cost = electricity cost + heat cost where applicable
                   + captured CO2 × specific variable cost
                   - co2_priced_captured × co2_price
```

This prevents zero-rated biogenic CO2 from incorrectly earning a full ETS credit.

### Amine CCS

`amine_ccs` is post-combustion capture. It applies `capture_efficiency`,
`minimum_capture_fraction`, total `max_capture_rate`, electrical SEC, thermal SEC,
variable capture cost and an explicit `heat_cost`.

It receives physical and priced CO2 from all configured preheater, calciner and kiln
blocks.

### Cryogenic CCS

`cryogenic_ccs` is the same high-level capture boundary but without regeneration heat.
It uses `capture_efficiency`, `minimum_capture_fraction`, total capture capacity,
electrical SEC and variable processing cost.

It also receives all configured preheater, calciner and kiln emissions.

### Oxyfuel CCS

`oxyfuel_ccs` is a CO2 recovery, purification and compression block. It uses
`recovery_efficiency`, `minimum_recovery_fraction`, total recovery capacity, electrical
SEC and variable processing cost. It includes no oxygen, ASU, heat or storage model.

It receives only the configured oxyfuel calciner and/or oxyfuel kiln stream. In a
partial oxyfuel route, conventional-stage emissions remain outside this CPU and remain
uncaptured by it.

## Optional support technologies

### Thermal storage

`thermal_storage` stores heat in MWh with charge/discharge power limits, a capacity,
charge/discharge efficiencies, losses and an initial state of charge. Its discharge is
added to calciner effective heat. Its charge electricity is included in total plant
electricity and the objective. Storage requires a calciner.

### Electrolyser and hydrogen buffer

An electrolyser consumes electricity and produces hydrogen and oxygen. Hydrogen can
serve hydrogen-fired stages directly or through `hydrogen_buffer_storage`. Oxyfuel
stages may use its oxygen coproduct; any oxygen not supplied this way is generated by
the oxyfuel stage at its configured electricity intensity.

## Required CSV inputs

`plants.csv` always needs the normal plant identity columns such as `name`,
`unit_type`, `technology`, `node` and `objective`, plus the technology-specific fields.
The following is a concise operational reference for kiln-line rows.

| Field | Applies to | Unit / interpretation |
| --- | --- | --- |
| `max_heat_out` | all stages | MW_th heat rating; `max_power` is accepted as an input alias. |
| `min_heat_out` | all stages | MW_th turndown floor while on. |
| `specific_heat_demand` | all stages | MWh_th/t output. |
| `fuel_type` | all stages | One of the four fuel modes above. |
| `eta_electric`, `eta_fossil` | relevant modes | Heat efficiency. |
| `fossil_ng_share` | fossil/hybrid | NG share within fossil remainder. |
| `biomass_share`, `rdf_share` | fossil/hybrid | Shares of total combustion energy. Their sum cannot exceed one. |
| `natural_gas_co2_factor`, `coal_co2_factor` | fossil/hybrid | tCO2/MWh_th fuel. |
| `biomass_co2_factor` | biomass share > 0 | Physical tCO2/MWh_th; must be positive. |
| `biomass_co2_accounting_share` | biomass | Priced share from 0 to 1. |
| `rdf_mixed_fossil_co2_factor` | RDF | Fossil tCO2/MWh_th total mixed RDF; default 0.243. |
| `rdf_mixed_biogenic_co2_factor` | RDF share > 0 | Physical biogenic tCO2/MWh_th total mixed RDF; required. |
| `rdf_biogenic_co2_accounting_share` | RDF | Priced biogenic share from 0 to 1. |
| `specific_electricity_aux` | all stages | MWh_el/t output. |
| `ramp_up`, `ramp_down` | all stages | MW_th change per timestep. |
| `min_operating_steps`, `min_down_steps` | all stages | Commitment durations in model timesteps. |
| `calcination_emission_factor` | calciner variants | t process CO2/t clinker. |
| `direct_separation_efficiency` | LEILAC | Fraction of calcination CO2 directly separated. |
| `*_oxygen_demand` | oxyfuel stages | t oxygen/MWh of the named fuel. Required for each used combustion fuel. |
| `specific_oxygen_electricity_consumption` | oxyfuel stages | MWh_el/t oxygen internally generated. |

CCS rows use `max_capture_rate` (or `max_co2_capture_rate`), the relevant
capture/recovery efficiency, minimum fraction, `specific_electricity_consumption`,
`specific_variable_cost`, and, for amine, `specific_heat_consumption` and `heat_cost`.

The common plant-level fields are:

| Field | Meaning |
| --- | --- |
| `demand` | Name of the per-timestep clinker-demand forecast column. Defaults to `<plant_name>_clinker_demand`. |
| `raw_meal_to_clinker_ratio` | Raw-meal requirement per t clinker. |
| `waste_heat_per_t_clinker` | Kiln waste heat available per t clinker. |
| `waste_heat_utilization_efficiency` | Usable fraction of available kiln waste heat. |
| `cement_route_id`, `route_id` or `route` | Optional route identifier used for defaults, notably R2. |

### R2 example

This example makes the R2 default choice explicit and supplies the physical factors
that cannot safely be inferred from a route label:

```csv
name,unit_type,route_id,technology,fuel_type,max_heat_out,specific_heat_demand,eta_fossil,biomass_co2_factor,rdf_mixed_biogenic_co2_factor
cement_r2,cement_plant,R2,simple_calciner,fossil,100,0.45,0.90,0.40,0.13433
cement_r2,cement_plant,R2,simple_kiln,fossil,100,0.80,0.90,0.40,0.13433
```

The R2 defaults supply the energy shares and the RDF fossil factor. Replace the
illustrative biomass and RDF biogenic factors with values from the selected fuel/RDF
specification.

## Forecast inputs and conditional requirements

The clinker-demand column and these base price signals are required for price-taking
dispatch:

```text
electricity price column selected by the strategy
natural_gas_price
hydrogen_price
co2_price
```

Additional columns are required only when a configured stage needs them:

| Column | Required when |
| --- | --- |
| `coal_price` | A fossil/hybrid stage has a positive coal share. |
| `biomass_price` | A fossil/hybrid stage has `biomass_share > 0`. |
| `rdf_price` | A fossil/hybrid stage has `rdf_share > 0`. |

The aFRR-down strategy uses the same commodity columns, plus its market-specific
forecast signals. In market mode, technology-level electricity prices are set to zero
because electricity settlement is performed once by the market layer; fuel and CO2
costs remain technology-level costs.

## Operating cost and hybrid fuel substitution

For a normal price-taking solve, each stage pays its auxiliary electricity, applicable
primary electricity, fuel purchases, priced CO2 and any stage-specific oxygen cost.

For a hybrid-electricity/fossil stage, the aFRR strategy calculates an electricity
benchmark from the fixed combustion blend:

```text
combustion blend cost = biomass share × biomass cost
                       + RDF share × RDF cost
                       + fossil remainder × NG/coal blend cost
```

The RDF cost includes `rdf_price` plus fossil RDF CO2 and the configured priced share
of biogenic RDF CO2. The benchmark converts this combustion cost using electric and
fossil heat efficiencies.

## Outputs

Every dispatch result includes plant identity, `cement_route`, demand/output,
electricity consumption, natural-gas/coal/hydrogen consumption, variable cost and
physical `co2_emissions_t`. Important conditional columns include:

| Output | Meaning |
| --- | --- |
| `biomass_consumption_MWh`, `rdf_consumption_MWh` | Total use across the kiln line. |
| `co2_fossil_t`, `co2_biogenic_t`, `co2_priced_t` | Gross stage-level accounting quantities. |
| `co2_rdf_t`, `co2_rdf_fossil_t`, `co2_rdf_biogenic_t` | RDF physical breakdown. |
| `gross_co2_emissions_t` | Physical CO2 before CCS. |
| `ccs_co2_input_t`, `ccs_co2_priced_input_t` | Physical and priced CO2 sent to CCS. |
| `co2_captured_t`, `co2_residual_t` | Physical CCS balance. |
| `co2_priced_captured_t`, `co2_unpriced_captured_t` | CCS accounting split. |
| `ccs_co2_priced_residual_t`, `co2_priced_emissions_t` | Remaining charged CO2 after eligible CCS. |
| `ccs_electricity_consumption_MWh`, `ccs_heat_consumption_MWh` | Capture energy use; heat applies only to amine CCS. |
| `co2_separated_t` | LEILAC directly separated process CO2. |
| `*_oxygen_*` | Oxyfuel oxygen demand, electrolyser contribution, internal generation and generation electricity. |

`co2_emissions_t` is physical residual CO2 after CCS. It is not the ETS-priced
emissions column; use `co2_priced_emissions_t` when a CCS block is configured and
carbon-accounting residuals are required.

## Supported research-route mapping

The following configurations express the current route catalogue. “Fossil” uses the
conventional default split unless overridden; R2 selects its alternative-fuel defaults.

| ID | Configuration |
| --- | --- |
| R1 | `preheater -> simple_calciner(fossil) -> simple_kiln(fossil)` |
| R2 | `preheater -> simple_calciner(fossil, route_id=R2) -> simple_kiln(fossil, route_id=R2)` |
| R3 | R1 + `amine_ccs` |
| R4 | R1 + `cryogenic_ccs` |
| R5 | `preheater -> simple_calciner(electricity) -> simple_kiln(fossil)` |
| R6 | `preheater -> simple_calciner(hybrid_electricity_fossil) -> simple_kiln(fossil)` |
| R7 | R5 + `amine_ccs` |
| R8 | `preheater -> simple_calciner(electricity) -> simple_kiln(electricity) -> cryogenic_ccs` |
| R9 | `preheater -> oxyfuel_calciner(fossil) -> oxyfuel_kiln(fossil) -> oxyfuel_ccs` |
| R10 | `preheater -> oxyfuel_calciner(fossil) -> simple_kiln(fossil) -> oxyfuel_ccs` |
| R11a | `preheater -> simple_calciner(hydrogen) -> simple_kiln(hydrogen) -> amine_ccs` with purchased hydrogen |
| R11b | R11a + `electrolyser` (and optional hydrogen buffer) |
| R12a | `preheater -> oxyfuel_calciner(hydrogen) -> oxyfuel_kiln(hydrogen) -> oxyfuel_ccs` with purchased hydrogen |
| R12b | R12a + `electrolyser` |
| R13 | `preheater -> leilac_calciner(fossil) -> simple_kiln(fossil)` |
| R14 | `preheater -> leilac_calciner(electricity) -> simple_kiln(fossil)` |
| R15a | `preheater -> leilac_calciner(hydrogen) -> simple_kiln(hydrogen)` with purchased hydrogen |
| R15b | R15a + `electrolyser` |
| R16 | R13 + `amine_ccs` |

For R10, the oxyfuel CPU sees the oxyfuel-calciner stream only. The simple-kiln
stream remains physical and priced plant emissions unless another capture system is
introduced in a future extension.

## Current modelling limits

- No cement grinding, blending, clinker inventory or product-quality constraints.
- No raw-material cost, quarrying, transport or material inventory model.
- No explicit flue-gas composition beyond the physical/priced CO2 split.
- No CO2 transport, permanent storage, utilisation revenues or storage eligibility
  validation; CCS credit is an external-boundary assumption.
- No detailed calciner/kiln thermal chemistry, air leakage, gas recycle, refractory
  dynamics, compressor train or refrigeration cycle.
- No kiln-owned calcination process CO2 in the current architecture.
- The priced/unpriced CCS split is a linear accounting formulation, not isotope-level
  separation physics; see the CCS section for its interpretation.

These limits are deliberate abstractions. New detail should be added only where it
changes a dispatch decision, an energy balance, a physical CO2 balance or a stated
policy-accounting result.
