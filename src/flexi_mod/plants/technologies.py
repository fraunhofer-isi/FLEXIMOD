# SPDX-FileCopyrightText: FLEXIMOD Developers
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd
import pyomo.environ as pyo

COAL = "coal"
NATURAL_GAS = "natural_gas"
HYDROGEN = "hydrogen"
HYBRID_HYDROGEN_NATURAL_GAS = "hybrid_hydrogen_natural_gas"
STEEL_FUEL_TYPES = frozenset({COAL, NATURAL_GAS, HYDROGEN, HYBRID_HYDROGEN_NATURAL_GAS})
CEMENT_ELECTRICITY = "electricity"
CEMENT_FOSSIL = "fossil"
CEMENT_HYBRID_ELECTRICITY_FOSSIL = "hybrid_electricity_fossil"
CEMENT_FUEL_TYPES = frozenset(
    {CEMENT_ELECTRICITY, CEMENT_FOSSIL, HYDROGEN, CEMENT_HYBRID_ELECTRICITY_FOSSIL}
)


class GenericStorage(ABC):
    """Common interface for energy stores and material inventories."""

    @classmethod
    @abstractmethod
    def from_row(cls, row: pd.Series) -> GenericStorage:
        """Build a storage technology from one ``plants.csv`` row."""

    @abstractmethod
    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add storage variables and physical constraints to a Pyomo block."""


@dataclass
class ThermalStorage(GenericStorage):
    """Electrically charged thermal storage / ETES component."""

    max_power_charge_mw: float
    max_power_discharge_mw: float
    max_capacity_mwh: float
    min_capacity_mwh: float
    initial_soc_mwh: float
    efficiency_charge: float
    efficiency_discharge: float
    storage_loss_rate: float
    storage_type: str = "short-term_with_generator"

    @classmethod
    def from_row(cls, row: pd.Series) -> ThermalStorage:
        max_capacity = _as_float(row.get("max_capacity"), "max_capacity")
        min_capacity = _as_float(row.get("min_capacity"), "min_capacity", default=0.0)
        initial_soc = _as_float(row.get("initial_soc"), "initial_soc", default=0.0)
        initial_soc = min(max(initial_soc, min_capacity), max_capacity)

        return cls(
            max_power_charge_mw=_as_float(row.get("max_power_charge"), "max_power_charge"),
            max_power_discharge_mw=_as_float(row.get("max_power_discharge"), "max_power_discharge"),
            max_capacity_mwh=max_capacity,
            min_capacity_mwh=min_capacity,
            initial_soc_mwh=initial_soc,
            efficiency_charge=_as_float(
                row.get("efficiency_charge"), "efficiency_charge", default=1.0
            ),
            efficiency_discharge=_as_float(
                row.get("efficiency_discharge"), "efficiency_discharge", default=1.0
            ),
            storage_loss_rate=_as_float(
                row.get("storage_loss_rate"), "storage_loss_rate", default=0.0
            ),
            storage_type=_clean(row.get("storage_type"), "short-term_with_generator"),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Pyomo Components:

        - **Parameters**: charge/discharge power limits, capacity bounds, the initial
          state of charge carried in from the previous rolling window, and round-trip
          efficiencies.
        - **Variables**: ``electric_charge_to_storage``, ``discharge_heat``, state of
          charge (``soc``, absolute MWh), and cost.
        - **Constraints**: ``storage_balance`` (charge/discharge/loss), and cost
          definitions. ``charge_allowed_limit`` is added only when the enclosing model
          defines a ``charge_allowed`` Param (a grid-fee high-load window gate).
        """
        dt_hours = float(context["dt_hours"])
        initial_soc = float(context.get("initial_soc_mwh", self.initial_soc_mwh))
        max_charge_mwh = self.max_power_charge_mw * dt_hours
        max_discharge_mwh = self.max_power_discharge_mw * dt_hours

        block.max_power_charge_mw = pyo.Param(initialize=self.max_power_charge_mw)
        block.max_power_discharge_mw = pyo.Param(initialize=self.max_power_discharge_mw)
        block.max_capacity_mwh = pyo.Param(initialize=self.max_capacity_mwh)
        block.min_capacity_mwh = pyo.Param(initialize=self.min_capacity_mwh)
        block.initial_soc_mwh = pyo.Param(initialize=initial_soc)
        block.efficiency_charge = pyo.Param(initialize=self.efficiency_charge)
        block.efficiency_discharge = pyo.Param(initialize=self.efficiency_discharge)
        block.storage_loss_rate = pyo.Param(initialize=self.storage_loss_rate)

        block.electric_charge_to_storage = pyo.Var(
            time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0.0, max_charge_mwh),
        )
        block.discharge_heat = pyo.Var(
            time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0.0, max_discharge_mwh),
        )
        block.soc = pyo.Var(
            time_steps,
            within=pyo.NonNegativeReals,
            bounds=(self.min_capacity_mwh, self.max_capacity_mwh),
        )
        block.electricity_consumption = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.electricity_cost = pyo.Var(time_steps, within=pyo.Reals)
        # Plants that total their costs by summing every block's ``operating_cost`` (the
        # cement plant does) would otherwise charge nothing for charging this storage and
        # treat its electricity as free. ``electricity_cost`` is kept because the steam
        # plant references it by name.
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def storage_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            previous_soc = b.initial_soc_mwh if t == 0 else b.soc[t - 1]
            return b.soc[t] == (
                previous_soc * (1.0 - b.storage_loss_rate)
                + b.electric_charge_to_storage[t] * b.efficiency_charge
                - b.discharge_heat[t] / b.efficiency_discharge
            )

        @block.Constraint(time_steps)
        def electricity_consumption_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.electricity_consumption[t] == b.electric_charge_to_storage[t]

        @block.Constraint(time_steps)
        def electricity_cost_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return (
                b.electricity_cost[t] == b.electricity_consumption[t] * model.electricity_price[t]
            )

        @block.Constraint(time_steps)
        def operating_cost_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.operating_cost[t] == b.electricity_cost[t]

        if hasattr(model, "charge_allowed"):

            @block.Constraint(time_steps)
            def charge_allowed_limit(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.electric_charge_to_storage[t] <= max_charge_mwh * model.charge_allowed[t]

        return block


@dataclass
class GasBoiler:
    """Natural-gas boiler component for industrial steam or heat supply."""

    max_heat_output_mw: float
    min_heat_output_mw: float
    efficiency: float
    fuel_type: str = "natural_gas"
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None

    @classmethod
    def from_row(cls, row: pd.Series) -> GasBoiler:
        fuel_type = _clean(row.get("fuel_type"), "natural_gas")
        if fuel_type != "natural_gas":
            raise ValueError("GasBoiler currently supports only fuel_type='natural_gas'")

        return cls(
            max_heat_output_mw=_as_float(row.get("max_power"), "max_power"),
            min_heat_output_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            efficiency=_as_float(row.get("efficiency"), "efficiency", default=0.9),
            fuel_type=fuel_type,
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up")),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down")),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Pyomo Components:

        - **Parameters**: rated heat output and conversion efficiency.
        - **Variables**: ``heat_out``, ``fuel_input``, operating cost, and CO2 cost.
        - **Constraints**: ``efficiency_constraint`` linking heat to fuel, and the
          operating-cost and CO2-cost definitions.
        """
        dt_hours = float(context["dt_hours"])
        max_heat_mwh = self.max_heat_output_mw * dt_hours
        max_fuel_input_mwh = max_heat_mwh / self.efficiency

        block.max_heat_output_mw = pyo.Param(initialize=self.max_heat_output_mw)
        block.min_heat_output_mw = pyo.Param(initialize=self.min_heat_output_mw)
        block.efficiency = pyo.Param(initialize=self.efficiency)
        block.heat_out = pyo.Var(
            time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0.0, max_heat_mwh),
        )
        block.fuel_input = pyo.Var(
            time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0.0, max_fuel_input_mwh),
        )
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)
        block.co2_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def efficiency_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.heat_out[t] == b.fuel_input[t] * b.efficiency

        @block.Constraint(time_steps)
        def operating_cost_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.operating_cost[t] == b.fuel_input[t] * model.gas_price[t]

        @block.Constraint(time_steps)
        def co2_cost_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_cost[t] == b.fuel_input[t] * model.co2_emission_factor * model.co2_price[t]

        return block


@dataclass
class Electrolyser:
    """Electricity-to-hydrogen converter used by a steel plant."""

    max_power_mw: float
    efficiency: float
    min_power_mw: float = 0.0
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    min_operating_steps: int = 1
    min_down_steps: int = 1
    initial_operational_status: int = 1
    oxygen_byproduct_t_per_mwh_hydrogen: float = 0.24

    def __post_init__(self) -> None:
        if self.oxygen_byproduct_t_per_mwh_hydrogen < 0.0:
            raise ValueError("oxygen_byproduct_t_per_mwh_hydrogen must be non-negative")

    @classmethod
    def from_row(cls, row: pd.Series) -> Electrolyser:
        max_power = _as_float(row.get("max_power"), "max_power")
        return cls(
            max_power_mw=max_power,
            efficiency=_as_float(row.get("efficiency"), "efficiency"),
            min_power_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up")),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down")),
            min_operating_steps=_as_int(
                _first_present(row.get("min_operating_steps"), row.get("min_operating_time")),
                default=1,
            ),
            min_down_steps=_as_int(
                _first_present(row.get("min_down_steps"), row.get("min_down_time")), default=1
            ),
            initial_operational_status=_as_int(row.get("initial_operational_status"), default=1),
            oxygen_byproduct_t_per_mwh_hydrogen=_as_float(
                row.get("oxygen_byproduct_t_per_mwh_hydrogen"),
                "oxygen_byproduct_t_per_mwh_hydrogen",
                default=0.24,
            ),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Pyomo Components:

        - **Parameters**: rating, ramp limits, and commitment state from
          ``_add_power_parameters`` (shared with the other steel power converters), plus
          conversion efficiency.
        - **Variables**: ``power_in``, ``hydrogen_out``, and operating cost.
        - **Constraints**: ``hydrogen_production_constraint`` linking output to input,
          operating cost, ramping (always), and commitment (only when
          ``min_operating_steps``, ``min_down_steps``, or ``min_power`` require it).
        """
        dt_hours = float(context["dt_hours"])
        _add_power_parameters(self, block, dt_hours, context)
        block.efficiency = pyo.Param(initialize=self.efficiency)
        block.power_in = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, self.max_power_mw * dt_hours)
        )
        block.hydrogen_out = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.oxygen_byproduct = pyo.Param(initialize=self.oxygen_byproduct_t_per_mwh_hydrogen)
        block.oxygen_out = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def hydrogen_production_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.hydrogen_out[t] == b.power_in[t] * b.efficiency

        @block.Constraint(time_steps)
        def oxygen_coproduct_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.oxygen_out[t] == b.hydrogen_out[t] * b.oxygen_byproduct

        @block.Constraint(time_steps)
        def operating_cost_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]

        _add_ramping_constraints(block, time_steps)
        if (
            int(pyo.value(block.min_operating_steps)) > 1
            or int(pyo.value(block.min_down_steps)) > 1
            or float(pyo.value(block.min_power)) > 0
        ):
            _add_commitment_constraints(block, time_steps)
        return block


@dataclass
class DRIPlant:
    """Direct-reduced-iron shaft using a configured reducing fuel.

    Balance structure follows the direct-reduction route taxonomy in the IEA
    Iron and Steel Technology Roadmap. Fuel and raw-material intensities remain
    explicit plant inputs, so case data can represent a specific shaft design.
    """

    specific_coal_consumption_mwh_per_t: float
    specific_hydrogen_consumption_mwh_per_t: float
    specific_natural_gas_consumption_mwh_per_t: float
    specific_electricity_consumption_mwh_per_t: float
    specific_iron_ore_consumption_t_per_t: float
    max_power_mw: float
    min_power_mw: float
    fuel_type: str
    natural_gas_co2_factor_t_per_mwh: float
    coal_co2_factor_t_per_mwh: float
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1

    @classmethod
    def from_row(cls, row: pd.Series) -> DRIPlant:
        fuel_type = _steel_fuel_type(row, "DRIPlant")
        return cls(
            specific_coal_consumption_mwh_per_t=_fuel_specific_consumption(
                row,
                "specific_coal_consumption",
                fuel_type,
                required_for={COAL},
            ),
            specific_hydrogen_consumption_mwh_per_t=_fuel_specific_consumption(
                row,
                "specific_hydrogen_consumption",
                fuel_type,
                required_for={HYDROGEN, HYBRID_HYDROGEN_NATURAL_GAS},
            ),
            specific_natural_gas_consumption_mwh_per_t=_fuel_specific_consumption(
                row,
                "specific_natural_gas_consumption",
                fuel_type,
                required_for={NATURAL_GAS, HYBRID_HYDROGEN_NATURAL_GAS},
            ),
            specific_electricity_consumption_mwh_per_t=_as_float(
                row.get("specific_electricity_consumption"),
                "specific_electricity_consumption",
            ),
            specific_iron_ore_consumption_t_per_t=_as_float(
                row.get("specific_iron_ore_consumption"), "specific_iron_ore_consumption"
            ),
            max_power_mw=_as_float(row.get("max_power"), "max_power"),
            min_power_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            fuel_type=fuel_type,
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up")),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down")),
            min_operating_steps=_as_int(
                _first_present(row.get("min_operating_steps"), row.get("min_operating_time")),
                default=0,
            ),
            min_down_steps=_as_int(
                _first_present(row.get("min_down_steps"), row.get("min_down_time")), default=0
            ),
            initial_operational_status=_as_int(row.get("initial_operational_status"), default=1),
            natural_gas_co2_factor_t_per_mwh=_fuel_co2_factor(
                row,
                "natural_gas_co2_factor",
                fuel_type,
                required_for={NATURAL_GAS, HYBRID_HYDROGEN_NATURAL_GAS},
            ),
            coal_co2_factor_t_per_mwh=_fuel_co2_factor(
                row,
                "coal_co2_factor",
                fuel_type,
                required_for={COAL},
            ),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Pyomo Components:

        - **Parameters**: rating, ramp limits, and commitment state from
          ``_add_power_parameters`` (shared with the other steel power converters), plus
          the specific consumption of every reducing agent this ``fuel_type`` can use.
        - **Variables**: ``power_in``, ``iron_ore_in``, ``coal_in``, ``natural_gas_in``,
          ``hydrogen_in``, ``co2_emission``, ``dri_output``, and operating cost.
        - **Constraints**: ``dri_output_constraint`` (fuel-specific, single fuel or the
          natural-gas/hydrogen hybrid blend), ``zero_unused_*_constraint`` for every
          reducing agent this ``fuel_type`` does not use, electricity and iron-ore
          consumption, CO2, and operating cost, plus ramping (always) and commitment
          (only when ``min_operating_steps``, ``min_down_steps``, or ``min_power``
          require it).
        """
        dt_hours = float(context["dt_hours"])
        _add_power_parameters(self, block, dt_hours, context)
        block.specific_coal_consumption = pyo.Param(
            initialize=self.specific_coal_consumption_mwh_per_t
        )
        block.specific_hydrogen_consumption = pyo.Param(
            initialize=self.specific_hydrogen_consumption_mwh_per_t
        )
        block.specific_natural_gas_consumption = pyo.Param(
            initialize=self.specific_natural_gas_consumption_mwh_per_t
        )
        block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption_mwh_per_t
        )
        block.specific_iron_ore_consumption = pyo.Param(
            initialize=self.specific_iron_ore_consumption_t_per_t
        )
        block.natural_gas_co2_factor = pyo.Param(initialize=self.natural_gas_co2_factor_t_per_mwh)
        block.coal_co2_factor = pyo.Param(initialize=self.coal_co2_factor_t_per_mwh)
        block.power_in = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, self.max_power_mw * dt_hours)
        )
        block.iron_ore_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.coal_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.natural_gas_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_emission = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.hydrogen_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.dri_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def dri_output_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            if self.fuel_type == COAL:
                return b.dri_output[t] == b.coal_in[t] / b.specific_coal_consumption
            if self.fuel_type == HYDROGEN:
                return b.dri_output[t] == b.hydrogen_in[t] / b.specific_hydrogen_consumption
            if self.fuel_type == NATURAL_GAS:
                return b.dri_output[t] == b.natural_gas_in[t] / b.specific_natural_gas_consumption
            return b.dri_output[t] == (
                b.natural_gas_in[t] / b.specific_natural_gas_consumption
                + b.hydrogen_in[t] / b.specific_hydrogen_consumption
            )

        @block.Constraint(time_steps)
        def zero_unused_coal_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            if self.fuel_type != COAL:
                return b.coal_in[t] == 0
            return pyo.Constraint.Skip

        @block.Constraint(time_steps)
        def zero_unused_natural_gas_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            if self.fuel_type not in {NATURAL_GAS, HYBRID_HYDROGEN_NATURAL_GAS}:
                return b.natural_gas_in[t] == 0
            return pyo.Constraint.Skip

        @block.Constraint(time_steps)
        def zero_unused_hydrogen_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            if self.fuel_type not in {HYDROGEN, HYBRID_HYDROGEN_NATURAL_GAS}:
                return b.hydrogen_in[t] == 0
            return pyo.Constraint.Skip

        @block.Constraint(time_steps)
        def electricity_consumption_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.power_in[t] == b.dri_output[t] * b.specific_electricity_consumption

        @block.Constraint(time_steps)
        def iron_ore_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.iron_ore_in[t] == b.dri_output[t] * b.specific_iron_ore_consumption

        @block.Constraint(time_steps)
        def co2_emission_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_emission[t] == (
                b.coal_in[t] * b.coal_co2_factor + b.natural_gas_in[t] * b.natural_gas_co2_factor
            )

        @block.Constraint(time_steps)
        def operating_cost_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            cost = (
                b.power_in[t] * model.electricity_price[t]
                + b.iron_ore_in[t] * model.iron_ore_price[t]
                + b.co2_emission[t] * model.co2_price[t]
            )
            if self.fuel_type == COAL:
                cost += b.coal_in[t] * model.coal_price[t]
            if self.fuel_type in {NATURAL_GAS, HYBRID_HYDROGEN_NATURAL_GAS}:
                cost += b.natural_gas_in[t] * model.natural_gas_price[t]
            if self.fuel_type in {HYDROGEN, HYBRID_HYDROGEN_NATURAL_GAS}:
                cost += b.hydrogen_in[t] * model.hydrogen_price[t]
            return b.operating_cost[t] == cost

        _add_ramping_constraints(block, time_steps)
        if (
            int(pyo.value(block.min_operating_steps)) > 1
            or int(pyo.value(block.min_down_steps)) > 1
            or float(pyo.value(block.min_power)) > 0
        ):
            _add_commitment_constraints(block, time_steps)
        return block


@dataclass
class ElectricArcFurnace:
    """Electric arc furnace converting DRI and lime into steel."""

    max_power_mw: float
    min_power_mw: float
    specific_electricity_consumption_mwh_per_t: float
    specific_dri_demand_t_per_t: float
    specific_lime_demand_t_per_t: float
    lime_co2_factor_t_per_t: float
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1

    @classmethod
    def from_row(cls, row: pd.Series) -> ElectricArcFurnace:
        return cls(
            max_power_mw=_as_float(row.get("max_power"), "max_power"),
            min_power_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            specific_electricity_consumption_mwh_per_t=_as_float(
                row.get("specific_electricity_consumption"),
                "specific_electricity_consumption",
            ),
            specific_dri_demand_t_per_t=_as_float(
                row.get("specific_dri_demand"), "specific_dri_demand"
            ),
            specific_lime_demand_t_per_t=_as_float(
                row.get("specific_lime_demand"), "specific_lime_demand"
            ),
            lime_co2_factor_t_per_t=_as_float(row.get("lime_co2_factor"), "lime_co2_factor"),
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up")),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down")),
            min_operating_steps=_as_int(
                _first_present(row.get("min_operating_steps"), row.get("min_operating_time")),
                default=0,
            ),
            min_down_steps=_as_int(
                _first_present(row.get("min_down_steps"), row.get("min_down_time")), default=0
            ),
            initial_operational_status=_as_int(row.get("initial_operational_status"), default=1),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Pyomo Components:

        - **Parameters**: rating, ramp limits, and commitment state from
          ``_add_power_parameters`` (shared with the other steel power converters), plus
          specific electricity/lime consumption and lime's CO2 factor.
        - **Variables**: ``power_in``, ``dri_input``, ``steel_output``, ``lime_demand``,
          ``co2_emission``, and operating cost.
        - **Constraints**: DRI-to-steel and power-to-steel relations, lime demand, CO2,
          and operating cost, plus ramping (always) and commitment (only when
          ``min_operating_steps``, ``min_down_steps``, or ``min_power`` require it).
        """
        dt_hours = float(context["dt_hours"])
        _add_power_parameters(self, block, dt_hours, context)
        block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption_mwh_per_t
        )
        block.specific_dri_demand = pyo.Param(initialize=self.specific_dri_demand_t_per_t)
        block.specific_lime_demand = pyo.Param(initialize=self.specific_lime_demand_t_per_t)
        block.lime_co2_factor = pyo.Param(initialize=self.lime_co2_factor_t_per_t)
        block.power_in = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, self.max_power_mw * dt_hours)
        )
        block.dri_input = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.steel_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)
        block.co2_emission = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.lime_demand = pyo.Var(time_steps, within=pyo.NonNegativeReals)

        @block.Constraint(time_steps)
        def steel_output_dri_relation_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.steel_output[t] == b.dri_input[t] / b.specific_dri_demand

        @block.Constraint(time_steps)
        def steel_output_power_relation_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.power_in[t] == b.steel_output[t] * b.specific_electricity_consumption

        @block.Constraint(time_steps)
        def lime_demand_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.lime_demand[t] == b.steel_output[t] * b.specific_lime_demand

        @block.Constraint(time_steps)
        def co2_emission_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_emission[t] == b.lime_demand[t] * b.lime_co2_factor

        @block.Constraint(time_steps)
        def operating_cost_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.operating_cost[t] == (
                b.power_in[t] * model.electricity_price[t]
                + b.co2_emission[t] * model.co2_price[t]
                + b.lime_demand[t] * model.lime_price[t]
            )

        _add_ramping_constraints(block, time_steps)
        if (
            int(pyo.value(block.min_operating_steps)) > 1
            or int(pyo.value(block.min_down_steps)) > 1
            or float(pyo.value(block.min_power)) > 0
        ):
            _add_commitment_constraints(block, time_steps)
        return block


@dataclass
class BasicOxygenFurnace:
    """Basic oxygen furnace converting DRI into steel.

    This is a linear converter balance for FLEXIMOD optimization. BOF process
    boundaries follow the JRC Iron and Steel BREF; the DRI-to-steel material
    balance follows the mass-balance formulation used for BOF/EAF converter
    modelling in Zhou et al. (2025). Consumption factors are case inputs.
    """

    max_power_mw: float
    min_power_mw: float
    specific_electricity_consumption_mwh_per_t: float
    specific_dri_demand_t_per_t: float
    specific_lime_demand_t_per_t: float
    lime_co2_factor_t_per_t: float
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1

    @classmethod
    def from_row(cls, row: pd.Series) -> BasicOxygenFurnace:
        return cls(
            max_power_mw=_as_float(row.get("max_power"), "max_power"),
            min_power_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            specific_electricity_consumption_mwh_per_t=_as_float(
                row.get("specific_electricity_consumption"),
                "specific_electricity_consumption",
            ),
            specific_dri_demand_t_per_t=_as_float(
                row.get("specific_dri_demand"), "specific_dri_demand"
            ),
            specific_lime_demand_t_per_t=_as_float(
                row.get("specific_lime_demand"), "specific_lime_demand"
            ),
            lime_co2_factor_t_per_t=_as_float(row.get("lime_co2_factor"), "lime_co2_factor"),
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up")),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down")),
            min_operating_steps=_as_int(
                _first_present(row.get("min_operating_steps"), row.get("min_operating_time")),
                default=0,
            ),
            min_down_steps=_as_int(
                _first_present(row.get("min_down_steps"), row.get("min_down_time")), default=0
            ),
            initial_operational_status=_as_int(row.get("initial_operational_status"), default=1),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Pyomo Components:

        - **Parameters**: rating, ramp limits, and commitment state from
          ``_add_power_parameters`` (shared with the other steel power converters), plus
          specific electricity/lime consumption and lime's CO2 factor.
        - **Variables**: ``power_in``, ``dri_input``, ``steel_output``, ``lime_demand``,
          ``co2_emission``, and operating cost.
        - **Constraints**: DRI-to-steel and power-to-steel relations, lime demand, CO2,
          and operating cost, plus ramping (always) and commitment (only when
          ``min_operating_steps``, ``min_down_steps``, or ``min_power`` require it).
        """
        dt_hours = float(context["dt_hours"])
        _add_power_parameters(self, block, dt_hours, context)
        block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption_mwh_per_t
        )
        block.specific_dri_demand = pyo.Param(initialize=self.specific_dri_demand_t_per_t)
        block.specific_lime_demand = pyo.Param(initialize=self.specific_lime_demand_t_per_t)
        block.lime_co2_factor = pyo.Param(initialize=self.lime_co2_factor_t_per_t)
        block.power_in = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, self.max_power_mw * dt_hours)
        )
        block.dri_input = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.steel_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)
        block.co2_emission = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.lime_demand = pyo.Var(time_steps, within=pyo.NonNegativeReals)

        @block.Constraint(time_steps)
        def steel_output_dri_relation_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.steel_output[t] == b.dri_input[t] / b.specific_dri_demand

        @block.Constraint(time_steps)
        def steel_output_power_relation_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.power_in[t] == b.steel_output[t] * b.specific_electricity_consumption

        @block.Constraint(time_steps)
        def lime_demand_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.lime_demand[t] == b.steel_output[t] * b.specific_lime_demand

        @block.Constraint(time_steps)
        def co2_emission_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_emission[t] == b.lime_demand[t] * b.lime_co2_factor

        @block.Constraint(time_steps)
        def operating_cost_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.operating_cost[t] == (
                b.power_in[t] * model.electricity_price[t]
                + b.co2_emission[t] * model.co2_price[t]
                + b.lime_demand[t] * model.lime_price[t]
            )

        _add_ramping_constraints(block, time_steps)
        if (
            int(pyo.value(block.min_operating_steps)) > 1
            or int(pyo.value(block.min_down_steps)) > 1
            or float(pyo.value(block.min_power)) > 0
        ):
            _add_commitment_constraints(block, time_steps)
        return block


@dataclass
class BlastFurnaceBasicOxygenFurnace:
    """Integrated BF-BOF steelmaking route.

    Process boundaries follow the JRC Iron and Steel BREF and IEA BF-BOF route
    taxonomy. The linear equations below are production-planning balances:
    explicit case coefficients convert final steel output into iron-bearing
    feedstock, reductant, electricity, lime, and direct CO2. This preserves
    literature structure without hard-coding one plant recipe as universal.
    """

    specific_coal_consumption_mwh_per_t: float
    specific_hydrogen_consumption_mwh_per_t: float
    specific_natural_gas_consumption_mwh_per_t: float
    specific_electricity_consumption_mwh_per_t: float
    specific_iron_ore_consumption_t_per_t: float
    specific_lime_demand_t_per_t: float
    max_power_mw: float
    min_power_mw: float
    fuel_type: str
    coal_co2_factor_t_per_mwh: float
    natural_gas_co2_factor_t_per_mwh: float
    lime_co2_factor_t_per_t: float
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1

    @classmethod
    def from_row(cls, row: pd.Series) -> BlastFurnaceBasicOxygenFurnace:
        fuel_type = _steel_fuel_type(row, "BlastFurnaceBasicOxygenFurnace")
        return cls(
            specific_coal_consumption_mwh_per_t=_fuel_specific_consumption(
                row,
                "specific_coal_consumption",
                fuel_type,
                required_for={COAL},
            ),
            specific_hydrogen_consumption_mwh_per_t=_fuel_specific_consumption(
                row,
                "specific_hydrogen_consumption",
                fuel_type,
                required_for={HYDROGEN, HYBRID_HYDROGEN_NATURAL_GAS},
            ),
            specific_natural_gas_consumption_mwh_per_t=_fuel_specific_consumption(
                row,
                "specific_natural_gas_consumption",
                fuel_type,
                required_for={NATURAL_GAS, HYBRID_HYDROGEN_NATURAL_GAS},
            ),
            specific_electricity_consumption_mwh_per_t=_as_float(
                row.get("specific_electricity_consumption"),
                "specific_electricity_consumption",
            ),
            specific_iron_ore_consumption_t_per_t=_as_float(
                row.get("specific_iron_ore_consumption"), "specific_iron_ore_consumption"
            ),
            specific_lime_demand_t_per_t=_as_float(
                row.get("specific_lime_demand"), "specific_lime_demand"
            ),
            max_power_mw=_as_float(row.get("max_power"), "max_power"),
            min_power_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            fuel_type=fuel_type,
            coal_co2_factor_t_per_mwh=_fuel_co2_factor(
                row,
                "coal_co2_factor",
                fuel_type,
                required_for={COAL},
            ),
            natural_gas_co2_factor_t_per_mwh=_fuel_co2_factor(
                row,
                "natural_gas_co2_factor",
                fuel_type,
                required_for={NATURAL_GAS, HYBRID_HYDROGEN_NATURAL_GAS},
            ),
            lime_co2_factor_t_per_t=_as_float(row.get("lime_co2_factor"), "lime_co2_factor"),
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up")),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down")),
            min_operating_steps=_as_int(
                _first_present(row.get("min_operating_steps"), row.get("min_operating_time")),
                default=0,
            ),
            min_down_steps=_as_int(
                _first_present(row.get("min_down_steps"), row.get("min_down_time")), default=0
            ),
            initial_operational_status=_as_int(row.get("initial_operational_status"), default=1),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Pyomo Components:

        - **Parameters**: rating, ramp limits, and commitment state from
          ``_add_power_parameters`` (shared with the other steel power converters), plus
          the specific consumption of every reducing agent this ``fuel_type`` can use,
          iron ore, and lime, and their CO2 factors.
        - **Variables**: ``power_in``, ``coal_in``, ``natural_gas_in``, ``hydrogen_in``,
          ``iron_ore_in``, ``lime_demand``, ``co2_emission``, ``steel_output``, and
          operating cost.
        - **Constraints**: ``steel_output_fuel_relation_constraint`` (fuel-specific,
          single fuel or the natural-gas/hydrogen hybrid blend), ``zero_unused_*``
          constraints for every reducing agent this ``fuel_type`` does not use,
          electricity/iron-ore/lime consumption, CO2, and operating cost, plus ramping
          (always) and commitment (only when ``min_operating_steps``,
          ``min_down_steps``, or ``min_power`` require it).
        """
        dt_hours = float(context["dt_hours"])
        _add_power_parameters(self, block, dt_hours, context)
        block.specific_coal_consumption = pyo.Param(
            initialize=self.specific_coal_consumption_mwh_per_t
        )
        block.specific_hydrogen_consumption = pyo.Param(
            initialize=self.specific_hydrogen_consumption_mwh_per_t
        )
        block.specific_natural_gas_consumption = pyo.Param(
            initialize=self.specific_natural_gas_consumption_mwh_per_t
        )
        block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption_mwh_per_t
        )
        block.specific_iron_ore_consumption = pyo.Param(
            initialize=self.specific_iron_ore_consumption_t_per_t
        )
        block.specific_lime_demand = pyo.Param(initialize=self.specific_lime_demand_t_per_t)
        block.coal_co2_factor = pyo.Param(initialize=self.coal_co2_factor_t_per_mwh)
        block.natural_gas_co2_factor = pyo.Param(initialize=self.natural_gas_co2_factor_t_per_mwh)
        block.lime_co2_factor = pyo.Param(initialize=self.lime_co2_factor_t_per_t)
        block.power_in = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, self.max_power_mw * dt_hours)
        )
        block.coal_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.natural_gas_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.hydrogen_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.iron_ore_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.lime_demand = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_emission = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.steel_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def steel_output_fuel_relation_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            if self.fuel_type == COAL:
                return b.steel_output[t] == b.coal_in[t] / b.specific_coal_consumption
            if self.fuel_type == HYDROGEN:
                return b.steel_output[t] == b.hydrogen_in[t] / b.specific_hydrogen_consumption
            if self.fuel_type == NATURAL_GAS:
                return b.steel_output[t] == (
                    b.natural_gas_in[t] / b.specific_natural_gas_consumption
                )
            return b.steel_output[t] == (
                b.natural_gas_in[t] / b.specific_natural_gas_consumption
                + b.hydrogen_in[t] / b.specific_hydrogen_consumption
            )

        @block.Constraint(time_steps)
        def zero_unused_coal_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            if self.fuel_type != COAL:
                return b.coal_in[t] == 0
            return pyo.Constraint.Skip

        @block.Constraint(time_steps)
        def zero_unused_natural_gas_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            if self.fuel_type not in {NATURAL_GAS, HYBRID_HYDROGEN_NATURAL_GAS}:
                return b.natural_gas_in[t] == 0
            return pyo.Constraint.Skip

        @block.Constraint(time_steps)
        def zero_unused_hydrogen_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            if self.fuel_type not in {HYDROGEN, HYBRID_HYDROGEN_NATURAL_GAS}:
                return b.hydrogen_in[t] == 0
            return pyo.Constraint.Skip

        @block.Constraint(time_steps)
        def electricity_consumption_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.power_in[t] == b.steel_output[t] * b.specific_electricity_consumption

        @block.Constraint(time_steps)
        def iron_ore_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.iron_ore_in[t] == b.steel_output[t] * b.specific_iron_ore_consumption

        @block.Constraint(time_steps)
        def lime_demand_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.lime_demand[t] == b.steel_output[t] * b.specific_lime_demand

        @block.Constraint(time_steps)
        def co2_emission_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_emission[t] == (
                b.coal_in[t] * b.coal_co2_factor
                + b.natural_gas_in[t] * b.natural_gas_co2_factor
                + b.lime_demand[t] * b.lime_co2_factor
            )

        @block.Constraint(time_steps)
        def operating_cost_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            cost = (
                b.power_in[t] * model.electricity_price[t]
                + b.iron_ore_in[t] * model.iron_ore_price[t]
                + b.lime_demand[t] * model.lime_price[t]
                + b.co2_emission[t] * model.co2_price[t]
            )
            if self.fuel_type == COAL:
                cost += b.coal_in[t] * model.coal_price[t]
            if self.fuel_type in {NATURAL_GAS, HYBRID_HYDROGEN_NATURAL_GAS}:
                cost += b.natural_gas_in[t] * model.natural_gas_price[t]
            if self.fuel_type in {HYDROGEN, HYBRID_HYDROGEN_NATURAL_GAS}:
                cost += b.hydrogen_in[t] * model.hydrogen_price[t]
            return b.operating_cost[t] == cost

        _add_ramping_constraints(block, time_steps)
        if (
            int(pyo.value(block.min_operating_steps)) > 1
            or int(pyo.value(block.min_down_steps)) > 1
            or float(pyo.value(block.min_power)) > 0
        ):
            _add_commitment_constraints(block, time_steps)
        return block


class CementKilnLineStage:
    """Shared physics for a fuel-switchable cement kiln-line stage.

    ``CementPreheater``, ``SimpleCementCalciner``, and ``SimpleCementKiln`` share the
    same heat balance, fuel-exclusivity, ramping, commitment, and CO2/cost machinery -
    only a handful of details differ per stage. Those differences are exactly what each
    subclass overrides:

    - ``_output_var_name``: the block attribute its throughput Var is stored under.
    - ``_add_stage_parameters``: any Params beyond the common set, e.g. the calciner's
      calcination factor (default: none).
    - ``_add_stage_variables``: any Vars beyond the common set (default: none).
    - ``_external_heat_expr``: heat entering from outside the stage, e.g. kiln waste
      heat reaching the preheater (default: none).
    - ``_add_stage_constraints``: what this stage's output actually equals, and its own
      process CO2 if it has any - written out directly by every subclass, even where the
      body is nearly identical across stages, so each class shows its own physics rather
      than delegating it to a shared, hook-parameterised constraint (default: none - a
      stage that skips this defines no relationship between its heat and its output).

    ``add_to_model`` is the template method: it calls the phases below in a fixed
    order and returns the populated block.
    """

    #: Attribute name the stage's throughput Var is stored under on the block.
    _output_var_name: ClassVar[str] = "clinker_out"

    #: Default combustion-energy shares used by the conventional fossil routes.
    default_fossil_ng_share: ClassVar[float] = 0.034
    default_biomass_share: ClassVar[float] = 0.0
    default_rdf_share: ClassVar[float] = 0.0

    #: R2 defaults: 0.9% NG and 25.7% coal within the 26.6% fossil remainder.
    r2_fossil_ng_share: ClassVar[float] = 0.009 / (0.009 + 0.257)
    r2_biomass_share: ClassVar[float] = 0.245
    r2_rdf_share: ClassVar[float] = 0.489

    #: Fossil CO2 emitted per MWh of total mixed RDF energy input.
    default_rdf_mixed_fossil_co2_factor_t_per_mwh: ClassVar[float] = 0.243

    @classmethod
    def _fuel_mix_defaults(
        cls,
        row: pd.Series,
        default_fuel_type: str,
    ) -> tuple[float, float, float]:
        """Return fossil-NG, separately procured biomass, and RDF share defaults."""
        fuel_type = _clean(row.get("fuel_type"), default_fuel_type).lower()
        if fuel_type not in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
            return cls.default_fossil_ng_share, 0.0, 0.0
        route_id = _clean(
            _first_present(row.get("cement_route_id"), row.get("route_id"), row.get("route")),
            "",
        ).upper()
        if route_id in {"R2", "2"}:
            return cls.r2_fossil_ng_share, cls.r2_biomass_share, cls.r2_rdf_share
        return cls.default_fossil_ng_share, cls.default_biomass_share, cls.default_rdf_share

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        dt_hours = float(context["dt_hours"])
        self._add_common_parameters(block, dt_hours, context)
        self._add_stage_parameters(block)
        self._add_common_variables(block, time_steps)
        self._add_stage_variables(block, time_steps)
        output = getattr(block, self._output_var_name)
        self._add_firing_constraints(block, time_steps, output)
        self._add_stage_constraints(block, time_steps, output)
        self._add_aux_and_operational_constraints(block, time_steps, output)
        self._add_emission_and_cost_constraints(model, block, time_steps, output)
        return block

    def _add_common_parameters(
        self, block: pyo.Block, dt_hours: float, context: dict[str, Any]
    ) -> None:
        """Pyomo Components:

        - **Parameters**: stage ratings (``max_heat_out``, ``min_heat_out`` - the
          turndown floor while the stage is on, 0 by default), efficiencies, ramp
          limits, CO2 factors, and the commitment state (min up/down steps, initial
          status) carried in from the previous rolling window.
        """
        max_heat_mwh = self.max_heat_out_mw * dt_hours
        min_heat_mwh = self.min_heat_out_mw * dt_hours
        max_electric_power_mw = (
            self.max_electric_power_mw
            if self.max_electric_power_mw is not None
            else self.max_heat_out_mw / max(1e-9, self.eta_electric)
        )
        if max_electric_power_mw < 0.0:
            raise ValueError("max_power must be non-negative")
        max_power_mwh = max_electric_power_mw * dt_hours
        min_power_mwh = self.min_electric_power_mw * dt_hours
        if self.min_electric_power_mw < 0.0:
            raise ValueError("min_power must be non-negative")
        if self.min_electric_power_mw > max_electric_power_mw:
            raise ValueError("min_power must not exceed max_power")
        ramp_up = (
            self.max_heat_out_mw if self.ramp_up_mw_per_step is None else self.ramp_up_mw_per_step
        )
        ramp_down = (
            self.max_heat_out_mw
            if self.ramp_down_mw_per_step is None
            else self.ramp_down_mw_per_step
        )
        if not 0.0 <= self.fossil_ng_share <= 1.0:
            raise ValueError("fossil_ng_share must be between 0 and 1")
        if not 0.0 <= self.biomass_share <= 1.0:
            raise ValueError("biomass_share must be between 0 and 1")
        if not 0.0 <= self.rdf_share <= 1.0:
            raise ValueError("rdf_share must be between 0 and 1")
        if self.biomass_share + self.rdf_share > 1.0 + 1e-12:
            raise ValueError("biomass_share + rdf_share must not exceed 1")
        if self.biomass_co2_factor_t_per_mwh < 0.0:
            raise ValueError("biomass_co2_factor must be non-negative")
        if self.biomass_share > 0.0 and self.biomass_co2_factor_t_per_mwh <= 0.0:
            raise ValueError("biomass_co2_factor must be positive when biomass_share is positive")
        if not 0.0 <= self.biomass_co2_accounting_share <= 1.0:
            raise ValueError("biomass_co2_accounting_share must be between 0 and 1")
        if self.rdf_mixed_fossil_co2_factor_t_per_mwh < 0.0:
            raise ValueError("rdf_mixed_fossil_co2_factor must be non-negative")
        if self.rdf_mixed_biogenic_co2_factor_t_per_mwh < 0.0:
            raise ValueError("rdf_mixed_biogenic_co2_factor must be non-negative")
        if not 0.0 <= self.rdf_biogenic_co2_accounting_share <= 1.0:
            raise ValueError("rdf_biogenic_co2_accounting_share must be between 0 and 1")

        block.max_heat_out = pyo.Param(initialize=max_heat_mwh)
        block.min_heat_out = pyo.Param(initialize=min_heat_mwh)
        block.max_power = pyo.Param(initialize=max_power_mwh)
        block.min_power = pyo.Param(initialize=min_power_mwh)
        block.specific_heat_demand = pyo.Param(initialize=self.specific_heat_demand_mwh_per_t)
        block.specific_electricity_aux = pyo.Param(
            initialize=self.specific_electricity_aux_mwh_per_t
        )
        block.eta_electric = pyo.Param(initialize=self.eta_electric)
        block.eta_fossil = pyo.Param(initialize=self.eta_fossil)
        block.fossil_ng_share = pyo.Param(initialize=self.fossil_ng_share, within=pyo.UnitInterval)
        block.biomass_share = pyo.Param(initialize=self.biomass_share, within=pyo.UnitInterval)
        block.rdf_share = pyo.Param(initialize=self.rdf_share, within=pyo.UnitInterval)
        block.ramp_up = pyo.Param(initialize=ramp_up * dt_hours)
        block.ramp_down = pyo.Param(initialize=ramp_down * dt_hours)
        block.natural_gas_co2_factor = pyo.Param(initialize=self.natural_gas_co2_factor_t_per_mwh)
        block.coal_co2_factor = pyo.Param(initialize=self.coal_co2_factor_t_per_mwh)
        block.biomass_co2_factor = pyo.Param(initialize=self.biomass_co2_factor_t_per_mwh)
        block.biomass_co2_accounting_share = pyo.Param(
            initialize=self.biomass_co2_accounting_share, within=pyo.UnitInterval
        )
        block.rdf_mixed_fossil_co2_factor = pyo.Param(
            initialize=self.rdf_mixed_fossil_co2_factor_t_per_mwh
        )
        block.rdf_mixed_biogenic_co2_factor = pyo.Param(
            initialize=self.rdf_mixed_biogenic_co2_factor_t_per_mwh
        )
        block.rdf_biogenic_co2_accounting_share = pyo.Param(
            initialize=self.rdf_biogenic_co2_accounting_share, within=pyo.UnitInterval
        )
        block.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        block.min_down_steps = pyo.Param(initialize=self.min_down_steps)
        initial_status = int(
            context.get("initial_operational_status", self.initial_operational_status)
        )
        default_consecutive_steps = max(self.min_operating_steps, self.min_down_steps, 1)
        block.initial_operational_status = pyo.Param(initialize=initial_status)
        block.initial_consecutive_status_steps = pyo.Param(
            initialize=int(
                context.get("initial_consecutive_status_steps", default_consecutive_steps)
            )
        )
        block.initial_heat_out = pyo.Param(initialize=float(context.get("initial_heat_out", 0.0)))

    def _add_common_variables(self, block: pyo.Block, time_steps: pyo.Set) -> None:
        """Pyomo Components:

        - **Variables**: heat and fuel flows every stage has (``heat_out``,
          ``power_in``, ``aux_power_in``, ``natural_gas_in``, ``coal_in``,
          ``biomass_in``, ``rdf_in``, ``hydrogen_in``, ``fossil_in``,
          ``combustion_in``), the
          stage's throughput output, physical/accounted CO2, operating cost, and the
          commitment binaries.
        """
        max_heat_mwh = float(pyo.value(block.max_heat_out))
        max_power_mwh = float(pyo.value(block.max_power))
        block.heat_out = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, max_heat_mwh)
        )
        block.power_in = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, max_power_mwh)
        )
        block.aux_power_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.natural_gas_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.coal_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.biomass_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.rdf_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.hydrogen_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.fossil_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.combustion_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        setattr(block, self._output_var_name, pyo.Var(time_steps, within=pyo.NonNegativeReals))
        block.co2_process = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_energy = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_fossil = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_biogenic = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_biomass = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_rdf = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_rdf_fossil = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_rdf_biogenic = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_physical = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_priced = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_emission = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)
        block.operational_status = pyo.Var(time_steps, within=pyo.Binary)
        block.start_up = pyo.Var(time_steps, within=pyo.Binary)
        block.shut_down = pyo.Var(time_steps, within=pyo.Binary)

    def _add_stage_parameters(self, block: pyo.Block) -> None:
        """Hook: Params beyond the common set, e.g. the calciner's own calcination
        factor. Default: none.

        Kept separate from ``_add_common_parameters`` so a stage-specific coefficient is
        materialised as a real Pyomo Param on the block - inspectable the same way every
        other coefficient is - rather than closed over as a plain Python attribute.
        """

    def _add_stage_variables(self, block: pyo.Block, time_steps: pyo.Set) -> None:
        """Hook: Vars beyond the common set. Default: none."""

    def _external_heat_expr(self, block: pyo.Block, t: int) -> pyo.Expression | float:
        """Hook: heat entering from outside the stage. Default: none."""
        return 0.0

    def _add_stage_constraints(
        self, block: pyo.Block, time_steps: pyo.Set, output: pyo.Var
    ) -> None:
        """Hook: what this stage's output equals, and its own process CO2, if any.

        Every concrete stage overrides this, even where the body barely differs from
        another stage's - a preheater and a kiln both drive their output straight off
        ``heat_out``, but each writes that ``output_from_heat`` constraint out for
        itself rather than sharing it, so a reader never has to leave the class to see
        what it does. Default: none.
        """

    def _additional_operating_cost_expr(
        self, model: pyo.ConcreteModel, block: pyo.Block, t: int
    ) -> pyo.Expression | float:
        """Hook: cost beyond fuel, electricity and CO2, e.g. an oxyfuel stage's oxygen
        supply. Default: none."""
        return 0.0

    def _add_firing_constraints(
        self, block: pyo.Block, time_steps: pyo.Set, output: pyo.Var
    ) -> None:
        """Pyomo Components:

        - **Constraints**: ``heat_balance`` (fuel-switchable heat generation, plus any
          external heat), fuel-exclusivity for the stage's ``fuel_type``, and the
          fossil natural-gas/coal split where fossil fuel is available. Linking that
          heat to the stage's own throughput is each stage's own concern - see
          ``_add_stage_constraints``.
        """
        fuel_type = self.fuel_type

        @block.Constraint(time_steps)
        def heat_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            if fuel_type == CEMENT_ELECTRICITY:
                generated_heat = b.power_in[t] * b.eta_electric
            elif fuel_type == CEMENT_FOSSIL:
                generated_heat = b.combustion_in[t] * b.eta_fossil
            elif fuel_type == CEMENT_HYBRID_ELECTRICITY_FOSSIL:
                generated_heat = b.power_in[t] * b.eta_electric + b.combustion_in[t] * b.eta_fossil
            else:
                generated_heat = b.hydrogen_in[t] * b.eta_fossil
            return b.heat_out[t] == generated_heat + self._external_heat_expr(b, t)

        if fuel_type == CEMENT_ELECTRICITY:

            @block.Constraint(time_steps)
            def zero_non_electric_inputs(b: pyo.Block, t: int) -> pyo.Constraint:
                return (
                    b.natural_gas_in[t]
                    + b.coal_in[t]
                    + b.biomass_in[t]
                    + b.rdf_in[t]
                    + b.hydrogen_in[t]
                    + b.fossil_in[t]
                    + b.combustion_in[t]
                    == 0
                )

        elif fuel_type == CEMENT_FOSSIL:

            @block.Constraint(time_steps)
            def zero_electric_and_hydrogen_inputs(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.power_in[t] + b.hydrogen_in[t] == 0

        elif fuel_type == CEMENT_HYBRID_ELECTRICITY_FOSSIL:

            @block.Constraint(time_steps)
            def zero_hydrogen_input(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.hydrogen_in[t] == 0

        else:

            @block.Constraint(time_steps)
            def zero_electric_and_fossil_inputs(b: pyo.Block, t: int) -> pyo.Constraint:
                return (
                    b.power_in[t]
                    + b.natural_gas_in[t]
                    + b.coal_in[t]
                    + b.biomass_in[t]
                    + b.rdf_in[t]
                    + b.fossil_in[t]
                    + b.combustion_in[t]
                    == 0
                )

        if fuel_type in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:

            @block.Constraint(time_steps)
            def combustion_sum_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.combustion_in[t] == b.biomass_in[t] + b.rdf_in[t] + b.fossil_in[t]

            @block.Constraint(time_steps)
            def biomass_split_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.biomass_in[t] == b.biomass_share * b.combustion_in[t]

            @block.Constraint(time_steps)
            def rdf_split_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.rdf_in[t] == b.rdf_share * b.combustion_in[t]

            @block.Constraint(time_steps)
            def fossil_sum_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.fossil_in[t] == b.natural_gas_in[t] + b.coal_in[t]

            @block.Constraint(time_steps)
            def fossil_split_natural_gas(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.natural_gas_in[t] == b.fossil_ng_share * b.fossil_in[t]

            @block.Constraint(time_steps)
            def fossil_split_coal(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.coal_in[t] == (1.0 - b.fossil_ng_share) * b.fossil_in[t]

    def _add_aux_and_operational_constraints(
        self, block: pyo.Block, time_steps: pyo.Set, output: pyo.Var
    ) -> None:
        """Pyomo Components:

        - **Constraints**: auxiliary power drawn from throughput, the heat/commitment
          coupling (``max_heat_if_on``/``min_heat_if_on`` - the stage may run anywhere
          between its turndown floor and its rating while on, and must be at 0 while
          off), heat ramp limits, and the full unit-commitment sub-model (state
          transition, minimum up/down time, and any residual time inherited from the
          previous rolling window).
        """
        ordered_steps = list(time_steps)

        @block.Constraint(time_steps)
        def auxiliary_power_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.aux_power_in[t] == output[t] * b.specific_electricity_aux

        @block.Constraint(time_steps)
        def max_heat_if_on(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.heat_out[t] <= b.max_heat_out * b.operational_status[t]

        @block.Constraint(time_steps)
        def min_heat_if_on(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.heat_out[t] >= b.min_heat_out * b.operational_status[t]

        @block.Constraint(time_steps)
        def min_power_if_on(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.power_in[t] >= b.min_power * b.operational_status[t]

        @block.Constraint(time_steps)
        def heat_ramp_up(b: pyo.Block, t: int) -> pyo.Constraint:
            position = ordered_steps.index(t)
            previous = (
                b.initial_heat_out if position == 0 else b.heat_out[ordered_steps[position - 1]]
            )
            return b.heat_out[t] - previous <= b.ramp_up

        @block.Constraint(time_steps)
        def heat_ramp_down(b: pyo.Block, t: int) -> pyo.Constraint:
            position = ordered_steps.index(t)
            if position == 0:
                return b.initial_heat_out - b.heat_out[t] <= b.ramp_down
            previous = b.heat_out[ordered_steps[position - 1]]
            return previous - b.heat_out[t] <= b.ramp_down

        @block.Constraint(time_steps)
        def state_transition_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            position = ordered_steps.index(t)
            previous = (
                b.initial_operational_status
                if position == 0
                else b.operational_status[ordered_steps[position - 1]]
            )
            return b.operational_status[t] - previous == b.start_up[t] - b.shut_down[t]

        @block.Constraint(time_steps)
        def prevent_simultaneous_startup_shutdown(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.start_up[t] + b.shut_down[t] <= 1

        @block.Constraint(time_steps)
        def min_operating_time_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            min_steps = int(pyo.value(b.min_operating_steps))
            if min_steps <= 1:
                return pyo.Constraint.Skip
            position = ordered_steps.index(t)
            if position + 1 < min_steps:
                return pyo.Constraint.Skip
            window = ordered_steps[position - min_steps + 1 : position + 1]
            return sum(b.start_up[i] for i in window) <= b.operational_status[t]

        @block.Constraint(time_steps)
        def min_down_time_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            min_steps = int(pyo.value(b.min_down_steps))
            if min_steps <= 1:
                return pyo.Constraint.Skip
            position = ordered_steps.index(t)
            if position + 1 < min_steps:
                return pyo.Constraint.Skip
            window = ordered_steps[position - min_steps + 1 : position + 1]
            return sum(b.shut_down[i] for i in window) <= 1 - b.operational_status[t]

        # Minimum up/down time that the previous rolling window had not yet served out. The
        # constraints above only see this window, so without this the stage could switch at
        # every commit boundary no matter how long it had just been running or standing.
        initial_status = int(pyo.value(block.initial_operational_status))
        residual_steps = max(
            0,
            int(pyo.value(block.min_operating_steps if initial_status else block.min_down_steps))
            - int(pyo.value(block.initial_consecutive_status_steps)),
        )
        if residual_steps:

            @block.Constraint(time_steps)
            def residual_status_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
                position = ordered_steps.index(t)
                if position >= residual_steps:
                    return pyo.Constraint.Skip
                return b.operational_status[t] == initial_status

    def _add_emission_and_cost_constraints(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        output: pyo.Var,
    ) -> None:
        """Pyomo Components:

        - **Constraints**: energy CO2, its sum with the stage's own process CO2 (set in
          ``_add_stage_constraints``), and operating cost (aux power always billed;
          primary fuel billed for whichever commodity the stage's ``fuel_type`` draws
          on; plus any stage-specific extra cost, e.g. an oxyfuel stage's oxygen
          supply).
        """
        fuel_type = self.fuel_type

        @block.Constraint(time_steps)
        def energy_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_energy[t] == b.co2_fossil[t] + b.co2_biogenic[t]

        @block.Constraint(time_steps)
        def biomass_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_biomass[t] == b.biomass_in[t] * b.biomass_co2_factor

        @block.Constraint(time_steps)
        def rdf_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_rdf[t] == b.co2_rdf_fossil[t] + b.co2_rdf_biogenic[t]

        @block.Constraint(time_steps)
        def rdf_fossil_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_rdf_fossil[t] == b.rdf_in[t] * b.rdf_mixed_fossil_co2_factor

        @block.Constraint(time_steps)
        def rdf_biogenic_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_rdf_biogenic[t] == b.rdf_in[t] * b.rdf_mixed_biogenic_co2_factor

        @block.Constraint(time_steps)
        def fossil_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_fossil[t] == (
                b.natural_gas_in[t] * b.natural_gas_co2_factor
                + b.coal_in[t] * b.coal_co2_factor
                + b.co2_rdf_fossil[t]
            )

        @block.Constraint(time_steps)
        def biogenic_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_biogenic[t] == b.co2_biomass[t] + b.co2_rdf_biogenic[t]

        @block.Constraint(time_steps)
        def physical_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_physical[t] == b.co2_process[t] + b.co2_fossil[t] + b.co2_biogenic[t]

        @block.Constraint(time_steps)
        def priced_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced[t] == (
                b.co2_process[t]
                + b.co2_fossil[t]
                + b.co2_biomass[t] * b.biomass_co2_accounting_share
                + b.co2_rdf_biogenic[t] * b.rdf_biogenic_co2_accounting_share
            )

        @block.Constraint(time_steps)
        def co2_emission_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_emission[t] == b.co2_physical[t]

        @block.Constraint(time_steps)
        def operating_cost_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            cost = b.aux_power_in[t] * model.electricity_price[t]
            if fuel_type in {CEMENT_ELECTRICITY, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
                cost += b.power_in[t] * model.electricity_price[t]
            if fuel_type in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
                cost += (
                    b.natural_gas_in[t] * model.natural_gas_price[t]
                    + b.coal_in[t] * model.coal_price[t]
                    + b.biomass_in[t] * model.biomass_price[t]
                    + b.rdf_in[t] * model.rdf_price[t]
                )
            if fuel_type == HYDROGEN:
                cost += b.hydrogen_in[t] * model.hydrogen_price[t]
            cost += b.co2_priced[t] * model.co2_price[t]
            cost += self._additional_operating_cost_expr(model, b, t)
            return b.operating_cost[t] == cost


@dataclass
class CementPreheater(CementKilnLineStage):
    """Fuel-switchable preheater converting raw-meal heat duty into raw meal output."""

    _output_var_name: ClassVar[str] = "raw_meal_out"

    max_heat_out_mw: float
    specific_heat_demand_mwh_per_t: float
    min_heat_out_mw: float = 0.0
    fuel_type: str = CEMENT_ELECTRICITY
    eta_electric: float = 0.98
    eta_fossil: float = 0.90
    fossil_ng_share: float = CementKilnLineStage.default_fossil_ng_share
    biomass_share: float = CementKilnLineStage.default_biomass_share
    rdf_share: float = CementKilnLineStage.default_rdf_share
    max_electric_power_mw: float | None = None
    min_electric_power_mw: float = 0.0
    specific_electricity_aux_mwh_per_t: float = 0.0
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    natural_gas_co2_factor_t_per_mwh: float = 0.202
    coal_co2_factor_t_per_mwh: float = 0.341
    biomass_co2_factor_t_per_mwh: float = 0.0
    biomass_co2_accounting_share: float = 0.0
    rdf_mixed_fossil_co2_factor_t_per_mwh: float = (
        CementKilnLineStage.default_rdf_mixed_fossil_co2_factor_t_per_mwh
    )
    rdf_mixed_biogenic_co2_factor_t_per_mwh: float = 0.0
    rdf_biogenic_co2_accounting_share: float = 0.0
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1

    @classmethod
    def from_row(cls, row: pd.Series) -> CementPreheater:
        max_heat = _as_float(
            _first_present(row.get("max_heat_out"), row.get("max_power")),
            "max_heat_out",
        )
        default_ng_share, default_biomass_share, default_rdf_share = cls._fuel_mix_defaults(
            row, CEMENT_ELECTRICITY
        )
        biomass_share = _as_float(
            row.get("biomass_share"), "biomass_share", default=default_biomass_share
        )
        rdf_share = _as_float(
            row.get("rdf_share"),
            "rdf_share",
            default=default_rdf_share,
        )
        return cls(
            max_heat_out_mw=max_heat,
            specific_heat_demand_mwh_per_t=_as_float(
                row.get("specific_heat_demand"), "specific_heat_demand"
            ),
            min_heat_out_mw=_as_float(row.get("min_heat_out"), "min_heat_out", default=0.0),
            fuel_type=_cement_fuel_type(row, "CementPreheater", default=CEMENT_ELECTRICITY),
            eta_electric=_as_float(row.get("eta_electric"), "eta_electric", default=0.98),
            eta_fossil=_as_float(row.get("eta_fossil"), "eta_fossil", default=0.90),
            fossil_ng_share=_as_float(
                row.get("fossil_ng_share"), "fossil_ng_share", default=default_ng_share
            ),
            biomass_share=biomass_share,
            rdf_share=rdf_share,
            max_electric_power_mw=_cement_max_electric_power(row),
            min_electric_power_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            specific_electricity_aux_mwh_per_t=_as_float(
                row.get("specific_electricity_aux"), "specific_electricity_aux", default=0.0
            ),
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up")),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down")),
            natural_gas_co2_factor_t_per_mwh=_as_float(
                _first_present(row.get("natural_gas_co2_factor"), row.get("ng_co2_factor")),
                "natural_gas_co2_factor",
                default=0.202,
            ),
            coal_co2_factor_t_per_mwh=_as_float(
                row.get("coal_co2_factor"), "coal_co2_factor", default=0.341
            ),
            biomass_co2_factor_t_per_mwh=_as_float(
                row.get("biomass_co2_factor"),
                "biomass_co2_factor",
                default=None if biomass_share > 0.0 else 0.0,
            ),
            biomass_co2_accounting_share=_as_float(
                row.get("biomass_co2_accounting_share"),
                "biomass_co2_accounting_share",
                default=0.0,
            ),
            rdf_mixed_fossil_co2_factor_t_per_mwh=_as_float(
                row.get("rdf_mixed_fossil_co2_factor"),
                "rdf_mixed_fossil_co2_factor",
                default=cls.default_rdf_mixed_fossil_co2_factor_t_per_mwh,
            ),
            rdf_mixed_biogenic_co2_factor_t_per_mwh=_as_float(
                row.get("rdf_mixed_biogenic_co2_factor"),
                "rdf_mixed_biogenic_co2_factor",
                default=None if rdf_share > 0.0 else 0.0,
            ),
            rdf_biogenic_co2_accounting_share=_as_float(
                _first_present(
                    row.get("rdf_biogenic_co2_accounting_share"),
                    row.get("rdf_biogenic_accounting_share"),
                ),
                "rdf_biogenic_co2_accounting_share",
                default=0.0,
            ),
            min_operating_steps=_as_int(
                _first_present(row.get("min_operating_steps"), row.get("min_operating_time")),
                default=0,
            ),
            min_down_steps=_as_int(
                _first_present(row.get("min_down_steps"), row.get("min_down_time")), default=0
            ),
            initial_operational_status=_as_int(row.get("initial_operational_status"), default=1),
        )

    def _add_stage_variables(self, block: pyo.Block, time_steps: pyo.Set) -> None:
        block.external_heat_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    def _external_heat_expr(self, block: pyo.Block, t: int) -> pyo.Expression:
        return block.external_heat_in[t]

    def _add_stage_constraints(
        self, block: pyo.Block, time_steps: pyo.Set, output: pyo.Var
    ) -> None:
        """Pyomo Components:

        - **Constraints**: ``output_from_heat``, raw meal produced from the preheater's
          own ``heat_out``. No process CO2 - preheating is not a chemical reaction.
        """

        @block.Constraint(time_steps)
        def output_from_heat(b: pyo.Block, t: int) -> pyo.Constraint:
            return output[t] == b.heat_out[t] / b.specific_heat_demand

        @block.Constraint(time_steps)
        def process_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_process[t] == 0.0


@dataclass
class SimpleCementCalciner(CementKilnLineStage):
    """Fuel-switchable calciner with calcination process CO2."""

    max_heat_out_mw: float
    specific_heat_demand_mwh_per_t: float
    min_heat_out_mw: float = 0.0
    fuel_type: str = CEMENT_ELECTRICITY
    eta_electric: float = 0.95
    eta_fossil: float = 0.90
    fossil_ng_share: float = CementKilnLineStage.default_fossil_ng_share
    biomass_share: float = CementKilnLineStage.default_biomass_share
    rdf_share: float = CementKilnLineStage.default_rdf_share
    max_electric_power_mw: float | None = None
    min_electric_power_mw: float = 0.0
    specific_electricity_aux_mwh_per_t: float = 0.0
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    calcination_emission_factor_t_per_t: float = 0.525
    natural_gas_co2_factor_t_per_mwh: float = 0.202
    coal_co2_factor_t_per_mwh: float = 0.341
    biomass_co2_factor_t_per_mwh: float = 0.0
    biomass_co2_accounting_share: float = 0.0
    rdf_mixed_fossil_co2_factor_t_per_mwh: float = (
        CementKilnLineStage.default_rdf_mixed_fossil_co2_factor_t_per_mwh
    )
    rdf_mixed_biogenic_co2_factor_t_per_mwh: float = 0.0
    rdf_biogenic_co2_accounting_share: float = 0.0
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1

    @classmethod
    def from_row(cls, row: pd.Series) -> SimpleCementCalciner:
        max_heat = _as_float(
            _first_present(row.get("max_heat_out"), row.get("max_power")),
            "max_heat_out",
        )
        default_ng_share, default_biomass_share, default_rdf_share = cls._fuel_mix_defaults(
            row, CEMENT_ELECTRICITY
        )
        biomass_share = _as_float(
            row.get("biomass_share"), "biomass_share", default=default_biomass_share
        )
        rdf_share = _as_float(
            row.get("rdf_share"),
            "rdf_share",
            default=default_rdf_share,
        )
        return cls(
            max_heat_out_mw=max_heat,
            specific_heat_demand_mwh_per_t=_as_float(
                row.get("specific_heat_demand"), "specific_heat_demand"
            ),
            min_heat_out_mw=_as_float(row.get("min_heat_out"), "min_heat_out", default=0.0),
            fuel_type=_cement_fuel_type(row, "SimpleCementCalciner", default=CEMENT_ELECTRICITY),
            eta_electric=_as_float(row.get("eta_electric"), "eta_electric", default=0.95),
            eta_fossil=_as_float(row.get("eta_fossil"), "eta_fossil", default=0.90),
            fossil_ng_share=_as_float(
                row.get("fossil_ng_share"), "fossil_ng_share", default=default_ng_share
            ),
            biomass_share=biomass_share,
            rdf_share=rdf_share,
            max_electric_power_mw=_cement_max_electric_power(row),
            min_electric_power_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            specific_electricity_aux_mwh_per_t=_as_float(
                row.get("specific_electricity_aux"), "specific_electricity_aux", default=0.0
            ),
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up")),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down")),
            calcination_emission_factor_t_per_t=_as_float(
                row.get("calcination_emission_factor"),
                "calcination_emission_factor",
                default=0.525,
            ),
            natural_gas_co2_factor_t_per_mwh=_as_float(
                _first_present(row.get("natural_gas_co2_factor"), row.get("ng_co2_factor")),
                "natural_gas_co2_factor",
                default=0.202,
            ),
            coal_co2_factor_t_per_mwh=_as_float(
                row.get("coal_co2_factor"), "coal_co2_factor", default=0.341
            ),
            biomass_co2_factor_t_per_mwh=_as_float(
                row.get("biomass_co2_factor"),
                "biomass_co2_factor",
                default=None if biomass_share > 0.0 else 0.0,
            ),
            biomass_co2_accounting_share=_as_float(
                row.get("biomass_co2_accounting_share"),
                "biomass_co2_accounting_share",
                default=0.0,
            ),
            rdf_mixed_fossil_co2_factor_t_per_mwh=_as_float(
                row.get("rdf_mixed_fossil_co2_factor"),
                "rdf_mixed_fossil_co2_factor",
                default=cls.default_rdf_mixed_fossil_co2_factor_t_per_mwh,
            ),
            rdf_mixed_biogenic_co2_factor_t_per_mwh=_as_float(
                row.get("rdf_mixed_biogenic_co2_factor"),
                "rdf_mixed_biogenic_co2_factor",
                default=None if rdf_share > 0.0 else 0.0,
            ),
            rdf_biogenic_co2_accounting_share=_as_float(
                _first_present(
                    row.get("rdf_biogenic_co2_accounting_share"),
                    row.get("rdf_biogenic_accounting_share"),
                ),
                "rdf_biogenic_co2_accounting_share",
                default=0.0,
            ),
            min_operating_steps=_as_int(
                _first_present(row.get("min_operating_steps"), row.get("min_operating_time")),
                default=0,
            ),
            min_down_steps=_as_int(
                _first_present(row.get("min_down_steps"), row.get("min_down_time")), default=0
            ),
            initial_operational_status=_as_int(row.get("initial_operational_status"), default=1),
        )

    def _add_stage_parameters(self, block: pyo.Block) -> None:
        """Pyomo Components:

        - **Parameters**: ``calcination_emission_factor``, t process CO2 per t clinker.
        """
        block.calcination_emission_factor = pyo.Param(
            initialize=self.calcination_emission_factor_t_per_t
        )

    def _add_stage_variables(self, block: pyo.Block, time_steps: pyo.Set) -> None:
        block.effective_heat_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    def _add_stage_constraints(
        self, block: pyo.Block, time_steps: pyo.Set, output: pyo.Var
    ) -> None:
        """Pyomo Components:

        - **Constraints**: ``output_from_heat``, clinker calcined from
          ``effective_heat_in`` (the calciner's own heat plus any thermal-storage
          discharge - see ``CementPlant.initialize_process_sequence``), and
          ``process_co2_constraint``, the calcination reaction's CO2, proportional to
          clinker output regardless of what fired the heat.
        """

        @block.Constraint(time_steps)
        def output_from_heat(b: pyo.Block, t: int) -> pyo.Constraint:
            return output[t] == b.effective_heat_in[t] / b.specific_heat_demand

        @block.Constraint(time_steps)
        def process_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_process[t] == output[t] * b.calcination_emission_factor


@dataclass
class LEILACCementCalciner(SimpleCementCalciner):
    """Indirectly heated calciner with direct separation of process CO2.

    The complete heat, fuel-switching, clinker-output, ramping, commitment, and
    fuel-cost formulation is inherited from ``SimpleCementCalciner``. Direct
    separation applies only to calcination CO2; combustion CO2 remains emitted.
    """

    direct_separation_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.direct_separation_efficiency <= 1.0:
            raise ValueError("direct_separation_efficiency must be between 0 and 1")

    @classmethod
    def from_row(cls, row: pd.Series) -> LEILACCementCalciner:
        """Parse the shared calciner fields via the parent, then separation efficiency."""
        base = SimpleCementCalciner.from_row(row)
        return cls(
            **vars(base),
            direct_separation_efficiency=_as_float(
                _first_present(
                    row.get("direct_separation_efficiency"),
                    row.get("process_co2_separation_efficiency"),
                ),
                "direct_separation_efficiency",
                default=1.0,
            ),
        )

    def _add_stage_parameters(self, block: pyo.Block) -> None:
        """Add the parent's calcination factor and the direct-separation fraction."""
        super()._add_stage_parameters(block)
        block.direct_separation_efficiency = pyo.Param(
            initialize=self.direct_separation_efficiency,
            within=pyo.UnitInterval,
        )

    def _add_stage_variables(self, block: pyo.Block, time_steps: pyo.Set) -> None:
        super()._add_stage_variables(block, time_steps)
        block.co2_separated = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_process_residual = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    def _add_emission_and_cost_constraints(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        output: pyo.Var,
    ) -> None:
        """Separate process CO2 while retaining all fuel-combustion emissions."""
        fuel_type = self.fuel_type

        @block.Constraint(time_steps)
        def separated_process_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_separated[t] == (b.co2_process[t] * b.direct_separation_efficiency)

        @block.Constraint(time_steps)
        def residual_process_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_process_residual[t] == b.co2_process[t] - b.co2_separated[t]

        @block.Constraint(time_steps)
        def energy_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_energy[t] == b.co2_fossil[t] + b.co2_biogenic[t]

        @block.Constraint(time_steps)
        def biomass_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_biomass[t] == b.biomass_in[t] * b.biomass_co2_factor

        @block.Constraint(time_steps)
        def rdf_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_rdf[t] == b.co2_rdf_fossil[t] + b.co2_rdf_biogenic[t]

        @block.Constraint(time_steps)
        def rdf_fossil_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_rdf_fossil[t] == b.rdf_in[t] * b.rdf_mixed_fossil_co2_factor

        @block.Constraint(time_steps)
        def rdf_biogenic_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_rdf_biogenic[t] == b.rdf_in[t] * b.rdf_mixed_biogenic_co2_factor

        @block.Constraint(time_steps)
        def fossil_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_fossil[t] == (
                b.natural_gas_in[t] * b.natural_gas_co2_factor
                + b.coal_in[t] * b.coal_co2_factor
                + b.co2_rdf_fossil[t]
            )

        @block.Constraint(time_steps)
        def biogenic_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_biogenic[t] == b.co2_biomass[t] + b.co2_rdf_biogenic[t]

        @block.Constraint(time_steps)
        def physical_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_physical[t] == (
                b.co2_process_residual[t] + b.co2_fossil[t] + b.co2_biogenic[t]
            )

        @block.Constraint(time_steps)
        def priced_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced[t] == (
                b.co2_process_residual[t]
                + b.co2_fossil[t]
                + b.co2_biomass[t] * b.biomass_co2_accounting_share
                + b.co2_rdf_biogenic[t] * b.rdf_biogenic_co2_accounting_share
            )

        @block.Constraint(time_steps)
        def co2_emission_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_emission[t] == b.co2_physical[t]

        @block.Constraint(time_steps)
        def operating_cost_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            cost = b.aux_power_in[t] * model.electricity_price[t]
            if fuel_type in {CEMENT_ELECTRICITY, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
                cost += b.power_in[t] * model.electricity_price[t]
            if fuel_type in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
                cost += (
                    b.natural_gas_in[t] * model.natural_gas_price[t]
                    + b.coal_in[t] * model.coal_price[t]
                    + b.biomass_in[t] * model.biomass_price[t]
                    + b.rdf_in[t] * model.rdf_price[t]
                )
            if fuel_type == HYDROGEN:
                cost += b.hydrogen_in[t] * model.hydrogen_price[t]
            cost += b.co2_priced[t] * model.co2_price[t]
            cost += self._additional_operating_cost_expr(model, b, t)
            return b.operating_cost[t] == cost


@dataclass
class OxyfuelCementCalciner(SimpleCementCalciner):
    """A calciner fired with oxygen instead of air, for CO2 capture.

    Inherits the complete heat, fuel-switching, clinker-output, ramping, commitment,
    CO2-emission, and cost formulation of ``SimpleCementCalciner`` unchanged. It adds
    combustion oxygen demand, any electrolyser-coproduct contribution, and electricity
    for generating the remaining oxygen. The cement plant constrains coproduct oxygen
    to actual electrolyser output, so oxygen can never appear as a free source.
    """

    natural_gas_oxygen_demand_t_per_mwh: float = 0.0
    coal_oxygen_demand_t_per_mwh: float = 0.0
    biomass_oxygen_demand_t_per_mwh: float = 0.0
    rdf_oxygen_demand_t_per_mwh: float = 0.0
    hydrogen_oxygen_demand_t_per_mwh: float = 0.0
    specific_oxygen_electricity_consumption_mwh_per_t: float = 0.0

    def __post_init__(self) -> None:
        """Require an oxygen coefficient for every combustion fuel this fuel_type uses."""
        if self.fuel_type == CEMENT_ELECTRICITY:
            raise ValueError(
                "OxyfuelCementCalciner cannot use fuel_type='electricity'. "
                "Use simple_calciner for a fully electric calciner."
            )
        if self.fuel_type in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
            if (
                self.biomass_share + self.rdf_share < 1.0
                and self.fossil_ng_share > 0.0
                and self.natural_gas_oxygen_demand_t_per_mwh <= 0.0
            ):
                raise ValueError(
                    "natural_gas_oxygen_demand must be positive when the oxyfuel calciner "
                    "uses natural gas."
                )
            if (
                self.biomass_share + self.rdf_share < 1.0
                and self.fossil_ng_share < 1.0
                and self.coal_oxygen_demand_t_per_mwh <= 0.0
            ):
                raise ValueError(
                    "coal_oxygen_demand must be positive when the oxyfuel calciner uses coal."
                )
            if self.biomass_share > 0.0 and self.biomass_oxygen_demand_t_per_mwh <= 0.0:
                raise ValueError(
                    "biomass_oxygen_demand must be positive when the oxyfuel calciner uses biomass."
                )
            if self.rdf_share > 0.0 and self.rdf_oxygen_demand_t_per_mwh <= 0.0:
                raise ValueError(
                    "rdf_oxygen_demand must be positive when the oxyfuel calciner uses RDF."
                )
        if self.fuel_type == HYDROGEN and self.hydrogen_oxygen_demand_t_per_mwh <= 0.0:
            raise ValueError(
                "hydrogen_oxygen_demand must be positive when the oxyfuel calciner uses hydrogen."
            )
        if self.specific_oxygen_electricity_consumption_mwh_per_t < 0.0:
            raise ValueError("specific_oxygen_electricity_consumption must be non-negative")

    @classmethod
    def from_row(cls, row: pd.Series) -> OxyfuelCementCalciner:
        """Parse the shared calciner fields via the parent, then this stage's own."""
        base = SimpleCementCalciner.from_row(row)
        fossil_route = base.fuel_type in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}
        natural_gas_required = (
            fossil_route
            and base.biomass_share + base.rdf_share < 1.0
            and base.fossil_ng_share > 0.0
        )
        coal_required = (
            fossil_route
            and base.biomass_share + base.rdf_share < 1.0
            and base.fossil_ng_share < 1.0
        )
        biomass_required = fossil_route and base.biomass_share > 0.0
        rdf_required = fossil_route and base.rdf_share > 0.0
        hydrogen_required = base.fuel_type == HYDROGEN

        return cls(
            **vars(base),
            natural_gas_oxygen_demand_t_per_mwh=_as_float(
                _first_present(row.get("natural_gas_oxygen_demand"), row.get("ng_oxygen_demand")),
                "natural_gas_oxygen_demand",
                default=None if natural_gas_required else 0.0,
            ),
            coal_oxygen_demand_t_per_mwh=_as_float(
                row.get("coal_oxygen_demand"),
                "coal_oxygen_demand",
                default=None if coal_required else 0.0,
            ),
            biomass_oxygen_demand_t_per_mwh=_as_float(
                row.get("biomass_oxygen_demand"),
                "biomass_oxygen_demand",
                default=None if biomass_required else 0.0,
            ),
            rdf_oxygen_demand_t_per_mwh=_as_float(
                row.get("rdf_oxygen_demand"),
                "rdf_oxygen_demand",
                default=None if rdf_required else 0.0,
            ),
            hydrogen_oxygen_demand_t_per_mwh=_as_float(
                _first_present(row.get("hydrogen_oxygen_demand"), row.get("h2_oxygen_demand")),
                "hydrogen_oxygen_demand",
                default=None if hydrogen_required else 0.0,
            ),
            specific_oxygen_electricity_consumption_mwh_per_t=_as_float(
                row.get("specific_oxygen_electricity_consumption"),
                "specific_oxygen_electricity_consumption",
            ),
        )

    def _add_stage_parameters(self, block: pyo.Block) -> None:
        """Pyomo Components:

        - **Parameters**: the parent's ``calcination_emission_factor``, plus this
          stage's own oxygen-demand coefficients, one per combustion fuel.
        """
        super()._add_stage_parameters(block)
        block.natural_gas_oxygen_demand = pyo.Param(
            initialize=self.natural_gas_oxygen_demand_t_per_mwh
        )
        block.coal_oxygen_demand = pyo.Param(initialize=self.coal_oxygen_demand_t_per_mwh)
        block.biomass_oxygen_demand = pyo.Param(initialize=self.biomass_oxygen_demand_t_per_mwh)
        block.rdf_oxygen_demand = pyo.Param(initialize=self.rdf_oxygen_demand_t_per_mwh)
        block.hydrogen_oxygen_demand = pyo.Param(initialize=self.hydrogen_oxygen_demand_t_per_mwh)
        block.specific_oxygen_electricity_consumption = pyo.Param(
            initialize=self.specific_oxygen_electricity_consumption_mwh_per_t
        )

    def _add_stage_variables(self, block: pyo.Block, time_steps: pyo.Set) -> None:
        super()._add_stage_variables(block, time_steps)
        block.oxygen_demand = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.oxygen_from_electrolyser = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.oxygen_generated = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.electricity_consumption = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    def _add_firing_constraints(
        self, block: pyo.Block, time_steps: pyo.Set, output: pyo.Var
    ) -> None:
        """Add oxygen demand and generation electricity to the firing physics."""
        super()._add_firing_constraints(block, time_steps, output)

        @block.Constraint(time_steps)
        def oxygen_demand_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.oxygen_demand[t] == (
                b.natural_gas_in[t] * b.natural_gas_oxygen_demand
                + b.coal_in[t] * b.coal_oxygen_demand
                + b.biomass_in[t] * b.biomass_oxygen_demand
                + b.rdf_in[t] * b.rdf_oxygen_demand
                + b.hydrogen_in[t] * b.hydrogen_oxygen_demand
            )

        @block.Constraint(time_steps)
        def electrolyser_oxygen_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.oxygen_from_electrolyser[t] <= b.oxygen_demand[t]

        @block.Constraint(time_steps)
        def oxygen_generation_requirement(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.oxygen_generated[t] == (b.oxygen_demand[t] - b.oxygen_from_electrolyser[t])

        @block.Constraint(time_steps)
        def oxygen_electricity_consumption(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.electricity_consumption[t] == (
                b.oxygen_generated[t] * b.specific_oxygen_electricity_consumption
            )

    def _additional_operating_cost_expr(
        self, model: pyo.ConcreteModel, block: pyo.Block, t: int
    ) -> pyo.Expression:
        return block.electricity_consumption[t] * model.electricity_price[t]


@dataclass
class SimpleCementKiln(CementKilnLineStage):
    """Fuel-switchable rotary kiln for final clinkerisation."""

    max_heat_out_mw: float
    specific_heat_demand_mwh_per_t: float
    min_heat_out_mw: float = 0.0
    fuel_type: str = CEMENT_FOSSIL
    eta_electric: float = 0.95
    eta_fossil: float = 0.90
    fossil_ng_share: float = CementKilnLineStage.default_fossil_ng_share
    biomass_share: float = CementKilnLineStage.default_biomass_share
    rdf_share: float = CementKilnLineStage.default_rdf_share
    max_electric_power_mw: float | None = None
    min_electric_power_mw: float = 0.0
    specific_electricity_aux_mwh_per_t: float = 0.0
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    natural_gas_co2_factor_t_per_mwh: float = 0.202
    coal_co2_factor_t_per_mwh: float = 0.341
    biomass_co2_factor_t_per_mwh: float = 0.0
    biomass_co2_accounting_share: float = 0.0
    rdf_mixed_fossil_co2_factor_t_per_mwh: float = (
        CementKilnLineStage.default_rdf_mixed_fossil_co2_factor_t_per_mwh
    )
    rdf_mixed_biogenic_co2_factor_t_per_mwh: float = 0.0
    rdf_biogenic_co2_accounting_share: float = 0.0
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1

    @classmethod
    def from_row(cls, row: pd.Series) -> SimpleCementKiln:
        max_heat = _as_float(
            _first_present(row.get("max_heat_out"), row.get("max_power")),
            "max_heat_out",
        )
        default_ng_share, default_biomass_share, default_rdf_share = cls._fuel_mix_defaults(
            row, CEMENT_FOSSIL
        )
        biomass_share = _as_float(
            row.get("biomass_share"), "biomass_share", default=default_biomass_share
        )
        rdf_share = _as_float(
            row.get("rdf_share"),
            "rdf_share",
            default=default_rdf_share,
        )
        return cls(
            max_heat_out_mw=max_heat,
            specific_heat_demand_mwh_per_t=_as_float(
                row.get("specific_heat_demand"), "specific_heat_demand"
            ),
            min_heat_out_mw=_as_float(row.get("min_heat_out"), "min_heat_out", default=0.0),
            fuel_type=_cement_fuel_type(row, "SimpleCementKiln", default=CEMENT_FOSSIL),
            eta_electric=_as_float(row.get("eta_electric"), "eta_electric", default=0.95),
            eta_fossil=_as_float(row.get("eta_fossil"), "eta_fossil", default=0.90),
            fossil_ng_share=_as_float(
                row.get("fossil_ng_share"), "fossil_ng_share", default=default_ng_share
            ),
            biomass_share=biomass_share,
            rdf_share=rdf_share,
            max_electric_power_mw=_cement_max_electric_power(row),
            min_electric_power_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            specific_electricity_aux_mwh_per_t=_as_float(
                row.get("specific_electricity_aux"), "specific_electricity_aux", default=0.0
            ),
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up")),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down")),
            natural_gas_co2_factor_t_per_mwh=_as_float(
                _first_present(row.get("natural_gas_co2_factor"), row.get("ng_co2_factor")),
                "natural_gas_co2_factor",
                default=0.202,
            ),
            coal_co2_factor_t_per_mwh=_as_float(
                row.get("coal_co2_factor"), "coal_co2_factor", default=0.341
            ),
            biomass_co2_factor_t_per_mwh=_as_float(
                row.get("biomass_co2_factor"),
                "biomass_co2_factor",
                default=None if biomass_share > 0.0 else 0.0,
            ),
            biomass_co2_accounting_share=_as_float(
                row.get("biomass_co2_accounting_share"),
                "biomass_co2_accounting_share",
                default=0.0,
            ),
            rdf_mixed_fossil_co2_factor_t_per_mwh=_as_float(
                row.get("rdf_mixed_fossil_co2_factor"),
                "rdf_mixed_fossil_co2_factor",
                default=cls.default_rdf_mixed_fossil_co2_factor_t_per_mwh,
            ),
            rdf_mixed_biogenic_co2_factor_t_per_mwh=_as_float(
                row.get("rdf_mixed_biogenic_co2_factor"),
                "rdf_mixed_biogenic_co2_factor",
                default=None if rdf_share > 0.0 else 0.0,
            ),
            rdf_biogenic_co2_accounting_share=_as_float(
                _first_present(
                    row.get("rdf_biogenic_co2_accounting_share"),
                    row.get("rdf_biogenic_accounting_share"),
                ),
                "rdf_biogenic_co2_accounting_share",
                default=0.0,
            ),
            min_operating_steps=_as_int(
                _first_present(row.get("min_operating_steps"), row.get("min_operating_time")),
                default=0,
            ),
            min_down_steps=_as_int(
                _first_present(row.get("min_down_steps"), row.get("min_down_time")), default=0
            ),
            initial_operational_status=_as_int(row.get("initial_operational_status"), default=1),
        )

    # No external heat: the base class's default (none) is correct as-is.

    def _add_stage_constraints(
        self, block: pyo.Block, time_steps: pyo.Set, output: pyo.Var
    ) -> None:
        """Pyomo Components:

        - **Constraints**: ``output_from_heat``, clinker calcined from the kiln's own
          ``heat_out`` directly - a bare kiln performs the calcination reaction itself,
          with no separate calciner upstream. No process CO2 modelled here.
        """

        @block.Constraint(time_steps)
        def output_from_heat(b: pyo.Block, t: int) -> pyo.Constraint:
            return output[t] == b.heat_out[t] / b.specific_heat_demand

        @block.Constraint(time_steps)
        def process_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_process[t] == 0.0


@dataclass
class OxyfuelCementKiln(SimpleCementKiln):
    """Oxyfuel rotary kiln with electricity-based oxygen generation.

    The complete kiln heat, fuel-switching, auxiliary-load, ramping, commitment,
    combustion-CO2, and cost formulation is inherited from ``SimpleCementKiln``.
    Oxygen demand follows actual fuel use. Electrolyser coproduct can cover part of
    that demand; the remainder is generated at the configured electricity intensity.
    """

    natural_gas_oxygen_demand_t_per_mwh: float = 0.0
    coal_oxygen_demand_t_per_mwh: float = 0.0
    biomass_oxygen_demand_t_per_mwh: float = 0.0
    rdf_oxygen_demand_t_per_mwh: float = 0.0
    hydrogen_oxygen_demand_t_per_mwh: float = 0.0
    specific_oxygen_electricity_consumption_mwh_per_t: float = 0.0

    def __post_init__(self) -> None:
        if self.fuel_type == CEMENT_ELECTRICITY:
            raise ValueError(
                "OxyfuelCementKiln cannot use fuel_type='electricity'. "
                "Use simple_kiln for a fully electric kiln."
            )
        if self.fuel_type in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
            if (
                self.biomass_share + self.rdf_share < 1.0
                and self.fossil_ng_share > 0.0
                and self.natural_gas_oxygen_demand_t_per_mwh <= 0.0
            ):
                raise ValueError(
                    "natural_gas_oxygen_demand must be positive when the oxyfuel kiln "
                    "uses natural gas."
                )
            if (
                self.biomass_share + self.rdf_share < 1.0
                and self.fossil_ng_share < 1.0
                and self.coal_oxygen_demand_t_per_mwh <= 0.0
            ):
                raise ValueError(
                    "coal_oxygen_demand must be positive when the oxyfuel kiln uses coal."
                )
            if self.biomass_share > 0.0 and self.biomass_oxygen_demand_t_per_mwh <= 0.0:
                raise ValueError(
                    "biomass_oxygen_demand must be positive when the oxyfuel kiln uses biomass."
                )
            if self.rdf_share > 0.0 and self.rdf_oxygen_demand_t_per_mwh <= 0.0:
                raise ValueError(
                    "rdf_oxygen_demand must be positive when the oxyfuel kiln uses RDF."
                )
        if self.fuel_type == HYDROGEN and self.hydrogen_oxygen_demand_t_per_mwh <= 0.0:
            raise ValueError(
                "hydrogen_oxygen_demand must be positive when the oxyfuel kiln uses hydrogen."
            )
        if self.specific_oxygen_electricity_consumption_mwh_per_t < 0.0:
            raise ValueError("specific_oxygen_electricity_consumption must be non-negative")

    @classmethod
    def from_row(cls, row: pd.Series) -> OxyfuelCementKiln:
        """Parse the shared kiln fields via the parent, then oxygen parameters."""
        base = SimpleCementKiln.from_row(row)
        fossil_route = base.fuel_type in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}
        natural_gas_required = (
            fossil_route
            and base.biomass_share + base.rdf_share < 1.0
            and base.fossil_ng_share > 0.0
        )
        coal_required = (
            fossil_route
            and base.biomass_share + base.rdf_share < 1.0
            and base.fossil_ng_share < 1.0
        )
        biomass_required = fossil_route and base.biomass_share > 0.0
        rdf_required = fossil_route and base.rdf_share > 0.0
        hydrogen_required = base.fuel_type == HYDROGEN

        return cls(
            **vars(base),
            natural_gas_oxygen_demand_t_per_mwh=_as_float(
                _first_present(row.get("natural_gas_oxygen_demand"), row.get("ng_oxygen_demand")),
                "natural_gas_oxygen_demand",
                default=None if natural_gas_required else 0.0,
            ),
            coal_oxygen_demand_t_per_mwh=_as_float(
                row.get("coal_oxygen_demand"),
                "coal_oxygen_demand",
                default=None if coal_required else 0.0,
            ),
            biomass_oxygen_demand_t_per_mwh=_as_float(
                row.get("biomass_oxygen_demand"),
                "biomass_oxygen_demand",
                default=None if biomass_required else 0.0,
            ),
            rdf_oxygen_demand_t_per_mwh=_as_float(
                row.get("rdf_oxygen_demand"),
                "rdf_oxygen_demand",
                default=None if rdf_required else 0.0,
            ),
            hydrogen_oxygen_demand_t_per_mwh=_as_float(
                _first_present(row.get("hydrogen_oxygen_demand"), row.get("h2_oxygen_demand")),
                "hydrogen_oxygen_demand",
                default=None if hydrogen_required else 0.0,
            ),
            specific_oxygen_electricity_consumption_mwh_per_t=_as_float(
                row.get("specific_oxygen_electricity_consumption"),
                "specific_oxygen_electricity_consumption",
            ),
        )

    def _add_stage_parameters(self, block: pyo.Block) -> None:
        super()._add_stage_parameters(block)
        block.natural_gas_oxygen_demand = pyo.Param(
            initialize=self.natural_gas_oxygen_demand_t_per_mwh
        )
        block.coal_oxygen_demand = pyo.Param(initialize=self.coal_oxygen_demand_t_per_mwh)
        block.biomass_oxygen_demand = pyo.Param(initialize=self.biomass_oxygen_demand_t_per_mwh)
        block.rdf_oxygen_demand = pyo.Param(initialize=self.rdf_oxygen_demand_t_per_mwh)
        block.hydrogen_oxygen_demand = pyo.Param(initialize=self.hydrogen_oxygen_demand_t_per_mwh)
        block.specific_oxygen_electricity_consumption = pyo.Param(
            initialize=self.specific_oxygen_electricity_consumption_mwh_per_t
        )

    def _add_stage_variables(self, block: pyo.Block, time_steps: pyo.Set) -> None:
        super()._add_stage_variables(block, time_steps)
        block.oxygen_demand = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.oxygen_from_electrolyser = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.oxygen_generated = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.electricity_consumption = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    def _add_firing_constraints(
        self, block: pyo.Block, time_steps: pyo.Set, output: pyo.Var
    ) -> None:
        """Add oxygen demand and generation electricity to the firing physics."""
        super()._add_firing_constraints(block, time_steps, output)

        @block.Constraint(time_steps)
        def oxygen_demand_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.oxygen_demand[t] == (
                b.natural_gas_in[t] * b.natural_gas_oxygen_demand
                + b.coal_in[t] * b.coal_oxygen_demand
                + b.biomass_in[t] * b.biomass_oxygen_demand
                + b.rdf_in[t] * b.rdf_oxygen_demand
                + b.hydrogen_in[t] * b.hydrogen_oxygen_demand
            )

        @block.Constraint(time_steps)
        def electrolyser_oxygen_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.oxygen_from_electrolyser[t] <= b.oxygen_demand[t]

        @block.Constraint(time_steps)
        def oxygen_generation_requirement(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.oxygen_generated[t] == (b.oxygen_demand[t] - b.oxygen_from_electrolyser[t])

        @block.Constraint(time_steps)
        def oxygen_electricity_consumption(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.electricity_consumption[t] == (
                b.oxygen_generated[t] * b.specific_oxygen_electricity_consumption
            )

    def _additional_operating_cost_expr(
        self, model: pyo.ConcreteModel, block: pyo.Block, t: int
    ) -> pyo.Expression:
        return block.electricity_consumption[t] * model.electricity_price[t]


@dataclass
class AmineCCS:
    """Post-combustion amine CO2 capture for a cement plant."""

    capture_efficiency: float
    specific_electricity_consumption_mwh_per_t: float
    specific_heat_consumption_mwh_per_t: float
    minimum_capture_fraction: float = 0.0
    specific_variable_cost_eur_per_t: float = 0.0
    heat_cost_eur_per_mwh: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.capture_efficiency <= 1.0:
            raise ValueError("capture_efficiency must be between 0 and 1")
        if not 0.0 <= self.minimum_capture_fraction <= self.capture_efficiency:
            raise ValueError(
                "minimum_capture_fraction must satisfy "
                "0 <= minimum_capture_fraction <= capture_efficiency"
            )
        if self.specific_electricity_consumption_mwh_per_t < 0.0:
            raise ValueError("specific_electricity_consumption_mwh_per_t must be non-negative")
        if self.specific_heat_consumption_mwh_per_t < 0.0:
            raise ValueError("specific_heat_consumption_mwh_per_t must be non-negative")
        if self.specific_variable_cost_eur_per_t < 0.0:
            raise ValueError("specific_variable_cost_eur_per_t must be non-negative")
        if self.heat_cost_eur_per_mwh < 0.0:
            raise ValueError("heat_cost_eur_per_mwh must be non-negative")

    @classmethod
    def from_row(cls, row: pd.Series) -> AmineCCS:
        return cls(
            capture_efficiency=_as_float(row.get("capture_efficiency"), "capture_efficiency"),
            specific_electricity_consumption_mwh_per_t=_as_float(
                _first_present(
                    row.get("specific_electricity_consumption"),
                    row.get("specific_capture_electricity"),
                ),
                "specific_electricity_consumption",
            ),
            specific_heat_consumption_mwh_per_t=_as_float(
                _first_present(
                    row.get("specific_heat_consumption"), row.get("specific_capture_heat")
                ),
                "specific_heat_consumption",
            ),
            minimum_capture_fraction=_as_float(
                row.get("minimum_capture_fraction"), "minimum_capture_fraction", default=0.0
            ),
            specific_variable_cost_eur_per_t=_as_float(
                _first_present(row.get("specific_variable_cost"), row.get("variable_capture_cost")),
                "specific_variable_cost",
                default=0.0,
            ),
            heat_cost_eur_per_mwh=_as_float(row.get("heat_cost"), "heat_cost", default=0.0),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add physical/accounting capture balances, energy demand, and priced credit."""
        dt_hours = float(context["dt_hours"])
        max_capture_rate_t_per_h = float(context["max_capture_rate_t_per_h"])
        if max_capture_rate_t_per_h < 0.0:
            raise ValueError("Plant-computed max_capture_rate_t_per_h must be non-negative")
        block.capture_efficiency = pyo.Param(
            initialize=self.capture_efficiency, within=pyo.UnitInterval
        )
        block.minimum_capture_fraction = pyo.Param(
            initialize=self.minimum_capture_fraction, within=pyo.UnitInterval
        )
        block.max_capture_per_step = pyo.Param(initialize=max_capture_rate_t_per_h * dt_hours)
        block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption_mwh_per_t
        )
        block.specific_heat_consumption = pyo.Param(
            initialize=self.specific_heat_consumption_mwh_per_t
        )
        block.specific_variable_cost = pyo.Param(initialize=self.specific_variable_cost_eur_per_t)
        block.heat_cost = pyo.Param(initialize=self.heat_cost_eur_per_mwh)

        block.co2_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_priced_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_unpriced_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_captured = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_priced_captured = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_unpriced_captured = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_residual = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_priced_residual = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_unpriced_residual = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.electricity_consumption = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.heat_consumption = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def co2_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_in[t] == b.co2_captured[t] + b.co2_residual[t]

        @block.Constraint(time_steps)
        def co2_input_accounting_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_in[t] == b.co2_priced_in[t] + b.co2_unpriced_in[t]

        @block.Constraint(time_steps)
        def priced_co2_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced_in[t] == b.co2_priced_captured[t] + b.co2_priced_residual[t]

        @block.Constraint(time_steps)
        def unpriced_co2_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_unpriced_in[t] == (b.co2_unpriced_captured[t] + b.co2_unpriced_residual[t])

        @block.Constraint(time_steps)
        def captured_co2_accounting_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_captured[t] == (b.co2_priced_captured[t] + b.co2_unpriced_captured[t])

        @block.Constraint(time_steps)
        def priced_capture_efficiency_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced_captured[t] <= b.co2_priced_in[t] * b.capture_efficiency

        @block.Constraint(time_steps)
        def unpriced_capture_efficiency_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_unpriced_captured[t] <= b.co2_unpriced_in[t] * b.capture_efficiency

        @block.Constraint(time_steps)
        def capture_capacity_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_captured[t] <= b.max_capture_per_step

        @block.Constraint(time_steps)
        def minimum_priced_capture_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced_captured[t] >= (b.co2_priced_in[t] * b.minimum_capture_fraction)

        @block.Constraint(time_steps)
        def minimum_unpriced_capture_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_unpriced_captured[t] >= (b.co2_unpriced_in[t] * b.minimum_capture_fraction)

        @block.Constraint(time_steps)
        def electricity_consumption_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return (
                b.electricity_consumption[t]
                == b.co2_captured[t] * b.specific_electricity_consumption
            )

        @block.Constraint(time_steps)
        def heat_consumption_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.heat_consumption[t] == b.co2_captured[t] * b.specific_heat_consumption

        @block.Constraint(time_steps)
        def operating_cost_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.operating_cost[t] == (
                b.electricity_consumption[t] * model.electricity_price[t]
                + b.heat_consumption[t] * b.heat_cost
                + b.co2_captured[t] * b.specific_variable_cost
                - b.co2_priced_captured[t] * model.co2_price[t]
            )

        return block


@dataclass
class CryogenicCCS:
    """Electricity-intensive cryogenic CO2 capture for a cement plant.

    No CO2 inventory, compressor-train dynamics, or refrigeration-cycle detail is
    represented. The cryogenic process is reduced to captured and residual CO2 plus
    its specific electricity demand and variable cost.
    """

    capture_efficiency: float
    specific_electricity_consumption_mwh_per_t: float
    minimum_capture_fraction: float = 0.0
    specific_variable_cost_eur_per_t: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.capture_efficiency <= 1.0:
            raise ValueError("capture_efficiency must be between 0 and 1")
        if not 0.0 <= self.minimum_capture_fraction <= self.capture_efficiency:
            raise ValueError(
                "minimum_capture_fraction must satisfy "
                "0 <= minimum_capture_fraction <= capture_efficiency"
            )
        if self.specific_electricity_consumption_mwh_per_t < 0.0:
            raise ValueError("specific_electricity_consumption_mwh_per_t must be non-negative")
        if self.specific_variable_cost_eur_per_t < 0.0:
            raise ValueError("specific_variable_cost_eur_per_t must be non-negative")

    @classmethod
    def from_row(cls, row: pd.Series) -> CryogenicCCS:
        return cls(
            capture_efficiency=_as_float(row.get("capture_efficiency"), "capture_efficiency"),
            specific_electricity_consumption_mwh_per_t=_as_float(
                _first_present(
                    row.get("specific_electricity_consumption"),
                    row.get("specific_capture_electricity"),
                ),
                "specific_electricity_consumption",
            ),
            minimum_capture_fraction=_as_float(
                row.get("minimum_capture_fraction"), "minimum_capture_fraction", default=0.0
            ),
            specific_variable_cost_eur_per_t=_as_float(
                _first_present(row.get("specific_variable_cost"), row.get("variable_capture_cost")),
                "specific_variable_cost",
                default=0.0,
            ),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add physical/accounting capture balances, electricity demand, and priced credit."""
        dt_hours = float(context["dt_hours"])
        max_capture_rate_t_per_h = float(context["max_capture_rate_t_per_h"])
        if max_capture_rate_t_per_h < 0.0:
            raise ValueError("Plant-computed max_capture_rate_t_per_h must be non-negative")
        block.capture_efficiency = pyo.Param(
            initialize=self.capture_efficiency, within=pyo.UnitInterval
        )
        block.minimum_capture_fraction = pyo.Param(
            initialize=self.minimum_capture_fraction, within=pyo.UnitInterval
        )
        block.max_capture_per_step = pyo.Param(initialize=max_capture_rate_t_per_h * dt_hours)
        block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption_mwh_per_t
        )
        block.specific_variable_cost = pyo.Param(initialize=self.specific_variable_cost_eur_per_t)

        block.co2_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_priced_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_unpriced_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_captured = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_priced_captured = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_unpriced_captured = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_residual = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_priced_residual = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_unpriced_residual = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.electricity_consumption = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def co2_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_in[t] == b.co2_captured[t] + b.co2_residual[t]

        @block.Constraint(time_steps)
        def co2_input_accounting_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_in[t] == b.co2_priced_in[t] + b.co2_unpriced_in[t]

        @block.Constraint(time_steps)
        def priced_co2_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced_in[t] == b.co2_priced_captured[t] + b.co2_priced_residual[t]

        @block.Constraint(time_steps)
        def unpriced_co2_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_unpriced_in[t] == (b.co2_unpriced_captured[t] + b.co2_unpriced_residual[t])

        @block.Constraint(time_steps)
        def captured_co2_accounting_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_captured[t] == (b.co2_priced_captured[t] + b.co2_unpriced_captured[t])

        @block.Constraint(time_steps)
        def priced_capture_efficiency_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced_captured[t] <= b.co2_priced_in[t] * b.capture_efficiency

        @block.Constraint(time_steps)
        def unpriced_capture_efficiency_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_unpriced_captured[t] <= b.co2_unpriced_in[t] * b.capture_efficiency

        @block.Constraint(time_steps)
        def capture_capacity_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_captured[t] <= b.max_capture_per_step

        @block.Constraint(time_steps)
        def minimum_priced_capture_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced_captured[t] >= (b.co2_priced_in[t] * b.minimum_capture_fraction)

        @block.Constraint(time_steps)
        def minimum_unpriced_capture_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_unpriced_captured[t] >= (b.co2_unpriced_in[t] * b.minimum_capture_fraction)

        @block.Constraint(time_steps)
        def electricity_consumption_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return (
                b.electricity_consumption[t]
                == b.co2_captured[t] * b.specific_electricity_consumption
            )

        @block.Constraint(time_steps)
        def operating_cost_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.operating_cost[t] == (
                b.electricity_consumption[t] * model.electricity_price[t]
                + b.co2_captured[t] * b.specific_variable_cost
                - b.co2_priced_captured[t] * model.co2_price[t]
            )

        return block


@dataclass
class OxyfuelCCS:
    """CO2 recovery, purification, and compression for an oxyfuel cement route.

    Oxygen production and its electricity demand remain inside the upstream oxyfuel
    calciner and kiln. This block contains no oxygen, heat, or CO2-storage model.
    """

    recovery_efficiency: float
    specific_electricity_consumption_mwh_per_t: float
    minimum_recovery_fraction: float = 0.0
    specific_variable_cost_eur_per_t: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.recovery_efficiency <= 1.0:
            raise ValueError("recovery_efficiency must be between 0 and 1")
        if not 0.0 <= self.minimum_recovery_fraction <= self.recovery_efficiency:
            raise ValueError(
                "minimum_recovery_fraction must satisfy "
                "0 <= minimum_recovery_fraction <= recovery_efficiency"
            )
        if self.specific_electricity_consumption_mwh_per_t < 0.0:
            raise ValueError("specific_electricity_consumption_mwh_per_t must be non-negative")
        if self.specific_variable_cost_eur_per_t < 0.0:
            raise ValueError("specific_variable_cost_eur_per_t must be non-negative")

    @classmethod
    def from_row(cls, row: pd.Series) -> OxyfuelCCS:
        return cls(
            recovery_efficiency=_as_float(row.get("recovery_efficiency"), "recovery_efficiency"),
            specific_electricity_consumption_mwh_per_t=_as_float(
                _first_present(
                    row.get("specific_electricity_consumption"),
                    row.get("specific_capture_electricity"),
                ),
                "specific_electricity_consumption",
            ),
            minimum_recovery_fraction=_as_float(
                row.get("minimum_recovery_fraction"),
                "minimum_recovery_fraction",
                default=0.0,
            ),
            specific_variable_cost_eur_per_t=_as_float(
                _first_present(row.get("specific_variable_cost"), row.get("variable_capture_cost")),
                "specific_variable_cost",
                default=0.0,
            ),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add physical/accounting recovery and purification/compression demand."""
        dt_hours = float(context["dt_hours"])
        max_capture_rate_t_per_h = float(context["max_capture_rate_t_per_h"])
        if max_capture_rate_t_per_h < 0.0:
            raise ValueError("Plant-computed max_capture_rate_t_per_h must be non-negative")
        block.recovery_efficiency = pyo.Param(
            initialize=self.recovery_efficiency, within=pyo.UnitInterval
        )
        block.minimum_recovery_fraction = pyo.Param(
            initialize=self.minimum_recovery_fraction, within=pyo.UnitInterval
        )
        block.max_capture_per_step = pyo.Param(initialize=max_capture_rate_t_per_h * dt_hours)
        block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption_mwh_per_t
        )
        block.specific_variable_cost = pyo.Param(initialize=self.specific_variable_cost_eur_per_t)

        block.co2_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_priced_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_unpriced_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_captured = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_priced_captured = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_unpriced_captured = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_residual = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_priced_residual = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_unpriced_residual = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.electricity_consumption = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def co2_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_in[t] == b.co2_captured[t] + b.co2_residual[t]

        @block.Constraint(time_steps)
        def co2_input_accounting_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_in[t] == b.co2_priced_in[t] + b.co2_unpriced_in[t]

        @block.Constraint(time_steps)
        def priced_co2_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced_in[t] == b.co2_priced_captured[t] + b.co2_priced_residual[t]

        @block.Constraint(time_steps)
        def unpriced_co2_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_unpriced_in[t] == (b.co2_unpriced_captured[t] + b.co2_unpriced_residual[t])

        @block.Constraint(time_steps)
        def captured_co2_accounting_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_captured[t] == (b.co2_priced_captured[t] + b.co2_unpriced_captured[t])

        @block.Constraint(time_steps)
        def priced_recovery_efficiency_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced_captured[t] <= b.co2_priced_in[t] * b.recovery_efficiency

        @block.Constraint(time_steps)
        def unpriced_recovery_efficiency_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_unpriced_captured[t] <= b.co2_unpriced_in[t] * b.recovery_efficiency

        @block.Constraint(time_steps)
        def capture_capacity_limit(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_captured[t] <= b.max_capture_per_step

        @block.Constraint(time_steps)
        def minimum_priced_recovery_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_priced_captured[t] >= (b.co2_priced_in[t] * b.minimum_recovery_fraction)

        @block.Constraint(time_steps)
        def minimum_unpriced_recovery_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_unpriced_captured[t] >= (
                b.co2_unpriced_in[t] * b.minimum_recovery_fraction
            )

        @block.Constraint(time_steps)
        def electricity_consumption_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return (
                b.electricity_consumption[t]
                == b.co2_captured[t] * b.specific_electricity_consumption
            )

        @block.Constraint(time_steps)
        def operating_cost_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.operating_cost[t] == (
                b.electricity_consumption[t] * model.electricity_price[t]
                + b.co2_captured[t] * b.specific_variable_cost
                - b.co2_priced_captured[t] * model.co2_price[t]
            )

        return block


@dataclass
class GenericInventoryStorage(GenericStorage):
    """Generic inventory store for hydrogen energy or DRI mass."""

    capacity: float
    min_soc: float = 0.0
    max_soc: float = 1.0
    max_power_charge: float | None = None
    max_power_discharge: float | None = None
    efficiency_charge: float = 1.0
    efficiency_discharge: float = 1.0
    initial_soc: float = 1.0
    ramp_up: float | None = None
    ramp_down: float | None = None
    storage_loss_rate: float = 0.0

    @classmethod
    def from_row(cls, row: pd.Series) -> GenericInventoryStorage:
        capacity = _as_float(
            _first_present(row.get("capacity"), row.get("max_capacity")), "capacity"
        )
        return cls(
            capacity=capacity,
            min_soc=_as_float(row.get("min_soc"), "min_soc", default=0.0),
            max_soc=_as_float(row.get("max_soc"), "max_soc", default=1.0),
            max_power_charge=_as_optional_float(row.get("max_power_charge")),
            max_power_discharge=_as_optional_float(row.get("max_power_discharge")),
            efficiency_charge=_as_float(
                row.get("efficiency_charge"), "efficiency_charge", default=1.0
            ),
            efficiency_discharge=_as_float(
                row.get("efficiency_discharge"), "efficiency_discharge", default=1.0
            ),
            initial_soc=_as_float(row.get("initial_soc"), "initial_soc", default=1.0),
            ramp_up=_as_optional_float(row.get("ramp_up")),
            ramp_down=_as_optional_float(row.get("ramp_down")),
            storage_loss_rate=_as_float(
                row.get("storage_loss_rate"), "storage_loss_rate", default=0.0
            ),
        )

    def __post_init__(self) -> None:
        if not 0 <= self.min_soc <= self.initial_soc <= self.max_soc <= 1:
            raise ValueError("Storage SOC fractions must satisfy 0 <= min <= initial <= max <= 1")

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Pyomo Components:

        - **Parameters**: capacity, SOC bounds (as a fraction of capacity), charge/
          discharge power limits, round-trip efficiencies, loss rate, ramp limits, and
          the initial SOC/charge/discharge carried in from the previous rolling window.
        - **Variables**: ``soc`` (fraction of capacity), ``charge``, ``discharge``, and
          ``status`` (mutual-exclusion binary between charging and discharging).
        - **Constraints**: ``soc_balance_constraint``, charge/discharge power limits
          gated by ``status``, and ramp limits on both charge and discharge.
        """
        dt_hours = float(context["dt_hours"])
        max_charge_rate = self.capacity if self.max_power_charge is None else self.max_power_charge
        max_discharge_rate = (
            self.capacity if self.max_power_discharge is None else self.max_power_discharge
        )
        ramp_up = max_charge_rate if self.ramp_up is None else self.ramp_up
        ramp_down = max_charge_rate if self.ramp_down is None else self.ramp_down
        block.capacity = pyo.Param(initialize=self.capacity)
        block.min_soc = pyo.Param(initialize=self.min_soc)
        block.max_soc = pyo.Param(initialize=self.max_soc)
        block.max_power_charge = pyo.Param(initialize=max_charge_rate * dt_hours)
        block.max_power_discharge = pyo.Param(initialize=max_discharge_rate * dt_hours)
        block.efficiency_charge = pyo.Param(initialize=self.efficiency_charge)
        block.efficiency_discharge = pyo.Param(initialize=self.efficiency_discharge)
        initial_soc = float(context.get("initial_soc", self.initial_soc))
        initial_charge = float(context.get("initial_charge", 0.0))
        initial_discharge = float(context.get("initial_discharge", 0.0))
        block.initial_soc = pyo.Param(initialize=initial_soc)
        block.initial_charge = pyo.Param(initialize=initial_charge)
        block.initial_discharge = pyo.Param(initialize=initial_discharge)
        block.storage_loss_rate = pyo.Param(initialize=self.storage_loss_rate)
        block.ramp_up = pyo.Param(initialize=ramp_up * dt_hours)
        block.ramp_down = pyo.Param(initialize=ramp_down * dt_hours)
        block.soc = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(self.min_soc, self.max_soc)
        )
        block.charge = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, max_charge_rate * dt_hours)
        )
        block.discharge = pyo.Var(
            time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0.0, max_discharge_rate * dt_hours),
        )
        block.status = pyo.Var(time_steps, within=pyo.Binary)

        ordered_steps = list(time_steps)

        @block.Constraint(time_steps)
        def soc_balance_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            position = ordered_steps.index(t)
            previous_soc = b.initial_soc if position == 0 else b.soc[ordered_steps[position - 1]]
            return (
                b.soc[t]
                == previous_soc
                + (
                    b.efficiency_charge * b.charge[t]
                    - b.discharge[t] / b.efficiency_discharge
                    - b.storage_loss_rate * previous_soc * b.capacity
                )
                / b.capacity
            )

        @block.Constraint(time_steps)
        def max_charge_power_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.charge[t] <= b.max_power_charge * b.status[t]

        @block.Constraint(time_steps)
        def max_discharge_power_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.discharge[t] <= b.max_power_discharge * (1 - b.status[t])

        @block.Constraint(time_steps)
        def charge_ramp_up_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            position = ordered_steps.index(t)
            previous = b.initial_charge if position == 0 else b.charge[ordered_steps[position - 1]]
            return b.charge[t] - previous <= b.ramp_up

        @block.Constraint(time_steps)
        def discharge_ramp_up_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            position = ordered_steps.index(t)
            previous = (
                b.initial_discharge if position == 0 else b.discharge[ordered_steps[position - 1]]
            )
            return b.discharge[t] - previous <= b.ramp_up

        @block.Constraint(time_steps)
        def charge_ramp_down_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            position = ordered_steps.index(t)
            if position == 0:
                return b.initial_charge - b.charge[t] <= b.ramp_down
            previous = b.charge[ordered_steps[position - 1]]
            return previous - b.charge[t] <= b.ramp_down

        @block.Constraint(time_steps)
        def discharge_ramp_down_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            position = ordered_steps.index(t)
            if position == 0:
                return b.initial_discharge - b.discharge[t] <= b.ramp_down
            previous = b.discharge[ordered_steps[position - 1]]
            return previous - b.discharge[t] <= b.ramp_down

        return block


@dataclass
class HydrogenBufferStorage(GenericInventoryStorage):
    """Hydrogen-energy buffer storage."""


@dataclass
class DRIStorage(GenericInventoryStorage):
    """Direct-reduced-iron mass storage."""


TECHNOLOGY_REGISTRY = {
    "thermal_storage": ThermalStorage,
    "boiler": GasBoiler,
    "electrolyser": Electrolyser,
    "dri_plant": DRIPlant,
    "eaf": ElectricArcFurnace,
    "bof": BasicOxygenFurnace,
    "bf_bof": BlastFurnaceBasicOxygenFurnace,
    "preheater": CementPreheater,
    "simple_calciner": SimpleCementCalciner,
    "leilac_calciner": LEILACCementCalciner,
    "oxyfuel_calciner": OxyfuelCementCalciner,
    "simple_kiln": SimpleCementKiln,
    "oxyfuel_kiln": OxyfuelCementKiln,
    "amine_ccs": AmineCCS,
    "cryogenic_ccs": CryogenicCCS,
    "oxyfuel_ccs": OxyfuelCCS,
    "generic_storage": GenericInventoryStorage,
    "hydrogen_buffer_storage": HydrogenBufferStorage,
    "dri_storage": DRIStorage,
}


def clean_value(value: Any, default: str | None = None) -> str:
    return _clean(value, default)


def first_non_empty(rows: pd.DataFrame, column: str, default: str = "") -> str:
    if column not in rows.columns:
        return default
    for value in rows[column].tolist():
        text = _clean(value)
        if text:
            return text
    return default


def _clean(value: Any, default: str | None = None) -> str:
    if pd.isna(value):
        return "" if default is None else default
    text = str(value).strip()
    if not text:
        return "" if default is None else default
    return text


def _as_float(value: Any, label: str, default: float | None = None) -> float:
    if pd.isna(value) or str(value).strip() == "":
        if default is None:
            raise ValueError(f"Missing required numeric plant parameter '{label}'")
        return float(default)
    return float(value)


def _as_optional_float(value: Any) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def _steel_fuel_type(row: pd.Series, owner: str) -> str:
    fuel_type = _clean(row.get("fuel_type"), HYDROGEN).lower()
    if fuel_type == "both":
        raise ValueError(
            f"{owner} fuel_type='both' is ambiguous; use fuel_type='hybrid_hydrogen_natural_gas'"
        )
    if fuel_type not in STEEL_FUEL_TYPES:
        allowed = ", ".join(sorted(STEEL_FUEL_TYPES))
        raise ValueError(f"{owner} fuel_type must be one of: {allowed}")
    return fuel_type


def _fuel_specific_consumption(
    row: pd.Series,
    column: str,
    fuel_type: str,
    *,
    required_for: set[str],
) -> float:
    value = _as_float(row.get(column), column, default=None if fuel_type in required_for else 0.0)
    if fuel_type in required_for and value <= 0:
        raise ValueError(
            f"Plant parameter '{column}' must be positive when fuel_type='{fuel_type}'"
        )
    return value


def _fuel_co2_factor(
    row: pd.Series,
    column: str,
    fuel_type: str,
    *,
    required_for: set[str],
) -> float:
    """Load an explicit positive fuel-emission factor when that fuel can be consumed."""

    value = _as_float(row.get(column), column, default=None if fuel_type in required_for else 0.0)
    if fuel_type in required_for and value <= 0:
        raise ValueError(
            f"Plant parameter '{column}' must be positive when fuel_type='{fuel_type}'"
        )
    return value


def _cement_fuel_type(row: pd.Series, owner: str, *, default: str) -> str:
    fuel_type = _clean(row.get("fuel_type"), default).lower()
    if fuel_type == "both":
        raise ValueError(
            f"{owner} fuel_type='both' is ambiguous; use fuel_type='hybrid_electricity_fossil'"
        )
    if fuel_type not in CEMENT_FUEL_TYPES:
        allowed = ", ".join(sorted(CEMENT_FUEL_TYPES))
        raise ValueError(f"{owner} fuel_type must be one of: {allowed}")
    return fuel_type


def _cement_max_electric_power(row: pd.Series) -> float | None:
    """Read the electrical rating without breaking the legacy heat-rating alias.

    Older cement inputs used ``max_power`` when ``max_heat_out`` was absent. Once the
    explicit heat rating is present, ``max_power`` has its normal meaning: the primary
    electrical-input rating of the stage.
    """
    explicit_alias = _first_present(
        row.get("max_electric_power"),
        row.get("max_power_electric"),
    )
    if explicit_alias is not None:
        return _as_optional_float(explicit_alias)
    if _first_present(row.get("max_heat_out")) is not None:
        return _as_optional_float(row.get("max_power"))
    return None


def _as_int(value: Any, default: int) -> int:
    if pd.isna(value) or str(value).strip() == "":
        return default
    return int(float(value))


def _first_present(*values: Any) -> Any:
    for value in values:
        if not pd.isna(value) and str(value).strip() != "":
            return value
    return None


def _add_power_parameters(
    component: Any,
    block: pyo.Block,
    dt_hours: float,
    context: dict[str, Any],
) -> None:
    ramp_up = (
        component.max_power_mw
        if component.ramp_up_mw_per_step is None
        else component.ramp_up_mw_per_step
    )
    ramp_down = (
        component.max_power_mw
        if component.ramp_down_mw_per_step is None
        else component.ramp_down_mw_per_step
    )
    block.max_power = pyo.Param(initialize=component.max_power_mw * dt_hours)
    block.min_power = pyo.Param(initialize=component.min_power_mw * dt_hours)
    block.ramp_up = pyo.Param(initialize=ramp_up * dt_hours)
    block.ramp_down = pyo.Param(initialize=ramp_down * dt_hours)
    block.min_operating_steps = pyo.Param(initialize=component.min_operating_steps)
    block.min_down_steps = pyo.Param(initialize=component.min_down_steps)
    initial_status = int(
        context.get("initial_operational_status", component.initial_operational_status)
    )
    default_consecutive_steps = max(
        component.min_operating_steps,
        component.min_down_steps,
        1,
    )
    block.initial_power_in = pyo.Param(initialize=float(context.get("initial_power_in", 0.0)))
    block.initial_operational_status = pyo.Param(initialize=initial_status)
    block.initial_consecutive_status_steps = pyo.Param(
        initialize=int(context.get("initial_consecutive_status_steps", default_consecutive_steps))
    )


def _add_ramping_constraints(block: pyo.Block, time_steps: pyo.Set) -> None:
    ordered_steps = list(time_steps)

    @block.Constraint(time_steps)
    def ramp_up_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
        position = ordered_steps.index(t)
        previous = b.initial_power_in if position == 0 else b.power_in[ordered_steps[position - 1]]
        return b.power_in[t] - previous <= b.ramp_up

    @block.Constraint(time_steps)
    def ramp_down_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
        position = ordered_steps.index(t)
        if position == 0:
            return b.initial_power_in - b.power_in[t] <= b.ramp_down
        previous = b.power_in[ordered_steps[position - 1]]
        return previous - b.power_in[t] <= b.ramp_down


def _add_commitment_constraints(block: pyo.Block, time_steps: pyo.Set) -> None:
    ordered_steps = list(time_steps)

    block.operational_status = pyo.Var(time_steps, within=pyo.Binary)
    block.start_up = pyo.Var(time_steps, within=pyo.Binary)
    block.shut_down = pyo.Var(time_steps, within=pyo.Binary)

    @block.Constraint(time_steps)
    def min_power_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
        return b.power_in[t] >= b.min_power * b.operational_status[t]

    @block.Constraint(time_steps)
    def max_power_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
        return b.power_in[t] <= b.max_power * b.operational_status[t]

    @block.Constraint(time_steps)
    def state_transition_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
        position = ordered_steps.index(t)
        previous = (
            b.initial_operational_status
            if position == 0
            else b.operational_status[ordered_steps[position - 1]]
        )
        return b.operational_status[t] - previous == b.start_up[t] - b.shut_down[t]

    @block.Constraint(time_steps)
    def prevent_simultaneous_startup_shutdown(b: pyo.Block, t: int) -> pyo.Constraint:
        return b.start_up[t] + b.shut_down[t] <= 1

    initial_status = int(pyo.value(block.initial_operational_status))
    consecutive_steps = int(pyo.value(block.initial_consecutive_status_steps))
    minimum_steps = int(
        pyo.value(block.min_operating_steps if initial_status else block.min_down_steps)
    )
    residual_steps = max(0, minimum_steps - consecutive_steps)
    if residual_steps:

        @block.Constraint(time_steps)
        def residual_status_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            position = ordered_steps.index(t)
            if position >= residual_steps:
                return pyo.Constraint.Skip
            return b.operational_status[t] == initial_status

    @block.Constraint(time_steps)
    def min_operating_time_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
        position = ordered_steps.index(t)
        duration = int(pyo.value(b.min_operating_steps))
        if duration <= 0 or position + 1 < duration:
            return pyo.Constraint.Skip
        window = ordered_steps[position - duration + 1 : position + 1]
        return sum(b.start_up[i] for i in window) <= b.operational_status[t]

    @block.Constraint(time_steps)
    def min_downtime_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
        position = ordered_steps.index(t)
        duration = int(pyo.value(b.min_down_steps))
        if duration <= 0 or position + 1 < duration:
            return pyo.Constraint.Skip
        window = ordered_steps[position - duration + 1 : position + 1]
        return sum(b.shut_down[i] for i in window) <= 1 - b.operational_status[t]
