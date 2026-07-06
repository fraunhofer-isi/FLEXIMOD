# SPDX-FileCopyrightText: FLEXIMOD Developers
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyomo.environ as pyo


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
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        dt_hours = float(context["dt_hours"])
        _add_power_parameters(self, block, dt_hours, context)
        block.efficiency = pyo.Param(initialize=self.efficiency)
        block.power_in = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, self.max_power_mw * dt_hours)
        )
        block.hydrogen_out = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def hydrogen_production_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.hydrogen_out[t] == b.power_in[t] * b.efficiency

        @block.Constraint(time_steps)
        def operating_cost_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]

        _add_power_operating_constraints(block, time_steps)
        return block


@dataclass
class DRIPlant:
    """Direct-reduced-iron shaft using hydrogen, natural gas, or a mixture."""

    specific_hydrogen_consumption_mwh_per_t: float
    specific_natural_gas_consumption_mwh_per_t: float
    specific_electricity_consumption_mwh_per_t: float
    specific_iron_ore_consumption_t_per_t: float
    max_power_mw: float
    min_power_mw: float
    fuel_type: str
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1
    natural_gas_co2_factor_t_per_mwh: float = 0.5

    @classmethod
    def from_row(cls, row: pd.Series) -> DRIPlant:
        fuel_type = _clean(row.get("fuel_type")).lower()
        if fuel_type not in {"hydrogen", "natural_gas", "both"}:
            raise ValueError("DRIPlant fuel_type must be 'hydrogen', 'natural_gas', or 'both'")
        return cls(
            specific_hydrogen_consumption_mwh_per_t=_as_float(
                row.get("specific_hydrogen_consumption"), "specific_hydrogen_consumption"
            ),
            specific_natural_gas_consumption_mwh_per_t=_as_float(
                row.get("specific_natural_gas_consumption"),
                "specific_natural_gas_consumption",
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
            natural_gas_co2_factor_t_per_mwh=_as_float(
                row.get("natural_gas_co2_factor"),
                "natural_gas_co2_factor",
                default=0.5,
            ),
        )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        dt_hours = float(context["dt_hours"])
        _add_power_parameters(self, block, dt_hours, context)
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
        block.power_in = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, self.max_power_mw * dt_hours)
        )
        block.iron_ore_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.natural_gas_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_emission = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.hydrogen_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.dri_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def dri_output_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            if self.fuel_type == "hydrogen":
                return b.dri_output[t] == b.hydrogen_in[t] / b.specific_hydrogen_consumption
            if self.fuel_type == "natural_gas":
                return b.dri_output[t] == b.natural_gas_in[t] / b.specific_natural_gas_consumption
            return b.dri_output[t] == (
                b.natural_gas_in[t] / b.specific_natural_gas_consumption
                + b.hydrogen_in[t] / b.specific_hydrogen_consumption
            )

        @block.Constraint(time_steps)
        def zero_unused_fuel_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            if self.fuel_type == "hydrogen":
                return b.natural_gas_in[t] == 0
            if self.fuel_type == "natural_gas":
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
            return b.co2_emission[t] == b.natural_gas_in[t] * b.natural_gas_co2_factor

        @block.Constraint(time_steps)
        def operating_cost_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            cost = (
                b.power_in[t] * model.electricity_price[t]
                + b.iron_ore_in[t] * model.iron_ore_price[t]
                + b.co2_emission[t] * model.co2_price[t]
            )
            if self.fuel_type in {"natural_gas", "both"}:
                cost += b.natural_gas_in[t] * model.natural_gas_price[t]
            if self.fuel_type in {"hydrogen", "both"}:
                cost += b.hydrogen_in[t] * model.hydrogen_price[t]
            return b.operating_cost[t] == cost

        _add_power_operating_constraints(block, time_steps)
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

        _add_power_operating_constraints(block, time_steps)
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


def _add_power_operating_constraints(block: pyo.Block, time_steps: pyo.Set) -> None:
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

    if (
        int(pyo.value(block.min_operating_steps)) <= 1
        and int(pyo.value(block.min_down_steps)) <= 1
        and float(pyo.value(block.min_power)) <= 0
    ):
        return

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
