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
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def hydrogen_production_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.hydrogen_out[t] == b.power_in[t] * b.efficiency

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

    ``CementPreheater``, ``CementCalciner``, and ``CementKiln`` share the same heat
    balance, fuel-exclusivity, ramping, commitment, and CO2/cost machinery - only a
    handful of details differ per stage. Those differences are exactly what each
    subclass overrides:

    - ``_output_var_name``: the block attribute its throughput Var is stored under.
    - ``_add_stage_variables``: any Vars beyond the common set (default: none).
    - ``_external_heat_expr``: heat entering from outside the stage, e.g. kiln waste
      heat reaching the preheater (default: none).
    - ``_output_heat_expr``: the heat that drives the stage's throughput - its own
      ``heat_out``, or a storage-blended ``effective_heat_in`` (default: ``heat_out``).
    - ``_process_co2_factor``: non-combustion CO2 per tonne of output, e.g. calcination
      (default: none).

    ``add_to_model`` is the template method: it calls the phases below in a fixed
    order and returns the populated block.
    """

    #: Attribute name the stage's throughput Var is stored under on the block.
    _output_var_name: ClassVar[str] = "clinker_out"

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        dt_hours = float(context["dt_hours"])
        self._add_common_parameters(block, dt_hours, context)
        self._add_common_variables(block, time_steps)
        self._add_stage_variables(block, time_steps)
        output = getattr(block, self._output_var_name)
        self._add_firing_constraints(block, time_steps, output)
        self._add_aux_and_operational_constraints(block, time_steps, output)
        self._add_emission_and_cost_constraints(model, block, time_steps, output)
        return block

    def _add_common_parameters(
        self, block: pyo.Block, dt_hours: float, context: dict[str, Any]
    ) -> None:
        """Pyomo Components:

        - **Parameters**: stage ratings, efficiencies, ramp limits, CO2 factors, and
          the commitment state (min up/down steps, initial status) carried in from
          the previous rolling window.
        """
        max_heat_mwh = self.max_heat_out_mw * dt_hours
        max_electric_power_mw = (
            self.max_electric_power_mw
            if self.max_electric_power_mw is not None
            else self.max_heat_out_mw / max(1e-9, self.eta_electric)
        )
        max_power_mwh = max_electric_power_mw * dt_hours
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

        block.max_heat_out = pyo.Param(initialize=max_heat_mwh)
        block.max_power = pyo.Param(initialize=max_power_mwh)
        block.specific_heat_demand = pyo.Param(initialize=self.specific_heat_demand_mwh_per_t)
        block.specific_electricity_aux = pyo.Param(
            initialize=self.specific_electricity_aux_mwh_per_t
        )
        block.eta_electric = pyo.Param(initialize=self.eta_electric)
        block.eta_fossil = pyo.Param(initialize=self.eta_fossil)
        block.fossil_ng_share = pyo.Param(initialize=self.fossil_ng_share, within=pyo.UnitInterval)
        block.ramp_up = pyo.Param(initialize=ramp_up * dt_hours)
        block.ramp_down = pyo.Param(initialize=ramp_down * dt_hours)
        block.natural_gas_co2_factor = pyo.Param(initialize=self.natural_gas_co2_factor_t_per_mwh)
        block.coal_co2_factor = pyo.Param(initialize=self.coal_co2_factor_t_per_mwh)
        block.process_co2_factor = pyo.Param(initialize=self._process_co2_factor())
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
          ``hydrogen_in``, ``fossil_in``), the stage's throughput output, CO2 and
          operating cost, and the commitment binaries.
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
        block.hydrogen_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.fossil_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        setattr(block, self._output_var_name, pyo.Var(time_steps, within=pyo.NonNegativeReals))
        block.co2_process = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_energy = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_emission = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)
        block.operational_status = pyo.Var(time_steps, within=pyo.Binary)
        block.start_up = pyo.Var(time_steps, within=pyo.Binary)
        block.shut_down = pyo.Var(time_steps, within=pyo.Binary)

    def _add_stage_variables(self, block: pyo.Block, time_steps: pyo.Set) -> None:
        """Hook: Vars beyond the common set. Default: none."""

    def _output_heat_expr(self, block: pyo.Block, t: int) -> pyo.Expression:
        """Hook: the heat that drives throughput. Default: the stage's own ``heat_out``."""
        return block.heat_out[t]

    def _external_heat_expr(self, block: pyo.Block, t: int) -> pyo.Expression | float:
        """Hook: heat entering from outside the stage. Default: none."""
        return 0.0

    def _process_co2_factor(self) -> float:
        """Hook: non-combustion CO2 per tonne of output. Default: none."""
        return 0.0

    def _add_firing_constraints(
        self, block: pyo.Block, time_steps: pyo.Set, output: pyo.Var
    ) -> None:
        """Pyomo Components:

        - **Constraints**: ``heat_balance`` (fuel-switchable heat generation, plus any
          external heat), fuel-exclusivity for the stage's ``fuel_type``, the fossil
          natural-gas/coal split where fossil fuel is available, and
          ``output_from_heat`` linking heat to throughput.
        """
        fuel_type = self.fuel_type

        @block.Constraint(time_steps)
        def heat_balance(b: pyo.Block, t: int) -> pyo.Constraint:
            if fuel_type == CEMENT_ELECTRICITY:
                generated_heat = b.power_in[t] * b.eta_electric
            elif fuel_type == CEMENT_FOSSIL:
                generated_heat = (b.natural_gas_in[t] + b.coal_in[t]) * b.eta_fossil
            elif fuel_type == CEMENT_HYBRID_ELECTRICITY_FOSSIL:
                generated_heat = (
                    b.power_in[t] * b.eta_electric
                    + (b.natural_gas_in[t] + b.coal_in[t]) * b.eta_fossil
                )
            else:
                generated_heat = b.hydrogen_in[t] * b.eta_fossil
            return b.heat_out[t] == generated_heat + self._external_heat_expr(b, t)

        if fuel_type == CEMENT_ELECTRICITY:

            @block.Constraint(time_steps)
            def zero_non_electric_inputs(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.natural_gas_in[t] + b.coal_in[t] + b.hydrogen_in[t] + b.fossil_in[t] == 0

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
                return b.power_in[t] + b.natural_gas_in[t] + b.coal_in[t] + b.fossil_in[t] == 0

        if fuel_type in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:

            @block.Constraint(time_steps)
            def fossil_sum_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.fossil_in[t] == b.natural_gas_in[t] + b.coal_in[t]

            @block.Constraint(time_steps)
            def fossil_split_natural_gas(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.natural_gas_in[t] == b.fossil_ng_share * b.fossil_in[t]

            @block.Constraint(time_steps)
            def fossil_split_coal(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.coal_in[t] == (1.0 - b.fossil_ng_share) * b.fossil_in[t]

        @block.Constraint(time_steps)
        def output_from_heat(b: pyo.Block, t: int) -> pyo.Constraint:
            return output[t] == self._output_heat_expr(b, t) / b.specific_heat_demand

    def _add_aux_and_operational_constraints(
        self, block: pyo.Block, time_steps: pyo.Set, output: pyo.Var
    ) -> None:
        """Pyomo Components:

        - **Constraints**: auxiliary power drawn from throughput, the heat/commitment
          coupling, heat ramp limits, and the full unit-commitment sub-model (state
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

        - **Constraints**: process and energy CO2, their sum, and operating cost
          (aux power always billed; primary fuel billed for whichever commodity the
          stage's ``fuel_type`` draws on).
        """
        fuel_type = self.fuel_type

        @block.Constraint(time_steps)
        def process_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_process[t] == output[t] * b.process_co2_factor

        @block.Constraint(time_steps)
        def energy_co2_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_energy[t] == (
                b.natural_gas_in[t] * b.natural_gas_co2_factor + b.coal_in[t] * b.coal_co2_factor
            )

        @block.Constraint(time_steps)
        def co2_emission_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.co2_emission[t] == b.co2_process[t] + b.co2_energy[t]

        @block.Constraint(time_steps)
        def operating_cost_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            cost = b.aux_power_in[t] * model.electricity_price[t]
            if fuel_type in {CEMENT_ELECTRICITY, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
                cost += b.power_in[t] * model.electricity_price[t]
            if fuel_type in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
                cost += (
                    b.natural_gas_in[t] * model.natural_gas_price[t]
                    + b.coal_in[t] * model.coal_price[t]
                )
            if fuel_type == HYDROGEN:
                cost += b.hydrogen_in[t] * model.hydrogen_price[t]
            cost += b.co2_emission[t] * model.co2_price[t]
            return b.operating_cost[t] == cost


@dataclass
class CementPreheater(CementKilnLineStage):
    """Fuel-switchable preheater converting raw-meal heat duty into raw meal output."""

    _output_var_name: ClassVar[str] = "raw_meal_out"

    max_heat_out_mw: float
    specific_heat_demand_mwh_per_t: float
    fuel_type: str = CEMENT_ELECTRICITY
    eta_electric: float = 0.98
    eta_fossil: float = 0.90
    fossil_ng_share: float = 1.0
    max_electric_power_mw: float | None = None
    specific_electricity_aux_mwh_per_t: float = 0.0
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    natural_gas_co2_factor_t_per_mwh: float = 0.202
    coal_co2_factor_t_per_mwh: float = 0.341
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1

    @classmethod
    def from_row(cls, row: pd.Series) -> CementPreheater:
        max_heat = _as_float(
            _first_present(row.get("max_heat_out"), row.get("max_power")),
            "max_heat_out",
        )
        return cls(
            max_heat_out_mw=max_heat,
            specific_heat_demand_mwh_per_t=_as_float(
                row.get("specific_heat_demand"), "specific_heat_demand"
            ),
            fuel_type=_cement_fuel_type(row, "CementPreheater", default=CEMENT_ELECTRICITY),
            eta_electric=_as_float(row.get("eta_electric"), "eta_electric", default=0.98),
            eta_fossil=_as_float(row.get("eta_fossil"), "eta_fossil", default=0.90),
            fossil_ng_share=_as_float(row.get("fossil_ng_share"), "fossil_ng_share", default=1.0),
            max_electric_power_mw=_as_optional_float(
                _first_present(row.get("max_electric_power"), row.get("max_power_electric"))
            ),
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


@dataclass
class CementCalciner(CementKilnLineStage):
    """Fuel-switchable calciner with calcination process CO2."""

    max_heat_out_mw: float
    specific_heat_demand_mwh_per_t: float
    fuel_type: str = CEMENT_ELECTRICITY
    eta_electric: float = 0.95
    eta_fossil: float = 0.90
    fossil_ng_share: float = 1.0
    max_electric_power_mw: float | None = None
    specific_electricity_aux_mwh_per_t: float = 0.0
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    calcination_emission_factor_t_per_t: float = 0.525
    natural_gas_co2_factor_t_per_mwh: float = 0.202
    coal_co2_factor_t_per_mwh: float = 0.341
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1

    @classmethod
    def from_row(cls, row: pd.Series) -> CementCalciner:
        max_heat = _as_float(
            _first_present(row.get("max_heat_out"), row.get("max_power")),
            "max_heat_out",
        )
        return cls(
            max_heat_out_mw=max_heat,
            specific_heat_demand_mwh_per_t=_as_float(
                row.get("specific_heat_demand"), "specific_heat_demand"
            ),
            fuel_type=_cement_fuel_type(row, "CementCalciner", default=CEMENT_ELECTRICITY),
            eta_electric=_as_float(row.get("eta_electric"), "eta_electric", default=0.95),
            eta_fossil=_as_float(row.get("eta_fossil"), "eta_fossil", default=0.90),
            fossil_ng_share=_as_float(row.get("fossil_ng_share"), "fossil_ng_share", default=1.0),
            max_electric_power_mw=_as_optional_float(
                _first_present(row.get("max_electric_power"), row.get("max_power_electric"))
            ),
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
        block.effective_heat_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    def _output_heat_expr(self, block: pyo.Block, t: int) -> pyo.Expression:
        return block.effective_heat_in[t]

    def _process_co2_factor(self) -> float:
        return self.calcination_emission_factor_t_per_t


@dataclass
class CementKiln(CementKilnLineStage):
    """Fuel-switchable rotary kiln for final clinkerisation."""

    max_heat_out_mw: float
    specific_heat_demand_mwh_per_t: float
    fuel_type: str = CEMENT_FOSSIL
    eta_electric: float = 0.95
    eta_fossil: float = 0.90
    fossil_ng_share: float = 1.0
    max_electric_power_mw: float | None = None
    specific_electricity_aux_mwh_per_t: float = 0.0
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None
    natural_gas_co2_factor_t_per_mwh: float = 0.202
    coal_co2_factor_t_per_mwh: float = 0.341
    min_operating_steps: int = 0
    min_down_steps: int = 0
    initial_operational_status: int = 1

    @classmethod
    def from_row(cls, row: pd.Series) -> CementKiln:
        max_heat = _as_float(
            _first_present(row.get("max_heat_out"), row.get("max_power")),
            "max_heat_out",
        )
        return cls(
            max_heat_out_mw=max_heat,
            specific_heat_demand_mwh_per_t=_as_float(
                row.get("specific_heat_demand"), "specific_heat_demand"
            ),
            fuel_type=_cement_fuel_type(row, "CementKiln", default=CEMENT_FOSSIL),
            eta_electric=_as_float(row.get("eta_electric"), "eta_electric", default=0.95),
            eta_fossil=_as_float(row.get("eta_fossil"), "eta_fossil", default=0.90),
            fossil_ng_share=_as_float(row.get("fossil_ng_share"), "fossil_ng_share", default=1.0),
            max_electric_power_mw=_as_optional_float(
                _first_present(row.get("max_electric_power"), row.get("max_power_electric"))
            ),
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
            min_operating_steps=_as_int(
                _first_present(row.get("min_operating_steps"), row.get("min_operating_time")),
                default=0,
            ),
            min_down_steps=_as_int(
                _first_present(row.get("min_down_steps"), row.get("min_down_time")), default=0
            ),
            initial_operational_status=_as_int(row.get("initial_operational_status"), default=1),
        )

    # No overrides: a kiln uses the base class's defaults for every hook - its own
    # ``heat_out``, no external heat, and no process CO2.


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
    "calciner": CementCalciner,
    "kiln": CementKiln,
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
