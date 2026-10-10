# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd
import pyomo.environ as pyo


class Boiler(ABC):
    """Common interface for technologies that convert energy into useful heat."""

    efficiency: float
    ramp_up_mw_per_step: float | None
    ramp_down_mw_per_step: float | None

    @classmethod
    @abstractmethod
    def from_row(cls, row: pd.Series) -> Boiler:
        """Construct a concrete boiler from one ``plants.csv`` row."""

    @abstractmethod
    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add the boiler's variables, parameters, and constraints to Pyomo."""


@dataclass
class ThermalStorage:
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

        component = cls(
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
        _validate_positive(component.max_power_charge_mw, "max_power_charge")
        _validate_positive(component.max_power_discharge_mw, "max_power_discharge")
        _validate_positive(component.max_capacity_mwh, "max_capacity")
        _validate_non_negative(component.min_capacity_mwh, "min_capacity")
        if component.min_capacity_mwh > component.max_capacity_mwh:
            raise ValueError("Plant parameter 'min_capacity' cannot exceed 'max_capacity'")
        _validate_efficiency(component.efficiency_charge, "efficiency_charge")
        _validate_efficiency(component.efficiency_discharge, "efficiency_discharge")
        _validate_fraction(component.storage_loss_rate, "storage_loss_rate")
        return component

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
class GasBoiler(Boiler):
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

        component = cls(
            max_heat_output_mw=_as_float(row.get("max_power"), "max_power"),
            min_heat_output_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            efficiency=_as_float(row.get("efficiency"), "efficiency", default=0.9),
            fuel_type=fuel_type,
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up"), "ramp_up"),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down"), "ramp_down"),
        )
        _validate_positive(component.max_heat_output_mw, "max_power")
        _validate_non_negative(component.min_heat_output_mw, "min_power")
        if component.min_heat_output_mw > component.max_heat_output_mw:
            raise ValueError("Plant parameter 'min_power' cannot exceed 'max_power'")
        _validate_efficiency(component.efficiency, "efficiency")
        _validate_optional_non_negative(component.ramp_up_mw_per_step, "ramp_up")
        _validate_optional_non_negative(component.ramp_down_mw_per_step, "ramp_down")
        return component

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
class ElectricBoiler(Boiler):
    """Direct electric boiler for industrial steam or heat supply.

    ``max_power`` and ``min_power`` in ``plants.csv`` are interpreted as
    electrical input power in MW_el. ``efficiency`` converts consumed
    electricity into useful heat in MWh_th/MWh_el.
    """

    max_electricity_input_mw: float
    min_electricity_input_mw: float
    efficiency: float
    ramp_up_mw_per_step: float | None = None
    ramp_down_mw_per_step: float | None = None

    @classmethod
    def from_row(cls, row: pd.Series) -> ElectricBoiler:
        component = cls(
            max_electricity_input_mw=_as_float(row.get("max_power"), "max_power"),
            min_electricity_input_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            efficiency=_as_float(row.get("efficiency"), "efficiency", default=1.0),
            ramp_up_mw_per_step=_as_optional_float(row.get("ramp_up"), "ramp_up"),
            ramp_down_mw_per_step=_as_optional_float(row.get("ramp_down"), "ramp_down"),
        )
        _validate_positive(component.max_electricity_input_mw, "max_power")
        _validate_non_negative(component.min_electricity_input_mw, "min_power")
        if component.min_electricity_input_mw > component.max_electricity_input_mw:
            raise ValueError("Plant parameter 'min_power' cannot exceed 'max_power'")
        _validate_efficiency(component.efficiency, "efficiency")
        _validate_optional_non_negative(component.ramp_up_mw_per_step, "ramp_up")
        _validate_optional_non_negative(component.ramp_down_mw_per_step, "ramp_down")
        return component

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        dt_hours = float(context["dt_hours"])
        max_electricity_mwh = self.max_electricity_input_mw * dt_hours
        max_heat_mwh = max_electricity_mwh * self.efficiency

        block.max_electricity_input_mw = pyo.Param(initialize=self.max_electricity_input_mw)
        block.min_electricity_input_mw = pyo.Param(initialize=self.min_electricity_input_mw)
        block.efficiency = pyo.Param(initialize=self.efficiency)
        block.electricity_consumption = pyo.Var(
            time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0.0, max_electricity_mwh),
        )
        block.heat_out = pyo.Var(
            time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0.0, max_heat_mwh),
        )
        block.electricity_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def efficiency_constraint(b: pyo.Block, t: int) -> pyo.Constraint:
            return b.heat_out[t] == b.electricity_consumption[t] * b.efficiency

        @block.Constraint(time_steps)
        def electricity_cost_definition(b: pyo.Block, t: int) -> pyo.Constraint:
            return (
                b.electricity_cost[t] == b.electricity_consumption[t] * model.electricity_price[t]
            )

        if hasattr(model, "charge_allowed"):

            @block.Constraint(time_steps)
            def charge_allowed_limit(b: pyo.Block, t: int) -> pyo.Constraint:
                return b.electricity_consumption[t] <= max_electricity_mwh * model.charge_allowed[t]

        return block


@dataclass
class ElectricVehicle:
    """Battery-electric vehicle or aggregated fleet.

    Battery capacity and charging power are specified per vehicle and scaled
    by ``fleet_size``. Availability and trip energy are time series supplied
    by the Building model.
    """

    battery_capacity_mwh: float
    fleet_size: int
    max_power_charge_mw: float
    max_power_discharge_mw: float
    min_soc_fraction: float
    max_soc_fraction: float
    initial_soc_fraction: float
    efficiency_charge: float
    efficiency_discharge: float
    storage_loss_rate_per_hour: float = 0.0
    mileage_mwh_per_km: float | None = None
    power_flow_directionality: str = "unidirectional"
    availability_column: str = ""
    trip_energy_column: str = ""
    trip_distance_column: str = ""
    terminal_soc_fraction: float | None = None
    degradation_cost_eur_per_mwh: float = 0.0

    @classmethod
    def from_row(cls, row: pd.Series) -> ElectricVehicle:
        directionality = _directionality(row.get("power_flow_directionality"))
        max_charge = _as_float(
            _first_value(row, "max_power_charge", "max_power"), "max_power_charge"
        )
        vehicle = cls(
            battery_capacity_mwh=_as_float(
                _first_value(row, "battery_capacity_mwh", "capacity", "max_capacity"),
                "battery_capacity_mwh",
            ),
            fleet_size=_as_positive_int(_first_value(row, "fleet_size", default=1), "fleet_size"),
            max_power_charge_mw=max_charge,
            max_power_discharge_mw=_as_float(
                row.get("max_power_discharge"),
                "max_power_discharge",
                default=max_charge if directionality == "bidirectional" else 0.0,
            ),
            min_soc_fraction=_as_float(row.get("min_soc"), "min_soc", default=0.0),
            max_soc_fraction=_as_float(row.get("max_soc"), "max_soc", default=1.0),
            initial_soc_fraction=_as_float(row.get("initial_soc"), "initial_soc", default=1.0),
            efficiency_charge=_as_float(
                row.get("efficiency_charge"), "efficiency_charge", default=1.0
            ),
            efficiency_discharge=_as_float(
                row.get("efficiency_discharge"), "efficiency_discharge", default=1.0
            ),
            storage_loss_rate_per_hour=_as_float(
                row.get("storage_loss_rate"), "storage_loss_rate", default=0.0
            ),
            mileage_mwh_per_km=_as_optional_float(
                _first_value(row, "mileage_mwh_per_km", "mileage"),
                "mileage_mwh_per_km",
            ),
            power_flow_directionality=directionality,
            availability_column=_clean(
                _first_value(row, "availability_column", "availability_profile")
            ),
            trip_energy_column=_clean(
                _first_value(row, "trip_energy_column", "trip_energy_consumption")
            ),
            trip_distance_column=_clean(_first_value(row, "trip_distance_column", "trip_distance")),
            terminal_soc_fraction=_as_optional_float(row.get("terminal_soc"), "terminal_soc"),
            degradation_cost_eur_per_mwh=_as_float(
                row.get("degradation_cost_eur_per_mwh"),
                "degradation_cost_eur_per_mwh",
                default=0.0,
            ),
        )
        _validate_vehicle(vehicle)
        return vehicle

    @property
    def total_capacity_mwh(self) -> float:
        return self.battery_capacity_mwh * self.fleet_size

    @property
    def initial_soc_mwh(self) -> float:
        return self.initial_soc_fraction * self.total_capacity_mwh

    @property
    def is_bidirectional(self) -> bool:
        return self.power_flow_directionality.strip().lower() == "bidirectional"

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add the vehicle battery to a Building Pyomo model."""

        _validate_vehicle(self)
        dt_hours = float(context["dt_hours"])
        steps = list(time_steps)
        availability, trip_energy = _vehicle_profiles(self, context, steps)

        # Fleet energy and power limits
        initial_soc = float(context.get("initial_soc_mwh", self.initial_soc_mwh))
        min_soc = self.min_soc_fraction * self.total_capacity_mwh
        max_soc = self.max_soc_fraction * self.total_capacity_mwh
        if not min_soc <= initial_soc <= max_soc:
            raise ValueError("initial fleet SOC must be within the configured SOC bounds")

        max_charge_mwh = self.max_power_charge_mw * self.fleet_size * dt_hours
        max_discharge_mwh = self.max_power_discharge_mw * self.fleet_size * dt_hours
        retention = (1.0 - self.storage_loss_rate_per_hour) ** dt_hours

        # Parameters
        block.total_capacity_mwh = pyo.Param(initialize=self.total_capacity_mwh)
        block.initial_soc_mwh = pyo.Param(initialize=initial_soc)
        block.efficiency_charge = pyo.Param(initialize=self.efficiency_charge)
        block.efficiency_discharge = pyo.Param(initialize=self.efficiency_discharge)
        block.availability = pyo.Param(
            time_steps, initialize={t: availability[i] for i, t in enumerate(steps)}
        )
        block.trip_energy_mwh = pyo.Param(
            time_steps, initialize={t: trip_energy[i] for i, t in enumerate(steps)}
        )

        # Variables: all energy flows are MWh per interval
        block.charge_mwh = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, max_charge_mwh)
        )
        block.discharge_mwh = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, max_discharge_mwh)
        )
        block.soc_mwh = pyo.Var(time_steps, within=pyo.NonNegativeReals, bounds=(min_soc, max_soc))
        block.charging_mode = pyo.Var(time_steps, within=pyo.Binary)

        # A connected bus may either charge or discharge, never both.
        @block.Constraint(time_steps)
        def charge_limit(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.charge_mwh[t] <= max_charge_mwh * b.availability[t] * b.charging_mode[t]

        @block.Constraint(time_steps)
        def discharge_limit(b: pyo.Block, t: Any) -> pyo.Constraint:
            if not self.is_bidirectional:
                return b.discharge_mwh[t] == 0.0
            return b.discharge_mwh[t] <= max_discharge_mwh * b.availability[t] * (
                1 - b.charging_mode[t]
            )

        # Battery SOC after charging, V2G discharge, and driving.
        first_step = steps[0]

        @block.Constraint(time_steps)
        def soc_balance(b: pyo.Block, t: Any) -> pyo.Constraint:
            previous_soc = b.initial_soc_mwh if t == first_step else b.soc_mwh[time_steps.prev(t)]
            return b.soc_mwh[t] == (
                previous_soc * retention
                + b.charge_mwh[t] * b.efficiency_charge
                - b.discharge_mwh[t] / b.efficiency_discharge
                - b.trip_energy_mwh[t]
            )

        terminal_soc = (
            initial_soc
            if self.terminal_soc_fraction is None
            else self.terminal_soc_fraction * self.total_capacity_mwh
        )
        block.terminal_soc = pyo.Constraint(expr=block.soc_mwh[steps[-1]] >= terminal_soc)
        return block


@dataclass
class ChargingStation:
    """One charging station or a group of identical charging points."""

    charger_count: int
    max_power_charge_mw: float
    max_power_discharge_mw: float
    power_flow_directionality: str = "unidirectional"
    availability_column: str = ""

    @classmethod
    def from_row(cls, row: pd.Series) -> ChargingStation:
        directionality = _directionality(row.get("power_flow_directionality"))
        max_charge = _as_float(
            _first_value(row, "max_power_charge", "max_power"), "max_power_charge"
        )
        station = cls(
            charger_count=_as_positive_int(
                _first_value(row, "charger_count", "number_of_chargers", default=1),
                "charger_count",
            ),
            max_power_charge_mw=max_charge,
            max_power_discharge_mw=_as_float(
                row.get("max_power_discharge"),
                "max_power_discharge",
                default=0.0 if directionality == "unidirectional" else max_charge,
            ),
            power_flow_directionality=directionality,
            availability_column=_clean(
                _first_value(row, "availability_column", "availability_profile")
            ),
        )
        _validate_charging_station(station)
        return station

    @property
    def is_bidirectional(self) -> bool:
        return self.power_flow_directionality.strip().lower() == "bidirectional"

    @property
    def total_max_power_charge_mw(self) -> float:
        return self.charger_count * self.max_power_charge_mw

    @property
    def total_max_power_discharge_mw(self) -> float:
        return self.charger_count * self.max_power_discharge_mw

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add charger power, availability, and direction constraints."""

        _validate_charging_station(self)
        dt_hours = float(context["dt_hours"])
        steps = list(time_steps)
        availability = _profile_values(
            context.get("availability"), steps, "charger availability", default=1.0
        )
        _validate_profile_range(availability, "charger availability", 0.0, 1.0)
        max_charge = self.total_max_power_charge_mw * dt_hours
        max_discharge = self.total_max_power_discharge_mw * dt_hours

        # Parameters
        block.charger_count = pyo.Param(initialize=self.charger_count)
        block.availability = pyo.Param(
            time_steps, initialize={t: availability[i] for i, t in enumerate(steps)}
        )

        # Grid-side energy through all charging points
        block.charge_mwh = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, max_charge)
        )
        block.discharge_mwh = pyo.Var(
            time_steps, within=pyo.NonNegativeReals, bounds=(0.0, max_discharge)
        )
        if self.is_bidirectional:
            block.charging_mode = pyo.Var(time_steps, within=pyo.Binary)

        # Available chargers may either charge or discharge, never both.
        @block.Constraint(time_steps)
        def charge_limit(b: pyo.Block, t: Any) -> pyo.Constraint:
            if self.is_bidirectional:
                return b.charge_mwh[t] <= max_charge * b.availability[t] * b.charging_mode[t]
            return b.charge_mwh[t] <= max_charge * b.availability[t]

        @block.Constraint(time_steps)
        def discharge_limit(b: pyo.Block, t: Any) -> pyo.Constraint:
            if not self.is_bidirectional:
                return b.discharge_mwh[t] == 0.0
            return b.discharge_mwh[t] <= max_discharge * b.availability[t] * (
                1 - b.charging_mode[t]
            )

        return block


@dataclass
class Electrolyser:
    """Electricity-to-hydrogen technology block for industrial plants.

    ``max_power`` in ``plants.csv`` is MW_el; all model variables use MWh per
    timestep, consistent with the existing FLEXIMOD technology blocks.
    """

    max_power_mw: float
    min_power_mw: float
    efficiency: float

    @classmethod
    def from_row(cls, row: pd.Series) -> Electrolyser:
        component = cls(
            max_power_mw=_as_float(row.get("max_power"), "max_power"),
            min_power_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            efficiency=_as_float(row.get("efficiency"), "efficiency"),
        )
        _validate_positive(component.max_power_mw, "max_power")
        _validate_non_negative(component.min_power_mw, "min_power")
        if component.min_power_mw > component.max_power_mw:
            raise ValueError("Plant parameter 'min_power' cannot exceed 'max_power'")
        _validate_efficiency(component.efficiency, "efficiency")
        return component

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add electricity consumption, hydrogen output, and operating cost."""

        if not hasattr(model, "electricity_price"):
            raise ValueError("Electrolyser requires model.electricity_price")
        dt_hours = float(context["dt_hours"])
        max_energy = self.max_power_mw * dt_hours
        block.max_power_mw = pyo.Param(initialize=self.max_power_mw)
        block.min_power_mw = pyo.Param(initialize=self.min_power_mw)
        block.efficiency = pyo.Param(initialize=self.efficiency)
        block.power_in = pyo.Var(
            time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0.0, max_energy),
        )
        block.hydrogen_out = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def hydrogen_production(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.hydrogen_out[t] == b.power_in[t] * b.efficiency

        @block.Constraint(time_steps)
        def operating_cost_definition(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]

        return block


@dataclass
class DRIPlant:
    """Direct-reduced-iron process block with hydrogen and/or natural-gas fuel."""

    max_power_mw: float
    specific_electricity_consumption: float
    specific_hydrogen_consumption: float
    specific_natural_gas_consumption: float
    specific_iron_ore_consumption: float
    natural_gas_co2_factor: float
    fuel_type: str

    @classmethod
    def from_row(cls, row: pd.Series) -> DRIPlant:
        fuel_type = _normalise_dri_fuel_type(row.get("fuel_type"))
        hydrogen = _as_float(
            row.get("specific_hydrogen_consumption"),
            "specific_hydrogen_consumption",
            default=0.0,
        )
        natural_gas = _as_float(
            row.get("specific_natural_gas_consumption"),
            "specific_natural_gas_consumption",
            default=0.0,
        )
        if fuel_type in {"hydrogen", "both"} and hydrogen <= 0.0:
            raise ValueError("Hydrogen DRI requires positive specific_hydrogen_consumption")
        if fuel_type in {"natural_gas", "both"} and natural_gas <= 0.0:
            raise ValueError("Natural-gas DRI requires positive specific_natural_gas_consumption")
        component = cls(
            max_power_mw=_as_float(row.get("max_power"), "max_power"),
            specific_electricity_consumption=_as_float(
                row.get("specific_electricity_consumption"),
                "specific_electricity_consumption",
            ),
            specific_hydrogen_consumption=hydrogen,
            specific_natural_gas_consumption=natural_gas,
            specific_iron_ore_consumption=_as_float(
                row.get("specific_iron_ore_consumption"),
                "specific_iron_ore_consumption",
            ),
            natural_gas_co2_factor=_as_float(
                row.get("natural_gas_co2_factor"),
                "natural_gas_co2_factor",
                default=0.0,
            ),
            fuel_type=fuel_type,
        )
        _validate_positive(component.max_power_mw, "max_power")
        _validate_non_negative(
            component.specific_electricity_consumption,
            "specific_electricity_consumption",
        )
        _validate_positive(component.specific_iron_ore_consumption, "specific_iron_ore_consumption")
        _validate_non_negative(component.natural_gas_co2_factor, "natural_gas_co2_factor")
        return component

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add DRI material, energy, emissions, and operating-cost equations."""

        _require_model_parameters(
            model,
            "electricity_price",
            "iron_ore_price",
            "co2_price",
        )
        if self.fuel_type in {"natural_gas", "both"}:
            _require_model_parameters(model, "natural_gas_price")
        if self.fuel_type in {"hydrogen", "both"}:
            _require_model_parameters(model, "hydrogen_price")

        max_energy = self.max_power_mw * float(context["dt_hours"])
        block.max_power_mw = pyo.Param(initialize=self.max_power_mw)
        block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption
        )
        block.specific_hydrogen_consumption = pyo.Param(
            initialize=self.specific_hydrogen_consumption
        )
        block.specific_natural_gas_consumption = pyo.Param(
            initialize=self.specific_natural_gas_consumption
        )
        block.specific_iron_ore_consumption = pyo.Param(
            initialize=self.specific_iron_ore_consumption
        )
        block.natural_gas_co2_factor = pyo.Param(initialize=self.natural_gas_co2_factor)
        block.power_in = pyo.Var(time_steps, within=pyo.NonNegativeReals, bounds=(0.0, max_energy))
        block.hydrogen_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.natural_gas_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.iron_ore_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.dri_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_emission = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def electricity_use(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.power_in[t] == b.dri_output[t] * b.specific_electricity_consumption

        @block.Constraint(time_steps)
        def iron_ore_use(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.iron_ore_in[t] == b.dri_output[t] * b.specific_iron_ore_consumption

        @block.Constraint(time_steps)
        def fuel_conversion(b: pyo.Block, t: Any) -> pyo.Constraint:
            if self.fuel_type == "hydrogen":
                return b.dri_output[t] == b.hydrogen_in[t] / b.specific_hydrogen_consumption
            if self.fuel_type == "natural_gas":
                return b.dri_output[t] == b.natural_gas_in[t] / b.specific_natural_gas_consumption
            return b.dri_output[t] == (
                b.hydrogen_in[t] / b.specific_hydrogen_consumption
                + b.natural_gas_in[t] / b.specific_natural_gas_consumption
            )

        @block.Constraint(time_steps)
        def unused_fuel_zero(b: pyo.Block, t: Any) -> pyo.Constraint:
            if self.fuel_type == "hydrogen":
                return b.natural_gas_in[t] == 0.0
            if self.fuel_type == "natural_gas":
                return b.hydrogen_in[t] == 0.0
            return pyo.Constraint.Skip

        @block.Constraint(time_steps)
        def emissions(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.co2_emission[t] == b.natural_gas_in[t] * b.natural_gas_co2_factor

        @block.Constraint(time_steps)
        def operating_cost_definition(b: pyo.Block, t: Any) -> pyo.Constraint:
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

        return block


@dataclass
class ElectricArcFurnace:
    """Electric arc furnace converting DRI, lime, and electricity into steel."""

    max_power_mw: float
    specific_dri_demand: float
    specific_electricity_consumption: float
    specific_lime_demand: float

    @classmethod
    def from_row(cls, row: pd.Series) -> ElectricArcFurnace:
        component = cls(
            max_power_mw=_as_float(row.get("max_power"), "max_power"),
            specific_dri_demand=_as_float(row.get("specific_dri_demand"), "specific_dri_demand"),
            specific_electricity_consumption=_as_float(
                row.get("specific_electricity_consumption"),
                "specific_electricity_consumption",
            ),
            specific_lime_demand=_as_float(
                row.get("specific_lime_demand"), "specific_lime_demand", default=0.0
            ),
        )
        _validate_positive(component.max_power_mw, "max_power")
        _validate_positive(component.specific_dri_demand, "specific_dri_demand")
        _validate_non_negative(
            component.specific_electricity_consumption,
            "specific_electricity_consumption",
        )
        _validate_non_negative(component.specific_lime_demand, "specific_lime_demand")
        return component

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add EAF input/output balances and costs to a Pyomo block."""

        _require_model_parameters(model, "electricity_price", "lime_price")
        max_energy = self.max_power_mw * float(context["dt_hours"])
        block.specific_dri_demand = pyo.Param(initialize=self.specific_dri_demand)
        block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption
        )
        block.specific_lime_demand = pyo.Param(initialize=self.specific_lime_demand)
        block.power_in = pyo.Var(time_steps, within=pyo.NonNegativeReals, bounds=(0.0, max_energy))
        block.dri_input = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.lime_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.steel_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)

        @block.Constraint(time_steps)
        def steel_production(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.steel_output[t] == b.dri_input[t] / b.specific_dri_demand

        @block.Constraint(time_steps)
        def electricity_use(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.power_in[t] == b.steel_output[t] * b.specific_electricity_consumption

        @block.Constraint(time_steps)
        def lime_use(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.lime_in[t] == b.steel_output[t] * b.specific_lime_demand

        @block.Constraint(time_steps)
        def operating_cost_definition(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.operating_cost[t] == (
                b.power_in[t] * model.electricity_price[t] + b.lime_in[t] * model.lime_price[t]
            )

        return block


@dataclass
class ThermalProcessStage:
    """Fuel-switchable thermal stage used by a cement kiln line.

    Subclasses declare their material output.  The shared block exposes the
    same modeller-facing ports for every stage: ``heat_out``, ``power_in``,
    ``natural_gas_in``, ``coal_in``, ``hydrogen_in``, and ``operating_cost``.
    """

    max_heat_output_mw: float
    specific_heat_demand: float
    specific_electricity_aux: float
    fuel_type: str
    eta_electric: float
    eta_fossil: float
    fossil_ng_share: float
    natural_gas_co2_factor: float
    coal_co2_factor: float

    output_name = "material_output"

    @classmethod
    def from_row(cls, row: pd.Series) -> ThermalProcessStage:
        component = cls(
            max_heat_output_mw=_as_float(row.get("max_heat_out"), "max_heat_out"),
            specific_heat_demand=_as_float(row.get("specific_heat_demand"), "specific_heat_demand"),
            specific_electricity_aux=_as_float(
                row.get("specific_electricity_aux"),
                "specific_electricity_aux",
                default=0.0,
            ),
            fuel_type=_normalise_cement_fuel_type(row.get("fuel_type")),
            eta_electric=_as_float(row.get("eta_electric"), "eta_electric", default=0.95),
            eta_fossil=_as_float(row.get("eta_fossil"), "eta_fossil", default=0.90),
            fossil_ng_share=_as_float(row.get("fossil_ng_share"), "fossil_ng_share", default=1.0),
            natural_gas_co2_factor=_as_float(
                row.get("ng_co2_factor"), "ng_co2_factor", default=0.0
            ),
            coal_co2_factor=_as_float(row.get("coal_co2_factor"), "coal_co2_factor", default=0.0),
        )
        _validate_positive(component.max_heat_output_mw, "max_heat_out")
        _validate_positive(component.specific_heat_demand, "specific_heat_demand")
        _validate_non_negative(component.specific_electricity_aux, "specific_electricity_aux")
        _validate_efficiency(component.eta_electric, "eta_electric")
        _validate_efficiency(component.eta_fossil, "eta_fossil")
        _validate_fraction(component.fossil_ng_share, "fossil_ng_share")
        _validate_non_negative(component.natural_gas_co2_factor, "ng_co2_factor")
        _validate_non_negative(component.coal_co2_factor, "coal_co2_factor")
        return component

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        """Add process heat, fuel, material-throughput, and cost equations."""

        _require_model_parameters(
            model,
            "electricity_price",
            "natural_gas_price",
            "coal_price",
            "hydrogen_price",
            "co2_price",
        )
        if not 0.0 <= self.fossil_ng_share <= 1.0:
            raise ValueError("fossil_ng_share must be between zero and one")
        max_heat = self.max_heat_output_mw * float(context["dt_hours"])
        block.specific_heat_demand = pyo.Param(initialize=self.specific_heat_demand)
        block.specific_electricity_aux = pyo.Param(initialize=self.specific_electricity_aux)
        block.eta_electric = pyo.Param(initialize=self.eta_electric)
        block.eta_fossil = pyo.Param(initialize=self.eta_fossil)
        block.fossil_ng_share = pyo.Param(initialize=self.fossil_ng_share)
        block.natural_gas_co2_factor = pyo.Param(initialize=self.natural_gas_co2_factor)
        block.coal_co2_factor = pyo.Param(initialize=self.coal_co2_factor)
        block.heat_out = pyo.Var(time_steps, within=pyo.NonNegativeReals, bounds=(0.0, max_heat))
        block.power_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.aux_power = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.natural_gas_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.coal_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.hydrogen_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.fossil_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.co2_energy = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        block.operating_cost = pyo.Var(time_steps, within=pyo.Reals)
        setattr(block, self.output_name, pyo.Var(time_steps, within=pyo.NonNegativeReals))
        output = getattr(block, self.output_name)

        @block.Constraint(time_steps)
        def heat_balance(b: pyo.Block, t: Any) -> pyo.Constraint:
            electric_heat = b.power_in[t] * b.eta_electric if self.uses_electricity else 0.0
            fossil_heat = b.fossil_in[t] * b.eta_fossil if self.uses_fossil else 0.0
            hydrogen_heat = b.hydrogen_in[t] * b.eta_fossil if self.uses_hydrogen else 0.0
            return b.heat_out[t] == electric_heat + fossil_heat + hydrogen_heat

        @block.Constraint(time_steps)
        def fossil_split(b: pyo.Block, t: Any) -> pyo.Constraint:
            if not self.uses_fossil:
                return b.fossil_in[t] + b.natural_gas_in[t] + b.coal_in[t] == 0.0
            return b.fossil_in[t] == b.natural_gas_in[t] + b.coal_in[t]

        @block.Constraint(time_steps)
        def natural_gas_share(b: pyo.Block, t: Any) -> pyo.Constraint:
            if not self.uses_fossil:
                return pyo.Constraint.Skip
            return b.natural_gas_in[t] == b.fossil_ng_share * b.fossil_in[t]

        @block.Constraint(time_steps)
        def unused_energy_carriers_zero(b: pyo.Block, t: Any) -> pyo.Constraint:
            if not self.uses_electricity:
                return b.power_in[t] == 0.0
            if not self.uses_hydrogen:
                return b.hydrogen_in[t] == 0.0
            return pyo.Constraint.Skip

        @block.Constraint(time_steps)
        def throughput_from_heat(b: pyo.Block, t: Any) -> pyo.Constraint:
            return output[t] == b.heat_out[t] / b.specific_heat_demand

        @block.Constraint(time_steps)
        def auxiliary_electricity(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.aux_power[t] == output[t] * b.specific_electricity_aux

        @block.Constraint(time_steps)
        def energy_emissions(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.co2_energy[t] == (
                b.natural_gas_in[t] * b.natural_gas_co2_factor + b.coal_in[t] * b.coal_co2_factor
            )

        @block.Constraint(time_steps)
        def operating_cost_definition(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.operating_cost[t] == (
                (b.power_in[t] + b.aux_power[t]) * model.electricity_price[t]
                + b.natural_gas_in[t] * model.natural_gas_price[t]
                + b.coal_in[t] * model.coal_price[t]
                + b.hydrogen_in[t] * model.hydrogen_price[t]
                + self.emission_cost_expression(b, t, model)
            )

        return block

    @property
    def uses_electricity(self) -> bool:
        return self.fuel_type in {"electricity", "both"}

    @property
    def uses_fossil(self) -> bool:
        return self.fuel_type in {"fossil", "both"}

    @property
    def uses_hydrogen(self) -> bool:
        return self.fuel_type == "hydrogen"

    def emission_cost_expression(
        self,
        block: pyo.Block,
        t: Any,
        model: pyo.ConcreteModel,
    ) -> pyo.NumericValue:
        return block.co2_energy[t] * model.co2_price[t]


class Preheater(ThermalProcessStage):
    """Cement preheater with raw-meal output in tonnes per timestep."""

    output_name = "raw_meal_out"


@dataclass
class Calciner(ThermalProcessStage):
    """Cement calciner with clinker output and process-CO₂ emissions."""

    calcination_emission_factor: float = 0.525
    output_name = "clinker_out"

    @classmethod
    def from_row(cls, row: pd.Series) -> Calciner:
        values = ThermalProcessStage.from_row(row)
        component = cls(
            **values.__dict__,
            calcination_emission_factor=_as_float(
                row.get("calcination_emission_factor"),
                "calcination_emission_factor",
                default=0.525,
            ),
        )
        _validate_non_negative(component.calcination_emission_factor, "calcination_emission_factor")
        return component

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        block: pyo.Block,
        time_steps: pyo.Set,
        context: dict[str, Any],
    ) -> pyo.Block:
        block = super().add_to_model(model, block, time_steps, context)
        block.calcination_emission_factor = pyo.Param(initialize=self.calcination_emission_factor)
        block.co2_process = pyo.Var(time_steps, within=pyo.NonNegativeReals)

        @block.Constraint(time_steps)
        def process_emissions(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.co2_process[t] == b.clinker_out[t] * b.calcination_emission_factor

        # Replace the inherited cost definition so process emissions are paid
        # alongside the energy-related emissions.
        block.del_component(block.operating_cost_definition)

        @block.Constraint(time_steps)
        def operating_cost_definition(b: pyo.Block, t: Any) -> pyo.Constraint:
            return b.operating_cost[t] == (
                (b.power_in[t] + b.aux_power[t]) * model.electricity_price[t]
                + b.natural_gas_in[t] * model.natural_gas_price[t]
                + b.coal_in[t] * model.coal_price[t]
                + b.hydrogen_in[t] * model.hydrogen_price[t]
                + (b.co2_energy[t] + b.co2_process[t]) * model.co2_price[t]
            )

        return block


class Kiln(ThermalProcessStage):
    """Cement kiln with clinker output in tonnes per timestep."""

    output_name = "clinker_out"


TECHNOLOGY_REGISTRY = {
    "thermal_storage": ThermalStorage,
    "boiler": GasBoiler,
    "electric_boiler": ElectricBoiler,
    "electric_vehicle": ElectricVehicle,
    "charging_station": ChargingStation,
    "dri_plant": DRIPlant,
    "eaf": ElectricArcFurnace,
    "electrolyser": Electrolyser,
    "preheater": Preheater,
    "calciner": Calciner,
    "kiln": Kiln,
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
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Plant parameter '{label}' must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"Plant parameter '{label}' must be a finite number")
    return number


def _as_optional_float(value: Any, label: str) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return _as_float(value, label)


def _first_value(row: pd.Series, *columns: str, default: Any = None) -> Any:
    for column in columns:
        value = row.get(column)
        if not pd.isna(value) and str(value).strip() != "":
            return value
    return default


def _as_positive_int(value: Any, label: str) -> int:
    number = _as_float(value, label)
    if not number.is_integer() or number <= 0:
        raise ValueError(f"Plant parameter '{label}' must be a positive integer")
    return int(number)


def _directionality(value: Any) -> str:
    directionality = _clean(value, "unidirectional").lower()
    if directionality not in {"unidirectional", "bidirectional"}:
        raise ValueError("power_flow_directionality must be 'unidirectional' or 'bidirectional'")
    return directionality


def _profile_values(
    values: Any,
    steps: list[Any],
    label: str,
    default: float,
) -> list[float]:
    if values is None:
        return [default] * len(steps)
    if isinstance(values, dict):
        result = [float(values[t]) for t in steps]
    else:
        result = [float(value) for value in values]
    if len(result) != len(steps):
        raise ValueError(f"{label} must have one value per optimization time step")
    if not all(pd.notna(value) for value in result):
        raise ValueError(f"{label} cannot contain missing values")
    return result


def _validate_profile_range(
    values: list[float],
    label: str,
    lower: float | None,
    upper: float | None,
) -> None:
    for value in values:
        if not float("-inf") < value < float("inf"):
            raise ValueError(f"{label} must contain finite values")
        if lower is not None and value < lower:
            raise ValueError(f"{label} cannot be below {lower}")
        if upper is not None and value > upper:
            raise ValueError(f"{label} cannot exceed {upper}")


def _validate_positive(value: float, label: str) -> None:
    if not 0.0 < value < float("inf"):
        raise ValueError(f"Plant parameter '{label}' must be finite and positive")


def _validate_non_negative(value: float, label: str) -> None:
    if not 0.0 <= value < float("inf"):
        raise ValueError(f"Plant parameter '{label}' must be finite and non-negative")


def _validate_fraction(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Plant parameter '{label}' must be between zero and one")


def _validate_optional_non_negative(value: float | None, label: str) -> None:
    if value is not None:
        _validate_non_negative(value, label)


def _validate_efficiency(value: float, label: str) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"Plant parameter '{label}' must be in (0, 1]")


def _vehicle_profiles(
    vehicle: ElectricVehicle,
    context: dict[str, Any],
    steps: list[Any],
) -> tuple[list[float], list[float]]:
    """Prepare the two time series used by the vehicle equations."""

    _validate_positive(float(context["dt_hours"]), "dt_hours")
    if not steps:
        raise ValueError("ElectricVehicle requires at least one optimization time step")

    availability = _profile_values(
        context.get("availability"), steps, "vehicle availability", default=1.0
    )
    trip_energy = _profile_values(context.get("trip_energy_mwh"), steps, "trip energy", default=0.0)
    trip_distance = _profile_values(
        context.get("trip_distance_km"), steps, "trip distance", default=0.0
    )

    if any(trip_energy) and any(trip_distance):
        raise ValueError("Provide trip energy or trip distance, not both")
    if any(trip_distance):
        if vehicle.mileage_mwh_per_km is None:
            raise ValueError("mileage_mwh_per_km is required with a trip-distance profile")
        trip_energy = [distance * vehicle.mileage_mwh_per_km for distance in trip_distance]

    _validate_profile_range(availability, "vehicle availability", 0.0, 1.0)
    _validate_profile_range(trip_energy, "trip energy", 0.0, None)
    return availability, trip_energy


def _validate_vehicle(vehicle: ElectricVehicle) -> None:
    """Validate configuration separately from the physical equations."""

    if not isinstance(vehicle.fleet_size, int) or vehicle.fleet_size <= 0:
        raise ValueError("fleet_size must be a positive integer")
    _directionality(vehicle.power_flow_directionality)
    _validate_positive(vehicle.battery_capacity_mwh, "battery_capacity_mwh")
    _validate_positive(vehicle.max_power_charge_mw, "max_power_charge")
    _validate_profile_range(
        [vehicle.min_soc_fraction, vehicle.max_soc_fraction, vehicle.initial_soc_fraction],
        "SOC fractions",
        0.0,
        1.0,
    )
    if vehicle.min_soc_fraction > vehicle.max_soc_fraction:
        raise ValueError("min_soc cannot exceed max_soc")
    if not vehicle.min_soc_fraction <= vehicle.initial_soc_fraction <= vehicle.max_soc_fraction:
        raise ValueError("initial_soc must be between min_soc and max_soc")
    if vehicle.terminal_soc_fraction is not None and not (
        vehicle.min_soc_fraction <= vehicle.terminal_soc_fraction <= vehicle.max_soc_fraction
    ):
        raise ValueError("terminal_soc must be between min_soc and max_soc")
    _validate_efficiency(vehicle.efficiency_charge, "efficiency_charge")
    _validate_efficiency(vehicle.efficiency_discharge, "efficiency_discharge")
    _validate_profile_range([vehicle.storage_loss_rate_per_hour], "storage_loss_rate", 0.0, 1.0)
    _validate_profile_range(
        [vehicle.max_power_discharge_mw, vehicle.degradation_cost_eur_per_mwh],
        "vehicle non-negative parameters",
        0.0,
        None,
    )
    if vehicle.mileage_mwh_per_km is not None and vehicle.mileage_mwh_per_km <= 0.0:
        raise ValueError("mileage_mwh_per_km must be positive when configured")


def _validate_charging_station(station: ChargingStation) -> None:
    """Validate configuration separately from the charger equations."""

    if not isinstance(station.charger_count, int) or station.charger_count <= 0:
        raise ValueError("charger_count must be a positive integer")
    _directionality(station.power_flow_directionality)
    _validate_positive(station.max_power_charge_mw, "max_power_charge")
    _validate_profile_range([station.max_power_discharge_mw], "max_power_discharge", 0.0, None)
    if station.is_bidirectional and station.max_power_discharge_mw <= 0.0:
        raise ValueError("A bidirectional charging station needs positive discharge power")


def _normalise_dri_fuel_type(value: Any) -> str:
    """Map the steel input vocabulary onto the three supported DRI fuel modes."""

    fuel_type = _clean(value, "").lower()
    aliases = {
        "hydrogen": "hydrogen",
        "natural_gas": "natural_gas",
        "both": "both",
        "hybrid_hydrogen_natural_gas": "both",
    }
    try:
        return aliases[fuel_type]
    except KeyError as exc:
        options = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported DRI fuel_type '{fuel_type}'. Expected one of: {options}"
        ) from exc


def _normalise_cement_fuel_type(value: Any) -> str:
    """Normalise the cement-stage firing modes used in the input data."""

    fuel_type = _clean(value, "electricity").lower()
    aliases = {
        "electricity": "electricity",
        "fossil": "fossil",
        "both": "both",
        "hydrogen": "hydrogen",
    }
    try:
        return aliases[fuel_type]
    except KeyError as exc:
        options = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported cement fuel_type '{fuel_type}'. Expected one of: {options}"
        ) from exc


def _require_model_parameters(model: pyo.ConcreteModel, *names: str) -> None:
    """Fail early when a component's declared price dependency is absent."""

    missing = [name for name in names if not hasattr(model, name)]
    if missing:
        raise ValueError("Model is missing required parameter(s): " + ", ".join(missing))
