# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
        return cls(
            max_electricity_input_mw=_as_float(row.get("max_power"), "max_power"),
            min_electricity_input_mw=_as_float(row.get("min_power"), "min_power", default=0.0),
            efficiency=_as_float(row.get("efficiency"), "efficiency", default=1.0),
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
                _first_value(row, "mileage_mwh_per_km", "mileage")
            ),
            power_flow_directionality=directionality,
            availability_column=_clean(
                _first_value(row, "availability_column", "availability_profile")
            ),
            trip_energy_column=_clean(
                _first_value(row, "trip_energy_column", "trip_energy_consumption")
            ),
            trip_distance_column=_clean(_first_value(row, "trip_distance_column", "trip_distance")),
            terminal_soc_fraction=_as_optional_float(row.get("terminal_soc")),
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


TECHNOLOGY_REGISTRY = {
    "thermal_storage": ThermalStorage,
    "boiler": GasBoiler,
    "electric_boiler": ElectricBoiler,
    "electric_vehicle": ElectricVehicle,
    "charging_station": ChargingStation,
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
