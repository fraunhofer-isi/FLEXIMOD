# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Building model for an electric bus depot with optional V2G."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.opt import SolverStatus, TerminationCondition

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import PlantInput
from flexi_mod.modeling.pyomo_utils import (
    available_pyomo_solvers,
    is_infeasible_termination,
)
from flexi_mod.modeling.validation import (
    component_from_row,
    numeric_forecast,
    time_parameter,
    validate_profile_range,
    validate_required_forecasts,
)
from flexi_mod.plants.base_plant import BasePlant
from flexi_mod.plants.technologies import ChargingStation, ElectricVehicle, first_non_empty


@dataclass
class Building(BasePlant):
    """A building containing an electric-bus fleet and charging stations.

    The buses and chargers are defined in ``plants.csv``. Time-dependent
    demand, bus availability, trips, and electricity prices come from
    ``forecasts_df.csv``.
    """

    required_technologies = ["electric_vehicle", "charging_station"]

    electric_demand_column: str = ""
    components: dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Assemble the two configured technologies
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, plant_input: PlantInput) -> Building:
        """Create a building model from the loader's parameters and components."""

        if plant_input.unit_type not in {"building", "bus_depot", "electric_bus_depot"}:
            raise ValueError(
                f"Plant '{plant_input.name}' has unit_type='{plant_input.unit_type}', "
                "not a supported building type"
            )
        return cls._assemble(plant_input.name, plant_input.component_table())

    @classmethod
    def _assemble(cls, building_name: str, component_table: pd.DataFrame) -> Building:
        """Create the building model from one plant's component table."""

        if component_table.empty:
            raise ValueError(f"Building '{building_name}' has no technology rows")

        rows = component_table.copy()
        rows["technology"] = rows["technology"].astype(str).str.strip().str.lower()
        _check_technology_rows(building_name, rows)

        vehicle_row = rows.loc[rows["technology"] == "electric_vehicle"].iloc[0]
        station_row = rows.loc[rows["technology"] == "charging_station"].iloc[0]
        vehicle = component_from_row(
            "Building",
            building_name,
            "electric_vehicle",
            ElectricVehicle.from_row,
            vehicle_row,
        )
        station = component_from_row(
            "Building",
            building_name,
            "charging_station",
            ChargingStation.from_row,
            station_row,
        )

        _check_vehicle_and_station(building_name, vehicle, station)

        # Conventional default profile names keep plants.csv short.
        if not vehicle.availability_column:
            vehicle.availability_column = f"{building_name}_vehicle_availability"
        if not vehicle.trip_energy_column and not vehicle.trip_distance_column:
            vehicle.trip_energy_column = f"{building_name}_vehicle_trip_energy"

        demand_column = first_non_empty(rows, "demand")
        if not demand_column:
            demand_column = first_non_empty(rows, "electricity_demand_column")
        if not demand_column:
            demand_column = f"{building_name}_electricity_demand"

        objective = first_non_empty(rows, "objective", default="min_variable_cost")
        if objective != "min_variable_cost":
            raise ValueError("Building currently supports only objective='min_variable_cost'")

        return cls(
            name=building_name,
            unit_type=first_non_empty(rows, "unit_type", default="building"),
            node=first_non_empty(rows, "node"),
            objective=objective,
            electric_demand_column=demand_column,
            components={
                "electric_vehicle": vehicle,
                "charging_station": station,
            },
        )

    @property
    def electric_vehicle(self) -> ElectricVehicle:
        return self.components["electric_vehicle"]  # type: ignore[return-value]

    @property
    def charging_station(self) -> ChargingStation:
        return self.components["charging_station"]  # type: ignore[return-value]

    @property
    def v2g_enabled(self) -> bool:
        return self.electric_vehicle.is_bidirectional

    def required_forecast_columns(self) -> set[str]:
        vehicle = self.electric_vehicle
        station = self.charging_station
        columns = {
            self.electric_demand_column,
            vehicle.availability_column,
            vehicle.trip_energy_column,
            vehicle.trip_distance_column,
            station.availability_column,
        }
        return {column for column in columns if column}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
        export_price_column: str | None = None,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        """Build, solve, and return one depot dispatch horizon."""

        model = self.build_model(
            config,
            forecasts,
            electricity_price_column,
            export_price_column,
            initial_soc_mwh,
        )
        solver_name = _solve_model(self.name, model, config)
        from flexi_mod.outputs.result_mappers import extract_dispatch_results

        return extract_dispatch_results(self, model, forecasts, solver_name)

    def build_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
        export_price_column: str | None = None,
        initial_soc_mwh: float | None = None,
    ) -> pyo.ConcreteModel:
        """Create the Pyomo model without solving it."""

        export_price_column = export_price_column or electricity_price_column
        self.validate_inputs(
            forecasts,
            electricity_price_column,
            export_price_column,
        )

        model = pyo.ConcreteModel(name=f"{self.name}_building_dispatch")
        model.T = pyo.Set(initialize=range(len(forecasts)), ordered=True)

        self.define_parameters(
            model,
            config,
            forecasts,
            electricity_price_column,
            export_price_column,
        )
        self.initialize_components(model, forecasts, initial_soc_mwh)
        self.define_variables(model)
        self.define_constraints(model)
        self.define_objective(model)
        return model

    def validate_inputs(
        self,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
        export_price_column: str | None = None,
    ) -> None:
        """Validate profiles and prices before creating building Pyomo objects."""

        export_price_column = export_price_column or electricity_price_column
        required_columns = self.required_forecast_columns()
        required_columns.update({electricity_price_column, export_price_column})
        validate_required_forecasts(forecasts, required_columns, f"Building '{self.name}'")

        numeric_forecast(
            forecasts,
            self.electric_demand_column,
            f"Building '{self.name}'",
            description="electricity demand",
            require_non_negative=True,
        )
        numeric_forecast(
            forecasts,
            electricity_price_column,
            f"Building '{self.name}'",
            description="import price",
        )
        numeric_forecast(
            forecasts,
            export_price_column,
            f"Building '{self.name}'",
            description="export price",
        )

        vehicle = self.electric_vehicle
        availability = numeric_forecast(
            forecasts,
            vehicle.availability_column,
            f"Building '{self.name}'",
            description="vehicle availability",
        )
        validate_profile_range(
            availability,
            f"Building '{self.name}'",
            "vehicle availability",
            0.0,
            1.0,
        )
        if vehicle.trip_energy_column:
            numeric_forecast(
                forecasts,
                vehicle.trip_energy_column,
                f"Building '{self.name}'",
                description="vehicle trip energy",
                require_non_negative=True,
            )
        if vehicle.trip_distance_column:
            numeric_forecast(
                forecasts,
                vehicle.trip_distance_column,
                f"Building '{self.name}'",
                description="vehicle trip distance",
                require_non_negative=True,
            )
        if self.charging_station.availability_column:
            station_availability = numeric_forecast(
                forecasts,
                self.charging_station.availability_column,
                f"Building '{self.name}'",
                description="charging-station availability",
            )
            validate_profile_range(
                station_availability,
                f"Building '{self.name}'",
                "charging-station availability",
                0.0,
                1.0,
            )

    # ------------------------------------------------------------------
    # Model definition: parameters -> components -> variables -> equations
    # ------------------------------------------------------------------
    def define_parameters(
        self,
        model: pyo.ConcreteModel,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        import_price_column: str,
        export_price_column: str,
    ) -> None:
        """Add timestep, building demand, and electricity prices."""

        dt_hours = config.timestep_minutes / 60.0
        demand = forecasts[self.electric_demand_column].astype(float) * dt_hours
        import_price = forecasts[import_price_column].astype(float)
        export_price = forecasts[export_price_column].astype(float)

        if not np.isfinite(demand).all() or (demand < 0.0).any():
            raise ValueError("Building electricity demand must be finite and non-negative")
        if not np.isfinite(import_price).all() or not np.isfinite(export_price).all():
            raise ValueError("Building electricity prices must be finite")

        model.dt_hours = pyo.Param(initialize=dt_hours)
        model.building_demand_mwh = time_parameter(model, demand)
        model.import_price = time_parameter(model, import_price)
        model.export_price = time_parameter(model, export_price)
        model.additional_import_charge = pyo.Param(
            initialize=self.additional_electricity_charge_eur_per_mwh
        )

    def initialize_components(
        self,
        model: pyo.ConcreteModel,
        forecasts: pd.DataFrame,
        initial_soc_mwh: float | None,
    ) -> None:
        """Add the electric-bus battery and charging-station blocks."""

        vehicle = self.electric_vehicle
        station = self.charging_station

        model.electric_vehicle = pyo.Block()
        vehicle.add_to_model(
            model,
            model.electric_vehicle,
            model.T,
            {
                "dt_hours": pyo.value(model.dt_hours),
                "initial_soc_mwh": (
                    vehicle.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
                ),
                "availability": forecasts[vehicle.availability_column].to_numpy(),
                "trip_energy_mwh": _optional_profile(forecasts, vehicle.trip_energy_column),
                "trip_distance_km": _optional_profile(forecasts, vehicle.trip_distance_column),
            },
        )

        model.charging_station = pyo.Block()
        station.add_to_model(
            model,
            model.charging_station,
            model.T,
            {
                "dt_hours": pyo.value(model.dt_hours),
                "availability": _optional_profile(forecasts, station.availability_column),
            },
        )

    def define_variables(self, model: pyo.ConcreteModel) -> None:
        """Add grid import/export variables."""

        dt_hours = pyo.value(model.dt_hours)
        station = self.charging_station
        max_demand = max(pyo.value(model.building_demand_mwh[t]) for t in model.T)
        max_import = max_demand + station.total_max_power_charge_mw * dt_hours
        max_export = station.total_max_power_discharge_mw * dt_hours

        model.max_grid_import_mwh = pyo.Param(initialize=max_import)
        model.max_grid_export_mwh = pyo.Param(initialize=max_export)
        model.grid_import_mwh = pyo.Var(
            model.T, within=pyo.NonNegativeReals, bounds=(0.0, max_import)
        )
        model.grid_export_mwh = pyo.Var(
            model.T, within=pyo.NonNegativeReals, bounds=(0.0, max_export)
        )
        model.grid_import_mode = pyo.Var(model.T, within=pyo.Binary)

    def define_constraints(self, model: pyo.ConcreteModel) -> None:
        """Connect buses, chargers, and the building electricity balance."""

        @model.Constraint(model.T)
        def vehicle_station_charge_link(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.electric_vehicle.charge_mwh[t] == m.charging_station.charge_mwh[t]

        @model.Constraint(model.T)
        def vehicle_station_discharge_link(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.electric_vehicle.discharge_mwh[t] == m.charging_station.discharge_mwh[t]

        @model.Constraint(model.T)
        def electricity_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return (
                m.grid_import_mwh[t] + m.charging_station.discharge_mwh[t]
                == m.building_demand_mwh[t]
                + m.charging_station.charge_mwh[t]
                + m.grid_export_mwh[t]
            )

        # The building cannot import and export in the same interval.
        @model.Constraint(model.T)
        def grid_import_limit(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.grid_import_mwh[t] <= m.max_grid_import_mwh * m.grid_import_mode[t]

        @model.Constraint(model.T)
        def grid_export_limit(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.grid_export_mwh[t] <= m.max_grid_export_mwh * (1 - m.grid_import_mode[t])

    def define_objective(self, model: pyo.ConcreteModel) -> None:
        """Minimize electricity cost plus battery cycling cost."""

        degradation_cost = self.electric_vehicle.degradation_cost_eur_per_mwh
        model.variable_cost = pyo.Expression(
            model.T,
            rule=lambda m, t: (
                m.grid_import_mwh[t] * (m.import_price[t] + m.additional_import_charge)
                - m.grid_export_mwh[t] * m.export_price[t]
                + degradation_cost
                * (m.electric_vehicle.charge_mwh[t] + m.electric_vehicle.discharge_mwh[t])
            ),
        )
        model.objective = pyo.Objective(
            expr=sum(model.variable_cost[t] for t in model.T),
            sense=pyo.minimize,
        )


# ----------------------------------------------------------------------
# Small input, solver, and result helpers kept below the physical model
# ----------------------------------------------------------------------
def _check_technology_rows(building_name: str, rows: pd.DataFrame) -> None:
    technologies = rows["technology"].tolist()
    unknown = set(technologies) - set(Building.required_technologies)
    if unknown:
        raise ValueError(
            f"Building '{building_name}' uses unsupported technology: " + ", ".join(sorted(unknown))
        )
    for technology in Building.required_technologies:
        count = technologies.count(technology)
        if count != 1:
            raise ValueError(
                f"Building '{building_name}' needs exactly one '{technology}' row; "
                "use fleet_size or charger_count to aggregate assets"
            )


def _check_vehicle_and_station(
    building_name: str,
    vehicle: ElectricVehicle,
    station: ChargingStation,
) -> None:
    if vehicle.is_bidirectional != station.is_bidirectional:
        raise ValueError(
            f"Building '{building_name}' must use the same power_flow_directionality "
            "for its vehicle and charging station"
        )
    if vehicle.is_bidirectional and vehicle.max_power_discharge_mw <= 0.0:
        raise ValueError("A bidirectional electric vehicle needs positive discharge power")
    if vehicle.trip_energy_column and vehicle.trip_distance_column:
        raise ValueError("Configure trip energy or trip distance, not both")


def _optional_profile(forecasts: pd.DataFrame, column: str) -> np.ndarray | None:
    return forecasts[column].to_numpy() if column else None


def _solve_model(
    building_name: str,
    model: pyo.ConcreteModel,
    config: CaseConfig,
) -> str:
    errors: list[str] = []
    for solver_name, solver in available_pyomo_solvers(config):
        try:
            result = solver.solve(model, tee=config.solver_tee)
        except NoFeasibleSolutionError as exc:
            raise RuntimeError(_infeasible_message(building_name, solver_name)) from exc
        except (ApplicationError, RuntimeError, OSError) as exc:
            errors.append(f"{solver_name}: {exc}")
            continue

        status = result.solver.status
        termination = result.solver.termination_condition
        if status == SolverStatus.ok and termination in {
            TerminationCondition.optimal,
            TerminationCondition.feasible,
        }:
            return solver_name
        if is_infeasible_termination(termination):
            raise RuntimeError(_infeasible_message(building_name, solver_name))
        errors.append(f"{solver_name}: status={status}, termination={termination}")

    raise RuntimeError("Building dispatch failed for all solvers. " + " | ".join(errors))


def _infeasible_message(building_name: str, solver_name: str) -> str:
    return (
        f"Building dispatch for '{building_name}' is infeasible with {solver_name}; "
        "check trips, availability, charger power, and SOC requirements"
    )


def building_profile_columns(plant_name: str, component_table: pd.DataFrame) -> set[str]:
    """Return profile columns needed by a building component table."""

    return Building._assemble(str(plant_name).strip(), component_table).required_forecast_columns()
