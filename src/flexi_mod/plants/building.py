# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Building model for an electric bus depot with optional V2G."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.opt import SolverStatus, TerminationCondition

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants.base_plant import BasePlant
from flexi_mod.plants.model_utils import (
    available_pyomo_solvers,
    is_infeasible_termination,
    pyomo_value,
)
from flexi_mod.plants.technologies import (
    ChargingStation,
    ElectricVehicle,
    PVPlant,
    first_non_empty,
)

InitialSoc = Mapping[str, float] | float | None
InitialRenewableSoc = Mapping[str, float] | float | None
MonthlyPeak = Mapping[str, float] | None


@dataclass
class Building(BasePlant):
    """A building containing buses, charging stations, and optional PV.

    The buses and chargers are defined in ``plants.csv``. Time-dependent
    demand, bus availability, trips, and electricity prices come from
    ``forecasts_df.csv``.
    """

    required_technologies = ["electric_vehicle", "charging_station"]
    optional_technologies = ["pv_plant"]

    electric_demand_column: str = ""
    components: dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Read every bus and charging-station row from plants.csv
    # ------------------------------------------------------------------
    @classmethod
    def from_rows(cls, building_name: str, rows: pd.DataFrame) -> Building:
        if rows.empty:
            raise ValueError(f"Building '{building_name}' has no technology rows")

        rows = rows.copy()
        rows["technology"] = rows["technology"].astype(str).str.strip().str.lower()
        _check_technology_rows(building_name, rows)

        components: dict[str, object] = {}
        vehicle_rows = rows.loc[rows["technology"] == "electric_vehicle"]
        station_rows = rows.loc[rows["technology"] == "charging_station"]
        pv_rows = rows.loc[rows["technology"] == "pv_plant"]

        for number, (_, row) in enumerate(vehicle_rows.iterrows(), start=1):
            vehicle = ElectricVehicle.from_row(row)
            vehicle.component_name = vehicle.component_name or f"bus_{number}"
            if not vehicle.availability_column:
                vehicle.availability_column = f"{vehicle.component_name}_availability"
            if not vehicle.trip_distance_column:
                vehicle.trip_distance_column = f"{vehicle.component_name}_trip_distance_km"
            _add_unique_component(building_name, components, vehicle.component_name, vehicle)

        for number, (_, row) in enumerate(station_rows.iterrows(), start=1):
            station = ChargingStation.from_row(row)
            station.component_name = station.component_name or f"charger_{number}"
            _add_unique_component(building_name, components, station.component_name, station)

        for number, (_, row) in enumerate(pv_rows.iterrows(), start=1):
            pv_plant = PVPlant.from_row(row)
            pv_plant.component_name = pv_plant.component_name or f"pv_{number}"
            if not pv_plant.generation_column:
                pv_plant.generation_column = f"{pv_plant.component_name}_generation_mw"
            _add_unique_component(building_name, components, pv_plant.component_name, pv_plant)

        _check_components(building_name, components)

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
            components=components,
        )

    @classmethod
    def from_plants_dataframe(cls, plants: pd.DataFrame) -> list[Building]:
        return [cls.from_rows(str(name), rows) for name, rows in plants.groupby("name", sort=False)]

    @property
    def electric_vehicle(self) -> ElectricVehicle:
        vehicles = self.electric_vehicles
        if len(vehicles) != 1:
            raise ValueError("electric_vehicle is available only for a one-bus building")
        return next(iter(vehicles.values()))

    @property
    def electric_vehicles(self) -> dict[str, ElectricVehicle]:
        return {
            name: component
            for name, component in self.components.items()
            if isinstance(component, ElectricVehicle)
        }

    @property
    def charging_station(self) -> ChargingStation:
        stations = self.charging_stations
        if len(stations) != 1:
            raise ValueError("charging_station is available only for a one-charger building")
        return next(iter(stations.values()))

    @property
    def charging_stations(self) -> dict[str, ChargingStation]:
        return {
            name: component
            for name, component in self.components.items()
            if isinstance(component, ChargingStation)
        }

    @property
    def pv_plants(self) -> dict[str, PVPlant]:
        return {
            name: component
            for name, component in self.components.items()
            if isinstance(component, PVPlant)
        }

    @property
    def v2g_enabled(self) -> bool:
        return any(vehicle.is_bidirectional for vehicle in self.electric_vehicles.values()) and any(
            station.is_bidirectional for station in self.charging_stations.values()
        )

    def required_forecast_columns(self) -> set[str]:
        columns = {self.electric_demand_column}
        for vehicle in self.electric_vehicles.values():
            columns.update(
                {
                    vehicle.availability_column,
                    vehicle.trip_distance_column,
                }
            )
        for station in self.charging_stations.values():
            columns.add(station.availability_column)
        for pv_plant in self.pv_plants.values():
            columns.add(pv_plant.generation_column)
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
        initial_soc_mwh: InitialSoc = None,
        initial_renewable_soc_mwh: InitialRenewableSoc = None,
        initial_monthly_peak_mw: MonthlyPeak = None,
    ) -> pd.DataFrame:
        """Build, solve, and return one depot dispatch horizon."""

        model = self.build_model(
            config,
            forecasts,
            electricity_price_column,
            export_price_column,
            initial_soc_mwh,
            initial_renewable_soc_mwh,
            initial_monthly_peak_mw,
        )
        dispatch_objective = _dispatch_objective(config)
        if dispatch_objective == "max_grid_support":
            solver_name = _solve_maximum_service(self.name, model, config, "grid")
        elif dispatch_objective == "max_renewable_shifting":
            solver_name = _solve_maximum_service(self.name, model, config, "renewable")
        else:
            solver_name = _solve_model(self.name, model, config)
        if dispatch_objective == "min_congestion_with_cost_budget":
            _switch_to_grid_support_objective(model, config)
            solver_name = _solve_model(self.name, model, config)
        elif dispatch_objective == "min_renewable_misalignment_with_cost_budget":
            _switch_to_renewable_alignment_objective(model, config)
            solver_name = _solve_model(self.name, model, config)
        return _extract_results(self, model, forecasts, solver_name)

    def build_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
        export_price_column: str | None = None,
        initial_soc_mwh: InitialSoc = None,
        initial_renewable_soc_mwh: InitialRenewableSoc = None,
        initial_monthly_peak_mw: MonthlyPeak = None,
    ) -> pyo.ConcreteModel:
        """Create the Pyomo model without solving it."""

        export_price_column = export_price_column or electricity_price_column
        required_columns = self.required_forecast_columns()
        required_columns.update({electricity_price_column, export_price_column})
        congestion_column = str(
            config.dispatch_setting("grid_congestion_weight_column", "")
        ).strip()
        if congestion_column:
            required_columns.add(congestion_column)
        renewable_column = str(
            config.dispatch_setting("renewable_availability_weight_column", "")
        ).strip()
        if renewable_column:
            required_columns.add(renewable_column)
        _check_forecasts(forecasts, required_columns)

        model = pyo.ConcreteModel(name=f"{self.name}_building_dispatch")
        model.T = pyo.Set(initialize=range(len(forecasts)), ordered=True)

        self.define_parameters(
            model,
            config,
            forecasts,
            electricity_price_column,
            export_price_column,
            initial_monthly_peak_mw,
        )
        self.initialize_components(
            model,
            forecasts,
            initial_soc_mwh,
            initial_renewable_soc_mwh,
        )
        self.define_variables(model)
        self.define_constraints(model)
        self.define_objective(model)
        return model

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
        initial_monthly_peak_mw: MonthlyPeak,
    ) -> None:
        """Add timestep, building demand, and electricity prices."""

        dt_hours = config.timestep_minutes / 60.0
        demand = forecasts[self.electric_demand_column].astype(float) * dt_hours
        import_price = forecasts[import_price_column].astype(float)
        export_price = forecasts[export_price_column].astype(float)
        currency = str(config.dispatch_setting("currency", "EUR")).strip().upper()
        demand_charge = float(config.dispatch_setting("demand_charge_per_kw_month", 0.0))
        export_limit = config.dispatch_setting("grid_export_limit_mw", None)
        discharge_enabled = bool(
            config.dispatch_setting("vehicle_discharge_enabled", self.v2g_enabled)
        )
        congestion_column = str(
            config.dispatch_setting("grid_congestion_weight_column", "")
        ).strip()
        congestion_weight = (
            forecasts[congestion_column].astype(float)
            if congestion_column
            else pd.Series(0.0, index=forecasts.index)
        )
        renewable_column = str(
            config.dispatch_setting("renewable_availability_weight_column", "")
        ).strip()
        renewable_availability = (
            forecasts[renewable_column].astype(float)
            if renewable_column
            else pd.Series(0.0, index=forecasts.index)
        )

        if not np.isfinite(demand).all() or (demand < 0.0).any():
            raise ValueError("Building electricity demand must be finite and non-negative")
        if not np.isfinite(import_price).all() or not np.isfinite(export_price).all():
            raise ValueError("Building electricity prices must be finite")
        if not np.isfinite(congestion_weight).all() or not congestion_weight.between(0, 1).all():
            raise ValueError("Grid congestion weights must be finite and between 0 and 1")
        if not np.isfinite(renewable_availability).all() or not renewable_availability.between(
            0, 1
        ).all():
            raise ValueError("Renewable availability weights must be finite and between 0 and 1")
        if not currency:
            raise ValueError("Building dispatch currency must not be empty")
        if not np.isfinite(demand_charge) or demand_charge < 0.0:
            raise ValueError("demand_charge_per_kw_month must be finite and non-negative")
        if export_limit is not None and (
            not np.isfinite(float(export_limit)) or float(export_limit) < 0.0
        ):
            raise ValueError("grid_export_limit_mw must be finite and non-negative")

        month_for_t = {
            t: pd.Timestamp(forecasts.index[t]).strftime("%Y-%m") for t in range(len(forecasts))
        }
        month_ids = list(dict.fromkeys(month_for_t.values()))
        previous_peaks = {
            month: float((initial_monthly_peak_mw or {}).get(month, 0.0)) for month in month_ids
        }
        if any(not np.isfinite(value) or value < 0.0 for value in previous_peaks.values()):
            raise ValueError("Initial monthly demand peaks must be finite and non-negative")

        model.dt_hours = pyo.Param(initialize=dt_hours)
        model.MONTH_IDS = pyo.Set(initialize=month_ids, ordered=True)
        model.month_for_t = pyo.Param(model.T, initialize=month_for_t, within=pyo.Any)
        model.building_demand_mwh = _time_parameter(model, demand)
        model.import_price = _time_parameter(model, import_price)
        model.export_price = _time_parameter(model, export_price)
        model.grid_congestion_weight = _time_parameter(model, congestion_weight)
        model.renewable_availability_weight = _time_parameter(model, renewable_availability)
        model.additional_import_charge = pyo.Param(
            initialize=self.additional_electricity_charge_eur_per_mwh
        )
        model.currency = pyo.Param(initialize=currency, within=pyo.Any)
        model.demand_charge_per_kw_month = pyo.Param(initialize=demand_charge)
        model.previous_monthly_peak_mw = pyo.Param(model.MONTH_IDS, initialize=previous_peaks)
        model.grid_export_limit_mw = pyo.Param(
            initialize=float(export_limit) if export_limit is not None else -1.0
        )
        model.track_renewable_energy = pyo.Param(
            initialize=int(_dispatch_objective(config) == "max_renewable_shifting"),
            within=pyo.Binary,
        )
        model.vehicle_discharge_enabled = pyo.Param(
            initialize=int(discharge_enabled), within=pyo.Binary
        )

    def initialize_components(
        self,
        model: pyo.ConcreteModel,
        forecasts: pd.DataFrame,
        initial_soc_mwh: InitialSoc,
        initial_renewable_soc_mwh: InitialRenewableSoc,
    ) -> None:
        """Add one Pyomo block for every bus, charger, and PV plant."""

        bus_names = list(self.electric_vehicles)
        charger_names = list(self.charging_stations)
        pv_names = list(self.pv_plants)
        model.BUS_IDS = pyo.Set(initialize=bus_names, ordered=True)
        model.CHARGER_IDS = pyo.Set(initialize=charger_names, ordered=True)
        model.PV_IDS = pyo.Set(initialize=pv_names, ordered=True)
        model.electric_vehicles = pyo.Block(model.BUS_IDS)
        model.charging_stations = pyo.Block(model.CHARGER_IDS)
        model.pv_plants = pyo.Block(model.PV_IDS)

        for bus_name, vehicle in self.electric_vehicles.items():
            vehicle.add_to_model(
                model,
                model.electric_vehicles[bus_name],
                model.T,
                {
                    "dt_hours": pyo.value(model.dt_hours),
                    "initial_soc_mwh": _initial_soc_for_bus(
                        bus_name, vehicle, initial_soc_mwh, len(bus_names)
                    ),
                    "availability": forecasts[vehicle.availability_column].to_numpy(),
                    "trip_distance_km": _optional_profile(forecasts, vehicle.trip_distance_column),
                },
            )
            if bool(pyo.value(model.track_renewable_energy)):
                _add_renewable_energy_tracking(
                    model,
                    model.electric_vehicles[bus_name],
                    vehicle,
                    _initial_renewable_soc_for_bus(
                        bus_name,
                        initial_renewable_soc_mwh,
                        len(bus_names),
                    ),
                )

        for charger_name, station in self.charging_stations.items():
            station.add_to_model(
                model,
                model.charging_stations[charger_name],
                model.T,
                {
                    "dt_hours": pyo.value(model.dt_hours),
                    "availability": _optional_profile(forecasts, station.availability_column),
                },
            )

        for pv_name, pv_plant in self.pv_plants.items():
            pv_plant.add_to_model(
                model,
                model.pv_plants[pv_name],
                model.T,
                {
                    "dt_hours": pyo.value(model.dt_hours),
                    "generation_mw": forecasts[pv_plant.generation_column].to_numpy(),
                },
            )

    def define_variables(self, model: pyo.ConcreteModel) -> None:
        """Add grid import/export variables."""

        dt_hours = pyo.value(model.dt_hours)
        max_demand = max(pyo.value(model.building_demand_mwh[t]) for t in model.T)
        max_import = (
            max_demand
            + sum(station.max_power_charge_mw for station in self.charging_stations.values())
            * dt_hours
        )
        max_export = (
            sum(station.max_power_discharge_mw for station in self.charging_stations.values())
            + sum(pv_plant.max_power_mw for pv_plant in self.pv_plants.values())
        ) * dt_hours
        configured_export_limit = pyo.value(model.grid_export_limit_mw)
        if configured_export_limit >= 0.0:
            max_export = min(max_export, configured_export_limit * dt_hours)

        model.max_grid_import_mwh = pyo.Param(initialize=max_import)
        model.max_grid_export_mwh = pyo.Param(initialize=max_export)
        model.grid_import_mwh = pyo.Var(
            model.T, within=pyo.NonNegativeReals, bounds=(0.0, max_import)
        )
        model.grid_export_mwh = pyo.Var(
            model.T, within=pyo.NonNegativeReals, bounds=(0.0, max_export)
        )
        model.grid_import_mode = pyo.Var(model.T, within=pyo.Binary)
        model.billing_peak_mw = pyo.Var(model.MONTH_IDS, within=pyo.NonNegativeReals)

    def define_constraints(self, model: pyo.ConcreteModel) -> None:
        """Connect buses, chargers, and the building electricity balance."""

        @model.Constraint(model.T)
        def vehicle_station_charge_link(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return sum(m.electric_vehicles[name].charge_mwh[t] for name in m.BUS_IDS) == sum(
                m.charging_stations[name].charge_mwh[t] for name in m.CHARGER_IDS
            )

        @model.Constraint(model.T)
        def vehicle_station_discharge_link(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return sum(m.electric_vehicles[name].discharge_mwh[t] for name in m.BUS_IDS) == sum(
                m.charging_stations[name].discharge_mwh[t] for name in m.CHARGER_IDS
            )

        # Identical chargers are interchangeable. Ordering their binary direction
        # modes removes label-permuted MILP solutions without changing the
        # building's feasible aggregate charge or discharge.
        charger_items = list(self.charging_stations.items())
        interchangeable_chargers = all(
            (
                station.max_power_charge_mw,
                station.max_power_discharge_mw,
                station.power_flow_directionality,
                station.availability_column,
            )
            == (
                charger_items[0][1].max_power_charge_mw,
                charger_items[0][1].max_power_discharge_mw,
                charger_items[0][1].power_flow_directionality,
                charger_items[0][1].availability_column,
            )
            for _, station in charger_items[1:]
        )
        if (
            len(charger_items) > 1
            and interchangeable_chargers
            and charger_items[0][1].is_bidirectional
        ):
            ordered_charger_names = [name for name, _ in charger_items]
            model.CHARGER_ORDER = pyo.RangeSet(0, len(ordered_charger_names) - 2)

            @model.Constraint(model.CHARGER_ORDER, model.T)
            def interchangeable_charger_mode_order(
                m: pyo.ConcreteModel, charger_index: int, t: int
            ) -> pyo.Constraint:
                first = ordered_charger_names[charger_index]
                second = ordered_charger_names[charger_index + 1]
                return (
                    m.charging_stations[first].charging_mode[t]
                    >= m.charging_stations[second].charging_mode[t]
                )

        @model.Constraint(model.T)
        def vehicle_discharge_mode(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            physical_limit = sum(
                station.max_power_discharge_mw for station in self.charging_stations.values()
            ) * pyo.value(m.dt_hours)
            return (
                sum(m.electric_vehicles[name].discharge_mwh[t] for name in m.BUS_IDS)
                <= physical_limit * m.vehicle_discharge_enabled
            )

        @model.Constraint(model.T)
        def electricity_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return (
                m.grid_import_mwh[t]
                + sum(m.charging_stations[name].discharge_mwh[t] for name in m.CHARGER_IDS)
                + sum(m.pv_plants[name].generation_mwh[t] for name in m.PV_IDS)
                == m.building_demand_mwh[t]
                + sum(m.charging_stations[name].charge_mwh[t] for name in m.CHARGER_IDS)
                + m.grid_export_mwh[t]
            )

        # The building cannot import and export in the same interval.
        @model.Constraint(model.T)
        def grid_import_limit(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.grid_import_mwh[t] <= m.max_grid_import_mwh * m.grid_import_mode[t]

        @model.Constraint(model.T)
        def grid_export_limit(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.grid_export_mwh[t] <= m.max_grid_export_mwh * (1 - m.grid_import_mode[t])

        @model.Constraint(model.T)
        def monthly_demand_peak(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.billing_peak_mw[m.month_for_t[t]] >= m.grid_import_mwh[t] / m.dt_hours

        @model.Constraint(model.MONTH_IDS)
        def previous_monthly_demand_peak(m: pyo.ConcreteModel, month: str) -> pyo.Constraint:
            return m.billing_peak_mw[month] >= m.previous_monthly_peak_mw[month]

    def define_objective(self, model: pyo.ConcreteModel) -> None:
        """Minimize net electricity cost."""

        model.energy_cost = pyo.Expression(
            model.T,
            rule=lambda m, t: (
                m.grid_import_mwh[t] * (m.import_price[t] + m.additional_import_charge)
                - m.grid_export_mwh[t] * m.export_price[t]
            ),
        )
        model.incremental_demand_charge = pyo.Expression(
            expr=sum(
                (model.billing_peak_mw[month] - model.previous_monthly_peak_mw[month])
                * 1000.0
                * model.demand_charge_per_kw_month
                for month in model.MONTH_IDS
            )
        )
        model.financial_cost = pyo.Expression(
            expr=sum(model.energy_cost[t] for t in model.T) + model.incremental_demand_charge
        )
        model.technical_service_cost = pyo.Expression(
            expr=sum(
                model.grid_import_mwh[t]
                * (model.import_price[t] + model.additional_import_charge)
                for t in model.T
            )
            + model.incremental_demand_charge
        )
        model.congestion_weighted_import = pyo.Expression(
            expr=sum(
                model.grid_import_mwh[t] * model.grid_congestion_weight[t] for t in model.T
            )
        )
        model.renewable_misalignment = pyo.Expression(
            expr=sum(
                sum(
                    model.electric_vehicles[name].charge_mwh[t]
                    * (1.0 - model.renewable_availability_weight[t])
                    + model.electric_vehicles[name].discharge_mwh[t]
                    * model.renewable_availability_weight[t]
                    for name in model.BUS_IDS
                )
                for t in model.T
            )
        )
        model.grid_support_value = pyo.Expression(
            expr=sum(
                model.grid_congestion_weight[t]
                * (model.grid_export_mwh[t] - model.grid_import_mwh[t])
                for t in model.T
            )
        )
        model.battery_throughput = pyo.Expression(
            expr=sum(
                model.electric_vehicles[name].charge_mwh[t]
                + model.electric_vehicles[name].discharge_mwh[t]
                for name in model.BUS_IDS
                for t in model.T
            )
        )
        model.renewable_deficit_support = pyo.Expression(
            expr=(
                sum(
                    (
                        sum(
                            model.electric_vehicles[name].renewable_discharge_mwh[t]
                            - model.electric_vehicles[name].charge_mwh[t]
                            for name in model.BUS_IDS
                        )
                        * (1.0 - model.renewable_availability_weight[t])
                    )
                    for t in model.T
                )
                if bool(pyo.value(model.track_renewable_energy))
                else 0.0
            )
        )
        model.objective = pyo.Objective(
            expr=model.financial_cost,
            sense=pyo.minimize,
        )


# ----------------------------------------------------------------------
# Small input, solver, and result helpers kept below the physical model
# ----------------------------------------------------------------------
def _check_technology_rows(building_name: str, rows: pd.DataFrame) -> None:
    technologies = rows["technology"].tolist()
    allowed = set(Building.required_technologies + Building.optional_technologies)
    unknown = set(technologies) - allowed
    if unknown:
        raise ValueError(
            f"Building '{building_name}' uses unsupported technology: " + ", ".join(sorted(unknown))
        )
    for technology in Building.required_technologies:
        if technology not in technologies:
            raise ValueError(f"Building '{building_name}' needs at least one '{technology}' row")

    for column in ("fleet_size", "charger_count"):
        if column not in rows:
            continue
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        if (values > 1).any():
            raise ValueError(
                f"Building '{building_name}' does not aggregate assets with '{column}'; "
                "repeat one row per bus or charging station"
            )


def _add_unique_component(
    building_name: str,
    components: dict[str, object],
    component_name: str,
    component: object,
) -> None:
    if component_name in components:
        raise ValueError(
            f"Building '{building_name}' has duplicate component_name '{component_name}'"
        )
    components[component_name] = component


def _check_components(building_name: str, components: dict[str, object]) -> None:
    vehicles = [item for item in components.values() if isinstance(item, ElectricVehicle)]
    stations = [item for item in components.values() if isinstance(item, ChargingStation)]
    directionality = {
        item.power_flow_directionality.strip().lower() for item in [*vehicles, *stations]
    }
    if len(directionality) != 1:
        raise ValueError(
            f"Building '{building_name}' must use the same power_flow_directionality "
            "for every bus and charging station"
        )
    for vehicle in vehicles:
        if vehicle.is_bidirectional and vehicle.max_power_discharge_mw <= 0.0:
            raise ValueError("A bidirectional electric vehicle needs positive discharge power")


def _initial_soc_for_bus(
    bus_name: str,
    vehicle: ElectricVehicle,
    initial_soc_mwh: InitialSoc,
    bus_count: int,
) -> float:
    if initial_soc_mwh is None:
        return vehicle.initial_soc_mwh
    if isinstance(initial_soc_mwh, Mapping):
        if bus_name not in initial_soc_mwh:
            raise ValueError(f"Missing rolling-horizon initial SOC for bus '{bus_name}'")
        return float(initial_soc_mwh[bus_name])
    if bus_count != 1:
        raise ValueError("Multiple buses require initial_soc_mwh as a mapping by component_name")
    return float(initial_soc_mwh)


def _initial_renewable_soc_for_bus(
    bus_name: str,
    initial_renewable_soc_mwh: InitialRenewableSoc,
    bus_count: int,
) -> float:
    if initial_renewable_soc_mwh is None:
        return 0.0
    if isinstance(initial_renewable_soc_mwh, Mapping):
        if bus_name not in initial_renewable_soc_mwh:
            raise ValueError(f"Missing renewable initial SOC for bus '{bus_name}'")
        return float(initial_renewable_soc_mwh[bus_name])
    if bus_count != 1:
        raise ValueError(
            "Multiple buses require initial_renewable_soc_mwh as a mapping by component_name"
        )
    return float(initial_renewable_soc_mwh)


def _add_renewable_energy_tracking(
    model: pyo.ConcreteModel,
    block: pyo.Block,
    vehicle: ElectricVehicle,
    initial_renewable_soc_mwh: float,
) -> None:
    """Track the renewable-equivalent portion of one bus battery."""

    initial_total_soc = pyomo_value(block.initial_soc_mwh)
    if (
        not np.isfinite(initial_renewable_soc_mwh)
        or initial_renewable_soc_mwh < -1e-7
        or initial_renewable_soc_mwh > initial_total_soc + 1e-7
    ):
        raise ValueError("Renewable initial SOC must be between zero and total initial SOC")
    initial_renewable_soc_mwh = min(max(initial_renewable_soc_mwh, 0.0), initial_total_soc)
    dt_hours = pyomo_value(model.dt_hours)
    max_charge_mwh = vehicle.max_power_charge_mw * dt_hours
    max_discharge_mwh = vehicle.max_power_discharge_mw * dt_hours
    retention = (1.0 - vehicle.storage_loss_rate_per_hour) ** dt_hours
    steps = list(model.T)
    first_step = steps[0]

    block.initial_renewable_soc_mwh = pyo.Param(initialize=initial_renewable_soc_mwh)
    block.renewable_charge_mwh = pyo.Var(
        model.T, within=pyo.NonNegativeReals, bounds=(0.0, max_charge_mwh)
    )
    block.renewable_discharge_mwh = pyo.Var(
        model.T, within=pyo.NonNegativeReals, bounds=(0.0, max_discharge_mwh)
    )
    block.renewable_trip_mwh = pyo.Var(model.T, within=pyo.NonNegativeReals)
    block.renewable_soc_mwh = pyo.Var(
        model.T,
        within=pyo.NonNegativeReals,
        bounds=(0.0, vehicle.total_capacity_mwh),
    )

    @block.Constraint(model.T)
    def renewable_charge_limit(b: pyo.Block, t: int) -> pyo.Constraint:
        return (
            b.renewable_charge_mwh[t]
            <= model.renewable_availability_weight[t] * b.charge_mwh[t]
        )

    @block.Constraint(model.T)
    def renewable_discharge_limit(b: pyo.Block, t: int) -> pyo.Constraint:
        return b.renewable_discharge_mwh[t] <= b.discharge_mwh[t]

    @block.Constraint(model.T)
    def renewable_discharge_from_opening_soc(b: pyo.Block, t: int) -> pyo.Constraint:
        previous = (
            b.initial_renewable_soc_mwh
            if t == first_step
            else b.renewable_soc_mwh[model.T.prev(t)]
        )
        return (
            b.renewable_discharge_mwh[t] / b.efficiency_discharge
            + b.renewable_trip_mwh[t]
            <= previous * retention
        )

    @block.Constraint(model.T)
    def renewable_trip_limit(b: pyo.Block, t: int) -> pyo.Constraint:
        return b.renewable_trip_mwh[t] <= b.trip_energy_mwh[t]

    @block.Constraint(model.T)
    def renewable_soc_limit(b: pyo.Block, t: int) -> pyo.Constraint:
        return b.renewable_soc_mwh[t] <= b.soc_mwh[t]

    @block.Constraint(model.T)
    def renewable_soc_balance(b: pyo.Block, t: int) -> pyo.Constraint:
        previous = (
            b.initial_renewable_soc_mwh
            if t == first_step
            else b.renewable_soc_mwh[model.T.prev(t)]
        )
        return b.renewable_soc_mwh[t] == (
            previous * retention
            + b.renewable_charge_mwh[t] * b.efficiency_charge
            - b.renewable_discharge_mwh[t] / b.efficiency_discharge
            - b.renewable_trip_mwh[t]
        )


def _check_forecasts(forecasts: pd.DataFrame, required_columns: set[str]) -> None:
    if forecasts.empty:
        raise ValueError("Building dispatch requires at least one forecast row")
    if not forecasts.index.is_unique:
        raise ValueError("Building dispatch forecast timestamps must be unique")
    missing = sorted(required_columns - set(forecasts.columns))
    if missing:
        raise ValueError("Building dispatch is missing forecast columns: " + ", ".join(missing))


def _time_parameter(model: pyo.ConcreteModel, values: pd.Series) -> pyo.Param:
    return pyo.Param(
        model.T,
        initialize={t: float(values.iloc[t]) for t in model.T},
    )


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


def _dispatch_objective(config: CaseConfig) -> str:
    objective = str(config.dispatch_setting("dispatch_objective", "min_cost")).strip().lower()
    allowed = {
        "min_cost",
        "min_congestion_with_cost_budget",
        "min_renewable_misalignment_with_cost_budget",
        "max_grid_support",
        "max_renewable_shifting",
    }
    if objective not in allowed:
        raise ValueError("dispatch_objective must be one of: " + ", ".join(sorted(allowed)))
    return objective


def _solve_maximum_service(
    building_name: str,
    model: pyo.ConcreteModel,
    config: CaseConfig,
    service: str,
) -> str:
    """Maximize service, then remove cycling, then minimize financial cost."""

    service_expression = (
        model.grid_support_value if service == "grid" else model.renewable_deficit_support
    )
    model.objective.deactivate()
    model.maximum_service_objective = pyo.Objective(
        expr=service_expression,
        sense=pyo.maximize,
    )
    _solve_model(building_name, model, config)
    maximum_service = pyomo_value(service_expression)
    tolerance = max(1e-5, abs(maximum_service) * 1e-7)
    model.maximum_service_floor = pyo.Constraint(
        expr=service_expression >= maximum_service - tolerance
    )

    model.maximum_service_objective.deactivate()
    model.minimum_throughput_objective = pyo.Objective(
        expr=model.battery_throughput,
        sense=pyo.minimize,
    )
    _solve_model(building_name, model, config)
    minimum_throughput = pyomo_value(model.battery_throughput)
    model.minimum_throughput_ceiling = pyo.Constraint(
        expr=model.battery_throughput
        <= minimum_throughput + max(1e-5, minimum_throughput * 1e-7)
    )

    model.minimum_throughput_objective.deactivate()
    model.technical_cost_objective = pyo.Objective(
        expr=model.technical_service_cost,
        sense=pyo.minimize,
    )
    return _solve_model(building_name, model, config)


def _switch_to_grid_support_objective(
    model: pyo.ConcreteModel,
    config: CaseConfig,
) -> None:
    """Keep cost near its minimum, then minimize imports in congested hours."""

    budget_fraction = float(
        config.dispatch_setting("grid_support_cost_budget_fraction_per_horizon", 0.0)
    )
    if not np.isfinite(budget_fraction) or budget_fraction < 0.0:
        raise ValueError(
            "grid_support_cost_budget_fraction_per_horizon must be finite and non-negative"
        )

    minimum_cost = pyomo_value(model.financial_cost)
    allowed_cost = minimum_cost + abs(minimum_cost) * budget_fraction + 1e-6
    model.objective.deactivate()
    model.grid_support_cost_limit = pyo.Constraint(expr=model.financial_cost <= allowed_cost)
    model.grid_support_objective = pyo.Objective(
        expr=model.congestion_weighted_import,
        sense=pyo.minimize,
    )


def _switch_to_renewable_alignment_objective(
    model: pyo.ConcreteModel,
    config: CaseConfig,
) -> None:
    """Keep cost near its minimum, then align bus operation with renewables."""

    budget_fraction = float(
        config.dispatch_setting("renewable_alignment_cost_budget_fraction_per_horizon", 0.0)
    )
    if not np.isfinite(budget_fraction) or budget_fraction < 0.0:
        raise ValueError(
            "renewable_alignment_cost_budget_fraction_per_horizon must be finite "
            "and non-negative"
        )

    minimum_cost = pyomo_value(model.financial_cost)
    allowed_cost = minimum_cost + abs(minimum_cost) * budget_fraction + 1e-6
    model.objective.deactivate()
    model.renewable_alignment_cost_limit = pyo.Constraint(
        expr=model.financial_cost <= allowed_cost
    )
    model.renewable_alignment_objective = pyo.Objective(
        expr=model.renewable_misalignment,
        sense=pyo.minimize,
    )


def _infeasible_message(building_name: str, solver_name: str) -> str:
    return (
        f"Building dispatch for '{building_name}' is infeasible with {solver_name}; "
        "check trips, availability, charger power, and SOC requirements"
    )


def _extract_results(
    building: Building,
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    solver_name: str,
) -> pd.DataFrame:
    dt_hours = pyo.value(model.dt_hours)
    currency = str(pyo.value(model.currency))
    demand_charge_rate = pyomo_value(model.demand_charge_per_kw_month)
    running_monthly_peaks = {
        str(month): pyomo_value(model.previous_monthly_peak_mw[month]) for month in model.MONTH_IDS
    }
    rows: list[dict[str, object]] = []
    for t, timestamp in enumerate(forecasts.index):
        grid_import = pyomo_value(model.grid_import_mwh[t])
        grid_export = pyomo_value(model.grid_export_mwh[t])
        grid_import_mw = grid_import / dt_hours
        month = str(pyo.value(model.month_for_t[t]))
        old_peak = running_monthly_peaks[month]
        new_peak = max(old_peak, grid_import_mw)
        demand_charge_cost = (new_peak - old_peak) * 1000.0 * demand_charge_rate
        running_monthly_peaks[month] = new_peak
        energy_cost = pyomo_value(model.energy_cost[t])
        total_cost = energy_cost + demand_charge_cost
        bus_charge = 0.0
        bus_discharge = 0.0
        bus_soc = 0.0
        bus_capacity = 0.0
        bus_availability = 0.0
        bus_trip_distance = 0.0
        bus_trip_energy = 0.0
        renewable_charge = 0.0
        renewable_discharge = 0.0
        renewable_trip_energy = 0.0
        renewable_soc = 0.0
        renewable_conversion_loss = 0.0
        pv_available_generation = 0.0
        pv_generation = 0.0
        pv_curtailment = 0.0
        row: dict[str, object] = {
            "datetime": timestamp,
            "plant_name": building.name,
            "building_demand_MWh": pyomo_value(model.building_demand_mwh[t]),
            "grid_import_MWh": grid_import,
            "grid_export_MWh": grid_export,
            "net_grid_import_MWh": grid_import - grid_export,
            "grid_import_MW": grid_import_mw,
            "grid_export_MW": grid_export / dt_hours,
            "billing_month": month,
            "billing_peak_MW": new_peak,
            "electricity_import_price_per_MWh": pyomo_value(model.import_price[t]),
            "electricity_export_price_per_MWh": pyomo_value(model.export_price[t]),
            "energy_cost": energy_cost,
            "demand_charge_cost": demand_charge_cost,
            "total_cost": total_cost,
            "currency": currency,
            "v2g_enabled": bool(pyo.value(model.vehicle_discharge_enabled)),
            "solver": solver_name,
        }
        row.update(
            {
                f"electricity_import_price_{currency}_per_MWh": row[
                    "electricity_import_price_per_MWh"
                ],
                f"electricity_export_price_{currency}_per_MWh": row[
                    "electricity_export_price_per_MWh"
                ],
                f"energy_cost_{currency}": energy_cost,
                f"demand_charge_cost_{currency}": demand_charge_cost,
                f"total_cost_{currency}": total_cost,
            }
        )
        for column in (
            "regional_grid_load_mw",
            "regional_grid_load_fraction",
            "grid_congestion_weight",
            "regional_grid_load_state",
            "grid_stress_threshold",
            "regional_grid_load_source_year",
            "regional_grid_load_profile",
            "regional_grid_load_interpolated",
            "solar_pv_capacity_factor",
            "wind_capacity_factor",
            "renewable_availability_weight",
            "renewable_availability_weight_2025_capacity",
            "modelled_vre_potential_mw",
            "national_system_demand_mw",
            "modelled_vre_potential_share_of_demand",
            "renewable_rich_threshold",
            "renewable_rich",
            "renewable_weather_source",
            "renewable_weather_year",
            "renewable_weather_scope",
            "renewable_capacity_basis_year",
            "renewable_weather_api_version",
        ):
            if column in forecasts:
                row[column] = forecasts.iloc[t][column]

        for bus_name, vehicle in building.electric_vehicles.items():
            block = model.electric_vehicles[bus_name]
            availability = pyomo_value(block.availability[t])
            trip_distance = pyomo_value(block.trip_distance_km[t])
            trip_energy = pyomo_value(block.trip_energy_mwh[t])
            charge = pyomo_value(block.charge_mwh[t])
            discharge = pyomo_value(block.discharge_mwh[t])
            soc = pyomo_value(block.soc_mwh[t])
            row.update(
                {
                    f"{bus_name}_availability_fraction": availability,
                    f"{bus_name}_trip_distance_km": trip_distance,
                    f"{bus_name}_trip_energy_MWh": trip_energy,
                    f"{bus_name}_charge_MWh": charge,
                    f"{bus_name}_discharge_MWh": discharge,
                    f"{bus_name}_soc_MWh": soc,
                    f"{bus_name}_soc_fraction": soc / vehicle.battery_capacity_mwh,
                }
            )
            if hasattr(block, "renewable_soc_mwh"):
                tagged_charge = pyomo_value(block.renewable_charge_mwh[t])
                tagged_discharge = pyomo_value(block.renewable_discharge_mwh[t])
                tagged_trip = pyomo_value(block.renewable_trip_mwh[t])
                tagged_soc = pyomo_value(block.renewable_soc_mwh[t])
                conversion_loss = tagged_charge * (1.0 - vehicle.efficiency_charge) + (
                    tagged_discharge * (1.0 / vehicle.efficiency_discharge - 1.0)
                )
                row.update(
                    {
                        f"{bus_name}_renewable_charge_MWh": tagged_charge,
                        f"{bus_name}_renewable_discharge_MWh": tagged_discharge,
                        f"{bus_name}_renewable_trip_energy_MWh": tagged_trip,
                        f"{bus_name}_renewable_soc_MWh": tagged_soc,
                        f"{bus_name}_renewable_conversion_loss_MWh": conversion_loss,
                    }
                )
                renewable_charge += tagged_charge
                renewable_discharge += tagged_discharge
                renewable_trip_energy += tagged_trip
                renewable_soc += tagged_soc
                renewable_conversion_loss += conversion_loss
            bus_availability += availability
            bus_trip_distance += trip_distance
            bus_trip_energy += trip_energy
            bus_charge += charge
            bus_discharge += discharge
            bus_soc += soc
            bus_capacity += vehicle.battery_capacity_mwh

        for charger_name in building.charging_stations:
            block = model.charging_stations[charger_name]
            row.update(
                {
                    f"{charger_name}_availability_fraction": pyomo_value(block.availability[t]),
                    f"{charger_name}_charge_MWh": pyomo_value(block.charge_mwh[t]),
                    f"{charger_name}_discharge_MWh": pyomo_value(block.discharge_mwh[t]),
                }
            )

        for pv_name in building.pv_plants:
            block = model.pv_plants[pv_name]
            available_generation = pyomo_value(block.available_generation_mwh[t])
            generation = pyomo_value(block.generation_mwh[t])
            curtailment = pyomo_value(block.curtailment_mwh[t])
            row.update(
                {
                    f"{pv_name}_available_generation_MWh": available_generation,
                    f"{pv_name}_generation_MWh": generation,
                    f"{pv_name}_curtailment_MWh": curtailment,
                }
            )
            pv_available_generation += available_generation
            pv_generation += generation
            pv_curtailment += curtailment

        bus_count = len(building.electric_vehicles)
        renewable_delivery = max(0.0, renewable_discharge - bus_charge)
        renewable_grid_export = min(grid_export, renewable_delivery)
        renewable_discharge_to_site = renewable_delivery - renewable_grid_export
        renewable_weight = pyomo_value(model.renewable_availability_weight[t])
        row.update(
            {
                "bus_availability_fraction": bus_availability / bus_count,
                "bus_trip_distance_km": bus_trip_distance,
                "bus_trip_energy_MWh": bus_trip_energy,
                "bus_charge_MWh": bus_charge,
                "bus_discharge_MWh": bus_discharge,
                "bus_soc_MWh": bus_soc,
                "bus_soc_fraction": bus_soc / bus_capacity,
                "unmet_trip_energy_MWh": 0.0,
                "renewable_equivalent_charge_MWh": renewable_charge,
                "renewable_equivalent_discharge_MWh": renewable_discharge,
                "renewable_equivalent_trip_energy_MWh": renewable_trip_energy,
                "renewable_equivalent_soc_MWh": renewable_soc,
                "renewable_equivalent_conversion_loss_MWh": renewable_conversion_loss,
                "renewable_equivalent_discharge_to_site_MWh": renewable_discharge_to_site,
                "renewable_equivalent_grid_export_MWh": renewable_grid_export,
                "renewable_deficit_weighted_discharge_MWh": renewable_delivery
                * (1.0 - renewable_weight),
                "pv_available_generation_MWh": pv_available_generation,
                "pv_generation_MWh": pv_generation,
                "pv_curtailment_MWh": pv_curtailment,
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows).set_index("datetime")
    numeric_columns = result.select_dtypes(include=["number"]).columns
    result[numeric_columns] = result[numeric_columns].mask(
        result[numeric_columns].abs() < 1e-9, 0.0
    )
    return result


def building_profile_columns(plant_name: str, rows: pd.DataFrame) -> set[str]:
    """Return profile columns needed by a building in ``plants.csv``."""

    return Building.from_rows(str(plant_name).strip(), rows).required_forecast_columns()
