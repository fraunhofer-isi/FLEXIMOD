# SPDX-FileCopyrightText: FLEXIMOD Developers
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.opt import SolverStatus, TerminationCondition

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants.base_plant import BasePlant
from flexi_mod.plants.technologies import (
    TECHNOLOGY_REGISTRY,
    DRIPlant,
    ElectricArcFurnace,
    first_non_empty,
)


@dataclass(frozen=True)
class SteelDispatchSignals:
    """Forecast-column mapping required for steel production dispatch."""

    electricity_price_col: str
    natural_gas_price_col: str
    hydrogen_price_col: str
    iron_ore_price_col: str
    lime_price_col: str
    co2_price_col: str
    steel_price_col: str | None = None


@dataclass
class SteelPlant(BasePlant):
    """Hydrogen/natural-gas DRI and EAF steel-production model.

    The physical chain is electricity/hydrogen/gas -> DRI -> EAF -> steel. Optional
    hydrogen and DRI stores can shift intermediate production across the horizon.
    A market strategy is deliberately not embedded in this class.
    """

    steel_demand_tonnes: float | None = None
    steel_demand_column: str = ""
    components: dict[str, object] = field(default_factory=dict)

    required_technologies = frozenset({"dri_plant", "eaf"})
    optional_technologies = frozenset({"electrolyser", "hydrogen_buffer_storage", "dri_storage"})

    @classmethod
    def from_rows(cls, plant_name: str, rows: pd.DataFrame) -> SteelPlant:
        normalised = rows.copy()
        normalised["technology_normalised"] = (
            normalised["technology"].astype(str).str.strip().str.lower()
        )
        allowed = cls.required_technologies | cls.optional_technologies
        components: dict[str, object] = {}
        for _, row in normalised.iterrows():
            technology = str(row["technology_normalised"])
            if technology not in allowed:
                raise ValueError(
                    f"Steel plant '{plant_name}' uses unsupported technology '{technology}'"
                )
            if technology in components:
                raise ValueError(
                    f"Steel plant '{plant_name}' defines technology '{technology}' more than once"
                )
            components[technology] = TECHNOLOGY_REGISTRY[technology].from_row(row)

        missing = cls.required_technologies - components.keys()
        if missing:
            raise ValueError(
                f"Steel plant '{plant_name}' is missing required technology/technologies: "
                f"{', '.join(sorted(missing))}"
            )

        steel_demand = _consistent_optional_total(rows, "steel_demand", plant_name)
        steel_demand_column = _consistent_optional_text(rows, "demand", plant_name)
        if not steel_demand_column:
            steel_demand_column = f"{plant_name}_steel_demand"

        return cls(
            name=plant_name,
            unit_type=first_non_empty(rows, "unit_type", default="steel_plant"),
            node=first_non_empty(rows, "node", default=""),
            objective=first_non_empty(rows, "objective", default="min_variable_cost"),
            steel_demand_tonnes=steel_demand,
            steel_demand_column=steel_demand_column,
            components=components,
        )

    @classmethod
    def from_plants_dataframe(cls, plants: pd.DataFrame) -> list[SteelPlant]:
        return [
            cls.from_rows(str(plant_name), rows)
            for plant_name, rows in plants.groupby("name", sort=False)
        ]

    @property
    def dri_plant(self) -> DRIPlant:
        component = self.components.get("dri_plant")
        if not isinstance(component, DRIPlant):
            raise ValueError(f"Steel plant '{self.name}' has no DRI plant")
        return component

    @property
    def eaf(self) -> ElectricArcFurnace:
        component = self.components.get("eaf")
        if not isinstance(component, ElectricArcFurnace):
            raise ValueError(f"Steel plant '{self.name}' has no EAF")
        return component

    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: SteelDispatchSignals,
    ) -> pd.DataFrame:
        if self.objective != "min_variable_cost":
            raise ValueError(f"Steel plant objective '{self.objective}' is not supported")
        if forecasts.empty:
            raise ValueError("Steel plant dispatch horizon must contain at least one timestep")
        self._validate_signal_columns(forecasts, signals)
        model = self._build_model(config, forecasts, signals)
        errors: list[str] = []
        for solver_name in dict.fromkeys([config.solver_name, *config.solver_fallbacks]):
            try:
                solver = pyo.SolverFactory(solver_name)
                if solver is None or not solver.available(exception_flag=False):
                    errors.append(f"{solver_name}: unavailable")
                    continue
                result = solver.solve(model, tee=config.solver_tee)
            except (ApplicationError, NoFeasibleSolutionError, RuntimeError) as exc:
                errors.append(f"{solver_name}: {exc}")
                continue
            status = result.solver.status
            termination = result.solver.termination_condition
            if status == SolverStatus.ok and termination in {
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal,
                TerminationCondition.globallyOptimal,
            }:
                return self._extract_results(model, forecasts, solver_name)
            errors.append(f"{solver_name}: status={status}, termination={termination}")
        raise RuntimeError("Steel plant dispatch could not be solved; " + "; ".join(errors))

    def _build_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: SteelDispatchSignals,
    ) -> pyo.ConcreteModel:
        dt_hours = config.timestep_minutes / 60.0
        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=range(len(forecasts)), ordered=True)

        def values(column: str) -> dict[int, float]:
            return {t: float(forecasts[column].iloc[t]) for t in model.T}

        model.electricity_price = pyo.Param(
            model.T, initialize=values(signals.electricity_price_col)
        )
        model.natural_gas_price = pyo.Param(
            model.T, initialize=values(signals.natural_gas_price_col)
        )
        hydrogen_prices = (
            {t: 0.0 for t in model.T}
            if "electrolyser" in self.components
            else values(signals.hydrogen_price_col)
        )
        model.hydrogen_price = pyo.Param(model.T, initialize=hydrogen_prices)
        model.iron_ore_price = pyo.Param(model.T, initialize=values(signals.iron_ore_price_col))
        model.lime_price = pyo.Param(model.T, initialize=values(signals.lime_price_col))
        model.co2_price = pyo.Param(model.T, initialize=values(signals.co2_price_col))
        if signals.steel_price_col:
            model.steel_price = pyo.Param(model.T, initialize=values(signals.steel_price_col))
        steel_demand_total, steel_demand_mode = self._resolve_steel_demand(forecasts)
        model.steel_demand = pyo.Param(initialize=steel_demand_total)
        model.steel_demand_from_forecast = pyo.Param(
            initialize=int(steel_demand_mode == "forecast_profile"), within=pyo.Binary
        )
        model.technology_blocks = pyo.Block(list(self.components))
        context: dict[str, Any] = {"dt_hours": dt_hours}
        for technology, component in self.components.items():
            component.add_to_model(model, model.technology_blocks[technology], model.T, context)

        dri = model.technology_blocks["dri_plant"]
        eaf = model.technology_blocks["eaf"]
        has_electrolyser = "electrolyser" in self.components
        has_hydrogen_storage = "hydrogen_buffer_storage" in self.components
        has_dri_storage = "dri_storage" in self.components

        if has_electrolyser:
            electrolyser = model.technology_blocks["electrolyser"]
            if has_hydrogen_storage:
                hydrogen_storage = model.technology_blocks["hydrogen_buffer_storage"]

                @model.Constraint(model.T)
                def hydrogen_flow_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                    return electrolyser.hydrogen_out[t] + hydrogen_storage.discharge[t] == (
                        dri.hydrogen_in[t] + hydrogen_storage.charge[t]
                    )

            else:

                @model.Constraint(model.T)
                def hydrogen_flow_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                    return electrolyser.hydrogen_out[t] == dri.hydrogen_in[t]

        if has_dri_storage:
            dri_storage = model.technology_blocks["dri_storage"]

            @model.Constraint(model.T)
            def dri_flow_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                return dri.dri_output[t] + dri_storage.discharge[t] == (
                    eaf.dri_input[t] + dri_storage.charge[t]
                )

        else:

            @model.Constraint(model.T)
            def dri_flow_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                return dri.dri_output[t] == eaf.dri_input[t]

        model.total_power_input = pyo.Var(model.T, within=pyo.NonNegativeReals)
        model.variable_cost = pyo.Var(model.T, within=pyo.Reals)

        @model.Constraint()
        def steel_output_association_constraint(m: pyo.ConcreteModel) -> pyo.Constraint:
            return sum(eaf.steel_output[t] for t in m.T) == m.steel_demand

        @model.Constraint(model.T)
        def total_power_input_constraint(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            power_input = eaf.power_in[t] + dri.power_in[t]
            if has_electrolyser:
                power_input += model.technology_blocks["electrolyser"].power_in[t]
            return m.total_power_input[t] == power_input

        @model.Constraint(model.T)
        def variable_cost_constraint(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            cost = eaf.operating_cost[t] + dri.operating_cost[t]
            if has_electrolyser:
                cost += model.technology_blocks["electrolyser"].operating_cost[t]
            return m.variable_cost[t] == cost

        model.objective = pyo.Objective(
            expr=sum(model.variable_cost[t] for t in model.T), sense=pyo.minimize
        )
        return model

    def _extract_results(
        self,
        model: pyo.ConcreteModel,
        forecasts: pd.DataFrame,
        solver_name: str,
    ) -> pd.DataFrame:
        dri = model.technology_blocks["dri_plant"]
        eaf = model.technology_blocks["eaf"]
        data: dict[str, list[float] | list[str]] = {
            "steel_demand_mode": [],
            "steel_demand_total_t": [],
            "total_electricity_consumption_MWh": [],
            "variable_cost_EUR": [],
            "dri_electricity_consumption_MWh": [],
            "dri_output_t": [],
            "hydrogen_consumption_MWh": [],
            "natural_gas_consumption_MWh": [],
            "iron_ore_consumption_t": [],
            "dri_co2_emissions_t": [],
            "eaf_electricity_consumption_MWh": [],
            "dri_input_t": [],
            "steel_output_t": [],
            "lime_consumption_t": [],
            "eaf_co2_emissions_t": [],
            "solver": [],
        }
        optional_variables = {
            "electrolyser_electricity_consumption_MWh": ("electrolyser", "power_in"),
            "electrolyser_hydrogen_output_MWh": ("electrolyser", "hydrogen_out"),
            "hydrogen_storage_charge_MWh": ("hydrogen_buffer_storage", "charge"),
            "hydrogen_storage_discharge_MWh": ("hydrogen_buffer_storage", "discharge"),
            "hydrogen_storage_soc": ("hydrogen_buffer_storage", "soc"),
            "dri_storage_charge_t": ("dri_storage", "charge"),
            "dri_storage_discharge_t": ("dri_storage", "discharge"),
            "dri_storage_soc": ("dri_storage", "soc"),
        }
        for column, (technology, _) in optional_variables.items():
            if technology in self.components:
                data[column] = []

        for t in model.T:
            data["steel_demand_mode"].append(
                "forecast_profile"
                if int(pyo.value(model.steel_demand_from_forecast))
                else "total_target"
            )
            data["steel_demand_total_t"].append(_value(model.steel_demand))
            data["total_electricity_consumption_MWh"].append(_value(model.total_power_input[t]))
            data["variable_cost_EUR"].append(_value(model.variable_cost[t]))
            data["dri_electricity_consumption_MWh"].append(_value(dri.power_in[t]))
            data["dri_output_t"].append(_value(dri.dri_output[t]))
            data["hydrogen_consumption_MWh"].append(_value(dri.hydrogen_in[t]))
            data["natural_gas_consumption_MWh"].append(_value(dri.natural_gas_in[t]))
            data["iron_ore_consumption_t"].append(_value(dri.iron_ore_in[t]))
            data["dri_co2_emissions_t"].append(_value(dri.co2_emission[t]))
            data["eaf_electricity_consumption_MWh"].append(_value(eaf.power_in[t]))
            data["dri_input_t"].append(_value(eaf.dri_input[t]))
            data["steel_output_t"].append(_value(eaf.steel_output[t]))
            data["lime_consumption_t"].append(_value(eaf.lime_demand[t]))
            data["eaf_co2_emissions_t"].append(_value(eaf.co2_emission[t]))
            data["solver"].append(solver_name)
            for column, (technology, variable) in optional_variables.items():
                if column in data:
                    data[column].append(
                        _value(getattr(model.technology_blocks[technology], variable)[t])
                    )
        return pd.DataFrame(data, index=forecasts.index)

    @staticmethod
    def _validate_signal_columns(forecasts: pd.DataFrame, signals: SteelDispatchSignals) -> None:
        columns = {
            signals.electricity_price_col,
            signals.natural_gas_price_col,
            signals.hydrogen_price_col,
            signals.iron_ore_price_col,
            signals.lime_price_col,
            signals.co2_price_col,
        }
        if signals.steel_price_col:
            columns.add(signals.steel_price_col)
        missing = columns - set(forecasts.columns)
        if missing:
            raise ValueError(
                "Steel dispatch forecasts are missing column(s): " + ", ".join(sorted(missing))
            )

    def _resolve_steel_demand(self, forecasts: pd.DataFrame) -> tuple[float, str]:
        if self.steel_demand_tonnes is not None:
            return self.steel_demand_tonnes, "total_target"

        if self.steel_demand_column not in forecasts.columns:
            raise ValueError(
                f"Steel plant '{self.name}' has no numeric steel_demand in plants.csv and "
                f"forecasts are missing demand column '{self.steel_demand_column}'"
            )

        raw_demand = forecasts[self.steel_demand_column]
        demand = pd.to_numeric(raw_demand, errors="coerce")
        if demand.isna().any():
            raise ValueError(
                f"Steel demand column '{self.steel_demand_column}' for plant '{self.name}' "
                "contains missing or non-numeric values"
            )
        if not demand.map(math.isfinite).all():
            raise ValueError(
                f"Steel demand column '{self.steel_demand_column}' for plant '{self.name}' "
                "contains non-finite values"
            )
        if (demand < 0).any():
            raise ValueError(
                f"Steel demand column '{self.steel_demand_column}' for plant '{self.name}' "
                "contains negative values"
            )
        return float(demand.sum()), "forecast_profile"


def _value(expression: Any) -> float:
    return float(pyo.value(expression))


def _consistent_optional_total(rows: pd.DataFrame, column: str, plant_name: str) -> float | None:
    values = _non_empty_values(rows, column)
    if not values:
        return None

    parsed: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Steel plant '{plant_name}' {column} must be a numeric simulation total"
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(f"Steel plant '{plant_name}' {column} must be finite")
        if numeric < 0:
            raise ValueError(f"Steel plant '{plant_name}' {column} must be non-negative")
        parsed.append(numeric)

    if len(set(parsed)) > 1:
        raise ValueError(
            f"Steel plant '{plant_name}' defines inconsistent {column} values across its rows"
        )
    return parsed[0]


def _consistent_optional_text(rows: pd.DataFrame, column: str, plant_name: str) -> str:
    values = [str(value).strip() for value in _non_empty_values(rows, column)]
    if len(set(values)) > 1:
        raise ValueError(
            f"Steel plant '{plant_name}' defines inconsistent {column} values across its rows"
        )
    return values[0] if values else ""


def _non_empty_values(rows: pd.DataFrame, column: str) -> list[Any]:
    if column not in rows.columns:
        return []
    return [value for value in rows[column].tolist() if not pd.isna(value) and str(value).strip()]
