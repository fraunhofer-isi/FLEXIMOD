# SPDX-FileCopyrightText: FLEXIMOD Developers
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
    CEMENT_FOSSIL,
    CEMENT_HYBRID_ELECTRICITY_FOSSIL,
    TECHNOLOGY_REGISTRY,
    first_non_empty,
)


@dataclass(frozen=True)
class CementDispatchSignals:
    """Forecast-column mapping required for cement clinker dispatch."""

    electricity_price_col: str
    natural_gas_price_col: str = "natural_gas_price"
    hydrogen_price_col: str = "hydrogen_price"
    coal_price_col: str = "coal_price"
    co2_price_col: str = "co2_price"


@dataclass
class CementPlant(BasePlant):
    """Cement clinker production model without grinding-mill integration.

    Supported technologies for this first FLEXIMOD cement version are
    ``preheater``, ``calciner``, ``kiln``, optional ``electrolyser``, optional
    ``hydrogen_buffer_storage``, and optional ``thermal_storage``. ``cement_mill``
    is intentionally rejected until grinding is modelled.
    """

    clinker_demand_tonnes: float | None = None
    clinker_demand_column: str = ""
    raw_meal_to_clinker_ratio: float = 1.55
    waste_heat_per_t_clinker_mwh: float = 0.22
    waste_heat_utilization_efficiency: float = 0.90
    components: dict[str, object] = field(default_factory=dict)

    allowed_technologies = frozenset(
        {
            "preheater",
            "calciner",
            "kiln",
            "electrolyser",
            "hydrogen_buffer_storage",
            "thermal_storage",
        }
    )
    excluded_technologies = frozenset({"cement_mill", "grinding_mill"})

    @classmethod
    def from_rows(cls, plant_name: str, rows: pd.DataFrame) -> CementPlant:
        normalised = rows.copy()
        normalised["technology_normalised"] = (
            normalised["technology"].astype(str).str.strip().str.lower()
        )
        components: dict[str, object] = {}
        for _, row in normalised.iterrows():
            technology = str(row["technology_normalised"])
            if technology in cls.excluded_technologies:
                raise ValueError(
                    f"Cement plant '{plant_name}' technology '{technology}' is not supported "
                    "yet; grinding mill integration is intentionally excluded"
                )
            if technology not in cls.allowed_technologies:
                raise ValueError(
                    f"Cement plant '{plant_name}' uses unsupported technology '{technology}'"
                )
            if technology in components:
                raise ValueError(
                    f"Cement plant '{plant_name}' defines technology '{technology}' more than once"
                )
            components[technology] = TECHNOLOGY_REGISTRY[technology].from_row(row)

        _validate_cement_route(components, plant_name)
        clinker_demand = _consistent_optional_total(rows, "clinker_demand", plant_name)
        clinker_demand_column = _consistent_optional_text(rows, "demand", plant_name)
        if not clinker_demand_column:
            clinker_demand_column = f"{plant_name}_clinker_demand"

        return cls(
            name=plant_name,
            unit_type=first_non_empty(rows, "unit_type", default="cement_plant"),
            node=first_non_empty(rows, "node", default=""),
            objective=first_non_empty(rows, "objective", default="min_variable_cost"),
            clinker_demand_tonnes=clinker_demand,
            clinker_demand_column=clinker_demand_column,
            raw_meal_to_clinker_ratio=_consistent_optional_total(
                rows,
                "raw_meal_to_clinker_ratio",
                plant_name,
                default=1.55,
            ),
            waste_heat_per_t_clinker_mwh=_consistent_optional_total(
                rows,
                "waste_heat_per_t_clinker",
                plant_name,
                default=0.22,
            ),
            waste_heat_utilization_efficiency=_consistent_optional_total(
                rows,
                "waste_heat_utilization_efficiency",
                plant_name,
                default=0.90,
            ),
            components=components,
        )

    @classmethod
    def from_plants_dataframe(cls, plants: pd.DataFrame) -> list[CementPlant]:
        return [
            cls.from_rows(str(plant_name), rows)
            for plant_name, rows in plants.groupby("name", sort=False)
        ]

    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: CementDispatchSignals,
    ) -> pd.DataFrame:
        if self.objective != "min_variable_cost":
            raise ValueError(f"Cement plant objective '{self.objective}' is not supported")
        if forecasts.empty:
            raise ValueError("Cement plant dispatch horizon must contain at least one timestep")
        self._validate_signal_columns(forecasts, signals)
        model = self._build_model(config, forecasts, signals)
        return self._solve_model(config, forecasts, model)

    def _build_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: CementDispatchSignals,
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
        coal_prices = (
            values(signals.coal_price_col)
            if self._requires_coal_price()
            else {t: 0.0 for t in model.T}
        )
        model.coal_price = pyo.Param(model.T, initialize=coal_prices)
        model.co2_price = pyo.Param(model.T, initialize=values(signals.co2_price_col))

        clinker_demand_total, demand_mode = self._resolve_clinker_demand(forecasts)
        model.clinker_demand = pyo.Param(initialize=clinker_demand_total)
        model.clinker_demand_from_forecast = pyo.Param(
            initialize=int(demand_mode == "forecast_profile"), within=pyo.Binary
        )
        model.raw_meal_to_clinker_ratio = pyo.Param(initialize=self.raw_meal_to_clinker_ratio)
        model.waste_heat_per_t_clinker = pyo.Param(initialize=self.waste_heat_per_t_clinker_mwh)
        model.waste_heat_utilization_efficiency = pyo.Param(
            initialize=self.waste_heat_utilization_efficiency
        )

        self._add_physical_system(model, model.T, dt_hours)
        model.objective = pyo.Objective(
            expr=sum(model.variable_cost[t] for t in model.T), sense=pyo.minimize
        )
        return model

    def _add_physical_system(
        self,
        model: pyo.ConcreteModel,
        time_steps: pyo.Set,
        dt_hours: float,
    ) -> None:
        model.technology_blocks = pyo.Block(list(self.components))
        for technology, component in self.components.items():
            component.add_to_model(
                model,
                model.technology_blocks[technology],
                time_steps,
                {"dt_hours": dt_hours},
            )

        blocks = model.technology_blocks
        preheater = blocks["preheater"] if "preheater" in self.components else None
        calciner = blocks["calciner"] if "calciner" in self.components else None
        kiln = blocks["kiln"] if "kiln" in self.components else None
        terminal = kiln if kiln is not None else calciner

        if preheater is not None and kiln is not None:

            @model.Constraint(time_steps)
            def waste_heat_to_preheater(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                return preheater.external_heat_in[t] == (
                    kiln.clinker_out[t]
                    * m.waste_heat_per_t_clinker
                    * m.waste_heat_utilization_efficiency
                )

        elif preheater is not None:

            @model.Constraint(time_steps)
            def no_external_preheater_heat(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                return preheater.external_heat_in[t] == 0.0

        if preheater is not None and calciner is not None:

            @model.Constraint(time_steps)
            def preheater_to_calciner_mass_link(
                m: pyo.ConcreteModel,
                t: int,
            ) -> pyo.Constraint:
                return (
                    calciner.clinker_out[t]
                    == preheater.raw_meal_out[t] / m.raw_meal_to_clinker_ratio
                )

        if calciner is not None:
            if "thermal_storage" in self.components:
                storage = blocks["thermal_storage"]

                @model.Constraint(time_steps)
                def calciner_effective_heat_with_storage(
                    m: pyo.ConcreteModel,
                    t: int,
                ) -> pyo.Constraint:
                    return calciner.effective_heat_in[t] == (
                        calciner.heat_out[t] + storage.discharge_heat[t]
                    )

            else:

                @model.Constraint(time_steps)
                def calciner_effective_heat_without_storage(
                    m: pyo.ConcreteModel,
                    t: int,
                ) -> pyo.Constraint:
                    return calciner.effective_heat_in[t] == calciner.heat_out[t]

        if calciner is not None and kiln is not None:

            @model.Constraint(time_steps)
            def calciner_to_kiln_mass_link(
                m: pyo.ConcreteModel,
                t: int,
            ) -> pyo.Constraint:
                return kiln.clinker_out[t] == calciner.clinker_out[t]

        if "electrolyser" in self.components:
            electrolyser = blocks["electrolyser"]
            hydrogen_demands = []
            if calciner is not None:
                hydrogen_demands.append(calciner.hydrogen_in)
            if kiln is not None:
                hydrogen_demands.append(kiln.hydrogen_in)

            def total_hydrogen_demand(t: int) -> pyo.Expression:
                return sum(demand[t] for demand in hydrogen_demands)

            if "hydrogen_buffer_storage" in self.components:
                hydrogen_storage = blocks["hydrogen_buffer_storage"]

                @model.Constraint(time_steps)
                def hydrogen_balance_with_storage(
                    m: pyo.ConcreteModel,
                    t: int,
                ) -> pyo.Constraint:
                    return electrolyser.hydrogen_out[t] + hydrogen_storage.discharge[t] == (
                        total_hydrogen_demand(t) + hydrogen_storage.charge[t]
                    )

            else:

                @model.Constraint(time_steps)
                def hydrogen_balance_without_storage(
                    m: pyo.ConcreteModel,
                    t: int,
                ) -> pyo.Constraint:
                    return electrolyser.hydrogen_out[t] == total_hydrogen_demand(t)

        model.total_power_input = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        model.variable_cost = pyo.Var(time_steps, within=pyo.Reals)

        @model.Constraint()
        def clinker_output_association_constraint(m: pyo.ConcreteModel) -> pyo.Constraint:
            return sum(terminal.clinker_out[t] for t in time_steps) == m.clinker_demand

        @model.Constraint(time_steps)
        def total_power_input_constraint(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            total = 0.0
            for block in m.technology_blocks.values():
                if hasattr(block, "power_in"):
                    total += block.power_in[t]
                if hasattr(block, "aux_power_in"):
                    total += block.aux_power_in[t]
                if hasattr(block, "electricity_consumption"):
                    total += block.electricity_consumption[t]
            return m.total_power_input[t] == total

        @model.Constraint(time_steps)
        def variable_cost_constraint(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.variable_cost[t] == sum(
                block.operating_cost[t]
                for block in m.technology_blocks.values()
                if hasattr(block, "operating_cost")
            )

    def _solve_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        model: pyo.ConcreteModel,
    ) -> pd.DataFrame:
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
        raise RuntimeError("Cement plant dispatch could not be solved; " + "; ".join(errors))

    def _extract_results(
        self,
        model: pyo.ConcreteModel,
        forecasts: pd.DataFrame,
        solver_name: str,
    ) -> pd.DataFrame:
        blocks = model.technology_blocks
        preheater = blocks["preheater"] if "preheater" in self.components else None
        calciner = blocks["calciner"] if "calciner" in self.components else None
        kiln = blocks["kiln"] if "kiln" in self.components else None
        terminal = kiln if kiln is not None else calciner

        def block_value(block: pyo.Block | None, variable: str, t: int) -> float:
            if block is None or not hasattr(block, variable):
                return 0.0
            return _value(getattr(block, variable)[t])

        data: dict[str, list[float] | list[str]] = {
            "plant_name": [],
            "plant_type": [],
            "clinker_demand_mode": [],
            "clinker_demand_total_t": [],
            "clinker_output_t": [],
            "total_electricity_consumption_MWh": [],
            "natural_gas_consumption_MWh": [],
            "coal_consumption_MWh": [],
            "hydrogen_consumption_MWh": [],
            "co2_emissions_t": [],
            "variable_cost_EUR": [],
            "preheater_heat_output_MWh": [],
            "preheater_raw_meal_output_t": [],
            "preheater_electricity_consumption_MWh": [],
            "calciner_heat_output_MWh": [],
            "calciner_effective_heat_MWh": [],
            "calciner_clinker_output_t": [],
            "calciner_electricity_consumption_MWh": [],
            "calciner_process_co2_emissions_t": [],
            "kiln_heat_output_MWh": [],
            "kiln_clinker_output_t": [],
            "kiln_electricity_consumption_MWh": [],
            "solver": [],
        }
        optional_variables = {
            "electrolyser_electricity_consumption_MWh": ("electrolyser", "power_in"),
            "electrolyser_hydrogen_output_MWh": ("electrolyser", "hydrogen_out"),
            "hydrogen_storage_charge_MWh": ("hydrogen_buffer_storage", "charge"),
            "hydrogen_storage_discharge_MWh": ("hydrogen_buffer_storage", "discharge"),
            "hydrogen_storage_soc": ("hydrogen_buffer_storage", "soc"),
            "thermal_storage_charge_MWh": ("thermal_storage", "electric_charge_to_storage"),
            "thermal_storage_discharge_MWh": ("thermal_storage", "discharge_heat"),
            "thermal_storage_soc_MWh": ("thermal_storage", "soc"),
        }
        for column, (technology, _) in optional_variables.items():
            if technology in self.components:
                data[column] = []

        for t in model.T:
            data["plant_name"].append(self.name)
            data["plant_type"].append(self.unit_type)
            data["clinker_demand_mode"].append(
                "forecast_profile"
                if int(pyo.value(model.clinker_demand_from_forecast))
                else "total_target"
            )
            data["clinker_demand_total_t"].append(_value(model.clinker_demand))
            data["clinker_output_t"].append(block_value(terminal, "clinker_out", t))
            data["total_electricity_consumption_MWh"].append(_value(model.total_power_input[t]))
            data["natural_gas_consumption_MWh"].append(
                sum(
                    block_value(block, "natural_gas_in", t) for block in [preheater, calciner, kiln]
                )
            )
            data["coal_consumption_MWh"].append(
                sum(block_value(block, "coal_in", t) for block in [preheater, calciner, kiln])
            )
            data["hydrogen_consumption_MWh"].append(
                sum(block_value(block, "hydrogen_in", t) for block in [preheater, calciner, kiln])
            )
            data["co2_emissions_t"].append(
                sum(block_value(block, "co2_emission", t) for block in [preheater, calciner, kiln])
            )
            data["variable_cost_EUR"].append(_value(model.variable_cost[t]))
            data["preheater_heat_output_MWh"].append(block_value(preheater, "heat_out", t))
            data["preheater_raw_meal_output_t"].append(block_value(preheater, "raw_meal_out", t))
            data["preheater_electricity_consumption_MWh"].append(
                block_value(preheater, "power_in", t) + block_value(preheater, "aux_power_in", t)
            )
            data["calciner_heat_output_MWh"].append(block_value(calciner, "heat_out", t))
            data["calciner_effective_heat_MWh"].append(
                block_value(calciner, "effective_heat_in", t)
            )
            data["calciner_clinker_output_t"].append(block_value(calciner, "clinker_out", t))
            data["calciner_electricity_consumption_MWh"].append(
                block_value(calciner, "power_in", t) + block_value(calciner, "aux_power_in", t)
            )
            data["calciner_process_co2_emissions_t"].append(block_value(calciner, "co2_process", t))
            data["kiln_heat_output_MWh"].append(block_value(kiln, "heat_out", t))
            data["kiln_clinker_output_t"].append(block_value(kiln, "clinker_out", t))
            data["kiln_electricity_consumption_MWh"].append(
                block_value(kiln, "power_in", t) + block_value(kiln, "aux_power_in", t)
            )
            data["solver"].append(solver_name)
            for column, (technology, variable) in optional_variables.items():
                if column in data:
                    block = blocks[technology]
                    data[column].append(_value(getattr(block, variable)[t]))

        return pd.DataFrame(data, index=forecasts.index)

    def _resolve_clinker_demand(self, forecasts: pd.DataFrame) -> tuple[float, str]:
        if self.clinker_demand_tonnes is not None:
            return self.clinker_demand_tonnes, "total_target"
        demand = self._validated_demand_profile(forecasts)
        return float(demand.sum()), "forecast_profile"

    def _validated_demand_profile(self, forecasts: pd.DataFrame) -> pd.Series:
        if self.clinker_demand_column not in forecasts.columns:
            raise ValueError(
                f"Cement plant '{self.name}' has no numeric clinker_demand in plants.csv and "
                f"forecasts are missing demand column '{self.clinker_demand_column}'"
            )
        demand = pd.to_numeric(forecasts[self.clinker_demand_column], errors="coerce")
        if demand.isna().any():
            raise ValueError(
                f"Clinker demand column '{self.clinker_demand_column}' for plant "
                f"'{self.name}' contains missing or non-numeric values"
            )
        if not demand.map(math.isfinite).all():
            raise ValueError(
                f"Clinker demand column '{self.clinker_demand_column}' for plant "
                f"'{self.name}' contains non-finite values"
            )
        if (demand < 0).any():
            raise ValueError(
                f"Clinker demand column '{self.clinker_demand_column}' for plant "
                f"'{self.name}' contains negative values"
            )
        return demand.astype(float)

    def _requires_coal_price(self) -> bool:
        for component in self.components.values():
            fuel_type = getattr(component, "fuel_type", "")
            fossil_share = float(getattr(component, "fossil_ng_share", 1.0))
            if fuel_type in {CEMENT_FOSSIL, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
                if fossil_share < 1.0 - 1e-12:
                    return True
        return False

    def _validate_signal_columns(
        self,
        forecasts: pd.DataFrame,
        signals: CementDispatchSignals,
    ) -> None:
        columns = {
            signals.electricity_price_col,
            signals.natural_gas_price_col,
            signals.hydrogen_price_col,
            signals.co2_price_col,
        }
        if self._requires_coal_price():
            columns.add(signals.coal_price_col)
        missing = columns - set(forecasts.columns)
        if missing:
            raise ValueError(
                "Cement dispatch forecasts are missing column(s): " + ", ".join(sorted(missing))
            )


def _validate_cement_route(components: dict[str, object], plant_name: str) -> None:
    if "calciner" not in components and "kiln" not in components:
        raise ValueError(
            f"Cement plant '{plant_name}' must define at least one terminal technology: "
            "calciner or kiln"
        )
    if "preheater" in components and "calciner" not in components:
        raise ValueError(
            f"Cement plant '{plant_name}' preheater currently requires a calciner downstream"
        )
    if "thermal_storage" in components and "calciner" not in components:
        raise ValueError(
            f"Cement plant '{plant_name}' thermal_storage currently buffers the calciner and "
            "therefore requires a calciner"
        )
    if "hydrogen_buffer_storage" in components and "electrolyser" not in components:
        raise ValueError(
            f"Cement plant '{plant_name}' defines hydrogen_buffer_storage without an electrolyser"
        )


def _consistent_optional_total(
    rows: pd.DataFrame,
    column: str,
    plant_name: str,
    default: float | None = None,
) -> float | None:
    values = _non_empty_values(rows, column)
    if not values:
        return default
    parsed: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cement plant '{plant_name}' {column} must be numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"Cement plant '{plant_name}' {column} must be finite")
        if numeric < 0:
            raise ValueError(f"Cement plant '{plant_name}' {column} must be non-negative")
        parsed.append(numeric)
    if len(set(parsed)) > 1:
        raise ValueError(
            f"Cement plant '{plant_name}' defines inconsistent {column} values across its rows"
        )
    return parsed[0]


def _consistent_optional_text(rows: pd.DataFrame, column: str, plant_name: str) -> str:
    values = [str(value).strip() for value in _non_empty_values(rows, column)]
    if len(set(values)) > 1:
        raise ValueError(
            f"Cement plant '{plant_name}' defines inconsistent {column} values across its rows"
        )
    return values[0] if values else ""


def _non_empty_values(rows: pd.DataFrame, column: str) -> list[Any]:
    if column not in rows.columns:
        return []
    return [value for value in rows[column].tolist() if not pd.isna(value) and str(value).strip()]


def _value(expression: Any) -> float:
    return float(pyo.value(expression))
