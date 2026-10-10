# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Modeler-facing cement kiln-line dispatch model.

Technology blocks own their firing, material-throughput, and cost equations.
This plant module contains only the flows between the preheater, calciner, and
kiln, plus the clinker-demand constraint.  It follows ASSUME's DSM-unit style
without changing FLEXIMOD's market runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    validate_required_forecasts,
)
from flexi_mod.plants.base_plant import BasePlant
from flexi_mod.plants.technologies import Calciner, Kiln, Preheater, first_non_empty

_TECHNOLOGY_ALIASES = {
    "simple_calciner": "calciner",
    "simple_kiln": "kiln",
}


@dataclass
class CementPlant(BasePlant):
    """Cement kiln line with a preheater, calciner, and kiln process sequence."""

    required_technologies = ("preheater", "calciner", "kiln")
    clinker_demand_column: str = ""
    raw_meal_to_clinker_ratio: float = 1.55
    components: dict[str, object] = field(default_factory=dict)

    @classmethod
    def create(cls, plant_input: PlantInput) -> CementPlant:
        """Create a cement plant from the loader's parameters and components."""

        if plant_input.unit_type != "cement_plant":
            raise ValueError(
                f"Plant '{plant_input.name}' has unit_type='{plant_input.unit_type}', "
                "not 'cement_plant'"
            )
        return cls._assemble(plant_input.name, plant_input.component_table())

    @classmethod
    def _assemble(cls, plant_name: str, component_table: pd.DataFrame) -> CementPlant:
        """Create the physical cement model from one plant's component table."""

        if component_table.empty:
            raise ValueError(f"Cement plant '{plant_name}' has no technology rows")
        normalised = component_table.copy()
        normalised["technology"] = (
            normalised["technology"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(_TECHNOLOGY_ALIASES)
        )
        technologies = normalised["technology"].tolist()
        unknown = sorted(set(technologies) - set(cls.required_technologies))
        if unknown:
            raise ValueError(
                f"Cement plant '{plant_name}' uses unsupported technology: " + ", ".join(unknown)
            )
        for technology in cls.required_technologies:
            if technologies.count(technology) != 1:
                raise ValueError(
                    f"Cement plant '{plant_name}' needs exactly one '{technology}' row"
                )

        component_rows = normalised.set_index("technology", drop=False)
        preheater = component_from_row(
            "Cement plant",
            plant_name,
            "preheater",
            Preheater.from_row,
            component_rows.loc["preheater"],
        )
        calciner = component_from_row(
            "Cement plant",
            plant_name,
            "calciner",
            Calciner.from_row,
            component_rows.loc["calciner"],
        )
        kiln = component_from_row(
            "Cement plant",
            plant_name,
            "kiln",
            Kiln.from_row,
            component_rows.loc["kiln"],
        )
        demand_column = first_non_empty(normalised, "clinker_demand", default="")
        if not demand_column:
            demand_column = first_non_empty(normalised, "demand", default="")
        if not demand_column:
            demand_column = f"{plant_name}_clinker_demand"

        objective = first_non_empty(normalised, "objective", default="min_variable_cost")
        if objective != "min_variable_cost":
            raise ValueError("CementPlant currently supports only objective='min_variable_cost'")

        return cls(
            name=plant_name,
            unit_type=first_non_empty(normalised, "unit_type", default="cement_plant"),
            node=first_non_empty(normalised, "node", default=""),
            objective=objective,
            clinker_demand_column=demand_column,
            raw_meal_to_clinker_ratio=_plant_number(
                normalised,
                "raw_meal_to_clinker_ratio",
                default=1.55,
            ),
            components={"preheater": preheater, "calciner": calciner, "kiln": kiln},
        )

    @property
    def preheater(self) -> Preheater:
        return self.components["preheater"]  # type: ignore[return-value]

    @property
    def calciner(self) -> Calciner:
        return self.components["calciner"]  # type: ignore[return-value]

    @property
    def kiln(self) -> Kiln:
        return self.components["kiln"]  # type: ignore[return-value]

    def required_forecast_columns(self) -> set[str]:
        return {
            self.clinker_demand_column,
            "natural_gas_price",
            "coal_price",
            "hydrogen_price",
            "co2_price",
        }

    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
    ) -> pd.DataFrame:
        """Minimise feasible cement production cost for one dispatch horizon."""

        model = self.build_model(config, forecasts, electricity_price_column)
        solver_name = _solve_model(self.name, model, config)
        from flexi_mod.outputs.result_mappers import extract_dispatch_results

        return extract_dispatch_results(self, model, forecasts, solver_name)

    def build_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
    ) -> pyo.ConcreteModel:
        self.validate_inputs(forecasts, electricity_price_column)
        model = pyo.ConcreteModel(name=f"{self.name}_cement_dispatch")
        model.T = pyo.Set(initialize=range(len(forecasts)), ordered=True)
        self.define_parameters(model, config, forecasts, electricity_price_column)
        self.initialize_components(model)
        self.define_variables(model)
        self.define_constraints(model)
        self.define_objective(model)
        return model

    def validate_inputs(
        self,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
    ) -> None:
        """Validate this plant's forecast contract before building Pyomo objects."""

        required = self.required_forecast_columns() | {electricity_price_column}
        validate_required_forecasts(forecasts, required, f"Cement plant '{self.name}'")
        for column in sorted(required):
            numeric_forecast(
                forecasts,
                column,
                f"Cement plant '{self.name}'",
                require_non_negative=column == self.clinker_demand_column,
            )

    def define_parameters(
        self,
        model: pyo.ConcreteModel,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
    ) -> None:
        """Add production demand and the shared market/fuel-price parameters."""

        model.dt_hours = pyo.Param(initialize=config.timestep_minutes / 60.0)
        model.clinker_demand = time_parameter(
            model,
            numeric_forecast(
                forecasts,
                self.clinker_demand_column,
                f"Cement plant '{self.name}'",
                require_non_negative=True,
            ),
        )
        for parameter_name, column in {
            "electricity_price": electricity_price_column,
            "natural_gas_price": "natural_gas_price",
            "coal_price": "coal_price",
            "hydrogen_price": "hydrogen_price",
            "co2_price": "co2_price",
        }.items():
            setattr(
                model,
                parameter_name,
                time_parameter(
                    model,
                    numeric_forecast(forecasts, column, f"Cement plant '{self.name}'"),
                ),
            )

    def initialize_components(self, model: pyo.ConcreteModel) -> None:
        """Add the independently modelled thermal process-stage blocks."""

        context = {"dt_hours": pyo.value(model.dt_hours)}
        for name, component in self.components.items():
            block = pyo.Block()
            setattr(model, name, block)
            component.add_to_model(model, block, model.T, context)  # type: ignore[union-attr]

    def define_variables(self, model: pyo.ConcreteModel) -> None:
        model.total_power_input = pyo.Var(model.T, within=pyo.NonNegativeReals)
        model.variable_cost = pyo.Var(model.T, within=pyo.Reals)

    def define_constraints(self, model: pyo.ConcreteModel) -> None:
        """Connect raw meal to clinker and enforce production requirements."""

        @model.Constraint(model.T)
        def raw_meal_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.preheater.raw_meal_out[t] == (
                m.calciner.clinker_out[t] * self.raw_meal_to_clinker_ratio
            )

        @model.Constraint(model.T)
        def clinker_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.calciner.clinker_out[t] == m.kiln.clinker_out[t]

        @model.Constraint(model.T)
        def demand_coverage(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.kiln.clinker_out[t] >= m.clinker_demand[t]

        @model.Constraint(model.T)
        def electricity_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.total_power_input[t] == sum(
                getattr(m, name).power_in[t] + getattr(m, name).aux_power[t]
                for name in self.components
            )

        @model.Constraint(model.T)
        def total_variable_cost(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.variable_cost[t] == sum(
                getattr(m, name).operating_cost[t] for name in self.components
            )

    def define_objective(self, model: pyo.ConcreteModel) -> None:
        model.objective = pyo.Objective(
            expr=pyo.quicksum(model.variable_cost[t] for t in model.T),
            sense=pyo.minimize,
        )


def _plant_number(rows: pd.DataFrame, column: str, default: float) -> float:
    if column not in rows.columns:
        return default
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    if values.empty:
        return default
    if not values.eq(values.iloc[0]).all():
        raise ValueError(f"Cement plant has inconsistent '{column}' values")
    return float(values.iloc[0])


def _solve_model(plant_name: str, model: pyo.ConcreteModel, config: CaseConfig) -> str:
    errors: list[str] = []
    for solver_name, solver in available_pyomo_solvers(config):
        try:
            result = solver.solve(model, tee=config.solver_tee)
            status = result.solver.status
            termination = result.solver.termination_condition
            if status == SolverStatus.ok and termination in {
                TerminationCondition.optimal,
                TerminationCondition.feasible,
            }:
                return solver_name
            if is_infeasible_termination(termination):
                raise RuntimeError(
                    f"Cement dispatch for '{plant_name}' is infeasible with {solver_name}"
                )
            errors.append(f"{solver_name}: status={status}, termination={termination}")
        except (ApplicationError, NoFeasibleSolutionError) as exc:
            errors.append(f"{solver_name}: {exc}")
    raise RuntimeError("Cement dispatch failed for all configured solvers. " + " | ".join(errors))
