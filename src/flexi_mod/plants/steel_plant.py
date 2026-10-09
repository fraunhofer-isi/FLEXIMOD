# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Modeler-facing DRI/EAF steel plant model.

The module follows the same modelling sequence used by ASSUME's DSM units:
read component rows, add independent Pyomo blocks for each technology, and
write only the plant-level material balances and objective here.  It is kept
independent from FLEXIMOD's market runner so a later steel strategy can reuse
the physical model without duplicating its equations.
"""

from __future__ import annotations

from collections.abc import Callable
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
    DRIPlant,
    ElectricArcFurnace,
    Electrolyser,
    first_non_empty,
)


@dataclass
class SteelPlant(BasePlant):
    """A DRI/EAF steel route with optional on-site hydrogen production."""

    required_technologies = ("dri_plant", "eaf")
    optional_technologies = ("electrolyser",)

    steel_demand_column: str = ""
    components: dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Input rows -> physical technology instances
    # ------------------------------------------------------------------
    @classmethod
    def from_rows(cls, plant_name: str, rows: pd.DataFrame) -> SteelPlant:
        if rows.empty:
            raise ValueError(f"Steel plant '{plant_name}' has no technology rows")

        normalised = rows.copy()
        normalised["technology"] = normalised["technology"].astype(str).str.strip().str.lower()
        technologies = normalised["technology"].tolist()
        allowed = set(cls.required_technologies) | set(cls.optional_technologies)
        unknown = sorted(set(technologies) - allowed)
        if unknown:
            raise ValueError(
                f"Steel plant '{plant_name}' uses unsupported technology: " + ", ".join(unknown)
            )
        for technology in cls.required_technologies:
            if technologies.count(technology) != 1:
                raise ValueError(f"Steel plant '{plant_name}' needs exactly one '{technology}' row")
        if technologies.count("electrolyser") > 1:
            raise ValueError(f"Steel plant '{plant_name}' can define at most one electrolyser")

        component_rows = normalised.set_index("technology", drop=False)
        dri = _component_from_row(
            plant_name,
            "dri_plant",
            DRIPlant.from_row,
            component_rows.loc["dri_plant"],
        )
        eaf = _component_from_row(
            plant_name,
            "eaf",
            ElectricArcFurnace.from_row,
            component_rows.loc["eaf"],
        )
        components: dict[str, object] = {"dri_plant": dri, "eaf": eaf}
        if "electrolyser" in component_rows.index:
            components["electrolyser"] = _component_from_row(
                plant_name,
                "electrolyser",
                Electrolyser.from_row,
                component_rows.loc["electrolyser"],
            )

        demand_column = first_non_empty(normalised, "steel_demand", default="")
        if not demand_column:
            demand_column = first_non_empty(normalised, "demand", default="")
        if not demand_column:
            demand_column = f"{plant_name}_steel_demand"

        objective = first_non_empty(normalised, "objective", default="min_variable_cost")
        if objective != "min_variable_cost":
            raise ValueError("SteelPlant currently supports only objective='min_variable_cost'")

        return cls(
            name=plant_name,
            unit_type=first_non_empty(normalised, "unit_type", default="steel_plant"),
            node=first_non_empty(normalised, "node", default=""),
            objective=objective,
            steel_demand_column=demand_column,
            components=components,
        )

    @property
    def dri_plant(self) -> DRIPlant:
        return self.components["dri_plant"]  # type: ignore[return-value]

    @property
    def eaf(self) -> ElectricArcFurnace:
        return self.components["eaf"]  # type: ignore[return-value]

    @property
    def electrolyser(self) -> Electrolyser | None:
        component = self.components.get("electrolyser")
        return component if isinstance(component, Electrolyser) else None

    def required_forecast_columns(self) -> set[str]:
        """Return physical demand and price profiles required by this route."""

        required = {
            self.steel_demand_column,
            "iron_ore_price",
            "lime_price",
            "co2_price",
        }
        if self.dri_plant.fuel_type in {"natural_gas", "both"}:
            required.add("natural_gas_price")
        if self.dri_plant.fuel_type in {"hydrogen", "both"} and self.electrolyser is None:
            required.add("hydrogen_price")
        return required

    # ------------------------------------------------------------------
    # Public dispatch API
    # ------------------------------------------------------------------
    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
    ) -> pd.DataFrame:
        """Minimise physical operating cost for one steel-dispatch horizon."""

        model = self.build_model(config, forecasts, electricity_price_column)
        solver_name = _solve_model(self.name, model, config)
        return _extract_results(self, model, forecasts, solver_name)

    def build_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
    ) -> pyo.ConcreteModel:
        """Build the Pyomo model without solving it."""

        self.validate_inputs(forecasts, electricity_price_column)
        model = pyo.ConcreteModel(name=f"{self.name}_steel_dispatch")
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
        _check_forecasts(forecasts, required, self.name)
        for column in sorted(required):
            _numeric_forecast(
                forecasts,
                column,
                self.name,
                require_non_negative=column == self.steel_demand_column,
            )

    # ------------------------------------------------------------------
    # Parameters -> components -> variables -> plant balances -> objective
    # ------------------------------------------------------------------
    def define_parameters(
        self,
        model: pyo.ConcreteModel,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        electricity_price_column: str,
    ) -> None:
        """Add unit-consistent demand and commodity-price parameters."""

        model.dt_hours = pyo.Param(initialize=config.timestep_minutes / 60.0)
        model.steel_demand = _time_parameter(
            model,
            _numeric_forecast(
                forecasts,
                self.steel_demand_column,
                self.name,
                require_non_negative=True,
            ),
        )
        for parameter_name, column in {
            "electricity_price": electricity_price_column,
            "natural_gas_price": "natural_gas_price",
            "hydrogen_price": "hydrogen_price",
            "iron_ore_price": "iron_ore_price",
            "lime_price": "lime_price",
            "co2_price": "co2_price",
        }.items():
            values = (
                _numeric_forecast(forecasts, column, self.name)
                if column in forecasts.columns
                else pd.Series(0.0, index=forecasts.index)
            )
            setattr(model, parameter_name, _time_parameter(model, values))

    def initialize_components(self, model: pyo.ConcreteModel) -> None:
        """Add independent DRI, EAF, and optional electrolyser blocks."""

        context = {"dt_hours": pyo.value(model.dt_hours)}
        model.dri_plant = pyo.Block()
        self.dri_plant.add_to_model(model, model.dri_plant, model.T, context)
        model.eaf = pyo.Block()
        self.eaf.add_to_model(model, model.eaf, model.T, context)
        if self.electrolyser is not None:
            model.electrolyser = pyo.Block()
            self.electrolyser.add_to_model(model, model.electrolyser, model.T, context)

    def define_variables(self, model: pyo.ConcreteModel) -> None:
        """Add plant-level reporting variables."""

        model.total_power_input = pyo.Var(model.T, within=pyo.NonNegativeReals)
        model.variable_cost = pyo.Var(model.T, within=pyo.Reals)

    def define_constraints(self, model: pyo.ConcreteModel) -> None:
        """Connect the DRI/EAF chain and enforce the steel-demand profile."""

        @model.Constraint(model.T)
        def dri_to_eaf_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.dri_plant.dri_output[t] == m.eaf.dri_input[t]

        if self.electrolyser is not None:

            @model.Constraint(model.T)
            def on_site_hydrogen_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                return m.electrolyser.hydrogen_out[t] == m.dri_plant.hydrogen_in[t]

        @model.Constraint(model.T)
        def demand_coverage(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.eaf.steel_output[t] >= m.steel_demand[t]

        @model.Constraint(model.T)
        def electricity_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            electrolyser_power = m.electrolyser.power_in[t] if hasattr(m, "electrolyser") else 0.0
            return m.total_power_input[t] == (
                m.dri_plant.power_in[t] + m.eaf.power_in[t] + electrolyser_power
            )

        @model.Constraint(model.T)
        def total_variable_cost(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            electrolyser_cost = (
                m.electrolyser.operating_cost[t] if hasattr(m, "electrolyser") else 0.0
            )
            return m.variable_cost[t] == (
                m.dri_plant.operating_cost[t] + m.eaf.operating_cost[t] + electrolyser_cost
            )

    def define_objective(self, model: pyo.ConcreteModel) -> None:
        """Minimise component operating costs over the model horizon."""

        model.objective = pyo.Objective(
            expr=pyo.quicksum(model.variable_cost[t] for t in model.T),
            sense=pyo.minimize,
        )


def _time_parameter(model: pyo.ConcreteModel, values: pd.Series) -> pyo.Param:
    return pyo.Param(
        model.T,
        initialize={t: float(values.iloc[t]) for t in model.T},
    )


def _numeric_forecast(
    forecasts: pd.DataFrame,
    column: str,
    plant_name: str,
    require_non_negative: bool = False,
) -> pd.Series:
    values = pd.to_numeric(forecasts[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(
            f"Steel plant '{plant_name}' forecast '{column}' must be finite and numeric"
        )
    if require_non_negative and (values < 0.0).any():
        raise ValueError(f"Steel plant '{plant_name}' forecast '{column}' cannot be negative")
    return values.astype(float)


def _check_forecasts(
    forecasts: pd.DataFrame,
    required_columns: set[str],
    plant_name: str,
) -> None:
    missing = sorted(required_columns - set(forecasts.columns))
    if missing:
        raise ValueError(
            f"Steel plant '{plant_name}' cannot build dispatch: forecasts_df.csv is missing "
            "required column(s): " + ", ".join(missing)
        )


def _component_from_row[T](
    plant_name: str,
    technology: str,
    factory: Callable[[pd.Series], T],
    row: pd.Series,
) -> T:
    """Add plant and technology context to component-row validation errors."""

    try:
        return factory(row)
    except ValueError as exc:
        raise ValueError(f"Steel plant '{plant_name}', technology '{technology}': {exc}") from exc


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
                    f"Steel dispatch for '{plant_name}' is infeasible with {solver_name}"
                )
            errors.append(f"{solver_name}: status={status}, termination={termination}")
        except (ApplicationError, NoFeasibleSolutionError) as exc:
            errors.append(f"{solver_name}: {exc}")
    raise RuntimeError("Steel dispatch failed for all configured solvers. " + " | ".join(errors))


def _extract_results(
    plant: SteelPlant,
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    solver_name: str,
) -> pd.DataFrame:
    records = []
    for t, _timestamp in enumerate(forecasts.index):
        row = {
            "plant_name": plant.name,
            "unit_type": plant.unit_type,
            "steel_demand_t": pyomo_value(model.steel_demand[t]),
            "steel_output_t": pyomo_value(model.eaf.steel_output[t]),
            "dri_output_t": pyomo_value(model.dri_plant.dri_output[t]),
            "electricity_consumption_MWh": pyomo_value(model.total_power_input[t]),
            "natural_gas_consumption_MWh": pyomo_value(model.dri_plant.natural_gas_in[t]),
            "hydrogen_consumption_MWh": pyomo_value(model.dri_plant.hydrogen_in[t]),
            "variable_cost_EUR": pyomo_value(model.variable_cost[t]),
            "solver": solver_name,
        }
        if hasattr(model, "electrolyser"):
            row["hydrogen_production_MWh"] = pyomo_value(model.electrolyser.hydrogen_out[t])
        records.append(row)
    return pd.DataFrame(records, index=forecasts.index)
