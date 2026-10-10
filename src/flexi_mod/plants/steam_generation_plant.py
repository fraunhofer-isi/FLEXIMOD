# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Steam-generation plant with shared market stages and route-specific physics.

The file is ordered for model users: plant construction and solve methods come
first, followed by market inputs, physical route equations, and low-level
helpers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.opt import SolverStatus, TerminationCondition

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import PlantInput
from flexi_mod.markets.afrr_capacity import BalancingCapacityAward
from flexi_mod.markets.afrr_energy import BalancingEnergyActivation
from flexi_mod.markets.day_ahead import DayAheadPosition
from flexi_mod.markets.electricity_settlement import (
    ElectricityMarketRequest,
    ElectricityMarketStage,
    ElectricityMarketStageInput,
    add_stage_to_model,
    market_name_for_stage,
    stage_label,
    stage_warning_label,
    validate_stage_input,
)
from flexi_mod.markets.intraday_continuous import IntradayAdjustment
from flexi_mod.modeling.pyomo_utils import (
    available_pyomo_solvers,
    forecast_values_or_zero,
    is_infeasible_termination,
    pyomo_value,
    series_or_zero,
    slice_dataclass_series,
)
from flexi_mod.modeling.validation import (
    component_from_row,
    numeric_forecast,
    validate_required_forecasts,
)
from flexi_mod.plants.base_plant import BasePlant
from flexi_mod.plants.technologies import (
    TECHNOLOGY_REGISTRY,
    ElectricBoiler,
    GasBoiler,
    ThermalStorage,
    first_non_empty,
)
from flexi_mod.simulation.market_stages import MarketStageInstruction

THERMAL_STORAGE_GAS_BOILER_ROUTE = "thermal_storage_gas_boiler"
DIRECT_ELECTRIC_GAS_BOILER_ROUTE = "direct_electric_gas_boiler"
DELIVERY_GUARANTEE_SLACK_PENALTY_EUR_PER_MWH = 1_000_000.0
THERMAL_STORAGE_SOC_STATE_KEY = "thermal_storage_soc_mwh"


# ---------------------------------------------------------------------------
# Plant model: start here
# ---------------------------------------------------------------------------
@dataclass
class SteamGenerationPlant(BasePlant):
    """Optimize one steam plant through DA, IDC, and aFRR market stages.

    Supported routes are thermal storage plus a gas boiler, or a direct
    electric boiler plus a gas boiler. The plant factory creates it from the
    loader's ``PlantInput``. Both routes use the same market-stage engine.
    """

    gas_emissions_factor_kg_per_mwh: float = 201.0
    components: dict[str, object] = field(default_factory=dict)

    # --- Construction -----------------------------------------------------

    @classmethod
    def create(cls, plant_input: PlantInput) -> SteamGenerationPlant:
        """Create a steam plant from parameters and components prepared by the loader."""

        if plant_input.unit_type != "steam_plant":
            raise ValueError(
                f"Plant '{plant_input.name}' has unit_type='{plant_input.unit_type}', "
                "not 'steam_plant'"
            )
        return cls._assemble(plant_input.name, plant_input.component_table())

    @classmethod
    def _assemble(cls, plant_name: str, component_table: pd.DataFrame) -> SteamGenerationPlant:
        """Create the physical Steam model from one plant's component table."""

        normalised = component_table.copy()
        normalised["technology_normalised"] = (
            normalised["technology"].astype(str).str.strip().str.lower()
        )

        components: dict[str, object] = {}
        for _, row in normalised.iterrows():
            technology = str(row["technology_normalised"])
            if technology not in TECHNOLOGY_REGISTRY:
                raise ValueError(f"Plant '{plant_name}' uses unsupported technology '{technology}'")
            if technology in components:
                raise ValueError(
                    f"Plant '{plant_name}' defines duplicate technology '{technology}'"
                )
            components[technology] = component_from_row(
                "Steam plant",
                plant_name,
                technology,
                TECHNOLOGY_REGISTRY[technology].from_row,
                row,
            )

        if "boiler" not in components:
            raise ValueError(f"Plant '{plant_name}' does not define a boiler row")
        has_storage = "thermal_storage" in components
        has_electric_boiler = "electric_boiler" in components
        if has_storage == has_electric_boiler:
            raise ValueError(
                f"Plant '{plant_name}' must define exactly one electric heat route: "
                "either thermal_storage or electric_boiler"
            )
        resolve_steam_route_process(components)

        heat_demand_column = first_non_empty(component_table, "demand", default="")
        if not heat_demand_column:
            heat_demand_column = f"{plant_name}_heat_demand"

        boiler_rows = normalised.loc[normalised["technology_normalised"] == "boiler"]
        raw_gas_emissions_factor = first_non_empty(
            boiler_rows,
            "gas_emissions_factor_kg_per_mwh",
            default="",
        )
        if raw_gas_emissions_factor == "":
            gas_emissions_factor = float(cls.gas_emissions_factor_kg_per_mwh)
        else:
            try:
                gas_emissions_factor = float(raw_gas_emissions_factor)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Plant '{plant_name}' gas_emissions_factor_kg_per_mwh on the "
                    "boiler row must be numeric"
                ) from exc
            if not np.isfinite(gas_emissions_factor) or gas_emissions_factor < 0.0:
                raise ValueError(
                    f"Plant '{plant_name}' gas_emissions_factor_kg_per_mwh on the "
                    "boiler row must be finite and non-negative"
                )

        return cls(
            name=plant_name,
            unit_type=first_non_empty(component_table, "unit_type", default="steam_plant"),
            node=first_non_empty(component_table, "node", default=""),
            objective=first_non_empty(
                component_table,
                "objective",
                default="min_variable_cost",
            ),
            heat_demand_column=heat_demand_column,
            components=components,
            gas_emissions_factor_kg_per_mwh=gas_emissions_factor,
        )

    # --- Configured route and technologies -------------------------------

    @property
    def route_process(self) -> SteamRouteProcess:
        return resolve_steam_route_process(self.components)

    @property
    def technology_route(self) -> str:
        return self.route_process.name

    @property
    def has_thermal_storage(self) -> bool:
        return self.technology_route == THERMAL_STORAGE_GAS_BOILER_ROUTE

    @property
    def etes(self) -> ThermalStorage:
        component = self.components.get("thermal_storage")
        if not isinstance(component, ThermalStorage):
            raise ValueError(f"Plant '{self.name}' has no thermal_storage component")
        return component

    @property
    def electric_boiler(self) -> ElectricBoiler:
        component = self.components.get("electric_boiler")
        if not isinstance(component, ElectricBoiler):
            raise ValueError(f"Plant '{self.name}' has no electric_boiler component")
        return component

    @property
    def gas_boiler(self) -> GasBoiler:
        component = self.components.get("boiler")
        if not isinstance(component, GasBoiler):
            raise ValueError(f"Plant '{self.name}' has no boiler component")
        return component

    # --- Public solve API -------------------------------------------------

    def solve_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: DayAheadPosition,
        initial_soc_mwh: float | None = None,
        *,
        capacity_award: BalancingCapacityAward | None = None,
    ) -> pd.DataFrame:
        return self._solve_rolling_windows(
            ElectricityMarketStage.DAY_AHEAD,
            config,
            forecasts,
            market_input,
            initial_soc_mwh,
            capacity_award,
        )

    def solve_intraday_adjustment_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: IntradayAdjustment,
        initial_soc_mwh: float | None = None,
        *,
        capacity_award: BalancingCapacityAward | None = None,
    ) -> pd.DataFrame:
        return self._solve_rolling_windows(
            ElectricityMarketStage.INTRADAY,
            config,
            forecasts,
            market_input,
            initial_soc_mwh,
            capacity_award,
        )

    def solve_afrr_down_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: BalancingEnergyActivation,
        initial_soc_mwh: float | None = None,
        *,
        capacity_award: BalancingCapacityAward | None = None,
    ) -> pd.DataFrame:
        return self._solve_rolling_windows(
            ElectricityMarketStage.AFRR_ENERGY,
            config,
            forecasts,
            market_input,
            initial_soc_mwh,
            capacity_award,
        )

    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: DayAheadPosition,
        initial_soc_mwh: float | None = None,
        *,
        capacity_award: BalancingCapacityAward | None = None,
    ) -> pd.DataFrame:
        return self._solve_stage(
            ElectricityMarketStage.DAY_AHEAD,
            config,
            forecasts,
            market_input,
            self.route_process.initial_soc(self.components, initial_soc_mwh),
            capacity_award,
        )

    def solve_intraday_adjustment_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: IntradayAdjustment,
        initial_soc_mwh: float | None = None,
        *,
        capacity_award: BalancingCapacityAward | None = None,
    ) -> pd.DataFrame:
        return self._solve_stage(
            ElectricityMarketStage.INTRADAY,
            config,
            forecasts,
            market_input,
            self.route_process.initial_soc(self.components, initial_soc_mwh),
            capacity_award,
        )

    def solve_afrr_down_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: BalancingEnergyActivation,
        initial_soc_mwh: float | None = None,
        *,
        capacity_award: BalancingCapacityAward | None = None,
    ) -> pd.DataFrame:
        return self._solve_stage(
            ElectricityMarketStage.AFRR_ENERGY,
            config,
            forecasts,
            market_input,
            self.route_process.initial_soc(self.components, initial_soc_mwh),
            capacity_award,
        )

    def solve_market_stage(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        stage: ElectricityMarketStage,
        market_input: ElectricityMarketStageInput,
        *,
        initial_soc_mwh: float | None = None,
        rolling: bool = False,
        capacity_award: BalancingCapacityAward | None = None,
    ) -> pd.DataFrame:
        """Check and execute one rule-based market instruction with Pyomo.

        Strategies determine the commercial position in ``market_input``. This
        plant method owns the stage-specific Pyomo feasibility and operation
        solve, keeping physical execution out of market orchestration code.
        """

        validate_stage_input(stage, market_input)
        if stage == ElectricityMarketStage.DAY_AHEAD:
            solver = self.solve_rolling if rolling else self.solve_horizon
        elif stage == ElectricityMarketStage.INTRADAY:
            solver = (
                self.solve_intraday_adjustment_rolling
                if rolling
                else self.solve_intraday_adjustment_horizon
            )
        elif stage == ElectricityMarketStage.AFRR_ENERGY:
            solver = self.solve_afrr_down_rolling if rolling else self.solve_afrr_down_horizon
        else:
            raise ValueError(f"Unsupported electricity market stage '{stage}'")

        return solver(
            config,
            forecasts,
            market_input,
            initial_soc_mwh=initial_soc_mwh,
            capacity_award=capacity_award,
        )

    def solve_market_instruction(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        instruction: MarketStageInstruction,
    ) -> pd.DataFrame:
        """Validate and execute a typed commercial instruction from a strategy.

        This is the common plant-facing execution contract used by the market
        runner.  ``solve_market_stage`` remains public as the lower-level,
        backwards-compatible API for notebooks and specialised integrations.
        """

        if not isinstance(instruction.payload, ElectricityMarketRequest):
            raise TypeError(
                "SteamGenerationPlant requires an ElectricityMarketRequest payload; "
                f"received {type(instruction.payload).__name__}"
            )
        request = instruction.payload
        expected_market = market_name_for_stage(request.stage)
        if instruction.market_name != expected_market:
            raise ValueError(
                f"Electricity stage '{request.stage}' must execute in market "
                f"'{expected_market}', not '{instruction.market_name}'"
            )

        execution_forecasts = (
            instruction.execution_forecasts
            if instruction.execution_forecasts is not None
            else forecasts
        )
        return self.solve_market_stage(
            config,
            execution_forecasts,
            request.stage,
            request.position,
            initial_soc_mwh=instruction.initial_state.get(THERMAL_STORAGE_SOC_STATE_KEY),
            rolling=instruction.rolling,
            capacity_award=request.capacity_award,
        )

    # --- Shared rolling and solver execution -----------------------------

    def _solve_rolling_windows(
        self,
        stage: ElectricityMarketStage,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: ElectricityMarketStageInput,
        initial_soc_mwh: float | None,
        capacity_award: BalancingCapacityAward | None,
    ) -> pd.DataFrame:
        dt_hours = config.timestep_minutes / 60.0
        horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 48))
        step_hours = float(config.dispatch_setting("rolling_step_hours", 24))
        horizon_steps = max(1, int(round(horizon_hours / dt_hours)))
        step_steps = max(1, int(round(step_hours / dt_hours)))

        route = self.route_process
        current_soc_mwh = route.initial_soc(self.components, initial_soc_mwh)
        implemented_frames: list[pd.DataFrame] = []
        position = 0

        while position < len(forecasts):
            horizon = forecasts.iloc[position : position + horizon_steps].copy()
            horizon_market_input = slice_dataclass_series(market_input, horizon.index)
            horizon_capacity_award = (
                slice_dataclass_series(capacity_award, horizon.index)
                if capacity_award is not None
                else None
            )
            horizon_result = self._solve_stage(
                stage,
                config,
                horizon,
                horizon_market_input,
                current_soc_mwh,
                horizon_capacity_award,
            )
            implement_count = min(step_steps, len(forecasts) - position)
            implemented = horizon_result.iloc[:implement_count].copy()
            implemented_frames.append(implemented)
            current_soc_mwh = route.next_soc(implemented.iloc[-1])
            position += implement_count

        return pd.concat(implemented_frames).sort_index()

    def _solve_stage(
        self,
        stage: ElectricityMarketStage,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: ElectricityMarketStageInput,
        initial_soc_mwh: float | None,
        capacity_award: BalancingCapacityAward | None,
    ) -> pd.DataFrame:
        route = self.route_process
        model = self.build_model(
            stage,
            config,
            forecasts,
            market_input,
            initial_soc_mwh,
            capacity_award=capacity_award,
        )
        solve_label = stage_label(stage)
        solve_errors: list[str] = []

        for candidate_name, solver in available_pyomo_solvers(config):
            try:
                result = solver.solve(model, tee=config.solver_tee)
            except NoFeasibleSolutionError as exc:
                raise RuntimeError(
                    route.infeasibility_message(solve_label, candidate_name)
                ) from exc
            except (ApplicationError, RuntimeError, OSError) as exc:
                solve_errors.append(f"{candidate_name}: {exc}")
                continue

            termination = result.solver.termination_condition
            status = result.solver.status
            if status == SolverStatus.ok and termination in {
                TerminationCondition.optimal,
                TerminationCondition.feasible,
            }:
                route.warn_after_solve(model, stage_warning_label(stage))
                from flexi_mod.outputs.result_mappers import extract_dispatch_results

                return extract_dispatch_results(
                    self,
                    model,
                    forecasts,
                    candidate_name,
                    config=config,
                    stage=stage,
                    market_input=market_input,
                    capacity_award=capacity_award,
                )

            if is_infeasible_termination(termination):
                raise RuntimeError(route.infeasibility_message(solve_label, candidate_name))
            solve_errors.append(f"{candidate_name}: status={status}, termination={termination}")

        raise RuntimeError(
            f"{solve_label} solve failed for all configured solvers. " + " | ".join(solve_errors)
        )

    # --- Shared physical and market model --------------------------------

    def build_model(
        self,
        stage: ElectricityMarketStage,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: ElectricityMarketStageInput,
        initial_soc_mwh: float | None = None,
        *,
        capacity_award: BalancingCapacityAward | None = None,
    ) -> pyo.ConcreteModel:
        """Build one market-stage model without solving it.

        The method follows the common plant-modelling sequence: define inputs,
        add market-stage terms, initialise technology blocks, then add the
        physical balance and objective.  Route classes retain the technology-
        specific equations so that this orchestration stays readable.
        """
        self.validate_inputs(stage, forecasts, market_input, initial_soc_mwh)
        model = pyo.ConcreteModel(name=f"{self.name}_{stage.value}")
        model.T = pyo.Set(initialize=range(len(forecasts)), ordered=True)
        heat_demand = self.define_parameters(
            model,
            config,
            forecasts,
            market_input,
            capacity_award,
        )
        self.initialize_electricity_market(model, stage, forecasts, market_input)
        self.initialize_components(model, config, initial_soc_mwh)
        self.define_variables(model)
        self.define_constraints(
            model,
            config,
            heat_demand,
            enforce_capacity_delivery=stage != ElectricityMarketStage.AFRR_ENERGY,
        )
        self.define_objective(model)
        return model

    def required_forecast_columns(self) -> set[str]:
        """Return plant-owned profiles independent of a selected market stage."""

        return {self.heat_demand_column}

    def required_stage_forecast_columns(
        self,
        stage: ElectricityMarketStage,
        market_input: ElectricityMarketStageInput,
    ) -> set[str]:
        """Return the forecast columns used by one configured market stage."""

        validate_stage_input(stage, market_input)
        required = {self.heat_demand_column, market_input.gas_price_col}
        if market_input.co2_price_col:
            required.add(market_input.co2_price_col)
        if stage == ElectricityMarketStage.DAY_AHEAD:
            required.add(cast(DayAheadPosition, market_input).electricity_price_col)
        elif stage == ElectricityMarketStage.INTRADAY:
            intraday_input = cast(IntradayAdjustment, market_input)
            required.update({intraday_input.da_price_col, intraday_input.idc_price_col})
        else:
            afrr_input = cast(BalancingEnergyActivation, market_input)
            required.update({afrr_input.da_price_col, afrr_input.idc_price_col})
        return required

    def validate_inputs(
        self,
        stage: ElectricityMarketStage,
        forecasts: pd.DataFrame,
        market_input: ElectricityMarketStageInput,
        initial_soc_mwh: float | None = None,
    ) -> None:
        """Validate one stage's physical and price inputs before Pyomo creation."""

        required = self.required_stage_forecast_columns(stage, market_input)
        subject = f"Steam plant '{self.name}'"
        validate_required_forecasts(forecasts, required, subject, operation=stage.value)
        missing_value_allowed = (
            cast(IntradayAdjustment, market_input).idc_price_col
            if stage == ElectricityMarketStage.INTRADAY
            else cast(BalancingEnergyActivation, market_input).idc_price_col
            if stage == ElectricityMarketStage.AFRR_ENERGY
            else None
        )
        for column in sorted(required):
            numeric_forecast(
                forecasts,
                column,
                subject,
                require_non_negative=column == self.heat_demand_column,
                allow_missing_values=column == missing_value_allowed,
            )

        for label, value in {
            "co2_emission_factor_t_per_mwh_fuel": market_input.co2_emission_factor_t_per_mwh_fuel,
            "tax_rate": market_input.tax_rate,
        }.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"Steam plant '{self.name}' signal '{label}' must be finite")

        if initial_soc_mwh is not None:
            if not np.isfinite(initial_soc_mwh):
                raise ValueError(f"Steam plant '{self.name}' initial_soc_mwh must be finite")
            storage = self.components.get("thermal_storage")
            if isinstance(storage, ThermalStorage) and not (
                storage.min_capacity_mwh <= initial_soc_mwh <= storage.max_capacity_mwh
            ):
                raise ValueError(
                    f"Steam plant '{self.name}' initial_soc_mwh must be between "
                    f"{storage.min_capacity_mwh} and {storage.max_capacity_mwh}"
                )

    def define_parameters(
        self,
        model: pyo.ConcreteModel,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: ElectricityMarketStageInput,
        capacity_award: BalancingCapacityAward | None,
    ) -> list[float]:
        """Attach time-series inputs common to every steam market stage."""
        steps = list(model.T)
        heat_demand = forecasts[self.heat_demand_column].astype(float).to_numpy() * (
            config.timestep_minutes / 60.0
        )
        additional_charge = series_or_zero(
            market_input.additional_electricity_charge_eur_per_mwh,
            forecasts.index,
        ).to_numpy()
        gas_price = forecasts[market_input.gas_price_col].astype(float).to_numpy()
        co2_price = forecast_values_or_zero(forecasts, market_input.co2_price_col)
        timestep_hours = config.timestep_minutes / 60.0
        reserved_capacity = (
            capacity_award.reserved_energy_mwh(forecasts.index, timestep_hours)
            if capacity_award is not None
            else pd.Series(0.0, index=forecasts.index)
        )
        self.route_process.validate_capacity_reservation(reserved_capacity)

        model.heat_demand = pyo.Param(
            model.T,
            initialize={t: float(heat_demand[t]) for t in steps},
        )
        model.additional_electricity_charge = pyo.Param(
            model.T,
            initialize={t: float(additional_charge[t]) for t in steps},
        )
        model.gas_price = pyo.Param(
            model.T,
            initialize={t: float(gas_price[t]) for t in steps},
        )
        model.co2_price = pyo.Param(
            model.T,
            initialize={t: float(co2_price[t]) for t in steps},
        )
        model.reserved_capacity_mwh = pyo.Param(
            model.T,
            initialize={t: float(reserved_capacity.iloc[t]) for t in steps},
        )
        model.co2_emission_factor = pyo.Param(
            initialize=float(market_input.co2_emission_factor_t_per_mwh_fuel)
        )
        model.tax_rate = pyo.Param(initialize=float(market_input.tax_rate))
        return [float(value) for value in heat_demand]

    def initialize_electricity_market(
        self,
        model: pyo.ConcreteModel,
        stage: ElectricityMarketStage,
        forecasts: pd.DataFrame,
        market_input: ElectricityMarketStageInput,
    ) -> None:
        """Attach one generic electricity-market position and settlement."""

        add_stage_to_model(model, stage, forecasts, market_input)

    def initialize_components(
        self,
        model: pyo.ConcreteModel,
        config: CaseConfig,
        initial_soc_mwh: float | None,
    ) -> None:
        """Add the configured steam-route technology blocks."""
        resolved_initial_soc_mwh = self.route_process.initial_soc(
            self.components,
            initial_soc_mwh,
        )
        self.route_process.add_technology_blocks(
            model,
            self.components,
            config.timestep_minutes / 60.0,
            resolved_initial_soc_mwh,
        )

    @staticmethod
    def define_variables(model: pyo.ConcreteModel) -> None:
        """Add route-independent physical decision variables."""
        model.electricity_consumption = pyo.Var(
            model.T,
            within=pyo.NonNegativeReals,
        )

    def define_constraints(
        self,
        model: pyo.ConcreteModel,
        config: CaseConfig,
        heat_demand_mwh: list[float],
        *,
        enforce_capacity_delivery: bool,
    ) -> None:
        """Connect route physics to the market-stage electricity position."""
        self.route_process.add_process_constraints(
            model,
            self.components,
            enforce_capacity_delivery,
            config.timestep_minutes / 60.0,
            heat_demand_mwh,
        )
        model.market_position_matches_physical_consumption = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                mm.electricity_consumption[t] == mm.required_electricity_consumption_mwh[t]
            ),
        )

    def define_objective(self, model: pyo.ConcreteModel) -> None:
        """Add costs and the route-specific market settlement objective."""
        _add_common_costs_and_objective(model, self.route_process)

    def _build_stage_model(
        self,
        stage: ElectricityMarketStage,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        market_input: ElectricityMarketStageInput,
        initial_soc_mwh: float | None,
        capacity_award: BalancingCapacityAward | None = None,
    ) -> pyo.ConcreteModel:
        """Compatibility wrapper for callers using the previous private API."""
        return self.build_model(
            stage,
            config,
            forecasts,
            market_input,
            initial_soc_mwh,
            capacity_award=capacity_award,
        )


# ---------------------------------------------------------------------------
# Route-specific physical processes
# ---------------------------------------------------------------------------
class SteamRouteProcess:
    """Small common interface for route-specific physical equations."""

    name: str
    required_components: frozenset[str]

    def initial_soc(
        self,
        components: dict[str, object],
        requested_soc_mwh: float | None,
    ) -> float | None:
        """Return rolling state; direct routes have no state."""

        return None

    def add_technology_blocks(
        self,
        model: pyo.ConcreteModel,
        components: dict[str, object],
        timestep_hours: float,
        initial_soc_mwh: float | None,
    ) -> None:
        """Add this route's configured technologies to the shared model."""

        model.technology_blocks = pyo.Block(list(components))
        for technology, component in components.items():
            context = {
                "dt_hours": timestep_hours,
                "initial_soc_mwh": (initial_soc_mwh if technology == "thermal_storage" else None),
            }
            component.add_to_model(
                model,
                model.technology_blocks[technology],
                model.T,
                context,
            )

    def validate_capacity_reservation(
        self,
        reserved_capacity_mwh: pd.Series,
    ) -> None:
        """Reject reserve by default; storage routes override this."""

        if (reserved_capacity_mwh.abs() > 1e-9).any():
            raise ValueError(f"Route '{self.name}' does not support aFRR-capacity reservations")

    def next_soc(self, delivered_row: pd.Series) -> float | None:
        """Return the next rolling state; direct routes stay stateless."""

        return None

    def objective_penalty(
        self,
        model: pyo.ConcreteModel,
        position: int,
    ) -> pyo.NumericValue | float:
        """Return a route-only feasibility penalty for the objective."""

        return 0.0

    def warn_after_solve(self, model: pyo.ConcreteModel, stage_label: str) -> None:
        """Report any route-specific feasibility relaxation."""

        return


class ThermalStorageGasBoilerProcess(SteamRouteProcess):
    """ETES charging/storage process backed by a natural-gas boiler."""

    name = THERMAL_STORAGE_GAS_BOILER_ROUTE
    required_components = frozenset({"thermal_storage", "boiler"})

    def initial_soc(
        self,
        components: dict[str, object],
        requested_soc_mwh: float | None,
    ) -> float:
        storage = _thermal_storage(components)
        if requested_soc_mwh is None:
            return float(storage.initial_soc_mwh)
        return float(requested_soc_mwh)

    def add_process_constraints(
        self,
        model: pyo.ConcreteModel,
        components: dict[str, object],
        enforce_capacity_delivery: bool,
        timestep_hours: float,
        heat_demand_mwh: list[float],
    ) -> None:
        storage = model.technology_blocks["thermal_storage"]
        boiler = model.technology_blocks["boiler"]

        model.electricity_balance = pyo.Constraint(
            model.T,
            rule=lambda mm, t: mm.electricity_consumption[t] == storage.electricity_consumption[t],
        )
        model.heat_balance = pyo.Constraint(
            model.T,
            rule=lambda mm, t: storage.discharge_heat[t] + boiler.heat_out[t] == mm.heat_demand[t],
        )

        if not enforce_capacity_delivery:
            return

        max_charge_mwh = _thermal_storage(components).max_power_charge_mw * timestep_hours
        model.reserve_charge_power_headroom = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                storage.electric_charge_to_storage[t] + mm.reserved_capacity_mwh[t]
                <= max_charge_mwh
            ),
        )
        model.reserve_storage_headroom = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                storage.soc[t] + mm.reserved_capacity_mwh[t] * storage.efficiency_charge
                <= storage.max_capacity_mwh
            ),
        )
        self._add_reserved_capacity_delivery_guarantee(
            model,
            components,
            heat_demand_mwh,
            timestep_hours,
        )

    @staticmethod
    def _add_reserved_capacity_delivery_guarantee(
        model: pyo.ConcreteModel,
        components: dict[str, object],
        heat_demand_mwh: list[float],
        timestep_hours: float,
    ) -> None:
        storage_component = _thermal_storage(components)
        storage = model.technology_blocks["thermal_storage"]
        max_discharge_mwh = storage_component.max_power_discharge_mw * timestep_hours
        model.max_heat_outlet_mwh = pyo.Param(
            model.T,
            initialize={
                t: min(float(heat_demand_mwh[t]), max_discharge_mwh)
                / storage_component.efficiency_discharge
                for t in model.T
            },
        )
        model.soc_under_full_activation_mwh = pyo.Var(
            model.T,
            within=pyo.NonNegativeReals,
        )
        model.delivery_guarantee_slack_mwh = pyo.Var(
            model.T,
            within=pyo.NonNegativeReals,
        )
        model.full_activation_soc_tracks_schedule = pyo.Constraint(
            model.T,
            rule=lambda mm, t: mm.soc_under_full_activation_mwh[t] >= storage.soc[t],
        )

        def full_activation_soc_balance(
            mm: pyo.ConcreteModel,
            t: int,
        ) -> pyo.Constraint:
            previous = (
                storage.initial_soc_mwh if t == 0 else mm.soc_under_full_activation_mwh[t - 1]
            )
            return mm.soc_under_full_activation_mwh[t] >= (
                previous * (1.0 - storage.storage_loss_rate)
                + (storage.electric_charge_to_storage[t] + mm.reserved_capacity_mwh[t])
                * storage.efficiency_charge
                - mm.max_heat_outlet_mwh[t]
            )

        model.full_activation_soc_balance = pyo.Constraint(
            model.T,
            rule=full_activation_soc_balance,
        )
        model.reserve_delivery_headroom = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                mm.soc_under_full_activation_mwh[t]
                <= storage.max_capacity_mwh + mm.delivery_guarantee_slack_mwh[t]
            ),
        )

    def validate_capacity_reservation(
        self,
        reserved_capacity_mwh: pd.Series,
    ) -> None:
        if (reserved_capacity_mwh < -1e-9).any():
            raise ValueError("Reserved aFRR capacity must be non-negative")

    def physical_result_fields(
        self,
        model: pyo.ConcreteModel,
        position: int,
    ) -> dict[str, object]:
        storage = model.technology_blocks["thermal_storage"]
        boiler = model.technology_blocks["boiler"]
        return {
            "etes_charge_MWh": pyomo_value(storage.electric_charge_to_storage[position]),
            "etes_discharge_MWh": pyomo_value(storage.discharge_heat[position]),
            "etes_soc_MWh": pyomo_value(storage.soc[position]),
            "gas_heat_MWh": pyomo_value(boiler.heat_out[position]),
            "gas_input_MWh": pyomo_value(boiler.fuel_input[position]),
        }

    def capacity_physical_result_fields(
        self,
        reserved_capacity_mwh: float,
        timestep_hours: float,
        final_planned_mwh: float,
        physical_fields: dict[str, object],
        components: dict[str, object],
    ) -> dict[str, object]:
        storage = _thermal_storage(components)
        max_charge_mwh = storage.max_power_charge_mw * timestep_hours
        return {
            "reserved_capacity_headroom_MWh": reserved_capacity_mwh,
            "available_charge_headroom_after_schedule_MWh": max(
                0.0,
                max_charge_mwh - float(final_planned_mwh),
            ),
            "available_storage_headroom_after_schedule_MWh": (
                max(
                    0.0,
                    storage.max_capacity_mwh - float(physical_fields["etes_soc_MWh"]),
                )
                / storage.efficiency_charge
            ),
        }

    def next_soc(self, delivered_row: pd.Series) -> float:
        return float(delivered_row["etes_soc_MWh"])

    def objective_penalty(
        self,
        model: pyo.ConcreteModel,
        position: int,
    ) -> pyo.NumericValue | float:
        if not hasattr(model, "delivery_guarantee_slack_mwh"):
            return 0.0
        return (
            DELIVERY_GUARANTEE_SLACK_PENALTY_EUR_PER_MWH
            * model.delivery_guarantee_slack_mwh[position]
        )

    def warn_after_solve(self, model: pyo.ConcreteModel, stage_label: str) -> None:
        if not hasattr(model, "delivery_guarantee_slack_mwh"):
            return
        worst = max(
            (float(pyo.value(model.delivery_guarantee_slack_mwh[t])) for t in model.T),
            default=0.0,
        )
        if worst > 1e-6:
            warnings.warn(
                f"{stage_label}: reserved aFRR-down capacity exceeds what the plant "
                f"can deliver under continuous activation by up to {worst:.3f} MWh "
                "per step even after rescheduling. The capacity bid sizing and the "
                "plant parameters are inconsistent.",
                stacklevel=2,
            )

    def infeasibility_message(self, stage_label: str, solver_name: str) -> str:
        return (
            f"{stage_label} solve is infeasible with solver '{solver_name}'. "
            "FLEXIMOD now enforces strict useful heat dispatch: gas heat plus "
            "storage discharge must equal heat demand in every timestep, with no "
            "unmet-heat or heat-dump slack. Check fixed market electricity "
            "positions, aFRR activation, ETES storage headroom, and heat demand."
        )


class DirectElectricGasBoilerProcess(SteamRouteProcess):
    """Direct electric-boiler process backed by a natural-gas boiler."""

    name = DIRECT_ELECTRIC_GAS_BOILER_ROUTE
    required_components = frozenset({"electric_boiler", "boiler"})

    def add_process_constraints(
        self,
        model: pyo.ConcreteModel,
        components: dict[str, object],
        _enforce_capacity_delivery: bool,
        timestep_hours: float,
        heat_demand_mwh: list[float],
    ) -> None:
        electric_boiler = model.technology_blocks["electric_boiler"]
        gas_boiler = model.technology_blocks["boiler"]
        model.electricity_balance = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                mm.electricity_consumption[t] == electric_boiler.electricity_consumption[t]
            ),
        )
        model.heat_balance = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                electric_boiler.heat_out[t] + gas_boiler.heat_out[t] == mm.heat_demand[t]
            ),
        )

    def validate_capacity_reservation(
        self,
        reserved_capacity_mwh: pd.Series,
    ) -> None:
        if (reserved_capacity_mwh.abs() > 1e-9).any():
            raise ValueError(
                "The electric-boiler + gas-boiler route does not support aFRR-capacity reservations"
            )

    def physical_result_fields(
        self,
        model: pyo.ConcreteModel,
        position: int,
    ) -> dict[str, object]:
        electric_boiler = model.technology_blocks["electric_boiler"]
        gas_boiler = model.technology_blocks["boiler"]
        return {
            "technology_route": self.name,
            "electric_boiler_electricity_consumption_MWh": pyomo_value(
                electric_boiler.electricity_consumption[position]
            ),
            "electric_boiler_heat_MWh": pyomo_value(electric_boiler.heat_out[position]),
            "gas_heat_MWh": pyomo_value(gas_boiler.heat_out[position]),
            "gas_input_MWh": pyomo_value(gas_boiler.fuel_input[position]),
        }

    def capacity_physical_result_fields(
        self,
        reserved_capacity_mwh: float,
        timestep_hours: float,
        final_planned_mwh: float,
        physical_fields: dict[str, object],
        components: dict[str, object],
    ) -> dict[str, object]:
        electric_boiler = _electric_boiler(components)
        max_load_mwh = electric_boiler.max_electricity_input_mw * timestep_hours
        return {
            "reserved_capacity_headroom_MWh": 0.0,
            "available_load_headroom_after_schedule_MWh": max(
                0.0,
                max_load_mwh - float(final_planned_mwh),
            ),
        }

    def infeasibility_message(self, stage_label: str, solver_name: str) -> str:
        return (
            f"{stage_label} solve is infeasible with solver '{solver_name}' for "
            "the direct electric-boiler + gas-boiler route. Gas heat plus "
            "electric-boiler heat must equal heat demand in every timestep, with "
            "no storage, unmet-heat, or heat-dump slack. Check fixed electricity "
            "positions, aFRR activation, electric-boiler input capacity, "
            "gas-boiler heat capacity, and heat demand."
        )


ROUTE_PROCESS_REGISTRY: dict[frozenset[str], SteamRouteProcess] = {
    ThermalStorageGasBoilerProcess.required_components: (ThermalStorageGasBoilerProcess()),
    DirectElectricGasBoilerProcess.required_components: (DirectElectricGasBoilerProcess()),
}


def resolve_steam_route_process(
    components: dict[str, object],
) -> SteamRouteProcess:
    """Resolve an exact component set to its physical steam process."""

    component_set = frozenset(components)
    try:
        return ROUTE_PROCESS_REGISTRY[component_set]
    except KeyError as exc:
        supported = ", ".join(
            " + ".join(sorted(route_components)) for route_components in ROUTE_PROCESS_REGISTRY
        )
        configured = " + ".join(sorted(component_set)) or "<none>"
        raise ValueError(
            f"Unsupported steam-plant component route '{configured}'. Supported routes: {supported}"
        ) from exc


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------
# The generic electricity-product formulation lives in
# ``markets.electricity_settlement``. Steam retains only heat-route physics,
# fuel economics, and plant-specific feasibility penalties.


# --- Common economics -------------------------------------------------


def _add_common_costs_and_objective(
    model: pyo.ConcreteModel,
    route: SteamRouteProcess,
) -> None:
    boiler = model.technology_blocks["boiler"]
    model.additional_electricity_charges_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: (
            mm.actual_electricity_consumption_mwh[t] * mm.additional_electricity_charge[t]
        ),
    )
    model.electricity_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: (
            mm.electricity_market_cost[t] + mm.additional_electricity_charges_cost[t]
        ),
    )
    model.tax_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.electricity_cost[t] * mm.tax_rate,
    )
    model.gas_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: boiler.operating_cost[t],
    )
    model.co2_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: boiler.co2_cost[t],
    )
    model.objective = pyo.Objective(
        sense=pyo.minimize,
        expr=pyo.quicksum(
            model.electricity_cost[t]
            + model.gas_cost[t]
            + model.co2_cost[t]
            + model.tax_cost[t]
            + route.objective_penalty(model, t)
            for t in model.T
        ),
    )


def _thermal_storage(components: dict[str, object]) -> ThermalStorage:
    component = components.get("thermal_storage")
    if not isinstance(component, ThermalStorage):
        raise ValueError("Thermal-storage route has no thermal_storage component")
    return component


def _electric_boiler(components: dict[str, object]) -> ElectricBoiler:
    component = components.get("electric_boiler")
    if not isinstance(component, ElectricBoiler):
        raise ValueError("Direct-boiler route has no electric_boiler component")
    return component
