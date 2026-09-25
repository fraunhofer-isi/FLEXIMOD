# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Steam-generation plant with shared market stages and route-specific physics.

The file is ordered for model users: plant construction and solve methods come
first, followed by market inputs, physical route equations, and low-level
helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.opt import SolverStatus, TerminationCondition

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants._helpers import (
    _stage_label,
    _stage_warning_label,
    _validate_stage_signal_type,
)
from flexi_mod.plants.base_plant import BasePlant
from flexi_mod.plants.constants import (
    THERMAL_STORAGE_GAS_BOILER_ROUTE,
)
from flexi_mod.plants.model_utils import (
    available_pyomo_solvers,
    forecast_values_or_zero,
    is_infeasible_termination,
    series_or_zero,
    slice_dataclass_series,
)
from flexi_mod.plants.objective import _add_common_costs_and_objective
from flexi_mod.plants.routes.base import SteamRouteProcess
from flexi_mod.plants.routes.registry import resolve_steam_route_process
from flexi_mod.plants.signals.afrr_down_signals import AFRRDownSignals
from flexi_mod.plants.signals.dispatch_signals import DispatchSignals
from flexi_mod.plants.signals.idc_adjustment_signals import IDCAdjustmentSignals
from flexi_mod.plants.signals.steam_signals import SteamSignals
from flexi_mod.plants.stage import SteamMarketStage
from flexi_mod.plants.stage_builders import (
    _add_afrr_energy_stage,
    _add_day_ahead_stage,
    _add_intraday_stage,
)
from flexi_mod.plants.stage_results import _extract_stage_results
from flexi_mod.plants.technologies import (
    TECHNOLOGY_REGISTRY,
    ElectricBoiler,
    GasBoiler,
    ThermalStorage,
    first_non_empty,
)


# ---------------------------------------------------------------------------
# Plant model: start here
# ---------------------------------------------------------------------------
@dataclass
class SteamGenerationPlant(BasePlant):
    """Optimize one steam plant through DA, IDC, and aFRR market stages.

    Supported routes are thermal storage plus a gas boiler, or a direct
    electric boiler plus a gas boiler. Construct the plant with
    ``from_rows()`` and run it through one of the public ``solve_*`` methods.
    Both routes use the same market-stage engine.
    """

    gas_emissions_factor_kg_per_mwh: float = 201.0
    components: dict[str, object] = field(default_factory=dict)

    # --- Construction -----------------------------------------------------

    @classmethod
    def from_rows(cls, plant_name: str, rows: pd.DataFrame) -> SteamGenerationPlant:
        normalised = rows.copy()
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
            components[technology] = TECHNOLOGY_REGISTRY[technology].from_row(row)

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

        heat_demand_column = first_non_empty(rows, "demand", default="")
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
            unit_type=first_non_empty(rows, "unit_type", default="steam_plant"),
            node=first_non_empty(rows, "node", default=""),
            objective=first_non_empty(
                rows,
                "objective",
                default="min_variable_cost",
            ),
            heat_demand_column=heat_demand_column,
            components=components,
            gas_emissions_factor_kg_per_mwh=gas_emissions_factor,
        )

    @classmethod
    def from_plants_dataframe(
        cls,
        plants: pd.DataFrame,
    ) -> list[SteamGenerationPlant]:
        return [
            cls.from_rows(str(plant_name), rows)
            for plant_name, rows in plants.groupby("name", sort=False)
        ]

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
        signals: DispatchSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        return self._solve_rolling_windows(
            SteamMarketStage.DAY_AHEAD,
            config,
            forecasts,
            signals,
            initial_soc_mwh,
        )

    def solve_intraday_adjustment_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: IDCAdjustmentSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        return self._solve_rolling_windows(
            SteamMarketStage.INTRADAY,
            config,
            forecasts,
            signals,
            initial_soc_mwh,
        )

    def solve_afrr_down_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: AFRRDownSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        return self._solve_rolling_windows(
            SteamMarketStage.AFRR_ENERGY,
            config,
            forecasts,
            signals,
            initial_soc_mwh,
        )

    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: DispatchSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        return self._solve_stage(
            SteamMarketStage.DAY_AHEAD,
            config,
            forecasts,
            signals,
            self.route_process.initial_soc(self.components, initial_soc_mwh),
        )

    def solve_intraday_adjustment_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: IDCAdjustmentSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        return self._solve_stage(
            SteamMarketStage.INTRADAY,
            config,
            forecasts,
            signals,
            self.route_process.initial_soc(self.components, initial_soc_mwh),
        )

    def solve_afrr_down_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: AFRRDownSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        return self._solve_stage(
            SteamMarketStage.AFRR_ENERGY,
            config,
            forecasts,
            signals,
            self.route_process.initial_soc(self.components, initial_soc_mwh),
        )

    # --- Shared rolling and solver execution -----------------------------

    def _solve_rolling_windows(
        self,
        stage: SteamMarketStage,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: SteamSignals,
        initial_soc_mwh: float | None,
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
            horizon_signals = slice_dataclass_series(signals, horizon.index)
            horizon_result = self._solve_stage(
                stage,
                config,
                horizon,
                horizon_signals,
                current_soc_mwh,
            )
            implement_count = min(step_steps, len(forecasts) - position)
            implemented = horizon_result.iloc[:implement_count].copy()
            implemented_frames.append(implemented)
            current_soc_mwh = route.next_soc(implemented.iloc[-1])
            position += implement_count

        return pd.concat(implemented_frames).sort_index()

    def _solve_stage(
        self,
        stage: SteamMarketStage,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: SteamSignals,
        initial_soc_mwh: float | None,
    ) -> pd.DataFrame:
        route = self.route_process
        model = self._build_stage_model(
            stage,
            config,
            forecasts,
            signals,
            initial_soc_mwh,
        )
        stage_label = _stage_label(stage)
        solve_errors: list[str] = []

        for candidate_name, solver in available_pyomo_solvers(config):
            try:
                result = solver.solve(model, tee=config.solver_tee)
            except NoFeasibleSolutionError as exc:
                raise RuntimeError(
                    route.infeasibility_message(stage_label, candidate_name)
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
                route.warn_after_solve(model, _stage_warning_label(stage))
                return _extract_stage_results(
                    self,
                    stage,
                    model,
                    config,
                    forecasts,
                    signals,
                    candidate_name,
                )

            if is_infeasible_termination(termination):
                raise RuntimeError(route.infeasibility_message(stage_label, candidate_name))
            solve_errors.append(f"{candidate_name}: status={status}, termination={termination}")

        raise RuntimeError(
            f"{stage_label} solve failed for all configured solvers. " + " | ".join(solve_errors)
        )

    # --- Shared physical and market model --------------------------------

    def _build_stage_model(
        self,
        stage: SteamMarketStage,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: SteamSignals,
        initial_soc_mwh: float | None,
    ) -> pyo.ConcreteModel:
        _validate_stage_signal_type(stage, signals)
        dt_hours = config.timestep_minutes / 60.0
        steps = list(range(len(forecasts)))
        heat_demand = forecasts[self.heat_demand_column].astype(float).to_numpy() * dt_hours
        additional_charge = series_or_zero(
            signals.additional_electricity_charge_eur_per_mwh,
            forecasts.index,
        ).to_numpy()
        gas_price = forecasts[signals.gas_price_col].astype(float).to_numpy()
        co2_price = forecast_values_or_zero(
            forecasts,
            signals.co2_price_col,
        )
        reserved_capacity = series_or_zero(
            signals.reserved_capacity_mwh,
            forecasts.index,
        )
        route = self.route_process
        route.validate_capacity_reservation(reserved_capacity)

        model = pyo.ConcreteModel(name=f"{self.name}_{stage.value}")
        model.T = pyo.Set(initialize=steps, ordered=True)
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
            initialize=float(signals.co2_emission_factor_t_per_mwh_fuel)
        )
        model.tax_rate = pyo.Param(initialize=float(signals.tax_rate))

        if stage == SteamMarketStage.DAY_AHEAD:
            _add_day_ahead_stage(
                model,
                forecasts,
                cast(DispatchSignals, signals),
            )
        elif stage == SteamMarketStage.INTRADAY:
            _add_intraday_stage(
                model,
                forecasts,
                cast(IDCAdjustmentSignals, signals),
            )
        else:
            _add_afrr_energy_stage(
                model,
                forecasts,
                cast(AFRRDownSignals, signals),
            )

        route.add_technology_blocks(
            model,
            self.components,
            dt_hours,
            initial_soc_mwh,
        )
        model.electricity_consumption = pyo.Var(
            model.T,
            within=pyo.NonNegativeReals,
        )
        route.add_process_constraints(
            model,
            self.components,
            stage,
            dt_hours,
            [float(value) for value in heat_demand],
        )
        model.market_position_matches_physical_consumption = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                mm.electricity_consumption[t] == mm.required_electricity_consumption_mwh[t]
            ),
        )
        _add_common_costs_and_objective(model, route)
        return model


# ---------------------------------------------------------------------------
# Market-stage input data
# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------


# --- Common result table ---------------------------------------------
