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
from enum import StrEnum
from typing import Any, cast

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
    forecast_values_or_zero,
    is_infeasible_termination,
    pyomo_value,
    series_float_or_nan,
    series_or_zero,
    series_value,
    slice_dataclass_series,
)
from flexi_mod.plants.technologies import (
    TECHNOLOGY_REGISTRY,
    ElectricBoiler,
    GasBoiler,
    ThermalStorage,
    first_non_empty,
)

DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL = 0.0
THERMAL_STORAGE_GAS_BOILER_ROUTE = "thermal_storage_gas_boiler"
DIRECT_ELECTRIC_GAS_BOILER_ROUTE = "direct_electric_gas_boiler"
DELIVERY_GUARANTEE_SLACK_PENALTY_EUR_PER_MWH = 1_000_000.0


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
@dataclass
class DispatchSignals:
    electricity_price_col: str
    gas_price_col: str
    gas_benchmark_eur_per_mwh_th: pd.Series
    charge_allowed: pd.Series
    additional_electricity_charge_eur_per_mwh: pd.Series | None = None
    tax_rate: float = 0.0
    reserved_capacity_mwh: pd.Series | None = None
    afrr_capacity_block_id: pd.Series | None = None
    afrr_capacity_block_duration_h: pd.Series | None = None
    afrr_capacity_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_pricing_rule: pd.Series | None = None
    afrr_capacity_bid_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_settlement_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_reserved_mw: pd.Series | None = None
    afrr_capacity_revenue_eur: pd.Series | None = None
    afrr_capacity_opportunity_cost_eur: pd.Series | None = None
    afrr_capacity_market_surplus_eur: pd.Series | None = None
    afrr_capacity_net_value_eur: pd.Series | None = None
    co2_price_col: str | None = None
    co2_emission_factor_t_per_mwh_fuel: float = DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL


@dataclass
class IDCAdjustmentSignals:
    da_price_col: str
    idc_price_col: str
    gas_price_col: str
    da_position_mwh: pd.Series
    idc_buy_upper_bound_mwh: pd.Series
    idc_sell_upper_bound_mwh: pd.Series
    gas_benchmark_eur_per_mwh_th: pd.Series
    electricity_trading_benchmark_eur_per_mwh_el: pd.Series
    additional_electricity_charge_eur_per_mwh: pd.Series | None = None
    tax_rate: float = 0.0
    reserved_capacity_mwh: pd.Series | None = None
    afrr_capacity_block_id: pd.Series | None = None
    afrr_capacity_block_duration_h: pd.Series | None = None
    afrr_capacity_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_pricing_rule: pd.Series | None = None
    afrr_capacity_bid_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_settlement_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_reserved_mw: pd.Series | None = None
    afrr_capacity_revenue_eur: pd.Series | None = None
    afrr_capacity_opportunity_cost_eur: pd.Series | None = None
    afrr_capacity_market_surplus_eur: pd.Series | None = None
    afrr_capacity_net_value_eur: pd.Series | None = None
    co2_price_col: str | None = None
    co2_emission_factor_t_per_mwh_fuel: float = DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL


@dataclass
class AFRRDownSignals:
    da_price_col: str
    idc_price_col: str
    gas_price_col: str
    da_position_mwh: pd.Series
    idc_buy_mwh: pd.Series
    idc_sell_mwh: pd.Series
    final_planned_electricity_mwh: pd.Series
    afrr_energy_price: pd.Series
    afrr_system_activation_mwh: pd.Series
    afrr_energy_bid_mwh: pd.Series
    afrr_energy_activated_mwh: pd.Series
    gas_benchmark_eur_per_mwh_th: pd.Series
    electricity_trading_benchmark_eur_per_mwh_el: pd.Series
    afrr_energy_bid_price: pd.Series | None = None
    afrr_energy_delivered_bid_price: pd.Series | None = None
    afrr_energy_capacity_backed_bid_mwh: pd.Series | None = None
    afrr_energy_free_bid_mwh: pd.Series | None = None
    afrr_energy_capacity_backed_activated_mwh: pd.Series | None = None
    afrr_energy_free_activated_mwh: pd.Series | None = None
    additional_electricity_charge_eur_per_mwh: pd.Series | None = None
    tax_rate: float = 0.0
    afrr_headroom_binding: pd.Series | None = None
    afrr_curtailment_mwh: pd.Series | None = None
    reserved_capacity_mwh: pd.Series | None = None
    afrr_capacity_block_id: pd.Series | None = None
    afrr_capacity_block_duration_h: pd.Series | None = None
    afrr_capacity_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_pricing_rule: pd.Series | None = None
    afrr_capacity_bid_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_settlement_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_reserved_mw: pd.Series | None = None
    afrr_capacity_revenue_eur: pd.Series | None = None
    afrr_capacity_opportunity_cost_eur: pd.Series | None = None
    afrr_capacity_market_surplus_eur: pd.Series | None = None
    afrr_capacity_net_value_eur: pd.Series | None = None
    co2_price_col: str | None = None
    co2_emission_factor_t_per_mwh_fuel: float = DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL


SteamSignals = DispatchSignals | IDCAdjustmentSignals | AFRRDownSignals


# ---------------------------------------------------------------------------
# Route-specific physical processes
# ---------------------------------------------------------------------------
class SteamMarketStage(StrEnum):
    """Market stage whose electricity position must be physically delivered."""

    DAY_AHEAD = "day_ahead"
    INTRADAY = "intraday"
    AFRR_ENERGY = "afrr_energy"


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

    def next_soc(self, committed_row: pd.Series) -> float | None:
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
        stage: SteamMarketStage,
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

        if stage not in {SteamMarketStage.DAY_AHEAD, SteamMarketStage.INTRADAY}:
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

    def capacity_result_fields(
        self,
        signals: Any,
        timestamp: pd.Timestamp,
        timestep_hours: float,
        final_planned_mwh: float,
        physical_fields: dict[str, object],
        components: dict[str, object],
    ) -> dict[str, object]:
        storage = _thermal_storage(components)
        values = _capacity_signal_values(signals, timestamp, timestep_hours)
        max_charge_mwh = storage.max_power_charge_mw * timestep_hours
        values.update(
            {
                "reserved_capacity_headroom_MWh": values["afrr_capacity_reserved_MWh"],
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
        )
        return values

    def next_soc(self, committed_row: pd.Series) -> float:
        return float(committed_row["etes_soc_MWh"])

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
        stage: SteamMarketStage,
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

    def capacity_result_fields(
        self,
        signals: Any,
        timestamp: pd.Timestamp,
        timestep_hours: float,
        final_planned_mwh: float,
        physical_fields: dict[str, object],
        components: dict[str, object],
    ) -> dict[str, object]:
        electric_boiler = _electric_boiler(components)
        max_load_mwh = electric_boiler.max_electricity_input_mw * timestep_hours
        return {
            **_empty_capacity_result_fields(),
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
# Market-stage equations, economics, output assembly, and solver discovery.
# Only the electricity positions differ between these three stages.
# Route-specific heat and electricity balances are attached afterward.


def _add_day_ahead_stage(
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    signals: DispatchSignals,
) -> None:
    steps = list(model.T)
    price = forecasts[signals.electricity_price_col].astype(float).to_numpy()
    charge_allowed = signals.charge_allowed.astype(bool).reindex(forecasts.index).fillna(False)
    model.da_price = pyo.Param(
        model.T,
        initialize={t: float(price[t]) for t in steps},
    )
    model.idc_price = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.afrr_energy_price = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.electricity_price = pyo.Param(
        model.T,
        initialize={
            t: float(price[t]) + float(pyo.value(model.additional_electricity_charge[t]))
            for t in steps
        },
    )
    model.charge_allowed = pyo.Param(
        model.T,
        within=pyo.Binary,
        initialize={t: int(bool(charge_allowed.iloc[t])) for t in steps},
    )
    model.da_position_mwh = pyo.Var(
        model.T,
        within=pyo.NonNegativeReals,
    )
    model.idc_buy_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.idc_sell_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.afrr_energy_bid_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.afrr_energy_activated_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.final_planned_electricity_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.da_position_mwh[t],
    )
    model.actual_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.final_planned_electricity_mwh[t],
    )
    model.required_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.actual_electricity_consumption_mwh[t],
    )
    _add_market_cost_expressions(model)


def _add_intraday_stage(
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    signals: IDCAdjustmentSignals,
) -> None:
    steps = list(model.T)
    da_price = forecasts[signals.da_price_col].astype(float).to_numpy()
    idc_price = forecasts[signals.idc_price_col].astype(float).fillna(0.0).to_numpy()
    da_position = series_or_zero(
        signals.da_position_mwh,
        forecasts.index,
    ).to_numpy()
    buy_upper = (
        series_or_zero(
            signals.idc_buy_upper_bound_mwh,
            forecasts.index,
        )
        .clip(lower=0.0)
        .to_numpy()
    )
    sell_upper = (
        series_or_zero(
            signals.idc_sell_upper_bound_mwh,
            forecasts.index,
        )
        .clip(lower=0.0)
        .to_numpy()
    )
    model.da_price = pyo.Param(
        model.T,
        initialize={t: float(da_price[t]) for t in steps},
    )
    model.idc_price = pyo.Param(
        model.T,
        initialize={t: float(idc_price[t]) for t in steps},
    )
    model.afrr_energy_price = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.electricity_price = pyo.Param(
        model.T,
        initialize={
            t: float(idc_price[t]) + float(pyo.value(model.additional_electricity_charge[t]))
            for t in steps
        },
    )
    model.da_position_mwh = pyo.Param(
        model.T,
        initialize={t: float(da_position[t]) for t in steps},
    )
    model.idc_buy_upper_bound_mwh = pyo.Param(
        model.T,
        initialize={t: float(buy_upper[t]) for t in steps},
    )
    model.idc_sell_upper_bound_mwh = pyo.Param(
        model.T,
        initialize={t: float(sell_upper[t]) for t in steps},
    )
    model.idc_buy_mwh = pyo.Var(
        model.T,
        within=pyo.NonNegativeReals,
    )
    model.idc_sell_mwh = pyo.Var(
        model.T,
        within=pyo.NonNegativeReals,
    )
    model.final_planned_electricity_mwh = pyo.Var(
        model.T,
        within=pyo.NonNegativeReals,
    )
    model.afrr_energy_bid_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.afrr_energy_activated_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.idc_buy_limit = pyo.Constraint(
        model.T,
        rule=lambda mm, t: mm.idc_buy_mwh[t] <= mm.idc_buy_upper_bound_mwh[t],
    )
    model.idc_sell_limit = pyo.Constraint(
        model.T,
        rule=lambda mm, t: mm.idc_sell_mwh[t] <= mm.idc_sell_upper_bound_mwh[t],
    )
    model.idc_sell_da_limit = pyo.Constraint(
        model.T,
        rule=lambda mm, t: mm.idc_sell_mwh[t] <= mm.da_position_mwh[t],
    )
    model.final_planned_position = pyo.Constraint(
        model.T,
        rule=lambda mm, t: (
            mm.final_planned_electricity_mwh[t]
            == mm.da_position_mwh[t] + mm.idc_buy_mwh[t] - mm.idc_sell_mwh[t]
        ),
    )
    model.actual_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.final_planned_electricity_mwh[t],
    )
    model.required_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.actual_electricity_consumption_mwh[t],
    )
    _add_market_cost_expressions(model)


def _add_afrr_energy_stage(
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    signals: AFRRDownSignals,
) -> None:
    steps = list(model.T)
    da_price = forecasts[signals.da_price_col].astype(float).to_numpy()
    idc_price = forecasts[signals.idc_price_col].astype(float).fillna(0.0).to_numpy()
    afrr_price = series_or_zero(
        signals.afrr_energy_price,
        forecasts.index,
    ).to_numpy()
    da_position = series_or_zero(
        signals.da_position_mwh,
        forecasts.index,
    ).to_numpy()
    idc_buy = series_or_zero(
        signals.idc_buy_mwh,
        forecasts.index,
    ).to_numpy()
    idc_sell = series_or_zero(
        signals.idc_sell_mwh,
        forecasts.index,
    ).to_numpy()
    final_planned = series_or_zero(
        signals.final_planned_electricity_mwh,
        forecasts.index,
    ).to_numpy()
    afrr_bid = series_or_zero(
        signals.afrr_energy_bid_mwh,
        forecasts.index,
    ).to_numpy()
    afrr_activation = series_or_zero(
        signals.afrr_energy_activated_mwh,
        forecasts.index,
    ).to_numpy()
    system_activation = series_or_zero(
        signals.afrr_system_activation_mwh,
        forecasts.index,
    ).to_numpy()
    _validate_fixed_afrr_instruction(
        bid_mwh=afrr_bid,
        activation_mwh=afrr_activation,
        system_activation_mwh=system_activation,
    )

    model.da_price = pyo.Param(
        model.T,
        initialize={t: float(da_price[t]) for t in steps},
    )
    model.idc_price = pyo.Param(
        model.T,
        initialize={t: float(idc_price[t]) for t in steps},
    )
    model.afrr_energy_price = pyo.Param(
        model.T,
        initialize={t: float(afrr_price[t]) for t in steps},
    )
    model.electricity_price = pyo.Param(
        model.T,
        initialize={
            t: float(afrr_price[t]) + float(pyo.value(model.additional_electricity_charge[t]))
            for t in steps
        },
    )
    model.da_position_mwh = pyo.Param(
        model.T,
        initialize={t: float(da_position[t]) for t in steps},
    )
    model.idc_buy_mwh = pyo.Param(
        model.T,
        initialize={t: float(idc_buy[t]) for t in steps},
    )
    model.idc_sell_mwh = pyo.Param(
        model.T,
        initialize={t: float(idc_sell[t]) for t in steps},
    )
    model.final_planned_electricity_mwh = pyo.Param(
        model.T,
        initialize={t: float(final_planned[t]) for t in steps},
    )
    model.afrr_energy_bid_mwh = pyo.Param(
        model.T,
        initialize={t: float(afrr_bid[t]) for t in steps},
    )
    model.afrr_energy_activated_mwh = pyo.Param(
        model.T,
        initialize={t: float(afrr_activation[t]) for t in steps},
    )
    model.actual_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.final_planned_electricity_mwh[t] + mm.afrr_energy_activated_mwh[t],
    )
    model.required_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.actual_electricity_consumption_mwh[t],
    )
    _add_market_cost_expressions(model)


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


# --- Common result table ---------------------------------------------


def _extract_stage_results(
    plant: SteamGenerationPlant,
    stage: SteamMarketStage,
    model: pyo.ConcreteModel,
    config: CaseConfig,
    forecasts: pd.DataFrame,
    signals: SteamSignals,
    solver_name: str,
) -> pd.DataFrame:
    route = plant.route_process
    direct_route = route.name == DIRECT_ELECTRIC_GAS_BOILER_ROUTE
    dt_hours = config.timestep_minutes / 60.0
    da_price_col = (
        signals.electricity_price_col
        if isinstance(signals, DispatchSignals)
        else signals.da_price_col
    )
    idc_price_col = (
        signals.idc_price_col
        if isinstance(signals, (IDCAdjustmentSignals, AFRRDownSignals))
        else None
    )
    electricity_benchmark = getattr(
        signals,
        "electricity_trading_benchmark_eur_per_mwh_el",
        None,
    )
    rows: list[dict[str, object]] = []

    for position, timestamp in enumerate(forecasts.index):
        physical = route.physical_result_fields(model, position)
        electricity = pyomo_value(model.electricity_consumption[position])
        da_position = pyomo_value(model.da_position_mwh[position])
        idc_buy = pyomo_value(model.idc_buy_mwh[position])
        idc_sell = pyomo_value(model.idc_sell_mwh[position])
        final_planned = pyomo_value(model.final_planned_electricity_mwh[position])
        actual_electricity = pyomo_value(model.actual_electricity_consumption_mwh[position])
        afrr_bid = pyomo_value(model.afrr_energy_bid_mwh[position])
        afrr_activation = pyomo_value(model.afrr_energy_activated_mwh[position])
        day_ahead_price = float(forecasts[da_price_col].iloc[position])
        idc_price = (
            series_float_or_nan(
                forecasts[idc_price_col],
                position,
            )
            if idc_price_col is not None
            else float("nan")
        )
        additional_charge = pyomo_value(model.additional_electricity_charge[position])
        afrr_price = (
            pyomo_value(model.afrr_energy_price[position])
            if stage == SteamMarketStage.AFRR_ENERGY
            else float("nan")
        )
        tax_rate = pyomo_value(model.tax_rate)
        benchmark = (
            float(electricity_benchmark.iloc[position])
            if electricity_benchmark is not None
            else float("nan")
        )
        default_raw_bid_price = (
            benchmark / (1.0 + tax_rate) - additional_charge
            if stage == SteamMarketStage.AFRR_ENERGY
            else float("nan")
        )
        afrr_bid_price = (
            float(
                series_value(
                    cast(AFRRDownSignals, signals).afrr_energy_bid_price,
                    timestamp,
                    default_raw_bid_price,
                )
            )
            if stage == SteamMarketStage.AFRR_ENERGY
            else float("nan")
        )
        afrr_delivered_price = (afrr_price + additional_charge) * (1.0 + tax_rate)
        afrr_delivered_bid_price = (
            float(
                series_value(
                    cast(AFRRDownSignals, signals).afrr_energy_delivered_bid_price,
                    timestamp,
                    (afrr_bid_price + additional_charge) * (1.0 + tax_rate),
                )
            )
            if stage == SteamMarketStage.AFRR_ENERGY
            else float("nan")
        )
        afrr_market_spread = (
            afrr_delivered_bid_price - afrr_price if stage == SteamMarketStage.AFRR_ENERGY else 0.0
        )
        afrr_net_spread = (
            benchmark - afrr_delivered_price if stage == SteamMarketStage.AFRR_ENERGY else 0.0
        )
        afrr_reward = afrr_activation * afrr_market_spread
        afrr_net_value = afrr_activation * afrr_net_spread
        co2_price = (
            float(forecasts[signals.co2_price_col].iloc[position])
            if signals.co2_price_col and signals.co2_price_col in forecasts.columns
            else 0.0
        )
        afrr_signals = (
            cast(AFRRDownSignals, signals) if stage == SteamMarketStage.AFRR_ENERGY else None
        )
        default_free_volume = afrr_bid if direct_route else 0.0
        free_bid = (
            float(
                series_value(
                    afrr_signals.afrr_energy_free_bid_mwh,
                    timestamp,
                    default_free_volume,
                )
            )
            if afrr_signals is not None
            else 0.0
        )
        free_activation = (
            float(
                series_value(
                    afrr_signals.afrr_energy_free_activated_mwh,
                    timestamp,
                    afrr_activation if direct_route else 0.0,
                )
            )
            if afrr_signals is not None
            else 0.0
        )
        capacity_backed_bid = (
            0.0
            if direct_route or afrr_signals is None
            else float(
                series_value(
                    afrr_signals.afrr_energy_capacity_backed_bid_mwh,
                    timestamp,
                    0.0,
                )
            )
        )
        capacity_backed_activation = (
            0.0
            if direct_route or afrr_signals is None
            else float(
                series_value(
                    afrr_signals.afrr_energy_capacity_backed_activated_mwh,
                    timestamp,
                    0.0,
                )
            )
        )
        electricity_market_cost = pyomo_value(model.electricity_market_cost[position])
        additional_cost = pyomo_value(model.additional_electricity_charges_cost[position])
        electricity_cost = pyomo_value(model.electricity_cost[position])
        tax_cost = pyomo_value(model.tax_cost[position])
        gas_cost = pyomo_value(model.gas_cost[position])
        co2_cost = pyomo_value(model.co2_cost[position])

        row: dict[str, object] = {
            "datetime": timestamp,
            "plant_name": plant.name,
            "heat_demand_MWh": (
                float(forecasts[plant.heat_demand_column].iloc[position]) * dt_hours
            ),
            "day_ahead_price_EUR_per_MWh": day_ahead_price,
            "IDC_price_EUR_per_MWh": idc_price,
            "additional_electricity_charge_EUR_per_MWh_el": (additional_charge),
            "day_ahead_delivered_price_EUR_per_MWh": (
                (day_ahead_price + additional_charge) * (1.0 + tax_rate)
            ),
            "IDC_delivered_price_EUR_per_MWh": ((idc_price + additional_charge) * (1.0 + tax_rate)),
            "afrr_energy_delivered_price_EUR_per_MWh": (afrr_delivered_price),
            "gas_price_EUR_per_MWh": float(forecasts[signals.gas_price_col].iloc[position]),
            "co2_price_EUR_per_t": co2_price,
            "day_ahead_price_signal": da_price_col,
            "IDC_price_signal": idc_price_col or "",
            "gas_price_signal": signals.gas_price_col,
            "co2_price_signal": signals.co2_price_col or "",
            "gas_based_heat_benchmark_EUR_per_MWh_th": float(
                signals.gas_benchmark_eur_per_mwh_th.iloc[position]
            ),
            "electricity_trading_benchmark_EUR_per_MWh_el": benchmark,
            **physical,
            "electricity_consumption_MWh": electricity,
            "DA_position_MWh": da_position,
            "IDC_buy_MWh": idc_buy,
            "IDC_sell_MWh": idc_sell,
            "final_planned_electricity_MWh": final_planned,
            "actual_electricity_consumption_MWh": actual_electricity,
            "DA_electricity_cost_EUR": pyomo_value(model.da_electricity_cost[position]),
            "IDC_buy_cost_EUR": pyomo_value(model.idc_buy_cost[position]),
            "IDC_sell_revenue_EUR": pyomo_value(model.idc_sell_revenue[position]),
            "afrr_energy_bid_MWh": afrr_bid,
            "afrr_energy_bid_MW": (afrr_bid / dt_hours if dt_hours > 0 else 0.0),
            "afrr_energy_activated_MWh": afrr_activation,
            "afrr_energy_price_EUR_per_MWh": afrr_price,
            "afrr_system_activation_MWh": (
                float(afrr_signals.afrr_system_activation_mwh.iloc[position])
                if afrr_signals is not None
                else 0.0
            ),
            "afrr_energy_bid_price_EUR_per_MWh": afrr_bid_price,
            "afrr_energy_delivered_bid_price_EUR_per_MWh": (afrr_delivered_bid_price),
            "afrr_energy_market_spread_EUR_per_MWh": afrr_market_spread,
            "afrr_energy_net_spread_EUR_per_MWh": afrr_net_spread,
            "afrr_energy_cost_EUR": pyomo_value(model.afrr_energy_cost[position]),
            "afrr_energy_savings_vs_benchmark_EUR": afrr_net_value,
            "afrr_energy_pay_as_cleared_reward_EUR": afrr_reward,
            "afrr_energy_net_value_after_charges_EUR": afrr_net_value,
            "afrr_energy_capacity_backed_bid_MWh": capacity_backed_bid,
            "afrr_energy_free_bid_MWh": free_bid,
            "afrr_energy_capacity_backed_activated_MWh": (capacity_backed_activation),
            "afrr_energy_free_activated_MWh": free_activation,
            "afrr_headroom_binding": (
                bool(
                    series_value(
                        afrr_signals.afrr_headroom_binding,
                        timestamp,
                        False,
                    )
                )
                if afrr_signals is not None
                else False
            ),
            "afrr_curtailment_MWh": (
                float(
                    series_value(
                        afrr_signals.afrr_curtailment_mwh,
                        timestamp,
                        0.0,
                    )
                )
                if afrr_signals is not None
                else 0.0
            ),
            "electricity_market_cost_EUR": electricity_market_cost,
            "additional_electricity_charges_cost_EUR": additional_cost,
            "electricity_cost_EUR": electricity_cost,
            "tax_cost_EUR": tax_cost,
            "gas_cost_EUR": gas_cost,
            "co2_cost_EUR": co2_cost,
            "operating_cost_EUR": (electricity_cost + gas_cost + co2_cost + tax_cost),
            "charge_allowed_by_strategy": _charge_allowed_flag(
                stage,
                signals,
                position,
                direct_route,
            ),
            "idc_buy_allowed_by_strategy": (
                bool(
                    cast(
                        IDCAdjustmentSignals,
                        signals,
                    ).idc_buy_upper_bound_mwh.iloc[position]
                    > 1e-12
                )
                if stage == SteamMarketStage.INTRADAY
                else False
            ),
            "idc_sell_allowed_by_strategy": (
                bool(
                    cast(
                        IDCAdjustmentSignals,
                        signals,
                    ).idc_sell_upper_bound_mwh.iloc[position]
                    > 1e-12
                )
                if stage == SteamMarketStage.INTRADAY
                else False
            ),
            "afrr_energy_bid_allowed_by_strategy": bool(afrr_bid > 1e-12),
            "solver": solver_name,
        }
        row.update(
            route.capacity_result_fields(
                signals,
                timestamp,
                dt_hours,
                final_planned,
                physical,
                plant.components,
            )
        )
        row["gross_operating_cost_EUR"] = row["operating_cost_EUR"]
        row["net_operating_cost_EUR"] = float(row["gross_operating_cost_EUR"]) - float(
            row["afrr_capacity_revenue_EUR"]
        )
        _preserve_legacy_stage_schema(row, stage, direct_route)
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("datetime")
    numeric_columns = frame.select_dtypes(include=["number"]).columns
    frame[numeric_columns] = frame[numeric_columns].mask(
        frame[numeric_columns].abs() < 1e-9,
        0.0,
    )
    return frame


def _add_market_cost_expressions(model: pyo.ConcreteModel) -> None:
    model.da_electricity_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.da_position_mwh[t] * mm.da_price[t],
    )
    model.idc_buy_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.idc_buy_mwh[t] * mm.idc_price[t],
    )
    model.idc_sell_revenue = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.idc_sell_mwh[t] * mm.idc_price[t],
    )
    model.afrr_energy_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.afrr_energy_activated_mwh[t] * mm.afrr_energy_price[t],
    )
    model.electricity_market_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: (
            mm.da_electricity_cost[t]
            + mm.idc_buy_cost[t]
            - mm.idc_sell_revenue[t]
            + mm.afrr_energy_cost[t]
        ),
    )


def _validate_stage_signal_type(
    stage: SteamMarketStage,
    signals: SteamSignals,
) -> None:
    expected_type: type[SteamSignals]
    if stage == SteamMarketStage.DAY_AHEAD:
        expected_type = DispatchSignals
    elif stage == SteamMarketStage.INTRADAY:
        expected_type = IDCAdjustmentSignals
    else:
        expected_type = AFRRDownSignals
    if not isinstance(signals, expected_type):
        raise TypeError(
            f"Stage '{stage.value}' requires {expected_type.__name__}, "
            f"received {type(signals).__name__}"
        )


def _stage_label(stage: SteamMarketStage) -> str:
    return {
        SteamMarketStage.DAY_AHEAD: "Dispatch",
        SteamMarketStage.INTRADAY: "IDC adjustment",
        SteamMarketStage.AFRR_ENERGY: "aFRR down adjustment",
    }[stage]


def _stage_warning_label(stage: SteamMarketStage) -> str:
    return {
        SteamMarketStage.DAY_AHEAD: "day-ahead dispatch",
        SteamMarketStage.INTRADAY: "intraday adjustment",
        SteamMarketStage.AFRR_ENERGY: "aFRR down adjustment",
    }[stage]


def _charge_allowed_flag(
    stage: SteamMarketStage,
    signals: SteamSignals,
    position: int,
    direct_route: bool,
) -> bool:
    if stage == SteamMarketStage.DAY_AHEAD:
        return bool(cast(DispatchSignals, signals).charge_allowed.iloc[position])
    if stage == SteamMarketStage.INTRADAY and not direct_route:
        return bool(
            cast(
                IDCAdjustmentSignals,
                signals,
            ).idc_buy_upper_bound_mwh.iloc[position]
            > 1e-12
        )
    return False


def _preserve_legacy_stage_schema(
    row: dict[str, object],
    stage: SteamMarketStage,
    direct_route: bool,
) -> None:
    if stage != SteamMarketStage.AFRR_ENERGY:
        row.pop("afrr_energy_delivered_bid_price_EUR_per_MWh", None)
    if direct_route:
        return
    if stage == SteamMarketStage.DAY_AHEAD:
        for column in (
            "IDC_price_signal",
            "IDC_delivered_price_EUR_per_MWh",
            "afrr_energy_delivered_price_EUR_per_MWh",
            "electricity_trading_benchmark_EUR_per_MWh_el",
            "idc_buy_allowed_by_strategy",
            "idc_sell_allowed_by_strategy",
            "afrr_energy_bid_allowed_by_strategy",
            "afrr_headroom_binding",
            "afrr_curtailment_MWh",
        ):
            row.pop(column, None)
    elif stage == SteamMarketStage.INTRADAY:
        for column in (
            "afrr_energy_delivered_price_EUR_per_MWh",
            "afrr_energy_bid_allowed_by_strategy",
            "afrr_headroom_binding",
            "afrr_curtailment_MWh",
        ):
            row.pop(column, None)


def _validate_fixed_afrr_instruction(
    *,
    bid_mwh: np.ndarray,
    activation_mwh: np.ndarray,
    system_activation_mwh: np.ndarray,
) -> None:
    tolerance = 1e-9
    if (bid_mwh < -tolerance).any() or (activation_mwh < -tolerance).any():
        raise ValueError("Fixed aFRR bid and activation values must be non-negative")
    if (activation_mwh > bid_mwh + tolerance).any():
        raise ValueError("Fixed aFRR activation cannot exceed the submitted aFRR bid")
    if (activation_mwh > system_activation_mwh + tolerance).any():
        raise ValueError("Fixed aFRR activation cannot exceed the system activation signal")


def _capacity_signal_values(
    signals: SteamSignals,
    timestamp: pd.Timestamp,
    timestep_hours: float,
) -> dict[str, object]:
    reserved_mwh = float(series_value(signals.reserved_capacity_mwh, timestamp, 0.0))
    if signals.afrr_capacity_reserved_mw is not None:
        reserved_mw = float(series_value(signals.afrr_capacity_reserved_mw, timestamp, 0.0))
    else:
        reserved_mw = reserved_mwh / timestep_hours if timestep_hours > 0 else 0.0
    clearing_price = float(
        series_value(
            signals.afrr_capacity_price_eur_per_mw_h,
            timestamp,
            0.0,
        )
    )
    return {
        "afrr_capacity_block_id": str(series_value(signals.afrr_capacity_block_id, timestamp, "")),
        "afrr_capacity_block_duration_h": float(
            series_value(
                signals.afrr_capacity_block_duration_h,
                timestamp,
                0.0,
            )
        ),
        "afrr_capacity_pricing_rule": str(
            series_value(signals.afrr_capacity_pricing_rule, timestamp, "")
        ),
        "afrr_capacity_bid_price_EUR_per_MW_h": float(
            series_value(
                signals.afrr_capacity_bid_price_eur_per_mw_h,
                timestamp,
                0.0,
            )
        ),
        "afrr_capacity_clearing_price_EUR_per_MW_h": clearing_price,
        "afrr_capacity_settlement_price_EUR_per_MW_h": float(
            series_value(
                signals.afrr_capacity_settlement_price_eur_per_mw_h,
                timestamp,
                0.0,
            )
        ),
        "afrr_capacity_down_price_EUR_per_MW_h": clearing_price,
        "afrr_capacity_reserved_MW": reserved_mw,
        "afrr_capacity_reserved_MWh": reserved_mwh,
        "afrr_capacity_revenue_EUR": float(
            series_value(signals.afrr_capacity_revenue_eur, timestamp, 0.0)
        ),
        "afrr_capacity_opportunity_cost_EUR": float(
            series_value(
                signals.afrr_capacity_opportunity_cost_eur,
                timestamp,
                0.0,
            )
        ),
        "afrr_capacity_market_surplus_EUR": float(
            series_value(
                signals.afrr_capacity_market_surplus_eur,
                timestamp,
                0.0,
            )
        ),
        "afrr_capacity_net_value_EUR": float(
            series_value(signals.afrr_capacity_net_value_eur, timestamp, 0.0)
        ),
    }


def _empty_capacity_result_fields() -> dict[str, object]:
    return {
        "afrr_capacity_block_id": "",
        "afrr_capacity_block_duration_h": 0.0,
        "afrr_capacity_pricing_rule": "",
        "afrr_capacity_bid_price_EUR_per_MW_h": 0.0,
        "afrr_capacity_clearing_price_EUR_per_MW_h": 0.0,
        "afrr_capacity_settlement_price_EUR_per_MW_h": 0.0,
        "afrr_capacity_down_price_EUR_per_MW_h": 0.0,
        "afrr_capacity_reserved_MW": 0.0,
        "afrr_capacity_reserved_MWh": 0.0,
        "afrr_capacity_revenue_EUR": 0.0,
        "afrr_capacity_opportunity_cost_EUR": 0.0,
        "afrr_capacity_market_surplus_EUR": 0.0,
        "afrr_capacity_net_value_EUR": 0.0,
    }


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
