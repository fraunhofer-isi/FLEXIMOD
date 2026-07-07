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
    COAL,
    TECHNOLOGY_REGISTRY,
    BasicOxygenFurnace,
    BlastFurnaceBasicOxygenFurnace,
    DRIPlant,
    ElectricArcFurnace,
    GenericInventoryStorage,
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
    coal_price_col: str = "coal_price"
    steel_price_col: str | None = None


@dataclass(frozen=True)
class SteelAFRRDownSignals:
    """Ontology-aligned market and commodity inputs for electrified steel."""

    da_price_col: str
    afrr_energy_price_col: str
    afrr_system_activation_col: str
    afrr_price_available_col: str
    afrr_capacity_block_id_col: str
    afrr_capacity_block_duration_col: str
    afrr_capacity_price_col: str
    afrr_capacity_missing_price_col: str
    iron_ore_price_col: str
    lime_price_col: str
    co2_price_col: str
    additional_electricity_charge_eur_per_mwh: float = 0.0
    afrr_energy_min_bid_mw: float = 1.0
    afrr_energy_bid_increment_mw: float = 1.0
    afrr_capacity_min_bid_mw: float = 1.0
    afrr_capacity_bid_increment_mw: float = 1.0
    afrr_capacity_product_duration_h: float = 4.0


@dataclass(frozen=True)
class SteelComponentState:
    power_in_mwh: float
    operational_status: int
    consecutive_status_steps: int


@dataclass(frozen=True)
class SteelInventoryState:
    soc: float
    charge: float = 0.0
    discharge: float = 0.0


@dataclass
class SteelRollingState:
    components: dict[str, SteelComponentState] = field(default_factory=dict)
    inventories: dict[str, SteelInventoryState] = field(default_factory=dict)
    cumulative_steel_output_t: float = 0.0
    demand_balance_t: float = 0.0


@dataclass
class SteelPlant(BasePlant):
    """Route-aware steel-production model.

    Supported terminal routes are DRI -> EAF, DRI -> BOF, and standalone BF-BOF.
    Optional hydrogen and DRI stores can shift intermediate production across the horizon.
    A market strategy is deliberately not embedded in this class.
    """

    steel_demand_tonnes: float | None = None
    steel_demand_column: str = ""
    components: dict[str, object] = field(default_factory=dict)
    steel_route: str = ""

    route_technologies = frozenset({"dri_plant", "eaf", "bof", "bf_bof"})
    optional_technologies = frozenset({"electrolyser", "hydrogen_buffer_storage", "dri_storage"})

    @classmethod
    def from_rows(cls, plant_name: str, rows: pd.DataFrame) -> SteelPlant:
        normalised = rows.copy()
        normalised["technology_normalised"] = (
            normalised["technology"].astype(str).str.strip().str.lower()
        )
        allowed = cls.route_technologies | cls.optional_technologies
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

        steel_route = _detect_steel_route(components, plant_name)

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
            steel_route=steel_route,
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

    @property
    def bof(self) -> BasicOxygenFurnace:
        component = self.components.get("bof")
        if not isinstance(component, BasicOxygenFurnace):
            raise ValueError(f"Steel plant '{self.name}' has no BOF")
        return component

    @property
    def bf_bof(self) -> BlastFurnaceBasicOxygenFurnace:
        component = self.components.get("bf_bof")
        if not isinstance(component, BlastFurnaceBasicOxygenFurnace):
            raise ValueError(f"Steel plant '{self.name}' has no BF-BOF")
        return component

    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: SteelDispatchSignals,
        initial_state: SteelRollingState | None = None,
    ) -> pd.DataFrame:
        if self.objective != "min_variable_cost":
            raise ValueError(f"Steel plant objective '{self.objective}' is not supported")
        if forecasts.empty:
            raise ValueError("Steel plant dispatch horizon must contain at least one timestep")
        self._validate_signal_columns(forecasts, signals)
        model = self._build_model(
            config,
            forecasts,
            signals,
            initial_state=initial_state or self._initial_rolling_state(),
        )
        return self._solve_model(config, forecasts, model)

    def solve_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: SteelDispatchSignals,
        initial_state: SteelRollingState | None = None,
    ) -> pd.DataFrame:
        if forecasts.empty:
            raise ValueError("Steel rolling dispatch requires at least one timestep")
        self._validate_signal_columns(forecasts, signals)
        dt_hours = config.timestep_minutes / 60.0
        horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 48))
        step_hours = float(config.dispatch_setting("rolling_step_hours", 24))
        _validate_rolling_window(dt_hours, horizon_hours, step_hours)
        horizon_steps = int(round(horizon_hours / dt_hours))
        step_steps = int(round(step_hours / dt_hours))

        demand_schedule, demand_mode = self._rolling_demand_schedule(forecasts)
        total_demand = float(demand_schedule.sum())
        state = initial_state or self._initial_rolling_state()
        if state.cumulative_steel_output_t > total_demand + 1e-8:
            raise ValueError(
                f"Steel rolling state already produced {state.cumulative_steel_output_t:g} t, "
                f"above the simulation target of {total_demand:g} t"
            )

        implemented_frames: list[pd.DataFrame] = []
        position = 0
        window_number = 1
        while position < len(forecasts):
            horizon = forecasts.iloc[position : position + horizon_steps].copy()
            commit_count = min(step_steps, len(horizon), len(forecasts) - position)
            horizon_schedule = demand_schedule.iloc[position : position + len(horizon)]
            commit_schedule = horizon_schedule.iloc[:commit_count]
            is_final_window = position + commit_count >= len(forecasts)
            remaining_demand = max(0.0, total_demand - state.cumulative_steel_output_t)
            horizon_target = max(0.0, state.demand_balance_t + float(horizon_schedule.sum()))
            if is_final_window:
                horizon_target = remaining_demand
            horizon_target = min(horizon_target, remaining_demand)
            minimum_commit_output = (
                remaining_demand if is_final_window else max(0.0, state.demand_balance_t)
            )

            model = self._build_model(
                config,
                horizon,
                signals,
                initial_state=state,
                steel_demand_override_t=horizon_target,
                commit_steps=commit_count,
                minimum_commit_output_t=minimum_commit_output,
                demand_mode=demand_mode,
            )
            try:
                horizon_result = self._solve_model(config, horizon, model)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Steel rolling window {window_number} starting "
                    f"{horizon.index[0]} is infeasible: target={horizon_target:g} t, "
                    f"inherited backlog={max(0.0, state.demand_balance_t):g} t"
                ) from exc

            implemented = horizon_result.iloc[:commit_count].copy()
            produced_before = state.cumulative_steel_output_t
            balance_before = state.demand_balance_t
            produced_by_row = implemented["steel_output_t"].cumsum()
            scheduled_by_row = commit_schedule.cumsum()
            implemented["rolling_window"] = window_number
            implemented["steel_horizon_target_t"] = horizon_target
            implemented["steel_committed_output_t"] = float(implemented["steel_output_t"].sum())
            implemented["cumulative_steel_output_t"] = produced_before + produced_by_row
            implemented["remaining_steel_demand_t"] = (
                total_demand - implemented["cumulative_steel_output_t"]
            ).clip(lower=0.0)
            implemented["steel_demand_balance_t"] = (
                balance_before + scheduled_by_row - produced_by_row
            )
            implemented["steel_demand_total_t"] = total_demand
            implemented_frames.append(implemented)

            state = self._state_after_commit(
                previous=state,
                committed=implemented,
                scheduled_demand_t=float(commit_schedule.sum()),
            )
            position += commit_count
            window_number += 1

        result = pd.concat(implemented_frames).sort_index()
        if abs(state.cumulative_steel_output_t - total_demand) > 1e-6:
            raise RuntimeError(
                "Steel rolling dispatch ended without satisfying total demand: "
                f"produced={state.cumulative_steel_output_t:g} t, target={total_demand:g} t"
            )
        if abs(state.demand_balance_t) > 1e-6:
            raise RuntimeError(
                f"Steel rolling dispatch ended with demand balance {state.demand_balance_t:g} t"
            )
        return result

    def solve_afrr_down_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: SteelAFRRDownSignals,
        initial_state: SteelRollingState | None = None,
    ) -> pd.DataFrame:
        """Optimize the next-day DA and aFRR-down portfolio on a rolling horizon."""

        if forecasts.empty:
            raise ValueError("Electrified-steel rolling dispatch requires at least one timestep")
        self._validate_electrified_steel()
        self._validate_afrr_signal_columns(forecasts, signals)
        _validate_bid_rules(
            "afrr_energy",
            signals.afrr_energy_min_bid_mw,
            signals.afrr_energy_bid_increment_mw,
        )
        _validate_bid_rules(
            "afrr_capacity",
            signals.afrr_capacity_min_bid_mw,
            signals.afrr_capacity_bid_increment_mw,
        )

        dt_hours = config.timestep_minutes / 60.0
        horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 48))
        step_hours = float(config.dispatch_setting("rolling_step_hours", 24))
        _validate_rolling_window(dt_hours, horizon_hours, step_hours)
        if horizon_hours + 1e-9 < step_hours + signals.afrr_capacity_product_duration_h:
            raise ValueError(
                "Electrified-steel dispatch_horizon_hours must cover rolling_step_hours plus "
                "one complete aFRR capacity product"
            )
        horizon_steps = int(round(horizon_hours / dt_hours))
        step_steps = int(round(step_hours / dt_hours))

        demand_schedule, demand_mode = self._rolling_demand_schedule(forecasts)
        total_demand = float(demand_schedule.sum())
        state = initial_state or self._initial_rolling_state()
        implemented_frames: list[pd.DataFrame] = []
        position = 0
        window_number = 1
        while position < len(forecasts):
            horizon = forecasts.iloc[position : position + horizon_steps].copy()
            commit_count = min(step_steps, len(horizon), len(forecasts) - position)
            horizon_schedule = demand_schedule.iloc[position : position + len(horizon)]
            commit_schedule = horizon_schedule.iloc[:commit_count]
            is_final_window = position + commit_count >= len(forecasts)
            remaining_demand = max(0.0, total_demand - state.cumulative_steel_output_t)
            horizon_target = max(0.0, state.demand_balance_t + float(horizon_schedule.sum()))
            if is_final_window:
                horizon_target = remaining_demand
            horizon_target = min(horizon_target, remaining_demand)
            minimum_commit_output = (
                remaining_demand if is_final_window else max(0.0, state.demand_balance_t)
            )

            baseline_model = self._build_afrr_down_model(
                config,
                horizon,
                signals,
                initial_state=state,
                steel_demand_override_t=horizon_target,
                commit_steps=commit_count,
                minimum_commit_output_t=minimum_commit_output,
                demand_mode=demand_mode,
                capacity_enabled=False,
                terminal_state_required=is_final_window,
            )
            market_model = self._build_afrr_down_model(
                config,
                horizon,
                signals,
                initial_state=state,
                steel_demand_override_t=horizon_target,
                commit_steps=commit_count,
                minimum_commit_output_t=minimum_commit_output,
                demand_mode=demand_mode,
                capacity_enabled=True,
                terminal_state_required=is_final_window,
            )
            try:
                baseline_result = self._solve_model(config, horizon, baseline_model)
                horizon_result = self._solve_model(config, horizon, market_model)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Electrified-steel rolling window {window_number} starting "
                    f"{horizon.index[0]} is infeasible: target={horizon_target:g} t, "
                    f"inherited backlog={max(0.0, state.demand_balance_t):g} t"
                ) from exc

            horizon_result = _attach_capacity_opportunity_cost(
                horizon_result,
                baseline_result,
                commit_steps=commit_count,
            )
            implemented = horizon_result.iloc[:commit_count].copy()
            produced_before = state.cumulative_steel_output_t
            balance_before = state.demand_balance_t
            produced_by_row = implemented["steel_output_t"].cumsum()
            scheduled_by_row = commit_schedule.cumsum()
            implemented["rolling_window"] = window_number
            implemented["steel_horizon_target_t"] = horizon_target
            implemented["steel_committed_output_t"] = float(implemented["steel_output_t"].sum())
            implemented["cumulative_steel_output_t"] = produced_before + produced_by_row
            implemented["remaining_steel_demand_t"] = (
                total_demand - implemented["cumulative_steel_output_t"]
            ).clip(lower=0.0)
            implemented["steel_demand_balance_t"] = (
                balance_before + scheduled_by_row - produced_by_row
            )
            implemented["steel_demand_total_t"] = total_demand
            implemented_frames.append(implemented)

            state = self._state_after_commit(
                previous=state,
                committed=implemented,
                scheduled_demand_t=float(commit_schedule.sum()),
            )
            position += commit_count
            window_number += 1

        result = pd.concat(implemented_frames).sort_index()
        if abs(state.cumulative_steel_output_t - total_demand) > 1e-6:
            raise RuntimeError(
                "Electrified-steel dispatch ended without satisfying total demand: "
                f"produced={state.cumulative_steel_output_t:g} t, target={total_demand:g} t"
            )
        if abs(state.demand_balance_t) > 1e-6:
            raise RuntimeError(
                f"Electrified-steel dispatch ended with demand balance {state.demand_balance_t:g} t"
            )
        return result

    def _build_afrr_down_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: SteelAFRRDownSignals,
        initial_state: SteelRollingState,
        steel_demand_override_t: float,
        commit_steps: int,
        minimum_commit_output_t: float,
        demand_mode: str,
        capacity_enabled: bool,
        terminal_state_required: bool,
    ) -> pyo.ConcreteModel:
        dt_hours = config.timestep_minutes / 60.0
        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=range(len(forecasts)), ordered=True)

        def values(column: str) -> dict[int, float]:
            return {t: float(forecasts[column].iloc[t]) for t in model.T}

        # Electricity is settled once at plant level below. Component electricity prices
        # are therefore zero to avoid double counting in technology operating costs.
        model.electricity_price = pyo.Param(model.T, initialize={t: 0.0 for t in model.T})
        model.natural_gas_price = pyo.Param(model.T, initialize={t: 0.0 for t in model.T})
        model.hydrogen_price = pyo.Param(model.T, initialize={t: 0.0 for t in model.T})
        model.coal_price = pyo.Param(model.T, initialize={t: 0.0 for t in model.T})
        model.iron_ore_price = pyo.Param(model.T, initialize=values(signals.iron_ore_price_col))
        model.lime_price = pyo.Param(model.T, initialize=values(signals.lime_price_col))
        model.co2_price = pyo.Param(model.T, initialize=values(signals.co2_price_col))
        model.steel_demand = pyo.Param(initialize=float(steel_demand_override_t))
        model.steel_demand_from_forecast = pyo.Param(
            initialize=int(demand_mode == "forecast_profile"), within=pyo.Binary
        )
        model.day_ahead_price = pyo.Param(model.T, initialize=values(signals.da_price_col))
        model.afrr_energy_price = pyo.Param(
            model.T, initialize=values(signals.afrr_energy_price_col)
        )
        model.afrr_system_activation_mwh = pyo.Param(
            model.T, initialize=values(signals.afrr_system_activation_col)
        )
        additional_charge = float(signals.additional_electricity_charge_eur_per_mwh)
        model.additional_electricity_charge = pyo.Param(initialize=additional_charge)

        delivered_da = forecasts[signals.da_price_col].astype(float) + additional_charge
        bid_price = delivered_da.iloc[::-1].cummax().iloc[::-1]
        delivered_afrr = forecasts[signals.afrr_energy_price_col].astype(float) + additional_charge
        price_available = forecasts[signals.afrr_price_available_col].astype(bool)
        free_bid_allowed = price_available & (delivered_afrr <= bid_price + 1e-9)
        model.afrr_energy_bid_price = pyo.Param(
            model.T, initialize={t: float(bid_price.iloc[t]) for t in model.T}
        )
        model.free_bid_allowed = pyo.Param(
            model.T,
            initialize={t: int(bool(free_bid_allowed.iloc[t])) for t in model.T},
            within=pyo.Binary,
        )

        block_ids = list(dict.fromkeys(forecasts[signals.afrr_capacity_block_id_col].astype(str)))
        model.B = pyo.Set(initialize=block_ids, ordered=True)
        block_by_t = {
            t: str(forecasts[signals.afrr_capacity_block_id_col].iloc[t]) for t in model.T
        }
        block_prices = {
            block_id: float(
                forecasts.loc[
                    forecasts[signals.afrr_capacity_block_id_col].astype(str) == block_id,
                    signals.afrr_capacity_price_col,
                ].iloc[0]
            )
            for block_id in block_ids
        }
        block_durations = {
            block_id: float(
                forecasts.loc[
                    forecasts[signals.afrr_capacity_block_id_col].astype(str) == block_id,
                    signals.afrr_capacity_block_duration_col,
                ].iloc[0]
            )
            for block_id in block_ids
        }
        block_available = {}
        for block_id in block_ids:
            mask = forecasts[signals.afrr_capacity_block_id_col].astype(str) == block_id
            missing = bool(forecasts.loc[mask, signals.afrr_capacity_missing_price_col].iloc[0])
            complete = (
                abs(float(mask.sum()) * dt_hours - signals.afrr_capacity_product_duration_h) < 1e-8
            )
            block_available[block_id] = int(
                capacity_enabled and not missing and complete and block_prices[block_id] > 0.0
            )

        model.afrr_capacity_price = pyo.Param(model.B, initialize=block_prices)
        model.afrr_capacity_block_duration = pyo.Param(model.B, initialize=block_durations)
        model.capacity_available = pyo.Param(model.B, initialize=block_available, within=pyo.Binary)

        aggregate_max_power_mw = sum(
            float(getattr(component, "max_power_mw", 0.0)) for component in self.components.values()
        )
        capacity_max_steps = int(
            math.floor(aggregate_max_power_mw / signals.afrr_capacity_bid_increment_mw + 1e-9)
        )
        energy_max_steps = int(
            math.floor(aggregate_max_power_mw / signals.afrr_energy_bid_increment_mw + 1e-9)
        )
        capacity_min_steps = int(
            math.ceil(signals.afrr_capacity_min_bid_mw / signals.afrr_capacity_bid_increment_mw)
        )
        energy_min_steps = int(
            math.ceil(signals.afrr_energy_min_bid_mw / signals.afrr_energy_bid_increment_mw)
        )

        model.capacity_selected = pyo.Var(model.B, within=pyo.Binary)
        model.capacity_bid_steps = pyo.Var(
            model.B, within=pyo.NonNegativeIntegers, bounds=(0, capacity_max_steps)
        )
        model.afrr_capacity_reserved_mw = pyo.Expression(
            model.B,
            rule=lambda m, b: signals.afrr_capacity_bid_increment_mw * m.capacity_bid_steps[b],
        )

        @model.Constraint(model.B)
        def capacity_minimum_bid(m: pyo.ConcreteModel, b: str) -> pyo.Constraint:
            return m.capacity_bid_steps[b] >= capacity_min_steps * m.capacity_selected[b]

        @model.Constraint(model.B)
        def capacity_selection_limit(m: pyo.ConcreteModel, b: str) -> pyo.Constraint:
            return m.capacity_bid_steps[b] <= capacity_max_steps * m.capacity_selected[b]

        @model.Constraint(model.B)
        def capacity_availability(m: pyo.ConcreteModel, b: str) -> pyo.Constraint:
            return m.capacity_selected[b] <= m.capacity_available[b]

        model.free_bid_selected = pyo.Var(model.T, within=pyo.Binary)
        model.free_bid_steps = pyo.Var(
            model.T, within=pyo.NonNegativeIntegers, bounds=(0, energy_max_steps)
        )
        model.afrr_energy_free_bid_mw = pyo.Expression(
            model.T,
            rule=lambda m, t: signals.afrr_energy_bid_increment_mw * m.free_bid_steps[t],
        )
        model.afrr_energy_capacity_backed_bid_mwh = pyo.Expression(
            model.T,
            rule=lambda m, t: m.afrr_capacity_reserved_mw[block_by_t[t]] * dt_hours,
        )
        model.afrr_energy_free_bid_mwh = pyo.Expression(
            model.T, rule=lambda m, t: m.afrr_energy_free_bid_mw[t] * dt_hours
        )
        model.afrr_energy_bid_mwh = pyo.Expression(
            model.T,
            rule=lambda m, t: (
                m.afrr_energy_capacity_backed_bid_mwh[t] + m.afrr_energy_free_bid_mwh[t]
            ),
        )

        @model.Constraint(model.T)
        def free_energy_minimum_bid(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.free_bid_steps[t] >= energy_min_steps * m.free_bid_selected[t]

        @model.Constraint(model.T)
        def free_energy_selection_limit(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.free_bid_steps[t] <= energy_max_steps * m.free_bid_selected[t]

        @model.Constraint(model.T)
        def free_energy_price_gate(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.free_bid_selected[t] <= m.free_bid_allowed[t]

        max_activation_mwh = max(
            aggregate_max_power_mw * dt_hours,
            float(forecasts[signals.afrr_system_activation_col].max()),
            1.0,
        )
        model.afrr_energy_activated_mwh = pyo.Var(
            model.T, within=pyo.NonNegativeReals, bounds=(0.0, max_activation_mwh)
        )
        model.activation_bid_is_minimum = pyo.Var(model.T, within=pyo.Binary)

        @model.Constraint(model.T)
        def activation_not_above_bid(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.afrr_energy_activated_mwh[t] <= m.afrr_energy_bid_mwh[t]

        @model.Constraint(model.T)
        def activation_not_above_request(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.afrr_energy_activated_mwh[t] <= m.afrr_system_activation_mwh[t]

        @model.Constraint(model.T)
        def activation_equals_bid_when_smaller(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.afrr_energy_activated_mwh[t] >= (
                m.afrr_energy_bid_mwh[t] - max_activation_mwh * m.activation_bid_is_minimum[t]
            )

        @model.Constraint(model.T)
        def activation_equals_request_when_smaller(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.afrr_energy_activated_mwh[t] >= (
                m.afrr_system_activation_mwh[t]
                - max_activation_mwh * (1 - m.activation_bid_is_minimum[t])
            )

        model.afrr_energy_capacity_backed_activated_mwh = pyo.Var(
            model.T, within=pyo.NonNegativeReals, bounds=(0.0, max_activation_mwh)
        )
        model.capacity_activation_is_minimum = pyo.Var(model.T, within=pyo.Binary)

        @model.Constraint(model.T)
        def capacity_activation_not_above_bid(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return (
                m.afrr_energy_capacity_backed_activated_mwh[t]
                <= m.afrr_energy_capacity_backed_bid_mwh[t]
            )

        @model.Constraint(model.T)
        def capacity_activation_not_above_total(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.afrr_energy_capacity_backed_activated_mwh[t] <= m.afrr_energy_activated_mwh[t]

        @model.Constraint(model.T)
        def capacity_activation_equals_bid_when_smaller(
            m: pyo.ConcreteModel, t: int
        ) -> pyo.Constraint:
            return m.afrr_energy_capacity_backed_activated_mwh[t] >= (
                m.afrr_energy_capacity_backed_bid_mwh[t]
                - max_activation_mwh * m.capacity_activation_is_minimum[t]
            )

        @model.Constraint(model.T)
        def capacity_activation_equals_total_when_smaller(
            m: pyo.ConcreteModel, t: int
        ) -> pyo.Constraint:
            return m.afrr_energy_capacity_backed_activated_mwh[t] >= (
                m.afrr_energy_activated_mwh[t]
                - max_activation_mwh * (1 - m.capacity_activation_is_minimum[t])
            )

        model.afrr_energy_free_activated_mwh = pyo.Expression(
            model.T,
            rule=lambda m, t: (
                m.afrr_energy_activated_mwh[t] - m.afrr_energy_capacity_backed_activated_mwh[t]
            ),
        )
        model.da_position_mwh = pyo.Var(model.T, within=pyo.NonNegativeReals)

        model.actual = pyo.Block()
        model.full_activation = pyo.Block()
        self._add_physical_system(
            model,
            model.actual,
            model.T,
            initial_state,
            dt_hours,
            commit_steps,
            minimum_commit_output_t,
        )
        self._add_physical_system(
            model,
            model.full_activation,
            model.T,
            initial_state,
            dt_hours,
            commit_steps,
            minimum_commit_output_t,
        )

        if terminal_state_required:
            final_t = list(model.T)[-1]
            inventory_names = [
                name
                for name, component in self.components.items()
                if isinstance(component, GenericInventoryStorage)
            ]

            @model.Constraint(inventory_names)
            def actual_terminal_inventory(m: pyo.ConcreteModel, technology: str) -> pyo.Constraint:
                target = float(self.components[technology].initial_soc)
                return m.actual.technology_blocks[technology].soc[final_t] == target

            @model.Constraint(inventory_names)
            def full_activation_terminal_inventory(
                m: pyo.ConcreteModel, technology: str
            ) -> pyo.Constraint:
                target = float(self.components[technology].initial_soc)
                return m.full_activation.technology_blocks[technology].soc[final_t] == target

        @model.Constraint(model.T)
        def actual_electricity_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.actual.total_power_input[t] == (
                m.da_position_mwh[t] + m.afrr_energy_activated_mwh[t]
            )

        @model.Constraint(model.T)
        def full_activation_electricity_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.full_activation.total_power_input[t] == (
                m.da_position_mwh[t] + m.afrr_energy_bid_mwh[t]
            )

        model.electricity_market_cost = pyo.Expression(
            model.T,
            rule=lambda m, t: (
                m.da_position_mwh[t] * m.day_ahead_price[t]
                + m.afrr_energy_activated_mwh[t] * m.afrr_energy_price[t]
            ),
        )
        model.additional_electricity_charges_cost = pyo.Expression(
            model.T,
            rule=lambda m, t: (m.actual.total_power_input[t] * m.additional_electricity_charge),
        )
        model.gross_operating_cost = pyo.Expression(
            model.T,
            rule=lambda m, t: (
                m.actual.variable_cost[t]
                + m.electricity_market_cost[t]
                + m.additional_electricity_charges_cost[t]
            ),
        )
        model.afrr_capacity_revenue = pyo.Expression(
            model.T,
            rule=lambda m, t: (
                m.afrr_capacity_reserved_mw[block_by_t[t]]
                * m.afrr_capacity_price[block_by_t[t]]
                * dt_hours
            ),
        )
        model.net_operating_cost = pyo.Expression(
            model.T,
            rule=lambda m, t: m.gross_operating_cost[t] - m.afrr_capacity_revenue[t],
        )
        bid_tiebreaker = 1e-6 * (
            sum(model.capacity_bid_steps[b] for b in model.B)
            + sum(model.free_bid_steps[t] for t in model.T)
        )
        model.objective = pyo.Objective(
            expr=sum(model.net_operating_cost[t] for t in model.T) + bid_tiebreaker,
            sense=pyo.minimize,
        )
        model._afrr_block_by_t = block_by_t
        model._afrr_timestep_hours = dt_hours
        model._afrr_aggregate_max_power_mw = aggregate_max_power_mw
        return model

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
        raise RuntimeError("Steel plant dispatch could not be solved; " + "; ".join(errors))

    def _build_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: SteelDispatchSignals,
        initial_state: SteelRollingState | None = None,
        steel_demand_override_t: float | None = None,
        commit_steps: int | None = None,
        minimum_commit_output_t: float = 0.0,
        demand_mode: str | None = None,
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
        model.iron_ore_price = pyo.Param(model.T, initialize=values(signals.iron_ore_price_col))
        model.lime_price = pyo.Param(model.T, initialize=values(signals.lime_price_col))
        model.co2_price = pyo.Param(model.T, initialize=values(signals.co2_price_col))
        if signals.steel_price_col:
            model.steel_price = pyo.Param(model.T, initialize=values(signals.steel_price_col))
        steel_demand_total, resolved_demand_mode = self._resolve_steel_demand(forecasts)
        if steel_demand_override_t is not None:
            steel_demand_total = float(steel_demand_override_t)
        steel_demand_mode = demand_mode or resolved_demand_mode
        model.steel_demand = pyo.Param(initialize=steel_demand_total)
        model.steel_demand_from_forecast = pyo.Param(
            initialize=int(steel_demand_mode == "forecast_profile"), within=pyo.Binary
        )
        state = initial_state or self._initial_rolling_state()
        self._add_physical_system(
            model=model,
            container=model,
            time_steps=model.T,
            initial_state=state,
            dt_hours=dt_hours,
            commit_steps=commit_steps,
            minimum_commit_output_t=minimum_commit_output_t,
        )

        model.objective = pyo.Objective(
            expr=sum(model.variable_cost[t] for t in model.T), sense=pyo.minimize
        )
        return model

    def _add_physical_system(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
        initial_state: SteelRollingState,
        dt_hours: float,
        commit_steps: int | None = None,
        minimum_commit_output_t: float = 0.0,
    ) -> None:
        """Attach one complete steel-production trajectory to ``container``."""

        container.technology_blocks = pyo.Block(list(self.components))
        for technology, component in self.components.items():
            context: dict[str, Any] = {"dt_hours": dt_hours}
            component_state = initial_state.components.get(technology)
            if component_state is not None:
                context.update(
                    {
                        "initial_power_in": component_state.power_in_mwh,
                        "initial_operational_status": component_state.operational_status,
                        "initial_consecutive_status_steps": (
                            component_state.consecutive_status_steps
                        ),
                    }
                )
            inventory_state = initial_state.inventories.get(technology)
            if inventory_state is not None:
                context.update(
                    {
                        "initial_soc": inventory_state.soc,
                        "initial_charge": inventory_state.charge,
                        "initial_discharge": inventory_state.discharge,
                    }
                )
            component.add_to_model(
                model,
                container.technology_blocks[technology],
                time_steps,
                context,
            )

        terminal_name = self._terminal_technology_name()
        terminal = container.technology_blocks[terminal_name]
        dri = container.technology_blocks["dri_plant"] if "dri_plant" in self.components else None
        has_electrolyser = "electrolyser" in self.components
        has_hydrogen_storage = "hydrogen_buffer_storage" in self.components
        has_dri_storage = "dri_storage" in self.components

        if has_electrolyser:
            electrolyser = container.technology_blocks["electrolyser"]
            hydrogen_demand = []
            if dri is not None:
                hydrogen_demand.append(dri.hydrogen_in)
            if hasattr(terminal, "hydrogen_in"):
                hydrogen_demand.append(terminal.hydrogen_in)

            def total_hydrogen_demand(t: int) -> pyo.Expression:
                return sum(demand[t] for demand in hydrogen_demand)

            if has_hydrogen_storage:
                hydrogen_storage = container.technology_blocks["hydrogen_buffer_storage"]

                @container.Constraint(time_steps)
                def hydrogen_flow_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                    return electrolyser.hydrogen_out[t] + hydrogen_storage.discharge[t] == (
                        total_hydrogen_demand(t) + hydrogen_storage.charge[t]
                    )

            else:

                @container.Constraint(time_steps)
                def hydrogen_flow_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                    return electrolyser.hydrogen_out[t] == total_hydrogen_demand(t)

        if has_dri_storage:
            dri_storage = container.technology_blocks["dri_storage"]

            @container.Constraint(time_steps)
            def dri_flow_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                return dri.dri_output[t] + dri_storage.discharge[t] == (
                    terminal.dri_input[t] + dri_storage.charge[t]
                )

        elif dri is not None:

            @container.Constraint(time_steps)
            def dri_flow_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                return dri.dri_output[t] == terminal.dri_input[t]

        container.total_power_input = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        container.variable_cost = pyo.Var(time_steps, within=pyo.Reals)

        @container.Constraint()
        def steel_output_association_constraint(m: pyo.ConcreteModel) -> pyo.Constraint:
            return sum(terminal.steel_output[t] for t in time_steps) == model.steel_demand

        if commit_steps is not None and minimum_commit_output_t > 0:
            committed_steps = list(time_steps)[:commit_steps]

            @container.Constraint()
            def inherited_backlog_recovery_constraint(
                m: pyo.ConcreteModel,
            ) -> pyo.Constraint:
                return (
                    sum(terminal.steel_output[t] for t in committed_steps)
                    >= minimum_commit_output_t
                )

        @container.Constraint(time_steps)
        def total_power_input_constraint(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.total_power_input[t] == sum(
                block.power_in[t]
                for block in container.technology_blocks.values()
                if hasattr(block, "power_in")
            )

        @container.Constraint(time_steps)
        def variable_cost_constraint(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.variable_cost[t] == sum(
                block.operating_cost[t]
                for block in container.technology_blocks.values()
                if hasattr(block, "operating_cost")
            )

    def _extract_results(
        self,
        model: pyo.ConcreteModel,
        forecasts: pd.DataFrame,
        solver_name: str,
    ) -> pd.DataFrame:
        market_model = hasattr(model, "actual")
        trajectory = model.actual if market_model else model
        blocks = trajectory.technology_blocks
        dri = blocks["dri_plant"] if "dri_plant" in self.components else None
        eaf = blocks["eaf"] if "eaf" in self.components else None
        bof = blocks["bof"] if "bof" in self.components else None
        bf_bof = blocks["bf_bof"] if "bf_bof" in self.components else None
        terminal = blocks[self._terminal_technology_name()]

        def block_value(block: pyo.Block | None, variable: str, t: int) -> float:
            if block is None or not hasattr(block, variable):
                return 0.0
            return _value(getattr(block, variable)[t])

        def block_status(block: pyo.Block | None, t: int) -> int:
            if block is None:
                return 0
            return _operational_status(block, t)

        data: dict[str, list[float] | list[str]] = {
            "plant_name": [],
            "plant_type": [],
            "steel_route": [],
            "steel_demand_mode": [],
            "steel_demand_total_t": [],
            "total_electricity_consumption_MWh": [],
            "variable_cost_EUR": [],
            "coal_consumption_MWh": [],
            "co2_emissions_t": [],
            "dri_electricity_consumption_MWh": [],
            "dri_output_t": [],
            "dri_coal_consumption_MWh": [],
            "hydrogen_consumption_MWh": [],
            "natural_gas_consumption_MWh": [],
            "iron_ore_consumption_t": [],
            "dri_co2_emissions_t": [],
            "dri_operational_status": [],
            "eaf_electricity_consumption_MWh": [],
            "dri_input_t": [],
            "steel_output_t": [],
            "lime_consumption_t": [],
            "eaf_co2_emissions_t": [],
            "eaf_operational_status": [],
            "bof_electricity_consumption_MWh": [],
            "bof_dri_input_t": [],
            "bof_steel_output_t": [],
            "bof_lime_consumption_t": [],
            "bof_co2_emissions_t": [],
            "bof_operational_status": [],
            "bf_bof_electricity_consumption_MWh": [],
            "bf_bof_steel_output_t": [],
            "bf_bof_coal_consumption_MWh": [],
            "bf_bof_natural_gas_consumption_MWh": [],
            "bf_bof_hydrogen_consumption_MWh": [],
            "bf_bof_iron_ore_consumption_t": [],
            "bf_bof_lime_consumption_t": [],
            "bf_bof_co2_emissions_t": [],
            "bf_bof_operational_status": [],
            "solver": [],
        }
        optional_variables = {
            "electrolyser_electricity_consumption_MWh": ("electrolyser", "power_in"),
            "electrolyser_hydrogen_output_MWh": ("electrolyser", "hydrogen_out"),
            "electrolyser_operational_status": ("electrolyser", "operational_status"),
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
        if market_model:
            for column in _STEEL_AFRR_RESULT_COLUMNS:
                data[column] = []

        for t in model.T:
            data["plant_name"].append(self.name)
            data["plant_type"].append(self.unit_type)
            data["steel_route"].append(self.steel_route)
            data["steel_demand_mode"].append(
                "forecast_profile"
                if int(pyo.value(model.steel_demand_from_forecast))
                else "total_target"
            )
            data["steel_demand_total_t"].append(_value(model.steel_demand))
            total_electricity = _value(trajectory.total_power_input[t])
            data["total_electricity_consumption_MWh"].append(total_electricity)
            data["variable_cost_EUR"].append(
                _value(model.gross_operating_cost[t])
                if market_model
                else _value(trajectory.variable_cost[t])
            )
            dri_coal = block_value(dri, "coal_in", t)
            dri_hydrogen = block_value(dri, "hydrogen_in", t)
            dri_natural_gas = block_value(dri, "natural_gas_in", t)
            dri_iron_ore = block_value(dri, "iron_ore_in", t)
            dri_co2 = block_value(dri, "co2_emission", t)
            bf_coal = block_value(bf_bof, "coal_in", t)
            bf_hydrogen = block_value(bf_bof, "hydrogen_in", t)
            bf_natural_gas = block_value(bf_bof, "natural_gas_in", t)
            bf_iron_ore = block_value(bf_bof, "iron_ore_in", t)
            bf_lime = block_value(bf_bof, "lime_demand", t)
            bf_co2 = block_value(bf_bof, "co2_emission", t)
            terminal_lime = block_value(terminal, "lime_demand", t)
            terminal_co2 = block_value(terminal, "co2_emission", t)
            data["coal_consumption_MWh"].append(dri_coal + bf_coal)
            data["co2_emissions_t"].append(dri_co2 + terminal_co2)
            data["dri_electricity_consumption_MWh"].append(block_value(dri, "power_in", t))
            data["dri_output_t"].append(block_value(dri, "dri_output", t))
            data["dri_coal_consumption_MWh"].append(dri_coal)
            data["hydrogen_consumption_MWh"].append(dri_hydrogen + bf_hydrogen)
            data["natural_gas_consumption_MWh"].append(dri_natural_gas + bf_natural_gas)
            data["iron_ore_consumption_t"].append(dri_iron_ore + bf_iron_ore)
            data["dri_co2_emissions_t"].append(dri_co2)
            data["dri_operational_status"].append(block_status(dri, t))
            data["eaf_electricity_consumption_MWh"].append(block_value(eaf, "power_in", t))
            data["dri_input_t"].append(block_value(terminal, "dri_input", t))
            data["steel_output_t"].append(block_value(terminal, "steel_output", t))
            data["lime_consumption_t"].append(terminal_lime)
            data["eaf_co2_emissions_t"].append(block_value(eaf, "co2_emission", t))
            data["eaf_operational_status"].append(block_status(eaf, t))
            data["bof_electricity_consumption_MWh"].append(block_value(bof, "power_in", t))
            data["bof_dri_input_t"].append(block_value(bof, "dri_input", t))
            data["bof_steel_output_t"].append(block_value(bof, "steel_output", t))
            data["bof_lime_consumption_t"].append(block_value(bof, "lime_demand", t))
            data["bof_co2_emissions_t"].append(block_value(bof, "co2_emission", t))
            data["bof_operational_status"].append(block_status(bof, t))
            data["bf_bof_electricity_consumption_MWh"].append(block_value(bf_bof, "power_in", t))
            data["bf_bof_steel_output_t"].append(block_value(bf_bof, "steel_output", t))
            data["bf_bof_coal_consumption_MWh"].append(bf_coal)
            data["bf_bof_natural_gas_consumption_MWh"].append(bf_natural_gas)
            data["bf_bof_hydrogen_consumption_MWh"].append(bf_hydrogen)
            data["bf_bof_iron_ore_consumption_t"].append(bf_iron_ore)
            data["bf_bof_lime_consumption_t"].append(bf_lime)
            data["bf_bof_co2_emissions_t"].append(bf_co2)
            data["bf_bof_operational_status"].append(block_status(bf_bof, t))
            data["solver"].append(solver_name)
            for column, (technology, variable) in optional_variables.items():
                if column in data:
                    block = trajectory.technology_blocks[technology]
                    if variable == "operational_status":
                        data[column].append(_operational_status(block, t))
                    else:
                        data[column].append(_value(getattr(block, variable)[t]))
            if market_model:
                _append_steel_afrr_result_row(data, model, t, total_electricity)
        return pd.DataFrame(data, index=forecasts.index)

    def _initial_rolling_state(self) -> SteelRollingState:
        components: dict[str, SteelComponentState] = {}
        inventories: dict[str, SteelInventoryState] = {}
        for technology, component in self.components.items():
            if hasattr(component, "max_power_mw"):
                initial_status = int(getattr(component, "initial_operational_status", 0))
                satisfied_steps = max(
                    int(getattr(component, "min_operating_steps", 0)),
                    int(getattr(component, "min_down_steps", 0)),
                    1,
                )
                components[technology] = SteelComponentState(
                    power_in_mwh=0.0,
                    operational_status=initial_status,
                    consecutive_status_steps=satisfied_steps,
                )
            if isinstance(component, GenericInventoryStorage):
                inventories[technology] = SteelInventoryState(soc=component.initial_soc)
        return SteelRollingState(components=components, inventories=inventories)

    def _rolling_demand_schedule(self, forecasts: pd.DataFrame) -> tuple[pd.Series, str]:
        if self.steel_demand_tonnes is None:
            return self._validated_demand_profile(forecasts), "forecast_profile"

        total = float(self.steel_demand_tonnes)
        schedule = pd.Series(total / len(forecasts), index=forecasts.index, dtype=float)
        if len(schedule):
            schedule.iloc[-1] += total - float(schedule.sum())
        return schedule, "total_target"

    def _state_after_commit(
        self,
        previous: SteelRollingState,
        committed: pd.DataFrame,
        scheduled_demand_t: float,
    ) -> SteelRollingState:
        power_columns = {
            "electrolyser": "electrolyser_electricity_consumption_MWh",
            "dri_plant": "dri_electricity_consumption_MWh",
            "eaf": "eaf_electricity_consumption_MWh",
            "bof": "bof_electricity_consumption_MWh",
            "bf_bof": "bf_bof_electricity_consumption_MWh",
        }
        status_columns = {
            "electrolyser": "electrolyser_operational_status",
            "dri_plant": "dri_operational_status",
            "eaf": "eaf_operational_status",
            "bof": "bof_operational_status",
            "bf_bof": "bf_bof_operational_status",
        }
        component_states: dict[str, SteelComponentState] = {}
        for technology, old_state in previous.components.items():
            power_column = power_columns[technology]
            status_column = status_columns[technology]
            statuses = committed[status_column].round().astype(int).tolist()
            final_status = statuses[-1]
            consecutive_steps = old_state.consecutive_status_steps
            current_status = old_state.operational_status
            for status in statuses:
                if status == current_status:
                    consecutive_steps += 1
                else:
                    current_status = status
                    consecutive_steps = 1
            component_states[technology] = SteelComponentState(
                power_in_mwh=float(committed[power_column].iloc[-1]),
                operational_status=final_status,
                consecutive_status_steps=consecutive_steps,
            )

        inventory_columns = {
            "hydrogen_buffer_storage": (
                "hydrogen_storage_soc",
                "hydrogen_storage_charge_MWh",
                "hydrogen_storage_discharge_MWh",
            ),
            "dri_storage": (
                "dri_storage_soc",
                "dri_storage_charge_t",
                "dri_storage_discharge_t",
            ),
        }
        inventory_states: dict[str, SteelInventoryState] = {}
        for technology in previous.inventories:
            soc_column, charge_column, discharge_column = inventory_columns[technology]
            inventory_states[technology] = SteelInventoryState(
                soc=float(committed[soc_column].iloc[-1]),
                charge=float(committed[charge_column].iloc[-1]),
                discharge=float(committed[discharge_column].iloc[-1]),
            )

        committed_output = float(committed["steel_output_t"].sum())
        balance = previous.demand_balance_t + scheduled_demand_t - committed_output
        if abs(balance) < 1e-9:
            balance = 0.0
        return SteelRollingState(
            components=component_states,
            inventories=inventory_states,
            cumulative_steel_output_t=(previous.cumulative_steel_output_t + committed_output),
            demand_balance_t=balance,
        )

    def _terminal_technology_name(self) -> str:
        if self.steel_route == "bf_bof":
            return "bf_bof"
        if self.steel_route == "dri_bof":
            return "bof"
        return "eaf"

    def _requires_coal_price(self) -> bool:
        return any(
            getattr(component, "fuel_type", "") == COAL for component in self.components.values()
        )

    def _validate_signal_columns(
        self,
        forecasts: pd.DataFrame,
        signals: SteelDispatchSignals,
    ) -> None:
        columns = {
            signals.electricity_price_col,
            signals.natural_gas_price_col,
            signals.hydrogen_price_col,
            signals.iron_ore_price_col,
            signals.lime_price_col,
            signals.co2_price_col,
        }
        if self._requires_coal_price():
            columns.add(signals.coal_price_col)
        if signals.steel_price_col:
            columns.add(signals.steel_price_col)
        missing = columns - set(forecasts.columns)
        if missing:
            raise ValueError(
                "Steel dispatch forecasts are missing column(s): " + ", ".join(sorted(missing))
            )

    def _validate_electrified_steel(self) -> None:
        if self.steel_route != "dri_eaf":
            raise ValueError(
                f"Electrified-steel strategy requires plant '{self.name}' route to be DRI + EAF"
            )
        if self.dri_plant.fuel_type != "hydrogen":
            raise ValueError(
                f"Electrified-steel strategy requires plant '{self.name}' DRI fuel_type "
                "to be 'hydrogen'"
            )
        if "electrolyser" not in self.components:
            raise ValueError(
                f"Electrified-steel strategy requires plant '{self.name}' to define an "
                "electrolyser; purchased hydrogen is not a fallback"
            )

    @staticmethod
    def _validate_afrr_signal_columns(
        forecasts: pd.DataFrame,
        signals: SteelAFRRDownSignals,
    ) -> None:
        columns = {
            signals.da_price_col,
            signals.afrr_energy_price_col,
            signals.afrr_system_activation_col,
            signals.afrr_price_available_col,
            signals.afrr_capacity_block_id_col,
            signals.afrr_capacity_block_duration_col,
            signals.afrr_capacity_price_col,
            signals.afrr_capacity_missing_price_col,
            signals.iron_ore_price_col,
            signals.lime_price_col,
            signals.co2_price_col,
        }
        missing = columns - set(forecasts.columns)
        if missing:
            raise ValueError(
                "Electrified-steel forecasts are missing column(s): " + ", ".join(sorted(missing))
            )

    def _resolve_steel_demand(self, forecasts: pd.DataFrame) -> tuple[float, str]:
        if self.steel_demand_tonnes is not None:
            return self.steel_demand_tonnes, "total_target"

        demand = self._validated_demand_profile(forecasts)
        return float(demand.sum()), "forecast_profile"

    def _validated_demand_profile(self, forecasts: pd.DataFrame) -> pd.Series:
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
        return demand.astype(float)


_STEEL_AFRR_RESULT_COLUMNS = (
    "DA_position_MWh",
    "IDC_buy_MWh",
    "IDC_sell_MWh",
    "final_planned_electricity_MWh",
    "actual_electricity_consumption_MWh",
    "day_ahead_price_EUR_per_MWh",
    "day_ahead_delivered_price_EUR_per_MWh",
    "additional_electricity_charge_EUR_per_MWh_el",
    "afrr_energy_bid_MW",
    "afrr_energy_bid_MWh",
    "afrr_energy_activated_MWh",
    "afrr_energy_price_EUR_per_MWh",
    "afrr_energy_delivered_price_EUR_per_MWh",
    "afrr_energy_bid_price_EUR_per_MWh",
    "afrr_energy_market_spread_EUR_per_MWh",
    "afrr_energy_net_spread_EUR_per_MWh",
    "afrr_energy_cost_EUR",
    "afrr_energy_savings_vs_benchmark_EUR",
    "afrr_energy_pay_as_cleared_reward_EUR",
    "afrr_energy_net_value_after_charges_EUR",
    "afrr_energy_capacity_backed_bid_MWh",
    "afrr_energy_free_bid_MWh",
    "afrr_energy_capacity_backed_activated_MWh",
    "afrr_energy_free_activated_MWh",
    "afrr_system_activation_MWh",
    "afrr_headroom_binding",
    "afrr_curtailment_MWh",
    "afrr_capacity_block_id",
    "afrr_capacity_block_duration_h",
    "afrr_capacity_pricing_rule",
    "afrr_capacity_bid_price_EUR_per_MW_h",
    "afrr_capacity_clearing_price_EUR_per_MW_h",
    "afrr_capacity_settlement_price_EUR_per_MW_h",
    "afrr_capacity_down_price_EUR_per_MW_h",
    "afrr_capacity_reserved_MW",
    "afrr_capacity_reserved_MWh",
    "afrr_capacity_revenue_EUR",
    "afrr_capacity_opportunity_cost_EUR",
    "afrr_capacity_market_surplus_EUR",
    "afrr_capacity_net_value_EUR",
    "reserved_capacity_headroom_MWh",
    "available_load_headroom_after_schedule_MWh",
    "DA_electricity_cost_EUR",
    "electricity_market_cost_EUR",
    "additional_electricity_charges_cost_EUR",
    "gross_operating_cost_EUR",
    "net_operating_cost_EUR",
    "operating_cost_EUR",
    "non_electric_variable_cost_EUR",
)


def _append_steel_afrr_result_row(
    data: dict[str, list[float] | list[str]],
    model: pyo.ConcreteModel,
    t: int,
    total_electricity_mwh: float,
) -> None:
    block_id = model._afrr_block_by_t[t]
    timestep_hours = float(model._afrr_timestep_hours)
    da_position = _value(model.da_position_mwh[t])
    afrr_bid = _value(model.afrr_energy_bid_mwh[t])
    afrr_activation = _value(model.afrr_energy_activated_mwh[t])
    capacity_bid = _value(model.afrr_energy_capacity_backed_bid_mwh[t])
    free_bid = _value(model.afrr_energy_free_bid_mwh[t])
    capacity_activated = _value(model.afrr_energy_capacity_backed_activated_mwh[t])
    free_activated = _value(model.afrr_energy_free_activated_mwh[t])
    day_ahead_price = _value(model.day_ahead_price[t])
    afrr_price = _value(model.afrr_energy_price[t])
    additional_charge = _value(model.additional_electricity_charge)
    afrr_bid_price = _value(model.afrr_energy_bid_price[t])
    delivered_afrr_price = afrr_price + additional_charge
    market_spread = afrr_bid_price - afrr_price
    net_spread = afrr_bid_price - delivered_afrr_price
    capacity_price = _value(model.afrr_capacity_price[block_id])
    reserved_mw = _value(model.afrr_capacity_reserved_mw[block_id])
    reserved_mwh = reserved_mw * timestep_hours
    capacity_revenue = _value(model.afrr_capacity_revenue[t])
    gross_cost = _value(model.gross_operating_cost[t])
    non_electric_cost = _value(model.actual.variable_cost[t])
    available_load_headroom = max(
        0.0,
        float(model._afrr_aggregate_max_power_mw) * timestep_hours - da_position,
    )

    values: dict[str, float | str | bool] = {
        "DA_position_MWh": da_position,
        "IDC_buy_MWh": 0.0,
        "IDC_sell_MWh": 0.0,
        "final_planned_electricity_MWh": da_position,
        "actual_electricity_consumption_MWh": total_electricity_mwh,
        "day_ahead_price_EUR_per_MWh": day_ahead_price,
        "day_ahead_delivered_price_EUR_per_MWh": day_ahead_price + additional_charge,
        "additional_electricity_charge_EUR_per_MWh_el": additional_charge,
        "afrr_energy_bid_MW": afrr_bid / timestep_hours,
        "afrr_energy_bid_MWh": afrr_bid,
        "afrr_energy_activated_MWh": afrr_activation,
        "afrr_energy_price_EUR_per_MWh": afrr_price,
        "afrr_energy_delivered_price_EUR_per_MWh": delivered_afrr_price,
        "afrr_energy_bid_price_EUR_per_MWh": afrr_bid_price,
        "afrr_energy_market_spread_EUR_per_MWh": market_spread,
        "afrr_energy_net_spread_EUR_per_MWh": net_spread,
        "afrr_energy_cost_EUR": afrr_activation * afrr_price,
        "afrr_energy_savings_vs_benchmark_EUR": afrr_activation * market_spread,
        "afrr_energy_pay_as_cleared_reward_EUR": afrr_activation * market_spread,
        "afrr_energy_net_value_after_charges_EUR": afrr_activation * net_spread,
        "afrr_energy_capacity_backed_bid_MWh": capacity_bid,
        "afrr_energy_free_bid_MWh": free_bid,
        "afrr_energy_capacity_backed_activated_MWh": capacity_activated,
        "afrr_energy_free_activated_MWh": free_activated,
        "afrr_system_activation_MWh": _value(model.afrr_system_activation_mwh[t]),
        "afrr_headroom_binding": bool(
            afrr_bid > 1e-9 and abs(afrr_bid - available_load_headroom) <= 1e-7
        ),
        "afrr_curtailment_MWh": 0.0,
        "afrr_capacity_block_id": block_id,
        "afrr_capacity_block_duration_h": _value(model.afrr_capacity_block_duration[block_id]),
        "afrr_capacity_pricing_rule": "pay_as_bid",
        "afrr_capacity_bid_price_EUR_per_MW_h": capacity_price,
        "afrr_capacity_clearing_price_EUR_per_MW_h": capacity_price,
        "afrr_capacity_settlement_price_EUR_per_MW_h": capacity_price,
        "afrr_capacity_down_price_EUR_per_MW_h": capacity_price,
        "afrr_capacity_reserved_MW": reserved_mw,
        "afrr_capacity_reserved_MWh": reserved_mwh,
        "afrr_capacity_revenue_EUR": capacity_revenue,
        "afrr_capacity_opportunity_cost_EUR": 0.0,
        "afrr_capacity_market_surplus_EUR": 0.0,
        "afrr_capacity_net_value_EUR": capacity_revenue,
        "reserved_capacity_headroom_MWh": reserved_mwh,
        "available_load_headroom_after_schedule_MWh": available_load_headroom,
        "DA_electricity_cost_EUR": da_position * day_ahead_price,
        "electricity_market_cost_EUR": _value(model.electricity_market_cost[t]),
        "additional_electricity_charges_cost_EUR": _value(
            model.additional_electricity_charges_cost[t]
        ),
        "gross_operating_cost_EUR": gross_cost,
        "net_operating_cost_EUR": _value(model.net_operating_cost[t]),
        "operating_cost_EUR": gross_cost,
        "non_electric_variable_cost_EUR": non_electric_cost,
    }
    for column, value in values.items():
        data[column].append(value)


def _value(expression: Any) -> float:
    return float(pyo.value(expression))


def _operational_status(block: pyo.Block, t: int) -> int:
    if hasattr(block, "operational_status"):
        return int(round(_value(block.operational_status[t])))
    return int(_value(block.power_in[t]) > 1e-9)


def _validate_bid_rules(market_name: str, min_bid_mw: float, bid_increment_mw: float) -> None:
    if min_bid_mw < 0:
        raise ValueError(f"{market_name}.product_rules.min_bid_mw must be non-negative")
    if bid_increment_mw <= 0:
        raise ValueError(f"{market_name}.product_rules.bid_increment_mw must be positive")


def _attach_capacity_opportunity_cost(
    market_result: pd.DataFrame,
    no_capacity_result: pd.DataFrame,
    commit_steps: int,
) -> pd.DataFrame:
    """Allocate cross-market capacity opportunity cost over reserved MW-hours."""

    result = market_result.copy()
    committed_result = result.iloc[:commit_steps]
    committed_baseline = no_capacity_result.iloc[:commit_steps]
    incremental_gross_cost = max(
        0.0,
        float(committed_result["gross_operating_cost_EUR"].sum())
        - float(committed_baseline["gross_operating_cost_EUR"].sum()),
    )
    weights = committed_result["afrr_capacity_reserved_MWh"].clip(lower=0.0)
    total_weight = float(weights.sum())
    result["afrr_capacity_opportunity_cost_EUR"] = 0.0
    if total_weight > 1e-12:
        result.loc[weights.index, "afrr_capacity_opportunity_cost_EUR"] = (
            incremental_gross_cost * weights / total_weight
        )
    result["afrr_capacity_net_value_EUR"] = (
        result["afrr_capacity_revenue_EUR"] - result["afrr_capacity_opportunity_cost_EUR"]
    )
    return result


def _validate_rolling_window(
    timestep_hours: float,
    horizon_hours: float,
    step_hours: float,
) -> None:
    if horizon_hours <= 0 or step_hours <= 0:
        raise ValueError("Steel rolling horizon and step hours must be positive")
    if step_hours > horizon_hours:
        raise ValueError("rolling_step_hours must not exceed dispatch_horizon_hours")
    for label, hours in {
        "dispatch_horizon_hours": horizon_hours,
        "rolling_step_hours": step_hours,
    }.items():
        steps = hours / timestep_hours
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"{label} must align with case.timestep_minutes")


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


def _detect_steel_route(components: dict[str, object], plant_name: str) -> str:
    has_dri = "dri_plant" in components
    has_eaf = "eaf" in components
    has_bof = "bof" in components
    has_bf_bof = "bf_bof" in components

    if "hydrogen_buffer_storage" in components and "electrolyser" not in components:
        raise ValueError(
            f"Steel plant '{plant_name}' defines hydrogen_buffer_storage without an electrolyser"
        )
    if "dri_storage" in components and not has_dri:
        raise ValueError(f"Steel plant '{plant_name}' defines dri_storage without a DRI plant")

    if has_bf_bof:
        conflicting = sorted(
            technology
            for technology, present in {
                "dri_plant": has_dri,
                "eaf": has_eaf,
                "bof": has_bof,
            }.items()
            if present
        )
        if conflicting:
            raise ValueError(
                f"Steel plant '{plant_name}' has ambiguous route: bf_bof cannot be combined "
                f"with {', '.join(conflicting)}"
            )
        return "bf_bof"

    if has_eaf and has_bof:
        raise ValueError(
            f"Steel plant '{plant_name}' has ambiguous route: choose either eaf or bof, not both"
        )
    if has_eaf:
        if not has_dri:
            raise ValueError(
                f"Steel plant '{plant_name}' route dri_eaf is missing required technology "
                "'dri_plant'"
            )
        return "dri_eaf"
    if has_bof:
        if not has_dri:
            raise ValueError(
                f"Steel plant '{plant_name}' route dri_bof is missing required technology "
                "'dri_plant'"
            )
        return "dri_bof"
    if has_dri:
        raise ValueError(
            f"Steel plant '{plant_name}' is missing required terminal technology 'eaf' or 'bof'"
        )
    raise ValueError(
        f"Steel plant '{plant_name}' must define exactly one route: "
        "dri_plant + eaf, dri_plant + bof, or bf_bof"
    )
