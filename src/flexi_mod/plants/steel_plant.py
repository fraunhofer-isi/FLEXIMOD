# SPDX-FileCopyrightText: FLEXIMOD Developers
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyomo.environ as pyo

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_energy import (
    AFRR_DOWN_RESULT_COLUMNS,
    AFRRDownMarketSignals,
    AFRRWindow,
    FuelSubstitution,
    TrajectoryRole,
    append_afrr_result_row,
    attach_capacity_opportunity_cost,
    build_afrr_down_market_model,
    validate_bid_rules,
)
from flexi_mod.plants.dispatch_plant import DispatchPlant
from flexi_mod.plants.rolling import (
    ROLLING_DEMAND_TOLERANCE_T,
    solve_dispatch_model,
    validate_rolling_window,
)
from flexi_mod.plants.technologies import (
    COAL,
    HYBRID_HYDROGEN_NATURAL_GAS,
    TECHNOLOGY_REGISTRY,
    BasicOxygenFurnace,
    BlastFurnaceBasicOxygenFurnace,
    DRIPlant,
    ElectricArcFurnace,
    GenericInventoryStorage,
    first_non_empty,
)

FINAL_WINDOW_RECONCILIATION_T = 1e-7


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
    electrolyser_allowed_col: str | None = None


@dataclass(frozen=True)
class SteelWindowPlan:
    """What one rolling window owes, carried opaquely through the shared market layer."""

    initial_state: SteelRollingState
    steel_demand_override_t: float
    minimum_commit_output_t: float
    demand_mode: str


@dataclass(frozen=True)
class SteelAFRRDownCommoditySignals:
    """Forecast columns for what a steel plant consumes, as opposed to what it buys power on.

    Iron ore and lime are steel's alone; the fuels and CO2 happen to be shared with other
    families but are still the plant's business, because only the plant knows which of
    them it actually prices (hydrogen is free when made on site, coal only matters on a
    coal route).
    """

    iron_ore_price_col: str
    lime_price_col: str
    co2_price_col: str
    natural_gas_price_col: str
    hydrogen_price_col: str
    coal_price_col: str = "coal_price"


@dataclass(frozen=True)
class SteelAFRRDownSignals:
    """Ontology-aligned market and commodity inputs for electrified steel.

    Kept flat for construction, but exposes the two halves separately: the market layer
    is shared across plant families and must never see ``iron_ore_price_col``, while the
    commodity half is handed straight back to the plant that owns it.
    """

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
    natural_gas_price_col: str
    hydrogen_price_col: str
    additional_electricity_charge_eur_per_mwh: float = 0.0
    afrr_energy_min_bid_mw: float = 1.0
    afrr_energy_bid_increment_mw: float = 1.0
    afrr_capacity_min_bid_mw: float = 1.0
    afrr_capacity_bid_increment_mw: float = 1.0
    afrr_capacity_product_duration_h: float = 4.0
    coal_price_col: str = "coal_price"

    @property
    def market(self) -> AFRRDownMarketSignals:
        """The half the shared market layer consumes."""
        return AFRRDownMarketSignals(
            da_price_col=self.da_price_col,
            afrr_energy_price_col=self.afrr_energy_price_col,
            afrr_system_activation_col=self.afrr_system_activation_col,
            afrr_price_available_col=self.afrr_price_available_col,
            afrr_capacity_block_id_col=self.afrr_capacity_block_id_col,
            afrr_capacity_block_duration_col=self.afrr_capacity_block_duration_col,
            afrr_capacity_price_col=self.afrr_capacity_price_col,
            afrr_capacity_missing_price_col=self.afrr_capacity_missing_price_col,
            additional_electricity_charge_eur_per_mwh=(
                self.additional_electricity_charge_eur_per_mwh
            ),
            afrr_energy_min_bid_mw=self.afrr_energy_min_bid_mw,
            afrr_energy_bid_increment_mw=self.afrr_energy_bid_increment_mw,
            afrr_capacity_min_bid_mw=self.afrr_capacity_min_bid_mw,
            afrr_capacity_bid_increment_mw=self.afrr_capacity_bid_increment_mw,
            afrr_capacity_product_duration_h=self.afrr_capacity_product_duration_h,
        )

    @property
    def commodities(self) -> SteelAFRRDownCommoditySignals:
        """The half only the steel plant consumes."""
        return SteelAFRRDownCommoditySignals(
            iron_ore_price_col=self.iron_ore_price_col,
            lime_price_col=self.lime_price_col,
            co2_price_col=self.co2_price_col,
            natural_gas_price_col=self.natural_gas_price_col,
            hydrogen_price_col=self.hydrogen_price_col,
            coal_price_col=self.coal_price_col,
        )


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
class SteelPlant(DispatchPlant):
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
        progress_callback: Callable[[pd.Timestamp, pd.Timestamp], None] | None = None,
    ) -> pd.DataFrame:
        if forecasts.empty:
            raise ValueError("Steel rolling dispatch requires at least one timestep")
        self._validate_signal_columns(forecasts, signals)
        dt_hours = config.timestep_minutes / 60.0
        horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 48))
        step_hours = float(config.dispatch_setting("rolling_step_hours", 24))
        validate_rolling_window(dt_hours, horizon_hours, step_hours, plant_label="Steel plant")
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
            if progress_callback is not None:
                progress_callback(
                    pd.Timestamp(horizon.index[0]),
                    pd.Timestamp(horizon.index[commit_count - 1]),
                )
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
                if is_final_window and remaining_demand > FINAL_WINDOW_RECONCILIATION_T:
                    reconciled_target = remaining_demand - FINAL_WINDOW_RECONCILIATION_T
                    reconciled_model = self._build_model(
                        config,
                        horizon,
                        signals,
                        initial_state=state,
                        steel_demand_override_t=reconciled_target,
                        commit_steps=commit_count,
                        minimum_commit_output_t=reconciled_target,
                        demand_mode=demand_mode,
                    )
                    try:
                        horizon_result = self._solve_model(
                            config,
                            horizon,
                            reconciled_model,
                        )
                    except RuntimeError:
                        raise RuntimeError(
                            f"Steel rolling window {window_number} starting "
                            f"{horizon.index[0]} is infeasible: target={horizon_target:g} t, "
                            f"inherited backlog={max(0.0, state.demand_balance_t):g} t"
                        ) from exc
                    horizon_target = reconciled_target
                else:
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
        if abs(state.cumulative_steel_output_t - total_demand) > ROLLING_DEMAND_TOLERANCE_T:
            raise RuntimeError(
                "Steel rolling dispatch ended without satisfying total demand: "
                f"produced={state.cumulative_steel_output_t:g} t, target={total_demand:g} t"
            )
        if abs(state.demand_balance_t) > ROLLING_DEMAND_TOLERANCE_T:
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
        progress_callback: Callable[[pd.Timestamp, pd.Timestamp], None] | None = None,
    ) -> pd.DataFrame:
        """Optimize the next-day DA and aFRR-down portfolio on a rolling horizon."""

        if forecasts.empty:
            raise ValueError("Electrified-steel rolling dispatch requires at least one timestep")
        self._validate_afrr_signal_columns(forecasts, signals)
        validate_bid_rules(
            "afrr_energy",
            signals.afrr_energy_min_bid_mw,
            signals.afrr_energy_bid_increment_mw,
        )
        validate_bid_rules(
            "afrr_capacity",
            signals.afrr_capacity_min_bid_mw,
            signals.afrr_capacity_bid_increment_mw,
        )

        dt_hours = config.timestep_minutes / 60.0
        horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 48))
        step_hours = float(config.dispatch_setting("rolling_step_hours", 24))
        validate_rolling_window(dt_hours, horizon_hours, step_hours, plant_label="Steel plant")
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
            if progress_callback is not None:
                progress_callback(
                    pd.Timestamp(horizon.index[0]),
                    pd.Timestamp(horizon.index[commit_count - 1]),
                )
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

            horizon_result = attach_capacity_opportunity_cost(
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
        if abs(state.cumulative_steel_output_t - total_demand) > ROLLING_DEMAND_TOLERANCE_T:
            raise RuntimeError(
                "Electrified-steel dispatch ended without satisfying total demand: "
                f"produced={state.cumulative_steel_output_t:g} t, target={total_demand:g} t"
            )
        if abs(state.demand_balance_t) > ROLLING_DEMAND_TOLERANCE_T:
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
        """Hand the shared market layer everything it needs to wrap this plant."""
        window = AFRRWindow(
            horizon=forecasts,
            commit_steps=commit_steps,
            is_final_window=terminal_state_required,
            payload=SteelWindowPlan(
                initial_state=initial_state,
                steel_demand_override_t=steel_demand_override_t,
                minimum_commit_output_t=minimum_commit_output_t,
                demand_mode=demand_mode,
            ),
        )
        return build_afrr_down_market_model(
            self,
            forecasts,
            signals.market,
            signals.commodities,
            window,
            dt_hours=config.timestep_minutes / 60.0,
            capacity_enabled=capacity_enabled,
        )

    # -- AFRRDownPlant protocol ---------------------------------------------------------

    def afrr_attach_commodity_params(
        self,
        model: pyo.ConcreteModel,
        forecasts: pd.DataFrame,
        commodity_signals: SteelAFRRDownCommoditySignals,
        window: AFRRWindow,
    ) -> None:
        plan: SteelWindowPlan = window.payload

        def values(column: str) -> dict[int, float]:
            return {t: float(forecasts[column].iloc[t]) for t in model.T}

        # Electricity is settled once at plant level by the market layer. The component
        # electricity price is therefore zero to avoid double counting in technology
        # operating costs. Other fuels (coal, natural gas, purchased hydrogen) are priced
        # like the cost-minimization model: real market price unless produced on-site.
        model.electricity_price = pyo.Param(model.T, initialize={t: 0.0 for t in model.T})
        model.natural_gas_price = pyo.Param(
            model.T, initialize=values(commodity_signals.natural_gas_price_col)
        )
        hydrogen_prices = (
            {t: 0.0 for t in model.T}
            if "electrolyser" in self.components
            else values(commodity_signals.hydrogen_price_col)
        )
        model.hydrogen_price = pyo.Param(model.T, initialize=hydrogen_prices)
        coal_prices = (
            values(commodity_signals.coal_price_col)
            if self._requires_coal_price()
            else {t: 0.0 for t in model.T}
        )
        model.coal_price = pyo.Param(model.T, initialize=coal_prices)
        model.iron_ore_price = pyo.Param(
            model.T, initialize=values(commodity_signals.iron_ore_price_col)
        )
        model.lime_price = pyo.Param(model.T, initialize=values(commodity_signals.lime_price_col))
        model.co2_price = pyo.Param(model.T, initialize=values(commodity_signals.co2_price_col))
        model.steel_demand = pyo.Param(initialize=float(plan.steel_demand_override_t))
        model.steel_demand_from_forecast = pyo.Param(
            initialize=int(plan.demand_mode == "forecast_profile"), within=pyo.Binary
        )

    def afrr_attach_trajectory(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block,
        role: TrajectoryRole,
        window: AFRRWindow,
        dt_hours: float,
    ) -> None:
        plan: SteelWindowPlan = window.payload
        # Steel's output constraint is an equality on the window total, which a different
        # power draw would fight, so the hypothetical branch must be freed from it.
        self._add_physical_system(
            model,
            container,
            model.T,
            plan.initial_state,
            dt_hours,
            window.commit_steps,
            plan.minimum_commit_output_t,
            enforce_output_total=(role == "actual"),
        )

    def afrr_attach_terminal_state(self, model: pyo.ConcreteModel, final_t: int) -> None:
        inventory_names = [
            name
            for name, component in self.components.items()
            if isinstance(component, GenericInventoryStorage)
        ]

        @model.Constraint(inventory_names)
        def actual_terminal_inventory(m: pyo.ConcreteModel, technology: str) -> pyo.Constraint:
            target = float(self.components[technology].initial_soc)
            return m.actual.technology_blocks[technology].soc[final_t] == target

    def afrr_fuel_substitution(self, forecasts: pd.DataFrame) -> FuelSubstitution | None:
        """Electricity price at which hydrogen via the electrolyser matches gas per tonne.

        Requires both an on-site electrolyser and a hybrid-fuel block that can genuinely
        swap hydrogen for gas; without both there is no choice to price. CO2 is included
        on the gas side because the exact MILP prices it there too. Iron ore and the
        block's own baseline electricity are route-independent and correctly excluded.
        """
        if "electrolyser" not in self.components:
            return None
        hybrid = self.components.get("dri_plant") or self.components.get("bf_bof")
        if hybrid is None or getattr(hybrid, "fuel_type", None) != HYBRID_HYDROGEN_NATURAL_GAS:
            return None

        natural_gas_price = forecasts["natural_gas_price"].astype(float)
        co2_price = forecasts["co2_price"].astype(float)
        gas_route_cost_per_t = float(hybrid.specific_natural_gas_consumption_mwh_per_t) * (
            natural_gas_price
            + float(hybrid.natural_gas_co2_factor_t_per_mwh) * co2_price
        )
        hydrogen_mwh_el_per_t = float(hybrid.specific_hydrogen_consumption_mwh_per_t) / float(
            self.components["electrolyser"].efficiency
        )
        benchmark = gas_route_cost_per_t / hydrogen_mwh_el_per_t
        benchmark.name = "gas_based_electricity_benchmark_EUR_per_MWh_el"
        return FuelSubstitution(
            benchmark_eur_per_mwh_el=benchmark,
            gate_column="__electrolyser_allowed",
            gated_load_column="electrolyser_electricity_consumption_MWh",
        )

    def afrr_aggregate_max_power_mw(self) -> float:
        return sum(
            float(getattr(component, "max_power_mw", 0.0)) for component in self.components.values()
        )

    def _solve_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        model: pyo.ConcreteModel,
    ) -> pd.DataFrame:
        return solve_dispatch_model(
            config,
            forecasts,
            model,
            self._extract_results,
            plant_label="Steel plant",
        )

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

        # What matters to the operator is the delivered cost of a MWh: the market price
        # plus the per-MWh grid charge. Optimising against the bare market price makes
        # electricity look cheaper than it is and over-consumes. Matches the cement
        # plant's convention; the charge is zero unless a grid-fee regulation is attached.
        electricity_market_prices = values(signals.electricity_price_col)
        additional_charge = float(self.additional_electricity_charge_eur_per_mwh)
        model.electricity_market_price = pyo.Param(model.T, initialize=electricity_market_prices)
        model.additional_electricity_charge = pyo.Param(initialize=additional_charge)
        model.electricity_price = pyo.Param(
            model.T,
            initialize={t: electricity_market_prices[t] + additional_charge for t in model.T},
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

        if signals.electrolyser_allowed_col and "electrolyser" in self.components:
            gate_open = {
                t: bool(forecasts[signals.electrolyser_allowed_col].iloc[t]) for t in model.T
            }
            electrolyser_block = model.technology_blocks["electrolyser"]

            @model.Constraint(model.T)
            def electrolyser_gate_constraint(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
                if gate_open[t]:
                    return pyo.Constraint.Skip
                return electrolyser_block.power_in[t] == 0.0

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
        enforce_output_total: bool = True,
    ) -> None:
        """Attach one complete steel-production trajectory to ``container``.

        ``enforce_output_total`` ties ``container``'s cumulative steel output to
        ``model.steel_demand`` and the inherited backlog. It is required for any
        trajectory that becomes the real committed dispatch, but must be dropped for a
        purely hypothetical "what if the full aFRR bid gets activated" trajectory: that
        branch is never realized (only ``model.actual`` feeds the rolling state), so it
        should only be constrained by real per-timestep power/ramp physics, not forced
        to reproduce the same production total under a different power draw.
        """
        self._attach_technology_blocks(model, container, time_steps, dt_hours, initial_state)
        self.initialize_process_sequence(model, container, time_steps)
        self._declare_plant_totals(container, time_steps)
        self.define_constraints(
            model,
            container,
            time_steps,
            commit_steps=commit_steps,
            minimum_commit_output_t=minimum_commit_output_t,
            enforce_output_total=enforce_output_total,
        )
        self._attach_total_power_input_constraint(container, time_steps)
        self._attach_variable_cost_constraint(container, time_steps)

    def _component_context(
        self,
        technology: str,
        dt_hours: float,
        initial_state: SteelRollingState,
    ) -> dict[str, Any]:
        """Build the ``add_to_model`` context that seeds a window from the last commit."""
        context: dict[str, Any] = {"dt_hours": dt_hours}
        component_state = initial_state.components.get(technology)
        if component_state is not None:
            context.update(
                {
                    "initial_power_in": component_state.power_in_mwh,
                    "initial_operational_status": component_state.operational_status,
                    "initial_consecutive_status_steps": (component_state.consecutive_status_steps),
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
        return context

    def initialize_process_sequence(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
    ) -> None:
        """Pyomo Components:

        - **Constraints**: the DRI plant's hydrogen and DRI flow linking to the
          electrolyser and terminal furnace, each present only for the components this
          plant actually has configured.
        """
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

    def define_constraints(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
        *,
        enforce_output_total: bool = True,
        commit_steps: int | None = None,
        minimum_commit_output_t: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Pyomo Components:

        - **Constraints**: ``steel_output_association_constraint`` and
          ``inherited_backlog_recovery_constraint``, both gated by
          ``enforce_output_total`` (dropped for the hypothetical full-activation
          trajectory - see ``_add_physical_system``).
        """
        terminal = container.technology_blocks[self._terminal_technology_name()]

        if enforce_output_total:

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
            for column in AFRR_DOWN_RESULT_COLUMNS:
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
                append_afrr_result_row(data, model, t, total_electricity)
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
        if signals.electrolyser_allowed_col:
            columns.add(signals.electrolyser_allowed_col)
        missing = columns - set(forecasts.columns)
        if missing:
            raise ValueError(
                "Steel dispatch forecasts are missing column(s): " + ", ".join(sorted(missing))
            )

    def _validate_afrr_signal_columns(
        self,
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
            signals.natural_gas_price_col,
            signals.hydrogen_price_col,
        }
        if self._requires_coal_price():
            columns.add(signals.coal_price_col)
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


def _value(expression: Any) -> float:
    return float(pyo.value(expression))


def _operational_status(block: pyo.Block, t: int) -> int:
    if hasattr(block, "operational_status"):
        return int(round(_value(block.operational_status[t])))
    return int(_value(block.power_in[t]) > 1e-9)


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
