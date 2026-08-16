# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd
import pyomo.environ as pyo

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_energy import validate_bid_rules
from flexi_mod.plants.afrr_down import (
    AFRR_DOWN_RESULT_COLUMNS,
    AFRRDownMarketSignals,
    AFRRWindow,
    FuelSubstitution,
    TrajectoryRole,
    append_afrr_result_row,
    attach_capacity_opportunity_cost,
    build_afrr_down_market_model,
)
from flexi_mod.plants.dispatch_plant import DispatchPlant
from flexi_mod.plants.rolling import (
    ROLLING_DEMAND_TOLERANCE_T,
    solve_dispatch_model,
    validate_rolling_window,
)
from flexi_mod.plants.technologies import (
    CEMENT_ELECTRICITY,
    CEMENT_FOSSIL,
    CEMENT_HYBRID_ELECTRICITY_FOSSIL,
    TECHNOLOGY_REGISTRY,
    GenericInventoryStorage,
    first_non_empty,
)

#: Kiln-line stages in flow order. The route name is those present, joined by ``_``.
CEMENT_LINE_STAGES = ("preheater", "calciner", "kiln")


@dataclass(frozen=True)
class CementDispatchSignals:
    """Forecast-column mapping required for cement clinker dispatch."""

    electricity_price_col: str
    natural_gas_price_col: str = "natural_gas_price"
    hydrogen_price_col: str = "hydrogen_price"
    coal_price_col: str = "coal_price"
    co2_price_col: str = "co2_price"


@dataclass(frozen=True)
class CementAFRRDownCommoditySignals:
    """Forecast columns for what a cement plant consumes.

    No iron ore or lime: cement's raw meal cost is not modelled, so this is fuels and
    CO2 only.
    """

    co2_price_col: str
    natural_gas_price_col: str
    hydrogen_price_col: str
    coal_price_col: str = "coal_price"


@dataclass(frozen=True)
class CementAFRRDownSignals:
    """Market and commodity inputs for a cement plant bidding aFRR down."""

    market: AFRRDownMarketSignals
    commodities: CementAFRRDownCommoditySignals


@dataclass(frozen=True)
class CementWindowPlan:
    """What one rolling window carries, opaque to the shared market layer."""

    initial_state: CementRollingState


@dataclass(frozen=True)
class CementStageState:
    """Where a thermal stage was left at the end of the previous commit window.

    Thermal stages commit on their heat output rather than their power draw, which is why
    they need their own state type separate from :class:`CementComponentState`.
    """

    heat_out_mwh: float
    operational_status: int
    consecutive_status_steps: int


@dataclass(frozen=True)
class CementComponentState:
    """End-of-window state of an electrically committed component (the electrolyser)."""

    power_in_mwh: float
    operational_status: int
    consecutive_status_steps: int


@dataclass(frozen=True)
class CementInventoryState:
    """End-of-window fill level of a storage.

    ``soc`` carries whatever unit the owning component uses: a fraction of capacity for
    the hydrogen buffer, absolute MWh for the thermal storage.
    """

    soc: float
    charge: float = 0.0
    discharge: float = 0.0


@dataclass
class CementRollingState:
    stages: dict[str, CementStageState] = field(default_factory=dict)
    components: dict[str, CementComponentState] = field(default_factory=dict)
    inventories: dict[str, CementInventoryState] = field(default_factory=dict)
    cumulative_clinker_output_t: float = 0.0


@dataclass
class CementPlant(DispatchPlant):
    """Cement clinker production model without grinding-mill integration.

    Supported technologies for this first FLEXIMOD cement version are
    ``preheater``, ``calciner``, ``kiln``, optional ``electrolyser``, optional
    ``hydrogen_buffer_storage``, and optional ``thermal_storage``. ``cement_mill``
    is intentionally rejected until grinding is modelled.
    """

    clinker_demand_column: str = ""
    raw_meal_to_clinker_ratio: float = 1.55
    waste_heat_per_t_clinker_mwh: float = 0.22
    waste_heat_utilization_efficiency: float = 0.90
    #: Headroom above the hourly demand, for a stage that cannot run below some load.
    clinker_overproduction_tolerance: float = 0.05
    components: dict[str, object] = field(default_factory=dict)
    cement_route: str = ""

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

    #: Every kiln-line stage has its own auxiliary load, and the thermal store's electric
    #: draw is named ``electricity_consumption`` rather than ``power_in`` - the base
    #: class's default checks only the latter.
    _power_consuming_attrs: ClassVar[tuple[str, ...]] = (
        "power_in",
        "aux_power_in",
        "electricity_consumption",
    )

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

        cement_route = _detect_cement_route(components, plant_name)
        if _non_empty_values(rows, "clinker_demand"):
            raise ValueError(
                f"Cement plant '{plant_name}' sets a scalar 'clinker_demand'; cement demand is "
                "now always a per-timestep forecast column. Name it in the 'demand' column of "
                f"plants.csv, or provide '{plant_name}_clinker_demand' in forecasts_df.csv"
            )
        clinker_demand_column = _consistent_optional_text(rows, "demand", plant_name)
        if not clinker_demand_column:
            clinker_demand_column = f"{plant_name}_clinker_demand"

        return cls(
            name=plant_name,
            unit_type=first_non_empty(rows, "unit_type", default="cement_plant"),
            node=first_non_empty(rows, "node", default=""),
            objective=first_non_empty(rows, "objective", default="min_variable_cost"),
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
            clinker_overproduction_tolerance=_consistent_optional_total(
                rows,
                "clinker_overproduction_tolerance",
                plant_name,
                default=0.05,
            ),
            components=components,
            cement_route=cement_route,
        )

    @classmethod
    def from_plants_dataframe(cls, plants: pd.DataFrame) -> list[CementPlant]:
        return [
            cls.from_rows(str(plant_name), rows)
            for plant_name, rows in plants.groupby("name", sort=False)
        ]

    def _terminal_technology_name(self) -> str:
        """The stage whose clinker output is the plant's output."""
        return "kiln" if "kiln" in self.components else "calciner"

    def _raw_meal_receiving_stage(self) -> str | None:
        """The stage the preheated raw meal feeds.

        The calciner where there is one, otherwise the kiln, which then performs the
        calcination reaction itself. ``None`` when there is no preheater to feed anything.
        """
        if "preheater" not in self.components:
            return None
        return "calciner" if "calciner" in self.components else "kiln"

    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: CementDispatchSignals,
        initial_state: CementRollingState | None = None,
    ) -> pd.DataFrame:
        if self.objective != "min_variable_cost":
            raise ValueError(f"Cement plant objective '{self.objective}' is not supported")
        if forecasts.empty:
            raise ValueError("Cement plant dispatch horizon must contain at least one timestep")
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
        signals: CementDispatchSignals,
        initial_state: CementRollingState | None = None,
        progress_callback: Callable[[pd.Timestamp, pd.Timestamp], None] | None = None,
    ) -> pd.DataFrame:
        """Dispatch in overlapping windows, committing only the leading steps of each.

        Simpler than the steel equivalent because cement demand is a per-timestep
        minimum: the plant can never fall behind, so there is no backlog to carry and no
        final-window reconciliation. Each window inherits only the physical state - stage
        on/off, storage fill - left by the previous commit.
        """

        if forecasts.empty:
            raise ValueError("Cement rolling dispatch requires at least one timestep")
        self._validate_signal_columns(forecasts, signals)
        dt_hours = config.timestep_minutes / 60.0
        horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 48))
        step_hours = float(config.dispatch_setting("rolling_step_hours", 24))
        validate_rolling_window(dt_hours, horizon_hours, step_hours, plant_label="Cement plant")
        horizon_steps = int(round(horizon_hours / dt_hours))
        step_steps = int(round(step_hours / dt_hours))

        demand_schedule = self._validated_demand_profile(forecasts)
        total_demand = float(demand_schedule.sum())
        state = initial_state or self._initial_rolling_state()

        implemented_frames: list[pd.DataFrame] = []
        position = 0
        window_number = 1
        while position < len(forecasts):
            horizon = forecasts.iloc[position : position + horizon_steps].copy()
            commit_count = min(step_steps, len(horizon), len(forecasts) - position)
            if progress_callback is not None:
                progress_callback(
                    pd.Timestamp(horizon.index[0]),
                    pd.Timestamp(horizon.index[commit_count - 1]),
                )

            model = self._build_model(config, horizon, signals, initial_state=state)
            try:
                horizon_result = self._solve_model(config, horizon, model)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Cement rolling window {window_number} starting {horizon.index[0]} is "
                    f"infeasible: demand={float(horizon[self.clinker_demand_column].sum()):g} t"
                ) from exc

            implemented = horizon_result.iloc[:commit_count].copy()
            produced_before = state.cumulative_clinker_output_t
            produced_by_row = implemented["clinker_output_t"].cumsum()
            implemented["rolling_window"] = window_number
            implemented["clinker_committed_output_t"] = float(implemented["clinker_output_t"].sum())
            implemented["cumulative_clinker_output_t"] = produced_before + produced_by_row
            implemented["clinker_demand_total_t"] = total_demand
            implemented["remaining_clinker_demand_t"] = (
                total_demand - implemented["cumulative_clinker_output_t"]
            ).clip(lower=0.0)
            implemented_frames.append(implemented)

            state = self._state_after_commit(previous=state, committed=implemented)
            position += commit_count
            window_number += 1

        result = pd.concat(implemented_frames).sort_index()
        produced = state.cumulative_clinker_output_t
        if produced < total_demand - ROLLING_DEMAND_TOLERANCE_T:
            raise RuntimeError(
                "Cement rolling dispatch ended below the demanded clinker: "
                f"produced={produced:g} t, demanded={total_demand:g} t"
            )
        return result

    def solve_afrr_down_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: CementAFRRDownSignals,
        initial_state: CementRollingState | None = None,
        progress_callback: Callable[[pd.Timestamp, pd.Timestamp], None] | None = None,
    ) -> pd.DataFrame:
        """Roll through the horizon bidding day-ahead against aFRR down.

        Each window is solved twice: once with capacity bidding disabled, to price what
        committing capacity actually costs, and once for real. Simpler than the steel
        equivalent because a per-timestep demand band leaves no backlog to carry.
        """
        if forecasts.empty:
            raise ValueError("Cement aFRR dispatch requires at least one timestep")
        self._validate_afrr_signal_columns(forecasts, signals)
        market = signals.market
        validate_bid_rules(
            "afrr_energy", market.afrr_energy_min_bid_mw, market.afrr_energy_bid_increment_mw
        )
        validate_bid_rules(
            "afrr_capacity", market.afrr_capacity_min_bid_mw, market.afrr_capacity_bid_increment_mw
        )

        dt_hours = config.timestep_minutes / 60.0
        horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 48))
        step_hours = float(config.dispatch_setting("rolling_step_hours", 24))
        validate_rolling_window(dt_hours, horizon_hours, step_hours, plant_label="Cement plant")
        if horizon_hours + 1e-9 < step_hours + market.afrr_capacity_product_duration_h:
            raise ValueError(
                "Cement aFRR dispatch_horizon_hours must cover rolling_step_hours plus one "
                "complete aFRR capacity product"
            )
        horizon_steps = int(round(horizon_hours / dt_hours))
        step_steps = int(round(step_hours / dt_hours))

        demand_schedule = self._validated_demand_profile(forecasts)
        total_demand = float(demand_schedule.sum())
        state = initial_state or self._initial_rolling_state()

        implemented_frames: list[pd.DataFrame] = []
        position = 0
        window_number = 1
        while position < len(forecasts):
            horizon = forecasts.iloc[position : position + horizon_steps].copy()
            commit_count = min(step_steps, len(horizon), len(forecasts) - position)
            if progress_callback is not None:
                progress_callback(
                    pd.Timestamp(horizon.index[0]),
                    pd.Timestamp(horizon.index[commit_count - 1]),
                )

            window = AFRRWindow(
                horizon=horizon,
                commit_steps=commit_count,
                is_final_window=position + commit_count >= len(forecasts),
                payload=CementWindowPlan(initial_state=state),
            )

            def solve(capacity_enabled: bool, window: AFRRWindow = window) -> pd.DataFrame:
                model = build_afrr_down_market_model(
                    self,
                    window.horizon,
                    signals.market,
                    signals.commodities,
                    window,
                    dt_hours=dt_hours,
                    capacity_enabled=capacity_enabled,
                )
                return self._solve_model(config, window.horizon, model)

            try:
                # The capacity-disabled twin prices what committing capacity costs.
                baseline_result = solve(False)
                horizon_result = solve(True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Cement aFRR window {window_number} starting {horizon.index[0]} is infeasible"
                ) from exc

            horizon_result = attach_capacity_opportunity_cost(
                horizon_result, baseline_result, commit_steps=commit_count
            )
            implemented = horizon_result.iloc[:commit_count].copy()
            produced_before = state.cumulative_clinker_output_t
            produced_by_row = implemented["clinker_output_t"].cumsum()
            implemented["rolling_window"] = window_number
            implemented["clinker_committed_output_t"] = float(implemented["clinker_output_t"].sum())
            implemented["cumulative_clinker_output_t"] = produced_before + produced_by_row
            implemented["clinker_demand_total_t"] = total_demand
            implemented["remaining_clinker_demand_t"] = (
                total_demand - implemented["cumulative_clinker_output_t"]
            ).clip(lower=0.0)
            implemented_frames.append(implemented)

            state = self._state_after_commit(previous=state, committed=implemented)
            position += commit_count
            window_number += 1

        result = pd.concat(implemented_frames).sort_index()
        produced = state.cumulative_clinker_output_t
        if produced < total_demand - ROLLING_DEMAND_TOLERANCE_T:
            raise RuntimeError(
                "Cement aFRR dispatch ended below the demanded clinker: "
                f"produced={produced:g} t, demanded={total_demand:g} t"
            )
        return result

    def _validate_afrr_signal_columns(
        self, forecasts: pd.DataFrame, signals: CementAFRRDownSignals
    ) -> None:
        market, commodities = signals.market, signals.commodities
        required = {
            market.da_price_col,
            market.afrr_energy_price_col,
            market.afrr_system_activation_col,
            market.afrr_price_available_col,
            market.afrr_capacity_block_id_col,
            market.afrr_capacity_block_duration_col,
            market.afrr_capacity_price_col,
            market.afrr_capacity_missing_price_col,
            commodities.natural_gas_price_col,
            commodities.co2_price_col,
            self.clinker_demand_column,
        }
        if "electrolyser" not in self.components:
            required.add(commodities.hydrogen_price_col)
        if self._requires_coal_price():
            required.add(commodities.coal_price_col)
        missing = required - set(forecasts.columns)
        if missing:
            raise ValueError(
                "Cement aFRR forecasts are missing column(s): " + ", ".join(sorted(missing))
            )

    # -- AFRRDownPlant protocol -----------------------------------------------------

    def afrr_attach_commodity_params(
        self,
        model: pyo.ConcreteModel,
        forecasts: pd.DataFrame,
        commodity_signals: CementAFRRDownCommoditySignals,
        window: AFRRWindow,
    ) -> None:
        def values(column: str) -> dict[int, float]:
            return {t: float(forecasts[column].iloc[t]) for t in model.T}

        # Electricity is settled once at plant level by the market layer, so the price the
        # components see must be zero; otherwise every stage would pay for its own power
        # and the bill would be counted twice.
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
        model.co2_price = pyo.Param(model.T, initialize=values(commodity_signals.co2_price_col))

        demand = self._validated_demand_profile(forecasts)
        model.clinker_demand_per_timestep = pyo.Param(
            model.T, initialize={t: float(demand.iloc[t]) for t in model.T}
        )
        model.clinker_overproduction_tolerance = pyo.Param(
            initialize=self.clinker_overproduction_tolerance
        )
        model.raw_meal_to_clinker_ratio = pyo.Param(initialize=self.raw_meal_to_clinker_ratio)
        model.waste_heat_per_t_clinker = pyo.Param(initialize=self.waste_heat_per_t_clinker_mwh)
        model.waste_heat_utilization_efficiency = pyo.Param(
            initialize=self.waste_heat_utilization_efficiency
        )

    def afrr_attach_trajectory(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block,
        role: TrajectoryRole,
        window: AFRRWindow,
        dt_hours: float,
    ) -> None:
        """Attach identical physics to both trajectories.

        ``role`` is deliberately ignored. Steel must drop its output constraint on the
        hypothetical branch because a cumulative equality fights a different power draw,
        but cement's demand is a per-timestep band that a higher draw cannot violate -
        and dropping it would be wrong twice over: on a route with storage the twin would
        absorb the whole bid while making no clinker, making the feasibility test
        meaningless; on a route without storage or electrolyser its only load is
        auxiliary power, which is proportional to output, so zero output would make the
        electricity balance unsatisfiable and silently force every bid to zero.
        """
        plan: CementWindowPlan = window.payload
        self._add_physical_system(model, container, model.T, dt_hours, plan.initial_state)

    def afrr_attach_terminal_state(self, model: pyo.ConcreteModel, final_t: int) -> None:
        """Return every store to where it started, on the committed trajectory only.

        Without this the final window profits by draining a store it never refills. Note
        the two storage families report in different units - the thermal store in
        absolute MWh, the hydrogen buffer as a fraction of capacity - so the target has
        to be read from the matching attribute.
        """
        for technology, component in self.components.items():
            if technology == "thermal_storage":
                target = float(component.initial_soc_mwh)
            elif isinstance(component, GenericInventoryStorage):
                target = float(component.initial_soc)
            else:
                continue

            model.add_component(
                f"actual_terminal_inventory_{technology}",
                pyo.Constraint(
                    expr=(model.actual.technology_blocks[technology].soc[final_t] == target)
                ),
            )

    def afrr_fuel_substitution(self, forecasts: pd.DataFrame) -> FuelSubstitution | None:
        """Electricity price at which electric heat matches burner heat on a hybrid stage.

        A ``hybrid_electricity_fossil`` stage forms heat as
        ``power_in * eta_electric + fossil_in * eta_fossil``, so the two routes break even
        at ``(gas price + CO2) * eta_electric / eta_fossil`` per MWh of electricity.

        Unlike steel's electrolyser this is a continuous blend, not an on/off decision:
        the stage can take any split. So there is no gate column and no gated load - the
        LP picks the split itself once it is shown the effective price.
        """
        hybrid = next(
            (
                self.components[stage]
                for stage in CEMENT_LINE_STAGES
                if stage in self.components
                and self.components[stage].fuel_type == CEMENT_HYBRID_ELECTRICITY_FOSSIL
            ),
            None,
        )
        if hybrid is None:
            return None

        natural_gas_price = forecasts["natural_gas_price"].astype(float)
        co2_price = forecasts["co2_price"].astype(float)
        fossil_cost_per_mwh_th = (
            natural_gas_price + float(hybrid.natural_gas_co2_factor_t_per_mwh) * co2_price
        )
        benchmark = fossil_cost_per_mwh_th * (
            float(hybrid.eta_electric) / float(hybrid.eta_fossil)
        )
        benchmark.name = "fossil_based_electricity_benchmark_EUR_per_MWh_el"
        return FuelSubstitution(benchmark_eur_per_mwh_el=benchmark)

    def afrr_aggregate_max_power_mw(self) -> float:
        """Greatest electric power the plant can draw at once.

        Four contributions, and the auxiliary one is the easy mistake: on a plain fossil
        route with no storage or electrolyser it is the *only* electric load, so omitting
        it would silently collapse every bid to zero rather than raise.
        """
        total = 0.0
        for technology, component in self.components.items():
            if technology in CEMENT_LINE_STAGES:
                if component.fuel_type in {CEMENT_ELECTRICITY, CEMENT_HYBRID_ELECTRICITY_FOSSIL}:
                    total += float(
                        component.max_electric_power_mw
                        if component.max_electric_power_mw is not None
                        else component.max_heat_out_mw / max(component.eta_electric, 1e-9)
                    )
                throughput_per_h = (
                    component.max_heat_out_mw / component.specific_heat_demand_mwh_per_t
                )
                total += throughput_per_h * component.specific_electricity_aux_mwh_per_t
            elif technology == "thermal_storage":
                total += float(component.max_power_charge_mw)
            else:
                total += float(getattr(component, "max_power_mw", 0.0))
        return total

    def _build_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: CementDispatchSignals,
        initial_state: CementRollingState | None = None,
    ) -> pyo.ConcreteModel:
        dt_hours = config.timestep_minutes / 60.0
        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=range(len(forecasts)), ordered=True)

        def values(column: str) -> dict[int, float]:
            return {t: float(forecasts[column].iloc[t]) for t in model.T}

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
        model.co2_price = pyo.Param(model.T, initialize=values(signals.co2_price_col))

        demand = self._validated_demand_profile(forecasts)
        model.clinker_demand_per_timestep = pyo.Param(
            model.T, initialize={t: float(demand.iloc[t]) for t in model.T}
        )
        model.clinker_overproduction_tolerance = pyo.Param(
            initialize=self.clinker_overproduction_tolerance
        )
        model.raw_meal_to_clinker_ratio = pyo.Param(initialize=self.raw_meal_to_clinker_ratio)
        model.waste_heat_per_t_clinker = pyo.Param(initialize=self.waste_heat_per_t_clinker_mwh)
        model.waste_heat_utilization_efficiency = pyo.Param(
            initialize=self.waste_heat_utilization_efficiency
        )

        self._add_physical_system(
            model,
            model,
            model.T,
            dt_hours,
            initial_state=initial_state or self._initial_rolling_state(),
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
        dt_hours: float,
        initial_state: CementRollingState,
    ) -> None:
        """Attach one complete clinker-production trajectory to ``container``.

        ``container`` is the price-taker's ``model`` itself, but a market model needs two
        of these side by side - the committed plan and a hypothetical "what if the whole
        aFRR bid is called" twin - and Pyomo gives every component exactly one parent, so
        the destination has to be a parameter rather than hard-coded.

        Anything shared between the two trajectories (prices, the demand profile, the
        ratios) stays on ``model`` and must be reached through the closed-over ``model``
        rather than through a rule's first argument, which is ``container``.
        """
        self._attach_technology_blocks(model, container, time_steps, dt_hours, initial_state)
        self.initialize_process_sequence(model, container, time_steps)
        self._declare_plant_totals(container, time_steps)
        self.define_constraints(model, container, time_steps)
        self._attach_total_power_input_constraint(container, time_steps)
        self._attach_variable_cost_constraint(container, time_steps)

    def initialize_process_sequence(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
    ) -> None:
        """Pyomo Components:

        - **Constraints**: the kiln-line's own flow linking - waste heat reaching the
          preheater, the raw-meal mass link, the calciner's storage-blended effective
          heat, the calciner-to-kiln clinker link, and the electrolyser's hydrogen
          balance - each present only for the stages and storages this plant actually
          has configured.
        """
        blocks = container.technology_blocks
        preheater = blocks["preheater"] if "preheater" in self.components else None
        calciner = blocks["calciner"] if "calciner" in self.components else None
        kiln = blocks["kiln"] if "kiln" in self.components else None

        if preheater is not None and kiln is not None:

            @container.Constraint(time_steps)
            def waste_heat_to_preheater(m: pyo.Block, t: int) -> pyo.Constraint:
                return preheater.external_heat_in[t] == (
                    kiln.clinker_out[t]
                    * model.waste_heat_per_t_clinker
                    * model.waste_heat_utilization_efficiency
                )

        elif preheater is not None:

            @container.Constraint(time_steps)
            def no_external_preheater_heat(m: pyo.Block, t: int) -> pyo.Constraint:
                return preheater.external_heat_in[t] == 0.0

        receiving_stage = self._raw_meal_receiving_stage()
        if receiving_stage is not None:
            receiver = blocks[receiving_stage]

            @container.Constraint(time_steps)
            def preheater_raw_meal_mass_link(
                m: pyo.Block,
                t: int,
            ) -> pyo.Constraint:
                return (
                    receiver.clinker_out[t]
                    == preheater.raw_meal_out[t] / model.raw_meal_to_clinker_ratio
                )

        if calciner is not None:
            if "thermal_storage" in self.components:
                storage = blocks["thermal_storage"]

                @container.Constraint(time_steps)
                def calciner_effective_heat_with_storage(
                    m: pyo.Block,
                    t: int,
                ) -> pyo.Constraint:
                    return calciner.effective_heat_in[t] == (
                        calciner.heat_out[t] + storage.discharge_heat[t]
                    )

            else:

                @container.Constraint(time_steps)
                def calciner_effective_heat_without_storage(
                    m: pyo.Block,
                    t: int,
                ) -> pyo.Constraint:
                    return calciner.effective_heat_in[t] == calciner.heat_out[t]

        if calciner is not None and kiln is not None:

            @container.Constraint(time_steps)
            def calciner_to_kiln_mass_link(
                m: pyo.Block,
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

                @container.Constraint(time_steps)
                def hydrogen_balance_with_storage(
                    m: pyo.Block,
                    t: int,
                ) -> pyo.Constraint:
                    return electrolyser.hydrogen_out[t] + hydrogen_storage.discharge[t] == (
                        total_hydrogen_demand(t) + hydrogen_storage.charge[t]
                    )

            else:

                @container.Constraint(time_steps)
                def hydrogen_balance_without_storage(
                    m: pyo.Block,
                    t: int,
                ) -> pyo.Constraint:
                    return electrolyser.hydrogen_out[t] == total_hydrogen_demand(t)

    def define_constraints(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
        **kwargs: Any,
    ) -> None:
        """Pyomo Components:

        - **Constraints**: the hourly clinker demand band -
          ``clinker_demand_constraint`` and ``clinker_overproduction_cap``.

        Takes no keyword arguments: unlike steel's demand association, cement's demand
        band never needs to be dropped or adjusted between the ``actual`` and
        ``full_activation`` trajectories (see ``afrr_attach_trajectory``).
        """
        terminal = container.technology_blocks[self._terminal_technology_name()]

        @container.Constraint(time_steps)
        def clinker_demand_constraint(m: pyo.Block, t: int) -> pyo.Constraint:
            """Meet the hourly clinker demand.

            A minimum rather than an equality, mirroring the reference model: cost
            minimisation never produces above it, but leaving headroom keeps the window
            feasible when a stage's minimum load forces a little extra clinker.
            """
            return terminal.clinker_out[t] >= model.clinker_demand_per_timestep[t]

        @container.Constraint(time_steps)
        def clinker_overproduction_cap(m: pyo.Block, t: int) -> pyo.Constraint:
            """Cap production just above demand.

            FLEXIMOD models no clinker silo, so surplus clinker would simply vanish.
            Without a cap the market model would discover that it can absorb paid-for
            aFRR energy by making clinker nobody ordered - and, worse, claim a larger
            deliverable bid on the strength of it. The small tolerance is headroom for a
            stage's minimum load, not a licence to over-produce.
            """
            return terminal.clinker_out[t] <= model.clinker_demand_per_timestep[t] * (
                1.0 + model.clinker_overproduction_tolerance
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
            plant_label="Cement plant",
        )

    def _extract_results(
        self,
        model: pyo.ConcreteModel,
        forecasts: pd.DataFrame,
        solver_name: str,
    ) -> pd.DataFrame:
        # A market model carries two trajectories; only the committed one is ever
        # realised, so it alone is reported and feeds the rolling state. The
        # full-activation twin exists purely to constrain the bid.
        market_model = hasattr(model, "actual")
        trajectory = model.actual if market_model else model
        blocks = trajectory.technology_blocks
        preheater = blocks["preheater"] if "preheater" in self.components else None
        calciner = blocks["calciner"] if "calciner" in self.components else None
        kiln = blocks["kiln"] if "kiln" in self.components else None
        terminal = blocks[self._terminal_technology_name()]

        def block_value(block: pyo.Block | None, variable: str, t: int) -> float:
            if block is None or not hasattr(block, variable):
                return 0.0
            return _value(getattr(block, variable)[t])

        data: dict[str, list[float] | list[str]] = {
            "plant_name": [],
            "plant_type": [],
            "cement_route": [],
            "clinker_demand_t": [],
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
        # In market mode these ten columns belong to the aFRR ontology and are written
        # once by append_afrr_result_row; writing them here too would append twice per
        # timestep and the frame would not assemble.
        price_taker_columns = (
            "DA_position_MWh",
            "final_planned_electricity_MWh",
            "actual_electricity_consumption_MWh",
            "day_ahead_price_EUR_per_MWh",
            "day_ahead_delivered_price_EUR_per_MWh",
            "additional_electricity_charge_EUR_per_MWh_el",
            "electricity_market_cost_EUR",
            "additional_electricity_charges_cost_EUR",
            "gross_operating_cost_EUR",
            "net_operating_cost_EUR",
        )
        if not market_model:
            for column in price_taker_columns:
                data[column] = []
        else:
            for column in AFRR_DOWN_RESULT_COLUMNS:
                data[column] = []

        optional_variables = {
            # Stage on/off states are what a rolling window carries across its boundary,
            # so they have to survive into the dispatch frame.
            "preheater_operational_status": ("preheater", "operational_status"),
            "calciner_operational_status": ("calciner", "operational_status"),
            "kiln_operational_status": ("kiln", "operational_status"),
            "electrolyser_electricity_consumption_MWh": ("electrolyser", "power_in"),
            "electrolyser_hydrogen_output_MWh": ("electrolyser", "hydrogen_out"),
            "electrolyser_operational_status": ("electrolyser", "operational_status"),
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

        horizon_demand_total = sum(_value(model.clinker_demand_per_timestep[t]) for t in model.T)
        for t in model.T:
            total_electricity = _value(trajectory.total_power_input[t])
            # In market mode the electricity bill is settled by the market layer, so the
            # trajectory's own variable cost is non-electric only and the full figure
            # comes from gross_operating_cost.
            variable_cost = (
                _value(model.gross_operating_cost[t])
                if market_model
                else _value(trajectory.variable_cost[t])
            )
            data["plant_name"].append(self.name)
            data["plant_type"].append(self.unit_type)
            data["cement_route"].append(self.cement_route)
            data["clinker_demand_t"].append(_value(model.clinker_demand_per_timestep[t]))
            data["clinker_demand_total_t"].append(horizon_demand_total)
            data["clinker_output_t"].append(block_value(terminal, "clinker_out", t))
            data["total_electricity_consumption_MWh"].append(total_electricity)
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
            data["variable_cost_EUR"].append(variable_cost)
            if not market_model:
                market_price = _value(model.electricity_market_price[t])
                additional_charge = _value(model.additional_electricity_charge)
                data["DA_position_MWh"].append(total_electricity)
                data["final_planned_electricity_MWh"].append(total_electricity)
                data["actual_electricity_consumption_MWh"].append(total_electricity)
                data["day_ahead_price_EUR_per_MWh"].append(market_price)
                data["day_ahead_delivered_price_EUR_per_MWh"].append(
                    market_price + additional_charge
                )
                data["additional_electricity_charge_EUR_per_MWh_el"].append(additional_charge)
                data["electricity_market_cost_EUR"].append(total_electricity * market_price)
                data["additional_electricity_charges_cost_EUR"].append(
                    total_electricity * additional_charge
                )
                data["gross_operating_cost_EUR"].append(variable_cost)
                data["net_operating_cost_EUR"].append(variable_cost)
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
            if market_model:
                append_afrr_result_row(data, model, t, total_electricity)
            for column, (technology, variable) in optional_variables.items():
                if column not in data:
                    continue
                block = blocks[technology]
                if not hasattr(block, variable):
                    # The electrolyser only gets an on/off state when a minimum load or
                    # minimum up/down time is configured. Without one it is always
                    # available, which is what the next rolling window should inherit.
                    data[column].append(1.0 if variable == "operational_status" else 0.0)
                    continue
                data[column].append(_value(getattr(block, variable)[t]))

        return pd.DataFrame(data, index=forecasts.index)

    def _component_context(
        self,
        technology: str,
        dt_hours: float,
        initial_state: CementRollingState,
    ) -> dict[str, Any]:
        """Build the ``add_to_model`` context that seeds a window from the last commit."""

        context: dict[str, Any] = {"dt_hours": dt_hours}
        stage_state = initial_state.stages.get(technology)
        if stage_state is not None:
            context.update(
                {
                    "initial_heat_out": stage_state.heat_out_mwh,
                    "initial_operational_status": stage_state.operational_status,
                    "initial_consecutive_status_steps": stage_state.consecutive_status_steps,
                }
            )
        component_state = initial_state.components.get(technology)
        if component_state is not None:
            context.update(
                {
                    "initial_power_in": component_state.power_in_mwh,
                    "initial_operational_status": component_state.operational_status,
                    "initial_consecutive_status_steps": component_state.consecutive_status_steps,
                }
            )
        inventory_state = initial_state.inventories.get(technology)
        if inventory_state is not None:
            # The two storage families disagree on both key and unit: ThermalStorage
            # reads absolute MWh from `initial_soc_mwh`, GenericInventoryStorage reads a
            # fraction of capacity from `initial_soc`. Writing the wrong one silently
            # resets the fill level or pushes it outside its own bounds.
            key = "initial_soc_mwh" if technology == "thermal_storage" else "initial_soc"
            context.update(
                {
                    key: inventory_state.soc,
                    "initial_charge": inventory_state.charge,
                    "initial_discharge": inventory_state.discharge,
                }
            )
        return context

    def _initial_rolling_state(self) -> CementRollingState:
        """State the very first window starts from, taken from the component defaults."""

        stages: dict[str, CementStageState] = {}
        components: dict[str, CementComponentState] = {}
        inventories: dict[str, CementInventoryState] = {}
        for technology, component in self.components.items():
            satisfied_steps = max(
                int(getattr(component, "min_operating_steps", 0)),
                int(getattr(component, "min_down_steps", 0)),
                1,
            )
            initial_status = int(getattr(component, "initial_operational_status", 1))
            if technology in CEMENT_LINE_STAGES:
                stages[technology] = CementStageState(
                    heat_out_mwh=0.0,
                    operational_status=initial_status,
                    consecutive_status_steps=satisfied_steps,
                )
            elif hasattr(component, "max_power_mw"):
                components[technology] = CementComponentState(
                    power_in_mwh=0.0,
                    operational_status=initial_status,
                    consecutive_status_steps=satisfied_steps,
                )
            if technology == "thermal_storage":
                inventories[technology] = CementInventoryState(soc=component.initial_soc_mwh)
            elif isinstance(component, GenericInventoryStorage):
                inventories[technology] = CementInventoryState(soc=component.initial_soc)
        return CementRollingState(stages=stages, components=components, inventories=inventories)

    def _state_after_commit(
        self,
        previous: CementRollingState,
        committed: pd.DataFrame,
    ) -> CementRollingState:
        """Carry the end of a committed window forward as the next window's start."""

        def run_length(
            technology: str, previous_steps: int, previous_status: int
        ) -> tuple[int, int]:
            """Final on/off state, and how many consecutive steps it has already held."""
            statuses = committed[f"{technology}_operational_status"].round().astype(int).tolist()
            steps = previous_steps
            current = previous_status
            for status in statuses:
                if status == current:
                    steps += 1
                else:
                    current = status
                    steps = 1
            return current, steps

        stages: dict[str, CementStageState] = {}
        for technology, old in previous.stages.items():
            status, steps = run_length(
                technology, old.consecutive_status_steps, old.operational_status
            )
            stages[technology] = CementStageState(
                heat_out_mwh=float(committed[f"{technology}_heat_output_MWh"].iloc[-1]),
                operational_status=status,
                consecutive_status_steps=steps,
            )

        components: dict[str, CementComponentState] = {}
        for technology, old in previous.components.items():
            status, steps = run_length(
                technology, old.consecutive_status_steps, old.operational_status
            )
            components[technology] = CementComponentState(
                power_in_mwh=float(committed[f"{technology}_electricity_consumption_MWh"].iloc[-1]),
                operational_status=status,
                consecutive_status_steps=steps,
            )

        inventory_columns = {
            "thermal_storage": (
                "thermal_storage_soc_MWh",
                "thermal_storage_charge_MWh",
                "thermal_storage_discharge_MWh",
            ),
            "hydrogen_buffer_storage": (
                "hydrogen_storage_soc",
                "hydrogen_storage_charge_MWh",
                "hydrogen_storage_discharge_MWh",
            ),
        }
        inventories: dict[str, CementInventoryState] = {}
        for technology in previous.inventories:
            soc_column, charge_column, discharge_column = inventory_columns[technology]
            inventories[technology] = CementInventoryState(
                soc=float(committed[soc_column].iloc[-1]),
                charge=float(committed[charge_column].iloc[-1]),
                discharge=float(committed[discharge_column].iloc[-1]),
            )

        return CementRollingState(
            stages=stages,
            components=components,
            inventories=inventories,
            cumulative_clinker_output_t=(
                previous.cumulative_clinker_output_t + float(committed["clinker_output_t"].sum())
            ),
        )

    def _validated_demand_profile(self, forecasts: pd.DataFrame) -> pd.Series:
        if self.clinker_demand_column not in forecasts.columns:
            raise ValueError(
                f"Cement plant '{self.name}' is missing its per-timestep clinker demand column "
                f"'{self.clinker_demand_column}' in the forecasts"
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


def _detect_cement_route(components: dict[str, object], plant_name: str) -> str:
    """Name the clinker route, and reject component sets that cannot form one.

    The name is the configured kiln-line stages in flow order joined by ``_``, giving
    ``preheater_calciner_kiln`` for a full line down to a bare ``kiln``. A preheater
    without a calciner is a valid single-stage line in which the kiln itself performs the
    calcination reaction.
    """

    if "calciner" not in components and "kiln" not in components:
        raise ValueError(
            f"Cement plant '{plant_name}' must define at least one terminal technology: "
            "calciner or kiln"
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
    return "_".join(stage for stage in CEMENT_LINE_STAGES if stage in components)


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
