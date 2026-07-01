# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import importlib.util
import shutil
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
    GasBoiler,
    ThermalStorage,
    first_non_empty,
)

DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL = 0.0


@dataclass
class DispatchSignals:
    electricity_price_col: str
    gas_price_col: str
    gas_benchmark_eur_per_mwh_th: pd.Series
    charge_allowed: pd.Series
    additional_electricity_charge_eur_per_mwh: float = 0.0
    reserved_capacity_mwh: pd.Series | None = None
    afrr_capacity_block_id: pd.Series | None = None
    afrr_capacity_block_duration_h: pd.Series | None = None
    afrr_capacity_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_reserved_mw: pd.Series | None = None
    afrr_capacity_revenue_eur: pd.Series | None = None
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
    additional_electricity_charge_eur_per_mwh: float = 0.0
    reserved_capacity_mwh: pd.Series | None = None
    afrr_capacity_block_id: pd.Series | None = None
    afrr_capacity_block_duration_h: pd.Series | None = None
    afrr_capacity_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_reserved_mw: pd.Series | None = None
    afrr_capacity_revenue_eur: pd.Series | None = None
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
    afrr_energy_capacity_backed_bid_mwh: pd.Series | None = None
    afrr_energy_free_bid_mwh: pd.Series | None = None
    afrr_energy_capacity_backed_activated_mwh: pd.Series | None = None
    afrr_energy_free_activated_mwh: pd.Series | None = None
    additional_electricity_charge_eur_per_mwh: float = 0.0
    afrr_headroom_binding: pd.Series | None = None
    afrr_curtailment_mwh: pd.Series | None = None
    reserved_capacity_mwh: pd.Series | None = None
    afrr_capacity_block_id: pd.Series | None = None
    afrr_capacity_block_duration_h: pd.Series | None = None
    afrr_capacity_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_reserved_mw: pd.Series | None = None
    afrr_capacity_revenue_eur: pd.Series | None = None
    co2_price_col: str | None = None
    co2_emission_factor_t_per_mwh_fuel: float = DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL


@dataclass
class SteamGenerationPlant(BasePlant):
    """Plant-level Pyomo model connecting steam/heat technologies on one heat bus."""

    components: dict[str, object] = field(default_factory=dict)

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
            components[technology] = TECHNOLOGY_REGISTRY[technology].from_row(row)

        if "thermal_storage" not in components:
            raise ValueError(f"Plant '{plant_name}' does not define a thermal_storage row")
        if "boiler" not in components:
            raise ValueError(f"Plant '{plant_name}' does not define a boiler row")

        heat_demand_column = first_non_empty(rows, "demand", default="")
        if not heat_demand_column:
            heat_demand_column = f"{plant_name}_heat_demand"

        return cls(
            name=plant_name,
            unit_type=first_non_empty(rows, "unit_type", default="steam_plant"),
            node=first_non_empty(rows, "node", default=""),
            objective=first_non_empty(rows, "objective", default="min_variable_cost"),
            heat_demand_column=heat_demand_column,
            components=components,
        )

    @classmethod
    def from_plants_dataframe(cls, plants: pd.DataFrame) -> list[SteamGenerationPlant]:
        return [
            cls.from_rows(str(plant_name), rows)
            for plant_name, rows in plants.groupby("name", sort=False)
        ]

    @property
    def etes(self) -> ThermalStorage:
        component = self.components.get("thermal_storage")
        if not isinstance(component, ThermalStorage):
            raise ValueError(f"Plant '{self.name}' has no thermal_storage component")
        return component

    @property
    def gas_boiler(self) -> GasBoiler:
        component = self.components.get("boiler")
        if not isinstance(component, GasBoiler):
            raise ValueError(f"Plant '{self.name}' has no boiler component")
        return component

    def solve_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: DispatchSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        dt_hours = config.timestep_minutes / 60.0
        horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 48))
        step_hours = float(config.dispatch_setting("rolling_step_hours", 24))
        horizon_steps = max(1, int(round(horizon_hours / dt_hours)))
        step_steps = max(1, int(round(step_hours / dt_hours)))

        implemented_frames: list[pd.DataFrame] = []
        initial_soc = self.etes.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
        position = 0

        while position < len(forecasts):
            horizon = forecasts.iloc[position : position + horizon_steps].copy()
            horizon_signals = DispatchSignals(
                electricity_price_col=signals.electricity_price_col,
                gas_price_col=signals.gas_price_col,
                co2_price_col=signals.co2_price_col,
                additional_electricity_charge_eur_per_mwh=(
                    signals.additional_electricity_charge_eur_per_mwh
                ),
                gas_benchmark_eur_per_mwh_th=signals.gas_benchmark_eur_per_mwh_th.loc[
                    horizon.index
                ],
                charge_allowed=signals.charge_allowed.loc[horizon.index],
                reserved_capacity_mwh=_optional_loc(signals.reserved_capacity_mwh, horizon.index),
                afrr_capacity_block_id=_optional_loc(signals.afrr_capacity_block_id, horizon.index),
                afrr_capacity_block_duration_h=_optional_loc(
                    signals.afrr_capacity_block_duration_h, horizon.index
                ),
                afrr_capacity_price_eur_per_mw_h=_optional_loc(
                    signals.afrr_capacity_price_eur_per_mw_h, horizon.index
                ),
                afrr_capacity_reserved_mw=_optional_loc(
                    signals.afrr_capacity_reserved_mw, horizon.index
                ),
                afrr_capacity_revenue_eur=_optional_loc(
                    signals.afrr_capacity_revenue_eur, horizon.index
                ),
                co2_emission_factor_t_per_mwh_fuel=signals.co2_emission_factor_t_per_mwh_fuel,
            )
            horizon_result = self.solve_horizon(
                config=config,
                forecasts=horizon,
                signals=horizon_signals,
                initial_soc_mwh=initial_soc,
            )
            implement_count = min(step_steps, len(forecasts) - position)
            implemented = horizon_result.iloc[:implement_count].copy()
            implemented_frames.append(implemented)

            initial_soc = float(implemented["etes_soc_MWh"].iloc[-1])
            position += implement_count

        return pd.concat(implemented_frames).sort_index()

    def solve_intraday_adjustment_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: IDCAdjustmentSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        dt_hours = config.timestep_minutes / 60.0
        horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 48))
        step_hours = float(config.dispatch_setting("rolling_step_hours", 24))
        horizon_steps = max(1, int(round(horizon_hours / dt_hours)))
        step_steps = max(1, int(round(step_hours / dt_hours)))

        implemented_frames: list[pd.DataFrame] = []
        initial_soc = self.etes.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
        position = 0

        while position < len(forecasts):
            horizon = forecasts.iloc[position : position + horizon_steps].copy()
            horizon_signals = IDCAdjustmentSignals(
                da_price_col=signals.da_price_col,
                idc_price_col=signals.idc_price_col,
                gas_price_col=signals.gas_price_col,
                co2_price_col=signals.co2_price_col,
                da_position_mwh=signals.da_position_mwh.loc[horizon.index],
                idc_buy_upper_bound_mwh=signals.idc_buy_upper_bound_mwh.loc[horizon.index],
                idc_sell_upper_bound_mwh=signals.idc_sell_upper_bound_mwh.loc[horizon.index],
                gas_benchmark_eur_per_mwh_th=signals.gas_benchmark_eur_per_mwh_th.loc[
                    horizon.index
                ],
                electricity_trading_benchmark_eur_per_mwh_el=(
                    signals.electricity_trading_benchmark_eur_per_mwh_el.loc[horizon.index]
                ),
                additional_electricity_charge_eur_per_mwh=(
                    signals.additional_electricity_charge_eur_per_mwh
                ),
                reserved_capacity_mwh=_optional_loc(signals.reserved_capacity_mwh, horizon.index),
                afrr_capacity_block_id=_optional_loc(signals.afrr_capacity_block_id, horizon.index),
                afrr_capacity_block_duration_h=_optional_loc(
                    signals.afrr_capacity_block_duration_h, horizon.index
                ),
                afrr_capacity_price_eur_per_mw_h=_optional_loc(
                    signals.afrr_capacity_price_eur_per_mw_h, horizon.index
                ),
                afrr_capacity_reserved_mw=_optional_loc(
                    signals.afrr_capacity_reserved_mw, horizon.index
                ),
                afrr_capacity_revenue_eur=_optional_loc(
                    signals.afrr_capacity_revenue_eur, horizon.index
                ),
                co2_emission_factor_t_per_mwh_fuel=signals.co2_emission_factor_t_per_mwh_fuel,
            )
            horizon_result = self.solve_intraday_adjustment_horizon(
                config=config,
                forecasts=horizon,
                signals=horizon_signals,
                initial_soc_mwh=initial_soc,
            )
            implement_count = min(step_steps, len(forecasts) - position)
            implemented = horizon_result.iloc[:implement_count].copy()
            implemented_frames.append(implemented)

            initial_soc = float(implemented["etes_soc_MWh"].iloc[-1])
            position += implement_count

        return pd.concat(implemented_frames).sort_index()

    def solve_afrr_down_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: AFRRDownSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        dt_hours = config.timestep_minutes / 60.0
        horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 48))
        step_hours = float(config.dispatch_setting("rolling_step_hours", 24))
        horizon_steps = max(1, int(round(horizon_hours / dt_hours)))
        step_steps = max(1, int(round(step_hours / dt_hours)))

        implemented_frames: list[pd.DataFrame] = []
        initial_soc = self.etes.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
        position = 0

        while position < len(forecasts):
            horizon = forecasts.iloc[position : position + horizon_steps].copy()
            horizon_signals = AFRRDownSignals(
                da_price_col=signals.da_price_col,
                idc_price_col=signals.idc_price_col,
                gas_price_col=signals.gas_price_col,
                co2_price_col=signals.co2_price_col,
                da_position_mwh=signals.da_position_mwh.loc[horizon.index],
                idc_buy_mwh=signals.idc_buy_mwh.loc[horizon.index],
                idc_sell_mwh=signals.idc_sell_mwh.loc[horizon.index],
                final_planned_electricity_mwh=signals.final_planned_electricity_mwh.loc[
                    horizon.index
                ],
                afrr_energy_price=signals.afrr_energy_price.loc[horizon.index],
                afrr_system_activation_mwh=signals.afrr_system_activation_mwh.loc[horizon.index],
                afrr_energy_bid_mwh=signals.afrr_energy_bid_mwh.loc[horizon.index],
                afrr_energy_activated_mwh=signals.afrr_energy_activated_mwh.loc[horizon.index],
                gas_benchmark_eur_per_mwh_th=signals.gas_benchmark_eur_per_mwh_th.loc[
                    horizon.index
                ],
                electricity_trading_benchmark_eur_per_mwh_el=(
                    signals.electricity_trading_benchmark_eur_per_mwh_el.loc[horizon.index]
                ),
                afrr_energy_bid_price=_optional_loc(signals.afrr_energy_bid_price, horizon.index),
                afrr_energy_capacity_backed_bid_mwh=_optional_loc(
                    signals.afrr_energy_capacity_backed_bid_mwh, horizon.index
                ),
                afrr_energy_free_bid_mwh=_optional_loc(
                    signals.afrr_energy_free_bid_mwh, horizon.index
                ),
                afrr_energy_capacity_backed_activated_mwh=_optional_loc(
                    signals.afrr_energy_capacity_backed_activated_mwh, horizon.index
                ),
                afrr_energy_free_activated_mwh=_optional_loc(
                    signals.afrr_energy_free_activated_mwh, horizon.index
                ),
                additional_electricity_charge_eur_per_mwh=(
                    signals.additional_electricity_charge_eur_per_mwh
                ),
                afrr_headroom_binding=_optional_loc(signals.afrr_headroom_binding, horizon.index),
                afrr_curtailment_mwh=_optional_loc(
                    signals.afrr_curtailment_mwh,
                    horizon.index,
                ),
                reserved_capacity_mwh=_optional_loc(signals.reserved_capacity_mwh, horizon.index),
                afrr_capacity_block_id=_optional_loc(signals.afrr_capacity_block_id, horizon.index),
                afrr_capacity_block_duration_h=_optional_loc(
                    signals.afrr_capacity_block_duration_h, horizon.index
                ),
                afrr_capacity_price_eur_per_mw_h=_optional_loc(
                    signals.afrr_capacity_price_eur_per_mw_h, horizon.index
                ),
                afrr_capacity_reserved_mw=_optional_loc(
                    signals.afrr_capacity_reserved_mw, horizon.index
                ),
                afrr_capacity_revenue_eur=_optional_loc(
                    signals.afrr_capacity_revenue_eur, horizon.index
                ),
                co2_emission_factor_t_per_mwh_fuel=signals.co2_emission_factor_t_per_mwh_fuel,
            )
            horizon_result = self.solve_afrr_down_horizon(
                config=config,
                forecasts=horizon,
                signals=horizon_signals,
                initial_soc_mwh=initial_soc,
            )
            implement_count = min(step_steps, len(forecasts) - position)
            implemented = horizon_result.iloc[:implement_count].copy()
            implemented_frames.append(implemented)

            initial_soc = float(implemented["etes_soc_MWh"].iloc[-1])
            position += implement_count

        return pd.concat(implemented_frames).sort_index()

    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: DispatchSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        model = self._build_model(
            config=config,
            forecasts=forecasts,
            signals=signals,
            initial_soc_mwh=self.etes.initial_soc_mwh
            if initial_soc_mwh is None
            else initial_soc_mwh,
        )
        solver_name = ""
        solve_errors: list[str] = []
        for candidate_name, solver in self._available_solvers(config):
            solver_name = candidate_name
            try:
                result = solver.solve(model, tee=config.solver_tee)
            except NoFeasibleSolutionError as exc:
                message = _strict_heat_infeasibility_message("Dispatch", candidate_name)
                raise RuntimeError(message) from exc
            except (ApplicationError, RuntimeError, OSError) as exc:
                solve_errors.append(f"{candidate_name}: {exc}")
                continue

            termination = result.solver.termination_condition
            status = result.solver.status
            if status == SolverStatus.ok and termination in {
                TerminationCondition.optimal,
                TerminationCondition.feasible,
            }:
                return self._extract_results(model, config, forecasts, signals, solver_name)

            if _is_infeasible_termination(termination):
                raise RuntimeError(_strict_heat_infeasibility_message("Dispatch", candidate_name))

            solve_errors.append(f"{candidate_name}: status={status}, termination={termination}")

        raise RuntimeError(
            "Dispatch solve failed for all configured solvers. " + " | ".join(solve_errors)
        )

    def solve_intraday_adjustment_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: IDCAdjustmentSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        model = self._build_intraday_adjustment_model(
            config=config,
            forecasts=forecasts,
            signals=signals,
            initial_soc_mwh=self.etes.initial_soc_mwh
            if initial_soc_mwh is None
            else initial_soc_mwh,
        )
        solver_name = ""
        solve_errors: list[str] = []
        for candidate_name, solver in self._available_solvers(config):
            solver_name = candidate_name
            try:
                result = solver.solve(model, tee=config.solver_tee)
            except NoFeasibleSolutionError as exc:
                raise RuntimeError(
                    _strict_heat_infeasibility_message("IDC adjustment", candidate_name)
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
                return self._extract_intraday_adjustment_results(
                    model,
                    config,
                    forecasts,
                    signals,
                    solver_name,
                )

            if _is_infeasible_termination(termination):
                raise RuntimeError(
                    _strict_heat_infeasibility_message("IDC adjustment", candidate_name)
                )

            solve_errors.append(f"{candidate_name}: status={status}, termination={termination}")

        raise RuntimeError(
            "IDC adjustment solve failed for all configured solvers. " + " | ".join(solve_errors)
        )

    def solve_afrr_down_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: AFRRDownSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        model = self._build_afrr_down_model(
            config=config,
            forecasts=forecasts,
            signals=signals,
            initial_soc_mwh=self.etes.initial_soc_mwh
            if initial_soc_mwh is None
            else initial_soc_mwh,
        )
        solver_name = ""
        solve_errors: list[str] = []
        for candidate_name, solver in self._available_solvers(config):
            solver_name = candidate_name
            try:
                result = solver.solve(model, tee=config.solver_tee)
            except NoFeasibleSolutionError as exc:
                raise RuntimeError(
                    _strict_heat_infeasibility_message("aFRR down adjustment", candidate_name)
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
                return self._extract_afrr_down_results(
                    model,
                    config,
                    forecasts,
                    signals,
                    solver_name,
                )

            if _is_infeasible_termination(termination):
                raise RuntimeError(
                    _strict_heat_infeasibility_message("aFRR down adjustment", candidate_name)
                )

            solve_errors.append(f"{candidate_name}: status={status}, termination={termination}")

        raise RuntimeError(
            "aFRR down adjustment solve failed for all configured solvers. "
            + " | ".join(solve_errors)
        )

    def _build_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: DispatchSignals,
        initial_soc_mwh: float,
    ) -> pyo.ConcreteModel:
        dt_hours = config.timestep_minutes / 60.0
        m = pyo.ConcreteModel(name=f"{self.name}_dispatch")
        steps = list(range(len(forecasts)))
        m.T = pyo.Set(initialize=steps, ordered=True)

        heat_demand_mwh = forecasts[self.heat_demand_column].astype(float).to_numpy() * dt_hours
        market_electricity_price = forecasts[signals.electricity_price_col].astype(float).to_numpy()
        additional_charge = float(signals.additional_electricity_charge_eur_per_mwh)
        electricity_price = market_electricity_price + additional_charge
        gas_price = forecasts[signals.gas_price_col].astype(float).to_numpy()
        if signals.co2_price_col and signals.co2_price_col in forecasts.columns:
            co2_price = forecasts[signals.co2_price_col].astype(float).to_numpy()
        else:
            co2_price = [0.0 for _ in steps]
        charge_allowed = signals.charge_allowed.astype(bool).reindex(forecasts.index).fillna(False)
        reserved_capacity_mwh = _series_or_zero(signals.reserved_capacity_mwh, forecasts.index)

        m.heat_demand = pyo.Param(m.T, initialize={t: heat_demand_mwh[t] for t in steps})
        m.market_electricity_price = pyo.Param(
            m.T,
            initialize={t: market_electricity_price[t] for t in steps},
        )
        m.additional_electricity_charge = pyo.Param(initialize=additional_charge)
        m.electricity_price = pyo.Param(m.T, initialize={t: electricity_price[t] for t in steps})
        m.gas_price = pyo.Param(m.T, initialize={t: gas_price[t] for t in steps})
        m.co2_price = pyo.Param(m.T, initialize={t: co2_price[t] for t in steps})
        m.charge_allowed = pyo.Param(
            m.T,
            within=pyo.Binary,
            initialize={t: int(bool(charge_allowed.iloc[t])) for t in steps},
        )
        m.reserved_capacity_mwh = pyo.Param(
            m.T, initialize={t: float(reserved_capacity_mwh.iloc[t]) for t in steps}
        )
        m.co2_emission_factor = pyo.Param(
            initialize=float(signals.co2_emission_factor_t_per_mwh_fuel)
        )

        m.technology_blocks = pyo.Block(list(self.components.keys()))
        for technology, component in self.components.items():
            context = {
                "dt_hours": dt_hours,
                "initial_soc_mwh": initial_soc_mwh if technology == "thermal_storage" else None,
            }
            component.add_to_model(m, m.technology_blocks[technology], m.T, context)

        m.electricity_consumption = pyo.Var(m.T, within=pyo.NonNegativeReals)

        @m.Constraint(m.T)
        def heat_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            boiler = mm.technology_blocks["boiler"]
            return storage.discharge_heat[t] + boiler.heat_out[t] == mm.heat_demand[t]

        @m.Constraint(m.T)
        def electricity_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            return mm.electricity_consumption[t] == storage.electricity_consumption[t]

        @m.Constraint(m.T)
        def reserve_charge_power_headroom(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            max_charge_mwh = storage.max_power_charge_mw * dt_hours
            return (
                storage.electric_charge_to_storage[t] + mm.reserved_capacity_mwh[t]
                <= max_charge_mwh
            )

        @m.Constraint(m.T)
        def reserve_storage_headroom(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            return (
                storage.soc[t] + mm.reserved_capacity_mwh[t] * storage.efficiency_charge
                <= storage.max_capacity_mwh
            )

        @m.Expression(m.T)
        def electricity_market_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.electricity_consumption[t] * mm.market_electricity_price[t]

        @m.Expression(m.T)
        def additional_electricity_charges_cost(
            mm: pyo.ConcreteModel,
            t: int,
        ) -> pyo.Expression:
            return mm.electricity_consumption[t] * mm.additional_electricity_charge

        @m.Expression(m.T)
        def electricity_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.technology_blocks["thermal_storage"].electricity_cost[t]

        @m.Expression(m.T)
        def gas_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.technology_blocks["boiler"].operating_cost[t]

        @m.Expression(m.T)
        def co2_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.technology_blocks["boiler"].co2_cost[t]

        @m.Objective(sense=pyo.minimize)
        def objective(mm: pyo.ConcreteModel) -> pyo.Expression:
            return pyo.quicksum(
                # CO2 cost is disabled for the first MVP and kept as a zero output column.
                mm.electricity_cost[t] + mm.gas_cost[t]
                for t in mm.T
            )

        return m

    def _build_intraday_adjustment_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: IDCAdjustmentSignals,
        initial_soc_mwh: float,
    ) -> pyo.ConcreteModel:
        dt_hours = config.timestep_minutes / 60.0
        m = pyo.ConcreteModel(name=f"{self.name}_idc_adjustment")
        steps = list(range(len(forecasts)))
        m.T = pyo.Set(initialize=steps, ordered=True)

        heat_demand_mwh = forecasts[self.heat_demand_column].astype(float).to_numpy() * dt_hours
        da_price = forecasts[signals.da_price_col].astype(float).to_numpy()
        idc_price = forecasts[signals.idc_price_col].astype(float).fillna(0.0).to_numpy()
        additional_charge = float(signals.additional_electricity_charge_eur_per_mwh)
        gas_price = forecasts[signals.gas_price_col].astype(float).to_numpy()
        if signals.co2_price_col and signals.co2_price_col in forecasts.columns:
            co2_price = forecasts[signals.co2_price_col].astype(float).to_numpy()
        else:
            co2_price = [0.0 for _ in steps]

        da_position = (
            signals.da_position_mwh.astype(float).reindex(forecasts.index).fillna(0.0).to_numpy()
        )
        idc_buy_upper_bound = (
            signals.idc_buy_upper_bound_mwh.astype(float)
            .reindex(forecasts.index)
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy()
        )
        idc_sell_upper_bound = (
            signals.idc_sell_upper_bound_mwh.astype(float)
            .reindex(forecasts.index)
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy()
        )
        reserved_capacity_mwh = _series_or_zero(signals.reserved_capacity_mwh, forecasts.index)

        m.heat_demand = pyo.Param(m.T, initialize={t: heat_demand_mwh[t] for t in steps})
        m.da_price = pyo.Param(m.T, initialize={t: da_price[t] for t in steps})
        m.idc_price = pyo.Param(m.T, initialize={t: idc_price[t] for t in steps})
        m.additional_electricity_charge = pyo.Param(initialize=additional_charge)
        m.electricity_price = pyo.Param(
            m.T,
            initialize={t: idc_price[t] + additional_charge for t in steps},
        )
        m.gas_price = pyo.Param(m.T, initialize={t: gas_price[t] for t in steps})
        m.co2_price = pyo.Param(m.T, initialize={t: co2_price[t] for t in steps})
        m.da_position_mwh = pyo.Param(m.T, initialize={t: da_position[t] for t in steps})
        m.idc_buy_upper_bound_mwh = pyo.Param(
            m.T,
            initialize={t: idc_buy_upper_bound[t] for t in steps},
        )
        m.idc_sell_upper_bound_mwh = pyo.Param(
            m.T,
            initialize={t: idc_sell_upper_bound[t] for t in steps},
        )
        m.reserved_capacity_mwh = pyo.Param(
            m.T, initialize={t: float(reserved_capacity_mwh.iloc[t]) for t in steps}
        )
        m.co2_emission_factor = pyo.Param(
            initialize=float(signals.co2_emission_factor_t_per_mwh_fuel)
        )

        m.technology_blocks = pyo.Block(list(self.components.keys()))
        for technology, component in self.components.items():
            context = {
                "dt_hours": dt_hours,
                "initial_soc_mwh": initial_soc_mwh if technology == "thermal_storage" else None,
            }
            component.add_to_model(m, m.technology_blocks[technology], m.T, context)

        m.electricity_consumption = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.idc_buy_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.idc_sell_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.final_planned_electricity_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)

        @m.Constraint(m.T)
        def idc_buy_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.idc_buy_mwh[t] <= mm.idc_buy_upper_bound_mwh[t]

        @m.Constraint(m.T)
        def idc_sell_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.idc_sell_mwh[t] <= mm.idc_sell_upper_bound_mwh[t]

        @m.Constraint(m.T)
        def idc_sell_da_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.idc_sell_mwh[t] <= mm.da_position_mwh[t]

        @m.Constraint(m.T)
        def final_planned_position(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return (
                mm.final_planned_electricity_mwh[t]
                == mm.da_position_mwh[t] + mm.idc_buy_mwh[t] - mm.idc_sell_mwh[t]
            )

        @m.Constraint(m.T)
        def etes_charge_matches_final_position(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            # For the current hybrid ETES + gas plant, the electricity market position maps
            # directly to ETES charging. TODO: Generalise this for industrial plants with
            # several electric processes sharing one market position.
            return storage.electric_charge_to_storage[t] == mm.final_planned_electricity_mwh[t]

        @m.Constraint(m.T)
        def reserve_charge_power_headroom(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            max_charge_mwh = storage.max_power_charge_mw * dt_hours
            return (
                mm.final_planned_electricity_mwh[t] + mm.reserved_capacity_mwh[t] <= max_charge_mwh
            )

        @m.Constraint(m.T)
        def heat_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            boiler = mm.technology_blocks["boiler"]
            return storage.discharge_heat[t] + boiler.heat_out[t] == mm.heat_demand[t]

        @m.Constraint(m.T)
        def electricity_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            return mm.electricity_consumption[t] == storage.electricity_consumption[t]

        @m.Constraint(m.T)
        def reserve_storage_headroom(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            return (
                storage.soc[t] + mm.reserved_capacity_mwh[t] * storage.efficiency_charge
                <= storage.max_capacity_mwh
            )

        @m.Expression(m.T)
        def da_electricity_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_position_mwh[t] * mm.da_price[t]

        @m.Expression(m.T)
        def idc_buy_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.idc_buy_mwh[t] * mm.idc_price[t]

        @m.Expression(m.T)
        def idc_sell_revenue(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.idc_sell_mwh[t] * mm.idc_price[t]

        @m.Expression(m.T)
        def electricity_market_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_electricity_cost[t] + mm.idc_buy_cost[t] - mm.idc_sell_revenue[t]

        @m.Expression(m.T)
        def additional_electricity_charges_cost(
            mm: pyo.ConcreteModel,
            t: int,
        ) -> pyo.Expression:
            return mm.final_planned_electricity_mwh[t] * mm.additional_electricity_charge

        @m.Expression(m.T)
        def electricity_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.electricity_market_cost[t] + mm.additional_electricity_charges_cost[t]

        @m.Expression(m.T)
        def gas_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.technology_blocks["boiler"].operating_cost[t]

        @m.Expression(m.T)
        def co2_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.technology_blocks["boiler"].co2_cost[t]

        @m.Objective(sense=pyo.minimize)
        def objective(mm: pyo.ConcreteModel) -> pyo.Expression:
            return pyo.quicksum(
                # TODO: Add CO2 cost consistently to the gas benchmark and plant objective.
                mm.electricity_cost[t] + mm.gas_cost[t]
                for t in mm.T
            )

        return m

    def _build_afrr_down_model(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: AFRRDownSignals,
        initial_soc_mwh: float,
    ) -> pyo.ConcreteModel:
        dt_hours = config.timestep_minutes / 60.0
        m = pyo.ConcreteModel(name=f"{self.name}_afrr_down")
        steps = list(range(len(forecasts)))
        m.T = pyo.Set(initialize=steps, ordered=True)

        heat_demand_mwh = forecasts[self.heat_demand_column].astype(float).to_numpy() * dt_hours
        da_price = forecasts[signals.da_price_col].astype(float).to_numpy()
        idc_price = forecasts[signals.idc_price_col].astype(float).fillna(0.0).to_numpy()
        additional_charge = float(signals.additional_electricity_charge_eur_per_mwh)
        gas_price = forecasts[signals.gas_price_col].astype(float).to_numpy()
        if signals.co2_price_col and signals.co2_price_col in forecasts.columns:
            co2_price = forecasts[signals.co2_price_col].astype(float).to_numpy()
        else:
            co2_price = [0.0 for _ in steps]

        da_position = signals.da_position_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        idc_buy = signals.idc_buy_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        idc_sell = signals.idc_sell_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        final_planned = (
            signals.final_planned_electricity_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        )
        afrr_price = signals.afrr_energy_price.astype(float).reindex(forecasts.index).fillna(0.0)
        afrr_bid = signals.afrr_energy_bid_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        afrr_activation = (
            signals.afrr_energy_activated_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        )
        reserved_capacity_mwh = _series_or_zero(signals.reserved_capacity_mwh, forecasts.index)

        m.heat_demand = pyo.Param(m.T, initialize={t: heat_demand_mwh[t] for t in steps})
        m.da_price = pyo.Param(m.T, initialize={t: da_price[t] for t in steps})
        m.idc_price = pyo.Param(m.T, initialize={t: idc_price[t] for t in steps})
        m.afrr_energy_price = pyo.Param(
            m.T, initialize={t: float(afrr_price.iloc[t]) for t in steps}
        )
        m.additional_electricity_charge = pyo.Param(initialize=additional_charge)
        m.electricity_price = pyo.Param(
            m.T, initialize={t: float(afrr_price.iloc[t]) + additional_charge for t in steps}
        )
        m.gas_price = pyo.Param(m.T, initialize={t: gas_price[t] for t in steps})
        m.co2_price = pyo.Param(m.T, initialize={t: co2_price[t] for t in steps})
        m.da_position_mwh = pyo.Param(
            m.T, initialize={t: float(da_position.iloc[t]) for t in steps}
        )
        m.idc_buy_mwh = pyo.Param(m.T, initialize={t: float(idc_buy.iloc[t]) for t in steps})
        m.idc_sell_mwh = pyo.Param(m.T, initialize={t: float(idc_sell.iloc[t]) for t in steps})
        m.final_planned_electricity_mwh = pyo.Param(
            m.T, initialize={t: float(final_planned.iloc[t]) for t in steps}
        )
        m.afrr_energy_bid_mwh = pyo.Param(
            m.T, initialize={t: float(afrr_bid.iloc[t]) for t in steps}
        )
        m.afrr_energy_activated_mwh = pyo.Param(
            m.T, initialize={t: float(afrr_activation.iloc[t]) for t in steps}
        )
        m.reserved_capacity_mwh = pyo.Param(
            m.T, initialize={t: float(reserved_capacity_mwh.iloc[t]) for t in steps}
        )
        m.co2_emission_factor = pyo.Param(
            initialize=float(signals.co2_emission_factor_t_per_mwh_fuel)
        )

        m.technology_blocks = pyo.Block(list(self.components.keys()))
        for technology, component in self.components.items():
            context = {
                "dt_hours": dt_hours,
                "initial_soc_mwh": initial_soc_mwh if technology == "thermal_storage" else None,
            }
            component.add_to_model(m, m.technology_blocks[technology], m.T, context)

        m.electricity_consumption = pyo.Var(m.T, within=pyo.NonNegativeReals)

        @m.Expression(m.T)
        def actual_electricity_consumption_mwh(
            mm: pyo.ConcreteModel,
            t: int,
        ) -> pyo.Expression:
            return mm.final_planned_electricity_mwh[t] + mm.afrr_energy_activated_mwh[t]

        @m.Constraint(m.T)
        def etes_charge_matches_actual_electricity(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            # For the current hybrid ETES + gas plant, activated aFRR down energy maps
            # directly to additional ETES charging. TODO: Generalise this for industrial
            # plants with multiple electric processes behind one market position.
            return storage.electric_charge_to_storage[t] == mm.actual_electricity_consumption_mwh[t]

        @m.Constraint(m.T)
        def electricity_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            return mm.electricity_consumption[t] == storage.electricity_consumption[t]

        @m.Constraint(m.T)
        def heat_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            boiler = mm.technology_blocks["boiler"]
            return storage.discharge_heat[t] + boiler.heat_out[t] == mm.heat_demand[t]

        @m.Expression(m.T)
        def da_electricity_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_position_mwh[t] * mm.da_price[t]

        @m.Expression(m.T)
        def idc_buy_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.idc_buy_mwh[t] * mm.idc_price[t]

        @m.Expression(m.T)
        def idc_sell_revenue(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.idc_sell_mwh[t] * mm.idc_price[t]

        @m.Expression(m.T)
        def afrr_energy_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.afrr_energy_activated_mwh[t] * mm.afrr_energy_price[t]

        @m.Expression(m.T)
        def electricity_market_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (
                mm.da_electricity_cost[t]
                + mm.idc_buy_cost[t]
                - mm.idc_sell_revenue[t]
                + mm.afrr_energy_cost[t]
            )

        @m.Expression(m.T)
        def additional_electricity_charges_cost(
            mm: pyo.ConcreteModel,
            t: int,
        ) -> pyo.Expression:
            return mm.electricity_consumption[t] * mm.additional_electricity_charge

        @m.Expression(m.T)
        def electricity_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.electricity_market_cost[t] + mm.additional_electricity_charges_cost[t]

        @m.Expression(m.T)
        def gas_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.technology_blocks["boiler"].operating_cost[t]

        @m.Expression(m.T)
        def co2_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.technology_blocks["boiler"].co2_cost[t]

        @m.Objective(sense=pyo.minimize)
        def objective(mm: pyo.ConcreteModel) -> pyo.Expression:
            return pyo.quicksum(
                # TODO: Add CO2 cost consistently to the gas benchmark and plant objective.
                mm.electricity_cost[t] + mm.gas_cost[t]
                for t in mm.T
            )

        return m

    def _extract_results(
        self,
        model: pyo.ConcreteModel,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: DispatchSignals,
        solver_name: str,
    ) -> pd.DataFrame:
        storage = model.technology_blocks["thermal_storage"]
        boiler = model.technology_blocks["boiler"]
        dt_hours = config.timestep_minutes / 60.0

        rows = []
        for t, timestamp in enumerate(forecasts.index):
            electricity_cost = _value(model.electricity_cost[t])
            electricity_market_cost = _value(model.electricity_market_cost[t])
            additional_charges_cost = _value(model.additional_electricity_charges_cost[t])
            gas_cost = _value(model.gas_cost[t])
            co2_cost = _value(model.co2_cost[t])
            additional_charge = float(signals.additional_electricity_charge_eur_per_mwh)
            day_ahead_price = float(forecasts[signals.electricity_price_col].iloc[t])
            co2_price = (
                float(forecasts[signals.co2_price_col].iloc[t])
                if signals.co2_price_col and signals.co2_price_col in forecasts.columns
                else 0.0
            )
            row = {
                "datetime": timestamp,
                "plant_name": self.name,
                "heat_demand_MWh": float(forecasts[self.heat_demand_column].iloc[t]) * dt_hours,
                "day_ahead_price_EUR_per_MWh": day_ahead_price,
                "additional_electricity_charge_EUR_per_MWh_el": additional_charge,
                "day_ahead_delivered_price_EUR_per_MWh": day_ahead_price + additional_charge,
                "gas_price_EUR_per_MWh": float(forecasts[signals.gas_price_col].iloc[t]),
                "co2_price_EUR_per_t": co2_price,
                "day_ahead_price_signal": signals.electricity_price_col,
                "gas_price_signal": signals.gas_price_col,
                "co2_price_signal": signals.co2_price_col or "",
                "gas_based_heat_benchmark_EUR_per_MWh_th": float(
                    signals.gas_benchmark_eur_per_mwh_th.iloc[t]
                ),
                "etes_charge_MWh": _value(storage.electric_charge_to_storage[t]),
                "etes_discharge_MWh": _value(storage.discharge_heat[t]),
                "etes_soc_MWh": _value(storage.soc[t]),
                "gas_heat_MWh": _value(boiler.heat_out[t]),
                "gas_input_MWh": _value(boiler.fuel_input[t]),
                "electricity_consumption_MWh": _value(model.electricity_consumption[t]),
                "DA_position_MWh": _value(model.electricity_consumption[t]),
                "IDC_buy_MWh": 0.0,
                "IDC_sell_MWh": 0.0,
                "IDC_price_EUR_per_MWh": float("nan"),
                "final_planned_electricity_MWh": _value(model.electricity_consumption[t]),
                "actual_electricity_consumption_MWh": _value(model.electricity_consumption[t]),
                "DA_electricity_cost_EUR": electricity_market_cost,
                "IDC_buy_cost_EUR": 0.0,
                "IDC_sell_revenue_EUR": 0.0,
                "afrr_energy_bid_MWh": 0.0,
                "afrr_energy_bid_MW": 0.0,
                "afrr_energy_activated_MWh": 0.0,
                "afrr_energy_price_EUR_per_MWh": float("nan"),
                "afrr_system_activation_MWh": 0.0,
                "afrr_energy_bid_price_EUR_per_MWh": float("nan"),
                "afrr_energy_market_spread_EUR_per_MWh": 0.0,
                "afrr_energy_net_spread_EUR_per_MWh": 0.0,
                "afrr_energy_cost_EUR": 0.0,
                "afrr_energy_savings_vs_benchmark_EUR": 0.0,
                "afrr_energy_pay_as_cleared_reward_EUR": 0.0,
                "afrr_energy_net_value_after_charges_EUR": 0.0,
                "afrr_energy_capacity_backed_bid_MWh": 0.0,
                "afrr_energy_free_bid_MWh": 0.0,
                "afrr_energy_capacity_backed_activated_MWh": 0.0,
                "afrr_energy_free_activated_MWh": 0.0,
                "electricity_market_cost_EUR": electricity_market_cost,
                "additional_electricity_charges_cost_EUR": additional_charges_cost,
                "electricity_cost_EUR": electricity_cost,
                "gas_cost_EUR": gas_cost,
                "co2_cost_EUR": co2_cost,
                "operating_cost_EUR": electricity_cost + gas_cost,
                "charge_allowed_by_strategy": bool(signals.charge_allowed.iloc[t]),
                "solver": solver_name,
            }
            row.update(
                _capacity_result_fields(
                    signals=signals,
                    timestamp=timestamp,
                    position=t,
                    timestep_hours=dt_hours,
                    final_planned_mwh=row["final_planned_electricity_MWh"],
                    soc_mwh=row["etes_soc_MWh"],
                    etes=self.etes,
                )
            )
            row["gross_operating_cost_EUR"] = row["operating_cost_EUR"]
            row["net_operating_cost_EUR"] = (
                row["gross_operating_cost_EUR"] - row["afrr_capacity_revenue_EUR"]
            )
            rows.append(row)

        frame = pd.DataFrame(rows).set_index("datetime")
        numeric_columns = frame.select_dtypes(include=["number"]).columns
        frame[numeric_columns] = frame[numeric_columns].mask(
            frame[numeric_columns].abs() < 1e-9, 0.0
        )
        return frame

    def _extract_intraday_adjustment_results(
        self,
        model: pyo.ConcreteModel,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: IDCAdjustmentSignals,
        solver_name: str,
    ) -> pd.DataFrame:
        storage = model.technology_blocks["thermal_storage"]
        boiler = model.technology_blocks["boiler"]
        dt_hours = config.timestep_minutes / 60.0

        rows = []
        for t, timestamp in enumerate(forecasts.index):
            electricity_cost = _value(model.electricity_cost[t])
            electricity_market_cost = _value(model.electricity_market_cost[t])
            additional_charges_cost = _value(model.additional_electricity_charges_cost[t])
            gas_cost = _value(model.gas_cost[t])
            co2_cost = _value(model.co2_cost[t])
            da_position = _value(model.da_position_mwh[t])
            idc_buy = _value(model.idc_buy_mwh[t])
            idc_sell = _value(model.idc_sell_mwh[t])
            final_planned = _value(model.final_planned_electricity_mwh[t])
            additional_charge = float(signals.additional_electricity_charge_eur_per_mwh)
            day_ahead_price = float(forecasts[signals.da_price_col].iloc[t])
            idc_price = float(forecasts[signals.idc_price_col].iloc[t])
            co2_price = (
                float(forecasts[signals.co2_price_col].iloc[t])
                if signals.co2_price_col and signals.co2_price_col in forecasts.columns
                else 0.0
            )
            row = {
                "datetime": timestamp,
                "plant_name": self.name,
                "heat_demand_MWh": float(forecasts[self.heat_demand_column].iloc[t]) * dt_hours,
                "day_ahead_price_EUR_per_MWh": day_ahead_price,
                "IDC_price_EUR_per_MWh": idc_price,
                "additional_electricity_charge_EUR_per_MWh_el": additional_charge,
                "day_ahead_delivered_price_EUR_per_MWh": day_ahead_price + additional_charge,
                "IDC_delivered_price_EUR_per_MWh": idc_price + additional_charge,
                "gas_price_EUR_per_MWh": float(forecasts[signals.gas_price_col].iloc[t]),
                "co2_price_EUR_per_t": co2_price,
                "day_ahead_price_signal": signals.da_price_col,
                "IDC_price_signal": signals.idc_price_col,
                "gas_price_signal": signals.gas_price_col,
                "co2_price_signal": signals.co2_price_col or "",
                "gas_based_heat_benchmark_EUR_per_MWh_th": float(
                    signals.gas_benchmark_eur_per_mwh_th.iloc[t]
                ),
                "electricity_trading_benchmark_EUR_per_MWh_el": float(
                    signals.electricity_trading_benchmark_eur_per_mwh_el.iloc[t]
                ),
                "etes_charge_MWh": _value(storage.electric_charge_to_storage[t]),
                "etes_discharge_MWh": _value(storage.discharge_heat[t]),
                "etes_soc_MWh": _value(storage.soc[t]),
                "gas_heat_MWh": _value(boiler.heat_out[t]),
                "gas_input_MWh": _value(boiler.fuel_input[t]),
                "electricity_consumption_MWh": _value(model.electricity_consumption[t]),
                "DA_position_MWh": da_position,
                "IDC_buy_MWh": idc_buy,
                "IDC_sell_MWh": idc_sell,
                "final_planned_electricity_MWh": final_planned,
                "actual_electricity_consumption_MWh": _value(model.electricity_consumption[t]),
                "DA_electricity_cost_EUR": _value(model.da_electricity_cost[t]),
                "IDC_buy_cost_EUR": _value(model.idc_buy_cost[t]),
                "IDC_sell_revenue_EUR": _value(model.idc_sell_revenue[t]),
                "afrr_energy_bid_MWh": 0.0,
                "afrr_energy_bid_MW": 0.0,
                "afrr_energy_activated_MWh": 0.0,
                "afrr_energy_price_EUR_per_MWh": float("nan"),
                "afrr_system_activation_MWh": 0.0,
                "afrr_energy_bid_price_EUR_per_MWh": float("nan"),
                "afrr_energy_market_spread_EUR_per_MWh": 0.0,
                "afrr_energy_net_spread_EUR_per_MWh": 0.0,
                "afrr_energy_cost_EUR": 0.0,
                "afrr_energy_savings_vs_benchmark_EUR": 0.0,
                "afrr_energy_pay_as_cleared_reward_EUR": 0.0,
                "afrr_energy_net_value_after_charges_EUR": 0.0,
                "afrr_energy_capacity_backed_bid_MWh": 0.0,
                "afrr_energy_free_bid_MWh": 0.0,
                "afrr_energy_capacity_backed_activated_MWh": 0.0,
                "afrr_energy_free_activated_MWh": 0.0,
                "electricity_market_cost_EUR": electricity_market_cost,
                "additional_electricity_charges_cost_EUR": additional_charges_cost,
                "electricity_cost_EUR": electricity_cost,
                "gas_cost_EUR": gas_cost,
                "co2_cost_EUR": co2_cost,
                "operating_cost_EUR": electricity_cost + gas_cost,
                "charge_allowed_by_strategy": bool(signals.idc_buy_upper_bound_mwh.iloc[t] > 1e-12),
                "idc_buy_allowed_by_strategy": bool(
                    signals.idc_buy_upper_bound_mwh.iloc[t] > 1e-12
                ),
                "idc_sell_allowed_by_strategy": bool(
                    signals.idc_sell_upper_bound_mwh.iloc[t] > 1e-12
                ),
                "solver": solver_name,
            }
            row.update(
                _capacity_result_fields(
                    signals=signals,
                    timestamp=timestamp,
                    position=t,
                    timestep_hours=dt_hours,
                    final_planned_mwh=final_planned,
                    soc_mwh=row["etes_soc_MWh"],
                    etes=self.etes,
                )
            )
            row["gross_operating_cost_EUR"] = row["operating_cost_EUR"]
            row["net_operating_cost_EUR"] = (
                row["gross_operating_cost_EUR"] - row["afrr_capacity_revenue_EUR"]
            )
            rows.append(row)

        frame = pd.DataFrame(rows).set_index("datetime")
        numeric_columns = frame.select_dtypes(include=["number"]).columns
        frame[numeric_columns] = frame[numeric_columns].mask(
            frame[numeric_columns].abs() < 1e-9, 0.0
        )
        return frame

    def _extract_afrr_down_results(
        self,
        model: pyo.ConcreteModel,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: AFRRDownSignals,
        solver_name: str,
    ) -> pd.DataFrame:
        storage = model.technology_blocks["thermal_storage"]
        boiler = model.technology_blocks["boiler"]
        dt_hours = config.timestep_minutes / 60.0

        rows = []
        for t, timestamp in enumerate(forecasts.index):
            electricity_cost = _value(model.electricity_cost[t])
            electricity_market_cost = _value(model.electricity_market_cost[t])
            additional_charges_cost = _value(model.additional_electricity_charges_cost[t])
            gas_cost = _value(model.gas_cost[t])
            co2_cost = _value(model.co2_cost[t])
            final_planned = _value(model.final_planned_electricity_mwh[t])
            afrr_bid = _value(model.afrr_energy_bid_mwh[t])
            afrr_activation = _value(model.afrr_energy_activated_mwh[t])
            actual_electricity = _value(model.electricity_consumption[t])
            afrr_price_clean = _value(model.afrr_energy_price[t])
            additional_charge = float(signals.additional_electricity_charge_eur_per_mwh)
            benchmark = float(signals.electricity_trading_benchmark_eur_per_mwh_el.iloc[t])
            afrr_bid_price = float(
                _series_value(signals.afrr_energy_bid_price, timestamp, benchmark)
            )
            afrr_delivered_price = afrr_price_clean + additional_charge
            afrr_market_spread = afrr_bid_price - afrr_price_clean
            afrr_net_spread = afrr_bid_price - afrr_delivered_price
            afrr_pay_as_cleared_reward = afrr_activation * afrr_market_spread
            afrr_net_value = afrr_activation * afrr_net_spread
            day_ahead_price = float(forecasts[signals.da_price_col].iloc[t])
            idc_price = float(forecasts[signals.idc_price_col].iloc[t])
            co2_price = (
                float(forecasts[signals.co2_price_col].iloc[t])
                if signals.co2_price_col and signals.co2_price_col in forecasts.columns
                else 0.0
            )
            row = {
                "datetime": timestamp,
                "plant_name": self.name,
                "heat_demand_MWh": float(forecasts[self.heat_demand_column].iloc[t]) * dt_hours,
                "day_ahead_price_EUR_per_MWh": day_ahead_price,
                "IDC_price_EUR_per_MWh": idc_price,
                "additional_electricity_charge_EUR_per_MWh_el": additional_charge,
                "day_ahead_delivered_price_EUR_per_MWh": day_ahead_price + additional_charge,
                "IDC_delivered_price_EUR_per_MWh": idc_price + additional_charge,
                "afrr_energy_delivered_price_EUR_per_MWh": afrr_delivered_price,
                "gas_price_EUR_per_MWh": float(forecasts[signals.gas_price_col].iloc[t]),
                "co2_price_EUR_per_t": co2_price,
                "day_ahead_price_signal": signals.da_price_col,
                "IDC_price_signal": signals.idc_price_col,
                "gas_price_signal": signals.gas_price_col,
                "co2_price_signal": signals.co2_price_col or "",
                "gas_based_heat_benchmark_EUR_per_MWh_th": float(
                    signals.gas_benchmark_eur_per_mwh_th.iloc[t]
                ),
                "electricity_trading_benchmark_EUR_per_MWh_el": benchmark,
                "etes_charge_MWh": _value(storage.electric_charge_to_storage[t]),
                "etes_discharge_MWh": _value(storage.discharge_heat[t]),
                "etes_soc_MWh": _value(storage.soc[t]),
                "gas_heat_MWh": _value(boiler.heat_out[t]),
                "gas_input_MWh": _value(boiler.fuel_input[t]),
                "electricity_consumption_MWh": _value(model.electricity_consumption[t]),
                "DA_position_MWh": _value(model.da_position_mwh[t]),
                "IDC_buy_MWh": _value(model.idc_buy_mwh[t]),
                "IDC_sell_MWh": _value(model.idc_sell_mwh[t]),
                "final_planned_electricity_MWh": final_planned,
                "actual_electricity_consumption_MWh": actual_electricity,
                "DA_electricity_cost_EUR": _value(model.da_electricity_cost[t]),
                "IDC_buy_cost_EUR": _value(model.idc_buy_cost[t]),
                "IDC_sell_revenue_EUR": _value(model.idc_sell_revenue[t]),
                "afrr_energy_bid_MWh": afrr_bid,
                "afrr_energy_bid_MW": afrr_bid / dt_hours if dt_hours > 0 else 0.0,
                "afrr_energy_activated_MWh": afrr_activation,
                "afrr_energy_price_EUR_per_MWh": afrr_price_clean,
                "afrr_system_activation_MWh": float(signals.afrr_system_activation_mwh.iloc[t]),
                "afrr_energy_bid_price_EUR_per_MWh": afrr_bid_price,
                "afrr_energy_market_spread_EUR_per_MWh": afrr_market_spread,
                "afrr_energy_net_spread_EUR_per_MWh": afrr_net_spread,
                "afrr_energy_cost_EUR": _value(model.afrr_energy_cost[t]),
                "afrr_energy_savings_vs_benchmark_EUR": afrr_net_value,
                "afrr_energy_pay_as_cleared_reward_EUR": afrr_pay_as_cleared_reward,
                "afrr_energy_net_value_after_charges_EUR": afrr_net_value,
                "afrr_energy_capacity_backed_bid_MWh": float(
                    _series_value(
                        signals.afrr_energy_capacity_backed_bid_mwh,
                        timestamp,
                        0.0,
                    )
                ),
                "afrr_energy_free_bid_MWh": float(
                    _series_value(signals.afrr_energy_free_bid_mwh, timestamp, 0.0)
                ),
                "afrr_energy_capacity_backed_activated_MWh": float(
                    _series_value(
                        signals.afrr_energy_capacity_backed_activated_mwh,
                        timestamp,
                        0.0,
                    )
                ),
                "afrr_energy_free_activated_MWh": float(
                    _series_value(signals.afrr_energy_free_activated_mwh, timestamp, 0.0)
                ),
                "afrr_headroom_binding": bool(
                    _series_value(signals.afrr_headroom_binding, timestamp, False)
                ),
                "afrr_curtailment_MWh": float(
                    _series_value(signals.afrr_curtailment_mwh, timestamp, 0.0)
                ),
                "electricity_market_cost_EUR": electricity_market_cost,
                "additional_electricity_charges_cost_EUR": additional_charges_cost,
                "electricity_cost_EUR": electricity_cost,
                "gas_cost_EUR": gas_cost,
                "co2_cost_EUR": co2_cost,
                "operating_cost_EUR": electricity_cost + gas_cost,
                "charge_allowed_by_strategy": False,
                "idc_buy_allowed_by_strategy": False,
                "idc_sell_allowed_by_strategy": False,
                "afrr_energy_bid_allowed_by_strategy": bool(afrr_bid > 1e-12),
                "solver": solver_name,
            }
            row.update(
                _capacity_result_fields(
                    signals=signals,
                    timestamp=timestamp,
                    position=t,
                    timestep_hours=dt_hours,
                    final_planned_mwh=final_planned,
                    soc_mwh=row["etes_soc_MWh"],
                    etes=self.etes,
                )
            )
            row["gross_operating_cost_EUR"] = row["operating_cost_EUR"]
            row["net_operating_cost_EUR"] = (
                row["gross_operating_cost_EUR"] - row["afrr_capacity_revenue_EUR"]
            )
            rows.append(row)

        frame = pd.DataFrame(rows).set_index("datetime")
        numeric_columns = frame.select_dtypes(include=["number"]).columns
        frame[numeric_columns] = frame[numeric_columns].mask(
            frame[numeric_columns].abs() < 1e-9, 0.0
        )
        return frame

    @staticmethod
    def _available_solvers(config: CaseConfig) -> list[tuple[str, Any]]:
        configured = [config.solver_name, *config.solver_fallbacks]
        expanded: list[str] = []
        for candidate in configured:
            if candidate == "highs":
                expanded.extend(["highs", "appsi_highs"])
            else:
                expanded.append(candidate)

        seen: set[str] = set()
        solvers: list[tuple[str, Any]] = []
        for candidate in expanded:
            if candidate in seen:
                continue
            seen.add(candidate)
            if not _solver_precheck(candidate):
                continue
            solver = pyo.SolverFactory(candidate)
            if solver.available(exception_flag=False):
                solvers.append((candidate, solver))

        if not solvers:
            raise RuntimeError(
                "No configured Pyomo solver is available. Tried: " + ", ".join(expanded)
            )
        return solvers


def _solver_precheck(candidate: str) -> bool:
    if candidate in {"highs", "appsi_highs"}:
        return importlib.util.find_spec("highspy") is not None or shutil.which("highs") is not None
    if candidate == "glpk":
        return shutil.which("glpsol") is not None
    if candidate == "cbc":
        return shutil.which("cbc") is not None
    return True


def _optional_loc(series: pd.Series | None, index: pd.Index) -> pd.Series | None:
    return None if series is None else series.loc[index]


def _series_or_zero(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index)
    return series.astype(float).reindex(index).fillna(0.0)


def _series_value(series: pd.Series | None, timestamp: pd.Timestamp, default: object) -> object:
    if series is None:
        return default
    try:
        value = series.loc[timestamp]
    except KeyError:
        return default
    if pd.isna(value):
        return default
    return value


def _capacity_result_fields(
    signals: DispatchSignals | IDCAdjustmentSignals | AFRRDownSignals,
    timestamp: pd.Timestamp,
    position: int,
    timestep_hours: float,
    final_planned_mwh: float,
    soc_mwh: float,
    etes: ThermalStorage,
) -> dict[str, object]:
    reserved_mwh = float(_series_value(signals.reserved_capacity_mwh, timestamp, 0.0))
    if signals.afrr_capacity_reserved_mw is not None:
        reserved_mw = float(_series_value(signals.afrr_capacity_reserved_mw, timestamp, 0.0))
    else:
        reserved_mw = reserved_mwh / timestep_hours if timestep_hours > 0 else 0.0
    block_id = str(_series_value(signals.afrr_capacity_block_id, timestamp, ""))
    block_duration = float(_series_value(signals.afrr_capacity_block_duration_h, timestamp, 0.0))
    price = float(_series_value(signals.afrr_capacity_price_eur_per_mw_h, timestamp, 0.0))
    revenue = float(_series_value(signals.afrr_capacity_revenue_eur, timestamp, 0.0))
    max_charge_mwh = etes.max_power_charge_mw * timestep_hours
    charge_headroom = max(0.0, max_charge_mwh - float(final_planned_mwh))
    storage_headroom = max(0.0, etes.max_capacity_mwh - float(soc_mwh)) / etes.efficiency_charge
    return {
        "afrr_capacity_block_id": block_id,
        "afrr_capacity_block_duration_h": block_duration,
        "afrr_capacity_down_price_EUR_per_MW_h": price,
        "afrr_capacity_reserved_MW": reserved_mw,
        "afrr_capacity_reserved_MWh": reserved_mwh,
        "afrr_capacity_revenue_EUR": revenue,
        "reserved_capacity_headroom_MWh": reserved_mwh,
        "available_charge_headroom_after_schedule_MWh": charge_headroom,
        "available_storage_headroom_after_schedule_MWh": storage_headroom,
    }


def _is_infeasible_termination(termination: TerminationCondition) -> bool:
    return termination in {
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.provenInfeasible,
    }


def _strict_heat_infeasibility_message(stage: str, solver_name: str) -> str:
    return (
        f"{stage} solve is infeasible with solver '{solver_name}'. FLEXIMOD now enforces "
        "strict useful heat dispatch: gas heat plus storage discharge must equal heat demand "
        "in every timestep, with no unmet-heat or heat-dump slack. Check fixed market "
        "electricity positions, aFRR activation, ETES storage headroom, and heat demand."
    )


def _value(expression: pyo.Expression) -> float:
    value = pyo.value(expression)
    if value is None:
        return 0.0
    return float(value)


def _series_float_or_nan(series: pd.Series, position: int) -> float:
    value = series.iloc[position]
    if pd.isna(value):
        return float("nan")
    return float(value)
