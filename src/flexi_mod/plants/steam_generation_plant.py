# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.opt import SolverStatus, TerminationCondition

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants.base_plant import DEFAULT_GAS_EMISSIONS_FACTOR_KG_PER_MWH, BasePlant
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
    future_position_mwh: pd.Series
    da_sell_mwh: pd.Series
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
    da_sell_mwh: pd.Series
    idc_buy_mwh: pd.Series
    idc_sell_da_mwh: pd.Series
    idc_sell_future_mwh: pd.Series
    future_position_mwh: pd.Series
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

        gas_emissions_factor = first_non_empty(
            rows, "gas_emissions_factor_kg_per_mwh", default=DEFAULT_GAS_EMISSIONS_FACTOR_KG_PER_MWH
        )
        if gas_emissions_factor is not None:
            try:
                gas_emissions_factor = float(gas_emissions_factor)
            except (ValueError, TypeError):
                gas_emissions_factor = DEFAULT_GAS_EMISSIONS_FACTOR_KG_PER_MWH

        return cls(
            name=plant_name,
            unit_type=first_non_empty(rows, "unit_type", default="steam_plant"),
            node=first_non_empty(rows, "node", default=""),
            objective=first_non_empty(rows, "objective", default="min_variable_cost"),
            heat_demand_column=heat_demand_column,
            components=components,
            gas_emissions_factor_kg_per_mwh=gas_emissions_factor,
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
                afrr_capacity_pricing_rule=_optional_loc(
                    signals.afrr_capacity_pricing_rule, horizon.index
                ),
                afrr_capacity_bid_price_eur_per_mw_h=_optional_loc(
                    signals.afrr_capacity_bid_price_eur_per_mw_h, horizon.index
                ),
                afrr_capacity_settlement_price_eur_per_mw_h=_optional_loc(
                    signals.afrr_capacity_settlement_price_eur_per_mw_h, horizon.index
                ),
                afrr_capacity_reserved_mw=_optional_loc(
                    signals.afrr_capacity_reserved_mw, horizon.index
                ),
                afrr_capacity_revenue_eur=_optional_loc(
                    signals.afrr_capacity_revenue_eur, horizon.index
                ),
                afrr_capacity_opportunity_cost_eur=_optional_loc(
                    signals.afrr_capacity_opportunity_cost_eur, horizon.index
                ),
                afrr_capacity_market_surplus_eur=_optional_loc(
                    signals.afrr_capacity_market_surplus_eur, horizon.index
                ),
                afrr_capacity_net_value_eur=_optional_loc(
                    signals.afrr_capacity_net_value_eur, horizon.index
                ),
                co2_emission_factor_t_per_mwh_fuel=signals.co2_emission_factor_t_per_mwh_fuel,
                tax_rate=signals.tax_rate,
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

            # Debug: SOC-Übergabe gegen Bilanz prüfen
            soc_calc = float(initial_soc)
            for _, r in implemented.iterrows():
                soc_calc = (
                    soc_calc * (1.0 - float(self.etes.storage_loss_rate))
                    + float(r["etes_charge_MWh"]) * float(self.etes.efficiency_charge)
                    - float(r["etes_discharge_MWh"]) / float(self.etes.efficiency_discharge)
                )
            soc_reported = float(implemented["etes_soc_MWh"].iloc[-1])
            if abs(soc_calc - soc_reported) > 1e-5:
                print(
                    f"[SOC-MISMATCH] plant={self.name} pos={position} "
                    f"calc={soc_calc:.6f} reported={soc_reported:.6f}"
                )

            initial_soc = soc_reported
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
                da_sell_mwh=signals.da_sell_mwh.loc[horizon.index],
                future_position_mwh=signals.future_position_mwh.loc[horizon.index],
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
                afrr_capacity_pricing_rule=_optional_loc(
                    signals.afrr_capacity_pricing_rule, horizon.index
                ),
                afrr_capacity_bid_price_eur_per_mw_h=_optional_loc(
                    signals.afrr_capacity_bid_price_eur_per_mw_h, horizon.index
                ),
                afrr_capacity_settlement_price_eur_per_mw_h=_optional_loc(
                    signals.afrr_capacity_settlement_price_eur_per_mw_h, horizon.index
                ),
                afrr_capacity_reserved_mw=_optional_loc(
                    signals.afrr_capacity_reserved_mw, horizon.index
                ),
                afrr_capacity_revenue_eur=_optional_loc(
                    signals.afrr_capacity_revenue_eur, horizon.index
                ),
                afrr_capacity_opportunity_cost_eur=_optional_loc(
                    signals.afrr_capacity_opportunity_cost_eur, horizon.index
                ),
                afrr_capacity_market_surplus_eur=_optional_loc(
                    signals.afrr_capacity_market_surplus_eur, horizon.index
                ),
                afrr_capacity_net_value_eur=_optional_loc(
                    signals.afrr_capacity_net_value_eur, horizon.index
                ),
                co2_emission_factor_t_per_mwh_fuel=signals.co2_emission_factor_t_per_mwh_fuel,
                tax_rate=signals.tax_rate,
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
                da_sell_mwh=signals.da_sell_mwh.loc[horizon.index],
                idc_buy_mwh=signals.idc_buy_mwh.loc[horizon.index],
                idc_sell_da_mwh=signals.idc_sell_da_mwh.loc[horizon.index],
                idc_sell_future_mwh=signals.idc_sell_future_mwh.loc[horizon.index],
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
                afrr_capacity_pricing_rule=_optional_loc(
                    signals.afrr_capacity_pricing_rule, horizon.index
                ),
                afrr_capacity_bid_price_eur_per_mw_h=_optional_loc(
                    signals.afrr_capacity_bid_price_eur_per_mw_h, horizon.index
                ),
                afrr_capacity_settlement_price_eur_per_mw_h=_optional_loc(
                    signals.afrr_capacity_settlement_price_eur_per_mw_h, horizon.index
                ),
                afrr_capacity_reserved_mw=_optional_loc(
                    signals.afrr_capacity_reserved_mw, horizon.index
                ),
                afrr_capacity_revenue_eur=_optional_loc(
                    signals.afrr_capacity_revenue_eur, horizon.index
                ),
                afrr_capacity_opportunity_cost_eur=_optional_loc(
                    signals.afrr_capacity_opportunity_cost_eur, horizon.index
                ),
                afrr_capacity_market_surplus_eur=_optional_loc(
                    signals.afrr_capacity_market_surplus_eur, horizon.index
                ),
                afrr_capacity_net_value_eur=_optional_loc(
                    signals.afrr_capacity_net_value_eur, horizon.index
                ),
                co2_emission_factor_t_per_mwh_fuel=signals.co2_emission_factor_t_per_mwh_fuel,
                tax_rate=signals.tax_rate,
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

        tax_rate = float(signals.tax_rate)
        heat_demand_mwh = forecasts[self.heat_demand_column].astype(float).to_numpy() * dt_hours
        market_electricity_price = forecasts[signals.electricity_price_col].astype(float).to_numpy()
        additional_charge = (
            signals.additional_electricity_charge_eur_per_mwh.astype(float).to_numpy()
            if signals.additional_electricity_charge_eur_per_mwh is not None
            else np.zeros(len(steps))
        )
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
        m.additional_electricity_charge = pyo.Param(
            m.T, initialize={t: additional_charge[t] for t in steps}
        )
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
        m.da_sell_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.da_buy_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.final_planned_electricity_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)

        @m.Expression(m.T)
        def future_positions(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            storage = mm.technology_blocks["thermal_storage"]
            return 0 #mm.heat_demand[t] / (storage.efficiency_discharge * storage.efficiency_charge)
        
        @m.Constraint(m.T)
        def da_sell_future_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.da_sell_mwh[t] <= mm.future_positions[t]

        @m.Constraint(m.T)
        def final_planned_position(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return (
                mm.final_planned_electricity_mwh[t]
                == mm.da_buy_mwh[t] + mm.future_positions[t] - mm.da_sell_mwh[t]
            )

        @m.Constraint(m.T)
        def etes_charge_matches_final_position(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            # For the current hybrid ETES + gas plant, the electricity market position maps
            # directly to ETES charging. TODO: Generalise this for industrial plants with
            # several electric processes sharing one market position.
            return storage.electric_charge_to_storage[t] == mm.final_planned_electricity_mwh[t]


        @m.Constraint(m.T)
        def heat_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            return storage.discharge_heat[t] == mm.heat_demand[t]

        
        @m.Constraint(m.T)
        def electricity_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            return mm.electricity_consumption[t] == storage.electricity_consumption[t]
        
        @m.Constraint(m.T)
        def heat_balance_boiler(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            boiler = mm.technology_blocks["boiler"]
            return boiler.heat_out[t] <= 0.0           

        

        @m.Expression(m.T)
        def additional_electricity_charges_cost(
            mm: pyo.ConcreteModel,
            t: int,
        ) -> pyo.Expression:
            return mm.electricity_consumption[t] * mm.additional_electricity_charge[t]
        
        @m.Expression(m.T)
        def da_positions_costs(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_buy_mwh[t] * mm.market_electricity_price[t]
        
        @m.Expression(m.T)
        def da_sell_revenue(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_sell_mwh[t] *( mm.market_electricity_price[t] - 1.0)
        
        @m.Expression(m.T)
        def electricity_market_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_positions_costs[t] - mm.da_sell_revenue[t]

        
        @m.Expression(m.T)
        def gas_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.future_positions[t]) * mm.gas_price[t]
        
        @m.Expression(m.T)
        def additional_costs_future(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.future_positions[t] - mm.da_sell_mwh[t]) * mm.additional_electricity_charge[t]
        
        @m.Expression(m.T)
        def additional_costs_da(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_buy_mwh[t] * mm.additional_electricity_charge[t]
        
        @m.Expression(m.T)
        def new_future_position(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.future_positions[t] - mm.da_sell_mwh[t])

        @m.Expression(m.T)
        def new_future_position_costs(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.new_future_position[t] * mm.gas_price[t]
        
        
        
        @m.Expression(m.T)
        def electricity_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.additional_electricity_charges_cost[t] 
                    + mm.da_positions_costs[t] 
                    - mm.da_sell_revenue[t])
        
        @m.Expression(m.T)
        def tax_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.electricity_cost[t]  + mm.gas_cost[t])* tax_rate
        
        @m.Objective(sense=pyo.minimize)
        def objective(mm: pyo.ConcreteModel) -> pyo.Expression:
            return pyo.quicksum(
                # CO2 cost is disabled for the first MVP and kept as a zero output column.
                mm.electricity_cost[t] + mm.tax_cost[t] + mm.gas_cost[t]
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

        tax_rate = float(signals.tax_rate)
        heat_demand_mwh = forecasts[self.heat_demand_column].astype(float).to_numpy() * dt_hours
        da_price = forecasts[signals.da_price_col].astype(float).to_numpy()
        idc_price = forecasts[signals.idc_price_col].astype(float).fillna(0.0).to_numpy()
        additional_charge = (
            signals.additional_electricity_charge_eur_per_mwh.astype(float).to_numpy()
            if signals.additional_electricity_charge_eur_per_mwh is not None
            else np.zeros(len(steps))
        )
        gas_price = forecasts[signals.gas_price_col].astype(float).to_numpy()
        if signals.co2_price_col and signals.co2_price_col in forecasts.columns:
            co2_price = forecasts[signals.co2_price_col].astype(float).to_numpy()
        else:
            co2_price = [0.0 for _ in steps]

        da_position = (
            signals.da_position_mwh.astype(float).reindex(forecasts.index).fillna(0.0).to_numpy()
        )

        da_sell = (
            signals.da_sell_mwh.astype(float).reindex(forecasts.index).fillna(0.0).to_numpy()
        )

        future_position = (
            signals.future_position_mwh.astype(float).reindex(forecasts.index).fillna(0.0).to_numpy()
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
        m.additional_electricity_charge = pyo.Param(
            m.T, initialize={t: additional_charge[t] for t in steps}
        )
        m.electricity_price = pyo.Param(
            m.T,
            initialize={t: idc_price[t] + additional_charge[t] for t in steps},
        )
        m.gas_price = pyo.Param(m.T, initialize={t: gas_price[t] for t in steps})
        m.co2_price = pyo.Param(m.T, initialize={t: co2_price[t] for t in steps})
        m.da_position_mwh = pyo.Param(m.T, initialize={t: da_position[t] for t in steps})
        m.da_sell_mwh = pyo.Param(m.T, initialize={t: da_sell[t] for t in steps})
        m.future_position_mwh = pyo.Param(m.T, initialize={t: future_position[t] for t in steps})
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
        m.idc_sell_future_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.idc_sell_da_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.final_planned_electricity_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)

        
        @m.Constraint(m.T)
        def idc_buy_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.idc_buy_mwh[t] <= mm.idc_buy_upper_bound_mwh[t]

        @m.Constraint(m.T)
        def idc_sell_future_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.idc_sell_future_mwh[t] <= mm.future_position_mwh[t]

        @m.Constraint(m.T)
        def idc_sell_da_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.idc_sell_da_mwh[t] <= mm.da_position_mwh[t]

        @m.Constraint(m.T)
        def final_planned_position(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return (
                mm.final_planned_electricity_mwh[t]
                == mm.da_position_mwh[t] + mm.idc_buy_mwh[t] - mm.idc_sell_da_mwh[t] + mm.future_position_mwh[t] - mm.idc_sell_future_mwh[t]
            )
              

        @m.Constraint(m.T)
        def heat_balance_boiler(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            boiler = mm.technology_blocks["boiler"]
            return boiler.heat_out[t] <= 0.0   
        
        @m.Constraint(m.T)
        def etes_charge_matches_final_position(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            # For the current hybrid ETES + gas plant, the electricity market position maps
            # directly to ETES charging. TODO: Generalise this for industrial plants with
            # several electric processes sharing one market position.
            return storage.electric_charge_to_storage[t] == mm.final_planned_electricity_mwh[t]


        @m.Constraint(m.T)
        def heat_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            boiler = mm.technology_blocks["boiler"]
            return storage.discharge_heat[t] + boiler.heat_out[t] == mm.heat_demand[t]

        @m.Constraint(m.T)
        def electricity_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            return mm.electricity_consumption[t] == storage.electricity_consumption[t]


        @m.Expression(m.T)
        def da_electricity_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_position_mwh[t] * mm.da_price[t]

        @m.Expression(m.T)
        def idc_buy_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.idc_buy_mwh[t] * mm.idc_price[t]
        

        @m.Expression(m.T)
        def idc_sell_revenue(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.idc_sell_da_mwh[t] + mm.idc_sell_future_mwh[t]) * (mm.idc_price[t] - 1.0)
        
        @m.Expression(m.T)
        def da_sell_revenue(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_sell_mwh[t] * mm.da_price[t]

        @m.Expression(m.T)
        def electricity_market_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.da_electricity_cost[t] 
                    + mm.idc_buy_cost[t] 
                    - mm.idc_sell_revenue[t] 
                    - mm.da_sell_revenue[t]
            )

        @m.Expression(m.T)
        def additional_electricity_charges_cost(
            mm: pyo.ConcreteModel,
            t: int,
        ) -> pyo.Expression:
            return (mm.final_planned_electricity_mwh[t]) * mm.additional_electricity_charge[t]
        
        
        @m.Expression(m.T)
        def new_future_position(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.future_position_mwh[t] - mm.idc_sell_future_mwh[t]

        @m.Expression(m.T)
        def additional_future_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.new_future_position[t] * mm.additional_electricity_charge[t]
        
        @m.Expression(m.T)
        def electricity_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.electricity_market_cost[t] + mm.additional_electricity_charges_cost[t]

        @m.Expression(m.T)
        def gas_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.future_position_mwh[t] + mm.da_sell_mwh[t]) * mm.gas_price[t]

        @m.Expression(m.T)
        def tax_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.electricity_cost[t]  + mm.gas_cost[t])* tax_rate
        

        @m.Objective(sense=pyo.minimize)
        def objective(mm: pyo.ConcreteModel) -> pyo.Expression:
            return pyo.quicksum(
                # TODO: Add CO2 cost consistently to the gas benchmark and plant objective.
                mm.electricity_cost[t] + mm.gas_cost[t] + mm.tax_cost[t]
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

        tax_rate = float(signals.tax_rate)
        heat_demand_mwh = forecasts[self.heat_demand_column].astype(float).to_numpy() * dt_hours
        da_price = forecasts[signals.da_price_col].astype(float).to_numpy()
        idc_price = forecasts[signals.idc_price_col].astype(float).fillna(0.0).to_numpy()
        additional_charge = (
            signals.additional_electricity_charge_eur_per_mwh.astype(float).to_numpy()
            if signals.additional_electricity_charge_eur_per_mwh is not None
            else np.zeros(len(steps))
        )
        gas_price = forecasts[signals.gas_price_col].astype(float).to_numpy()
        if signals.co2_price_col and signals.co2_price_col in forecasts.columns:
            co2_price = forecasts[signals.co2_price_col].astype(float).to_numpy()
        else:
            co2_price = [0.0 for _ in steps]
        da_position = signals.da_position_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        da_sell = signals.da_sell_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        idc_buy = signals.idc_buy_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        idc_sell_da = signals.idc_sell_da_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        idc_sell_future = signals.idc_sell_future_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        future_position = signals.future_position_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        final_planned = (
            signals.final_planned_electricity_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        )
        afrr_price = signals.afrr_energy_price.astype(float).reindex(forecasts.index).fillna(0.0)
        afrr_bid = signals.afrr_energy_bid_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        afrr_activation = (
            signals.afrr_energy_activated_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        )
        affr_market_activation_limit_mwh = signals.afrr_system_activation_mwh.astype(float).reindex(forecasts.index).fillna(0.0)
        actual_electricity = final_planned + afrr_activation
        reserved_capacity_mwh = _series_or_zero(signals.reserved_capacity_mwh, forecasts.index)

        m.heat_demand = pyo.Param(m.T, initialize={t: heat_demand_mwh[t] for t in steps})
        m.da_price = pyo.Param(m.T, initialize={t: da_price[t] for t in steps})
        m.idc_price = pyo.Param(m.T, initialize={t: idc_price[t] for t in steps})
        m.afrr_energy_price = pyo.Param(
            m.T, initialize={t: float(afrr_price.iloc[t]) for t in steps}
        )
        m.additional_electricity_charge = pyo.Param(
            m.T, initialize={t: additional_charge[t] for t in steps}
        )
        m.electricity_price = pyo.Param(
            m.T, initialize={t: float(afrr_price.iloc[t]) + additional_charge[t] for t in steps}
        )
        m.gas_price = pyo.Param(m.T, initialize={t: gas_price[t] for t in steps})
        m.co2_price = pyo.Param(m.T, initialize={t: co2_price[t] for t in steps})
        m.da_position_mwh = pyo.Param(
            m.T, initialize={t: float(da_position.iloc[t]) for t in steps}
        )
        m.da_sell_mwh = pyo.Param(m.T, initialize={t: float(da_sell.iloc[t]) for t in steps})
        m.idc_buy_mwh = pyo.Param(m.T, initialize={t: float(idc_buy.iloc[t]) for t in steps})
        m.idc_sell_da_mwh = pyo.Param(m.T, initialize={t: float(idc_sell_da.iloc[t]) for t in steps})
        m.idc_sell_future_mwh = pyo.Param(m.T, initialize={t: float(idc_sell_future.iloc[t]) for t in steps})
        m.future_position_mwh = pyo.Param(
            m.T, initialize={t: float(future_position.iloc[t]) for t in steps}
        )
        m.afrr_energy_bid_mwh = pyo.Param(
            m.T, initialize={t: float(afrr_bid.iloc[t]) for t in steps}
        )
        m.actual_electricity_consumption_mwh = pyo.Param(
            m.T, initialize={t: float(actual_electricity.iloc[t]) for t in steps}
        )
        m.reserved_capacity_mwh = pyo.Param(
            m.T, initialize={t: float(reserved_capacity_mwh.iloc[t]) for t in steps}
        )
        m.co2_emission_factor = pyo.Param(
            initialize=float(signals.co2_emission_factor_t_per_mwh_fuel)
        )

        m.affr_market_activation_limit_mwh = pyo.Param(
            m.T, initialize={t: float(affr_market_activation_limit_mwh.iloc[t]) for t in steps}
        )

        m.technology_blocks = pyo.Block(list(self.components.keys()))
        for technology, component in self.components.items():
            context = {
                "dt_hours": dt_hours,
                "initial_soc_mwh": initial_soc_mwh if technology == "thermal_storage" else None,
            }
            component.add_to_model(m, m.technology_blocks[technology], m.T, context)

        m.electricity_consumption = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.afrr_energy_activated_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.idc_resell_da_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.idc_resell_future_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.idc_resell_idc_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.afrr_resell_credit_mwh = pyo.Var(m.T, within=pyo.NonNegativeReals)

        @m.Expression(m.T)
        def idc_resell_total_mwh(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (
                mm.idc_resell_da_mwh[t]
                + mm.idc_resell_future_mwh[t]
                + mm.idc_resell_idc_mwh[t]
            )

        @m.Constraint(m.T)
        def afrr_resell_credit_init(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            if t == 0:
                # in t=0 darf nichts resold werden
                return mm.afrr_resell_credit_mwh[t] == 0.0
            return pyo.Constraint.Skip

        @m.Constraint(m.T)
        def afrr_resell_credit_transition(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            if t == 0:
                return pyo.Constraint.Skip
            # Aktivierung aus t-1 wird erst in t Änderungen bewirken, daher Übergangsgleichung für Credit von t-1 nach t:
            return mm.afrr_resell_credit_mwh[t] == (
                mm.afrr_resell_credit_mwh[t - 1]
                + mm.afrr_energy_activated_mwh[t - 1]
                - mm.idc_resell_total_mwh[t - 1]
            )

        @m.Constraint(m.T)
        def afrr_resell_credit_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            # Resell in t nur aus bis t-1 aufgebautem Credit
            return mm.idc_resell_total_mwh[t] <= mm.afrr_resell_credit_mwh[t]

        @m.Constraint(m.T)
        def idc_resell_idc_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.idc_resell_idc_mwh[t] <= mm.idc_buy_mwh[t]
        
        @m.Constraint(m.T)
        def idc_resell_future_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.idc_resell_future_mwh[t] <= mm.future_position_mwh[t]

        @m.Constraint(m.T)
        def idc_resell_da_limit(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.idc_resell_da_mwh[t] <= mm.da_position_mwh[t] - mm.idc_sell_da_mwh[t]
        
        @m.Constraint(m.T)
        def etes_charge_matches_actual_electricity(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            # For the current hybrid ETES + gas plant, activated aFRR down energy maps
            # directly to additional ETES charging. TODO: Generalise this for industrial
            # plants with multiple electric processes behind one market position.
            return storage.electric_charge_to_storage[t] == (
                mm.da_position_mwh[t]
                + mm.future_position_mwh[t] 
                + mm.idc_buy_mwh[t] 
                - mm.idc_sell_da_mwh[t]
                - mm.idc_resell_da_mwh[t]
                - mm.idc_resell_future_mwh[t] 
                - mm.idc_resell_idc_mwh[t]
                + mm.afrr_energy_activated_mwh[t]
             )

        @m.Constraint(m.T)
        def electricity_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            return mm.electricity_consumption[t] == storage.electricity_consumption[t]


        @m.Constraint(m.T)
        def affr_market_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return mm.afrr_energy_activated_mwh[t]<= mm.affr_market_activation_limit_mwh[t]
        
                
        @m.Constraint(m.T)
        def heat_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            boiler = mm.technology_blocks["boiler"]
            return storage.discharge_heat[t] + boiler.heat_out[t] == mm.heat_demand[t]

        @m.Constraint(m.T)
        def heat_balance_boiler(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            boiler = mm.technology_blocks["boiler"]
            return boiler.heat_out[t] <= 0.0    


        @m.Constraint(m.T)
        def affr_physical_balance(mm: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]
            return mm.afrr_energy_activated_mwh[t]<= (storage.max_power_charge_mw * dt_hours - (
                                                      mm.da_position_mwh[t]
                                                      + mm.future_position_mwh[t]
                                                      + mm.idc_buy_mwh[t]
                                                      - mm.idc_sell_da_mwh[t]
                                                      ))     

        @m.Constraint(m.T)
        def afrr_not_usable_for_discharge_in_same_t(
            mm: pyo.ConcreteModel, t: int
        ) -> pyo.Constraint:
            storage = mm.technology_blocks["thermal_storage"]

            prev_soc = storage.initial_soc_mwh if t == 0 else storage.soc[t - 1]

            # Geplante (nicht-aFRR) Ladung in t:
            planned_charge_without_afrr = (
                storage.electric_charge_to_storage[t] - mm.afrr_energy_activated_mwh[t]
            )

            # Entladung in t darf nur aus Vor-SOC + nicht-aFRR-Ladung in t kommen.
            return storage.discharge_heat[t] <= (
                prev_soc * storage.efficiency_discharge
                + planned_charge_without_afrr
                * storage.efficiency_charge
                * storage.efficiency_discharge
            )
        
        @m.Expression(m.T)
        def da_buy_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_position_mwh[t] * mm.da_price[t]
        
        @m.Expression(m.T)
        def da_sell_revenue(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.da_sell_mwh[t] * mm.da_price[t]

        @m.Expression(m.T)
        def idc_buy_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.idc_buy_mwh[t] * mm.idc_price[t]
        
        @m.Expression(m.T)
        def idc_total_sell_mwh(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.idc_sell_da_mwh[t] 
                    + mm.idc_sell_future_mwh[t] 
                    + mm.idc_resell_da_mwh[t] 
                    + mm.idc_resell_future_mwh[t] 
                    + mm.idc_resell_idc_mwh[t])

        @m.Expression(m.T)
        def idc_sell_revenue(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.idc_total_sell_mwh[t]) * (mm.idc_price[t] - 1.0)

        @m.Expression(m.T)
        def afrr_energy_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.afrr_energy_activated_mwh[t] * mm.afrr_energy_price[t]

        @m.Expression(m.T)
        def electricity_market_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (
                mm.da_buy_cost[t]
                - mm.da_sell_revenue[t]
                + mm.idc_buy_cost[t]
                - mm.idc_sell_revenue[t]
                + mm.afrr_energy_cost[t]
            )

        @m.Expression(m.T)
        def additional_electricity_charges_cost(
            mm: pyo.ConcreteModel,
            t: int,
        ) -> pyo.Expression:
            return mm.electricity_consumption[t] * mm.additional_electricity_charge[t]

        @m.Expression(m.T)
        def gas_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.future_position_mwh[t] 
                    + mm.da_sell_mwh[t]
                    + mm.idc_sell_future_mwh[t]
                    )* mm.gas_price[t]
        
        @m.Expression(m.T)
        def tax_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.electricity_market_cost[t] 
                    + mm.additional_electricity_charges_cost[t]  
                    + mm.gas_cost[t]
                    )* tax_rate
        
        @m.Expression(m.T)
        def electricity_cost(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.electricity_market_cost[t] 
                    + mm.additional_electricity_charges_cost[t] 
                    )

        @m.Expression(m.T)
        def new_future_position_mwh(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return mm.future_position_mwh[t] - mm.idc_resell_future_mwh[t]       

        @m.Expression(m.T)
        def final_planned_electricity_mwh(mm: pyo.ConcreteModel, t: int) -> pyo.Expression:
            return (mm.da_position_mwh[t]
                    + mm.future_position_mwh[t] 
                    + mm.idc_buy_mwh[t] 
                    + mm.afrr_energy_activated_mwh[t]
                    - mm.idc_sell_da_mwh[t]
                    - mm.idc_resell_da_mwh[t]
                    - mm.idc_resell_future_mwh[t]
                    - mm.idc_resell_idc_mwh[t]
                    )

        @m.Objective(sense=pyo.minimize)
        def objective(mm: pyo.ConcreteModel) -> pyo.Expression:
            return pyo.quicksum(
                # TODO: Add CO2 cost consistently to the gas benchmark and plant objective.
                mm.electricity_cost[t] + mm.gas_cost[t] + mm.tax_cost[t]
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
            electricity_cost = _value(model.electricity_market_cost[t]) + _value(model.additional_costs_da[t])
            electricity_market_cost = _value(model.electricity_market_cost[t])
            additional_charges_cost = _value(model.additional_electricity_charges_cost[t])
            tax_cost_val = _value(model.tax_cost[t])
            gas_cost = _value(model.gas_cost[t]) + _value(model.additional_costs_future[t])
            #co2_cost = _value(model.co2_cost[t])
            additional_charge = _value(model.additional_electricity_charge[t])
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
                "day_ahead_delivered_price_EUR_per_MWh": (day_ahead_price + additional_charge)*(1 + signals.tax_rate),
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
                "DA_position_MWh": _value(model.da_buy_mwh[t]),
                "DA_sell_MWh": _value(model.da_sell_mwh[t]),
                "IDC_buy_MWh": 0.0,
                "IDC_sell_MWh": 0.0,
                "IDC_price_EUR_per_MWh": float("nan"),
                "final_planned_electricity_MWh": _value(model.da_buy_mwh[t]) + _value(model.new_future_position[t]),
                "actual_electricity_consumption_MWh": _value(model.da_buy_mwh[t]),
                "DA_electricity_cost_EUR":  _value(model.da_positions_costs[t]),
                "DA_sell_gains_EUR": _value(model.da_sell_revenue[t]),
                "Future_positions_MWh": _value(model.new_future_position[t]),
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
                "tax_cost": tax_cost_val,
                "electricity_cost_EUR": electricity_cost,
                "gas_cost_EUR": gas_cost,
                "co2_cost_EUR": 0, #co2_cost,
                "operating_cost_EUR": electricity_cost + gas_cost + tax_cost_val,
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
            tax_cost_val = _value(model.tax_cost[t])
            #co2_cost = _value(model.co2_cost[t])
            da_position = _value(model.da_position_mwh[t])
            da_sell_mwh = _value(model.da_sell_mwh[t])
            idc_buy = _value(model.idc_buy_mwh[t])
            idc_sell_da = _value(model.idc_sell_da_mwh[t])
            idc_sell_future = _value(model.idc_sell_future_mwh[t])
            future_position = _value(model.new_future_position[t])
            final_planned = _value(model.final_planned_electricity_mwh[t])
            additional_charge = _value(model.additional_electricity_charge[t])
            day_ahead_price = float(forecasts[signals.da_price_col].iloc[t])
            idc_price = float(forecasts[signals.idc_price_col].iloc[t])
            co2_price = (
                float(forecasts[signals.co2_price_col].iloc[t])
                if signals.co2_price_col and signals.co2_price_col in forecasts.columns
                else 0.0
            )
            print(da_position.sum(), (forecasts[self.heat_demand_column]*dt_hours).sum())
            row = {
                "datetime": timestamp,
                "plant_name": self.name,
                "heat_demand_MWh": float(forecasts[self.heat_demand_column].iloc[t]) * dt_hours,
                "day_ahead_price_EUR_per_MWh": day_ahead_price,
                "IDC_price_EUR_per_MWh": idc_price,
                "additional_electricity_charge_EUR_per_MWh_el": additional_charge,
                "day_ahead_delivered_price_EUR_per_MWh": (day_ahead_price + additional_charge)*(1 + signals.tax_rate),
                "IDC_delivered_price_EUR_per_MWh": (idc_price + additional_charge)*(1 + signals.tax_rate),
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
                "DA_sell_MWh": da_sell_mwh,
                "IDC_buy_MWh": idc_buy,
                "IDC_sell_da_MWh": idc_sell_da,
                "IDC_sell_future_MWh": idc_sell_future,
                "Future_positions_MWh": future_position,
                "final_planned_electricity_MWh": final_planned,
                "actual_electricity_consumption_MWh": _value(model.electricity_consumption[t]),
                "DA_electricity_cost_EUR": _value(model.da_electricity_cost[t]),
                "DA_sell_gains_EUR": _value(model.da_sell_revenue[t]),
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
                "tax_cost": _value(model.tax_cost[t]),
                "electricity_cost_EUR": electricity_cost,
                "tax_cost_EUR": tax_cost_val,
                "gas_cost_EUR": gas_cost,
                "co2_cost_EUR": 0,#co2_cost,
                "operating_cost_EUR": electricity_cost + gas_cost + tax_cost_val,
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
            tax_cost_val = _value(model.tax_cost[t])
            #co2_cost = _value(model.co2_cost[t])
            final_planned = _value(model.final_planned_electricity_mwh[t]) - _value(model.new_future_position_mwh[t])
            afrr_bid = _value(model.afrr_energy_bid_mwh[t])
            afrr_activation = _value(model.afrr_energy_activated_mwh[t])
            actual_electricity = _value(model.electricity_consumption[t]) - _value(model.new_future_position_mwh[t])
            afrr_price_clean = _value(model.afrr_energy_price[t])
            additional_charge =  _value(model.additional_electricity_charge[t])
            benchmark = float(signals.electricity_trading_benchmark_eur_per_mwh_el.iloc[t])
            afrr_bid_price = float(
                _series_value(signals.afrr_energy_bid_price, timestamp, benchmark)
            )
            afrr_delivered_price = (afrr_price_clean + additional_charge)*(1 + signals.tax_rate)
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
                "day_ahead_delivered_price_EUR_per_MWh": (1 + signals.tax_rate) *(day_ahead_price + additional_charge),
                "IDC_delivered_price_EUR_per_MWh": (idc_price + additional_charge)*(1 + signals.tax_rate),
                "afrr_energy_delivered_price_EUR_per_MWh": afrr_delivered_price,
                "gas_price_EUR_per_MWh": float(forecasts[signals.gas_price_col].iloc[t]),
                "co2_price_EUR_per_t": co2_price,
                "tax_rate": signals.tax_rate,
                "day_ahead_price_signal": signals.da_price_col,
                "IDC_price_signal": signals.idc_price_col,
                "gas_price_signal": signals.gas_price_col,
                "co2_price_signal": signals.co2_price_col or "",
                "gas_based_heat_benchmark_EUR_per_MWh_th": float(
                    signals.gas_benchmark_eur_per_mwh_th.iloc[t]
                ),
                "electricity_trading_benchmark_EUR_per_MWh_el": benchmark,
                "etes_charge_MWh": _value(storage.electric_charge_to_storage[t]) - _value(model.new_future_position_mwh[t]),
                "etes_discharge_MWh": _value(storage.discharge_heat[t]) - _value(model.new_future_position_mwh[t])*0.95,
                "etes_soc_MWh": _value(storage.soc[t]),
                "gas_heat_MWh": _value(model.new_future_position_mwh[t])*0.95,
                "gas_input_MWh": _value(model.new_future_position_mwh[t]),
                "electricity_consumption_MWh": _value(model.electricity_consumption[t]) - _value(model.new_future_position_mwh[t]),
                "DA_position_MWh": _value(model.da_position_mwh[t]),
                "DA_sell_MWh": _value(model.da_sell_mwh[t]),
                "IDC_buy_MWh": _value(model.idc_buy_mwh[t]),
                "IDC_sell_da_MWh": _value(model.idc_sell_da_mwh[t]),
                "IDC_sell_future_MWh": _value(model.idc_sell_future_mwh[t]),
                "IDC_resell_idc_MWh": _value(model.idc_resell_idc_mwh[t]),
                "IDC_resell_da_MWh": _value(model.idc_resell_da_mwh[t]),
                "IDC_resell_future_MWh": _value(model.idc_resell_future_mwh[t]),
                "IDC_sell_MWh": _value(model.idc_total_sell_mwh[t]),
                "Future_positions_MWh": _value(model.new_future_position_mwh[t]),
                "final_planned_electricity_MWh": final_planned,
                "actual_electricity_consumption_MWh": actual_electricity,
                "DA_electricity_cost_EUR": _value(model.da_buy_cost[t]) - _value(model.da_sell_revenue[t]),
                "DA_electricity_buy_cost_EUR": _value(model.da_buy_cost[t]),
                "DA_sell_revenue_EUR": _value(model.da_sell_revenue[t]),
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
                "tax_cost": _value(model.tax_cost[t]),
                "electricity_cost_EUR": electricity_cost,
                "gas_cost_EUR": gas_cost,
                "co2_cost_EUR": 0,#co2_cost,
                "tax_cost_EUR": tax_cost_val,
                "operating_cost_EUR": electricity_cost + gas_cost + tax_cost_val,
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
    def _available_solvers(config: CaseConfig) -> list[tuple[str, pyo.SolverFactory]]:
        configured = [config.solver_name, *config.solver_fallbacks]
        expanded: list[str] = []
        for candidate in configured:
            if candidate == "highs":
                expanded.extend(["highs", "appsi_highs"])
            else:
                expanded.append(candidate)

        seen: set[str] = set()
        solvers: list[tuple[str, pyo.SolverFactory]] = []
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
