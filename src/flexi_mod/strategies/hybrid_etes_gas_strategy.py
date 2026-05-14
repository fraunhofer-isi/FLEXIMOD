# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import warnings

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_energy import clean_afrr_down_data
from flexi_mod.plants.steam_generation_plant import (
    AFRRDownSignals,
    DispatchSignals,
    IDCAdjustmentSignals,
    SteamGenerationPlant,
)
from flexi_mod.strategies.base_strategy import BaseStrategy

GAS_PRICE_SIGNAL = "natural_gas_price"
ELECTRICITY_PRICE_SAFETY_MARGIN_EUR_PER_MWH = 0.0
# TODO: Move IDC_MARGIN_EUR_PER_MWH to config.yaml once multi-country cases
# or sensitivity analyses are implemented.
IDC_MARGIN_EUR_PER_MWH = 10.0
# TODO: Move AFRR_ENERGY_MARGIN_EUR_PER_MWH to config.yaml once sensitivity
# analysis or multi-country cases are implemented.
AFRR_ENERGY_MARGIN_EUR_PER_MWH = 0.0


class HybridETESGasStrategy(BaseStrategy):
    """Rule-based market wrapper around the Pyomo plant dispatch model."""

    def __init__(self, config: CaseConfig):
        self.config = config
        self.afrr_energy_data_quality_summary = pd.DataFrame()

    def required_forecast_columns(self) -> set[str]:
        required = {GAS_PRICE_SIGNAL}
        if "intraday_continuous" in self.config.enabled_markets:
            required.add(self.config.market_signal("intraday_continuous", "price"))
        if "afrr_energy" in self.config.enabled_markets:
            required.add(self.config.market_signal("afrr_energy", "price"))
            required.add(self.config.market_signal("afrr_energy", "system_activation"))
        return required

    def decide_day_ahead(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
    ) -> pd.DataFrame:
        price_col = self.config.market_signal("day_ahead", "price")

        benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        charge_allowed = self._calculate_charge_gate(
            plant=plant,
            electricity_price=forecasts[price_col].astype(float),
            benchmark=benchmark,
        )

        signals = DispatchSignals(
            electricity_price_col=price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            gas_benchmark_eur_per_mwh_th=benchmark,
            charge_allowed=charge_allowed,
        )
        return plant.solve_rolling(self.config, forecasts, signals)

    def decide_intraday_continuous(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
    ) -> pd.DataFrame:
        idc_price_col = self.config.market_signal("intraday_continuous", "price")
        da_price_col = self.config.market_signal("day_ahead", "price")

        da_position = self._fixed_da_position(fixed_positions, forecasts.index)
        idc_price = forecasts[idc_price_col].astype(float)
        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_heat_benchmark,
        )

        missing_price = idc_price.isna()
        if missing_price.any():
            warnings.warn(
                "IDC price contains missing values. IDC action is set to zero for "
                f"{int(missing_price.sum())} timestep(s).",
                stacklevel=2,
            )

        buy_allowed = (
            idc_price < (electricity_benchmark - IDC_MARGIN_EUR_PER_MWH)
        ) & ~missing_price
        sell_allowed = (
            idc_price > (electricity_benchmark + IDC_MARGIN_EUR_PER_MWH)
        ) & ~missing_price

        timestep_hours = self.config.timestep_minutes / 60.0
        max_charge_mwh = plant.etes.max_power_charge_mw * timestep_hours
        idc_buy_upper_bound = pd.Series(0.0, index=forecasts.index)
        idc_sell_upper_bound = pd.Series(0.0, index=forecasts.index)
        idc_buy_upper_bound.loc[buy_allowed] = (max_charge_mwh - da_position.loc[buy_allowed]).clip(
            lower=0.0
        )
        idc_sell_upper_bound.loc[sell_allowed] = da_position.loc[sell_allowed].clip(lower=0.0)

        signals = IDCAdjustmentSignals(
            da_price_col=da_price_col,
            idc_price_col=idc_price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            da_position_mwh=da_position,
            idc_buy_upper_bound_mwh=idc_buy_upper_bound,
            idc_sell_upper_bound_mwh=idc_sell_upper_bound,
            gas_benchmark_eur_per_mwh_th=gas_heat_benchmark,
            electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
        )
        return plant.solve_intraday_adjustment_rolling(self.config, forecasts, signals)

    def decide_afrr_energy(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
    ) -> pd.DataFrame:
        price_col = self.config.market_signal("afrr_energy", "price")
        activation_col = self.config.market_signal("afrr_energy", "system_activation")
        da_price_col = self.config.market_signal("day_ahead", "price")
        idc_price_col = self.config.market_signal("intraday_continuous", "price")
        if idc_price_col not in forecasts.columns:
            forecasts = forecasts.copy()
            forecasts[idc_price_col] = 0.0
        timestep_hours = self.config.timestep_minutes / 60.0
        afrr_market = self.config.market("afrr_energy")
        interpretation = afrr_market.get("interpretation", {})
        activation_unit = str(interpretation.get("activation_unit", "MW"))
        product_rules = afrr_market.get("product_rules", {})
        min_bid_mw = float(product_rules.get("min_bid_mw", 0.0))

        cleaned = clean_afrr_down_data(
            forecasts=forecasts,
            price_col=price_col,
            quantity_col=activation_col,
            activation_unit=activation_unit,
            timestep_hours=timestep_hours,
        )
        self.afrr_energy_data_quality_summary = cleaned.quality_summary
        clean_afrr = cleaned.frame

        da_position = self._series_from_fixed_positions(
            fixed_positions,
            "DA_position_MWh",
            forecasts.index,
        )
        idc_buy = self._series_from_fixed_positions(
            fixed_positions,
            "IDC_buy_MWh",
            forecasts.index,
            default=0.0,
        )
        idc_sell = self._series_from_fixed_positions(
            fixed_positions,
            "IDC_sell_MWh",
            forecasts.index,
            default=0.0,
        )
        final_planned = self._series_from_fixed_positions(
            fixed_positions,
            "final_planned_electricity_MWh",
            forecasts.index,
            default=None,
        )
        if final_planned is None:
            final_planned = da_position + idc_buy - idc_sell

        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_heat_benchmark,
        )

        max_charge_mwh = plant.etes.max_power_charge_mw * timestep_hours
        charge_power_headroom = (max_charge_mwh - final_planned).clip(lower=0.0)
        storage_soc = self._series_from_fixed_positions(
            fixed_positions,
            "etes_soc_MWh",
            forecasts.index,
            default=plant.etes.initial_soc_mwh,
        )
        storage_capacity_headroom = (plant.etes.max_capacity_mwh - storage_soc).clip(
            lower=0.0
        ) / plant.etes.efficiency_charge
        feasible_bid_potential = pd.concat(
            [charge_power_headroom, storage_capacity_headroom],
            axis=1,
        ).min(axis=1)

        valid_price = clean_afrr["afrr_price_available"]
        price_allowed = (
            clean_afrr["afrr_energy_down_price_EUR_per_MWh"]
            <= electricity_benchmark - AFRR_ENERGY_MARGIN_EUR_PER_MWH
        ) & valid_price
        feasible_bid_potential = feasible_bid_potential.where(price_allowed, 0.0)
        remaining_headroom_mw = feasible_bid_potential / timestep_hours
        bid_upper_bound = feasible_bid_potential.where(remaining_headroom_mw >= min_bid_mw, 0.0)
        bid_upper_bound = bid_upper_bound.clip(lower=0.0)
        # TODO: Enforce strict bid increments using integer or discretised variables
        # if exact market-compliant bid granularity is required later.

        activated = pd.concat(
            [bid_upper_bound, clean_afrr["afrr_system_activation_MWh"]],
            axis=1,
        ).min(axis=1)
        activated = activated.clip(lower=0.0)

        signals = AFRRDownSignals(
            da_price_col=da_price_col,
            idc_price_col=idc_price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            da_position_mwh=da_position,
            idc_buy_mwh=idc_buy,
            idc_sell_mwh=idc_sell,
            final_planned_electricity_mwh=final_planned,
            afrr_energy_price=clean_afrr["afrr_energy_down_price_EUR_per_MWh"],
            afrr_system_activation_mwh=clean_afrr["afrr_system_activation_MWh"],
            afrr_energy_bid_mwh=bid_upper_bound,
            afrr_energy_activated_mwh=activated,
            gas_benchmark_eur_per_mwh_th=gas_heat_benchmark,
            electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
        )
        return plant.solve_afrr_down_rolling(self.config, forecasts, signals)

    def decide_afrr_capacity(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
    ) -> pd.DataFrame:
        # TODO: Add reserve capacity awards and pre-DA headroom reservation.
        return super().decide_afrr_capacity(plant, forecasts)

    def calculate_gas_based_heat_cost(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
    ) -> pd.Series:
        if plant.gas_boiler is None:
            raise ValueError(f"Plant '{plant.name}' has no gas boiler")

        gas_input_per_mwh_heat = 1.0 / plant.gas_boiler.efficiency
        benchmark = forecasts[GAS_PRICE_SIGNAL].astype(float) * gas_input_per_mwh_heat
        benchmark.name = "gas_based_heat_benchmark_EUR_per_MWh_th"
        # TODO: Add CO2 cost to this benchmark when CO2 is enabled in gas cost accounting.
        return benchmark

    def calculate_electricity_trading_benchmark(
        self,
        plant: SteamGenerationPlant,
        gas_heat_benchmark: pd.Series,
    ) -> pd.Series:
        if plant.etes is None:
            raise ValueError(f"Plant '{plant.name}' has no ETES component")

        delivered_heat_per_mwh_electric = (
            plant.etes.efficiency_charge * plant.etes.efficiency_discharge
        )
        if delivered_heat_per_mwh_electric <= 0:
            raise ValueError("ETES charge/discharge efficiencies must be positive")

        benchmark = gas_heat_benchmark.astype(float) * delivered_heat_per_mwh_electric
        benchmark.name = "electricity_trading_benchmark_EUR_per_MWh_el"
        return benchmark

    def _calculate_charge_gate(
        self,
        plant: SteamGenerationPlant,
        electricity_price: pd.Series,
        benchmark: pd.Series,
    ) -> pd.Series:
        if plant.etes is None:
            raise ValueError(f"Plant '{plant.name}' has no ETES component")

        safety_margin = ELECTRICITY_PRICE_SAFETY_MARGIN_EUR_PER_MWH
        delivered_heat_per_mwh_electric = (
            plant.etes.efficiency_charge * plant.etes.efficiency_discharge
        )
        if delivered_heat_per_mwh_electric <= 0:
            raise ValueError("ETES charge/discharge efficiencies must be positive")

        effective_electric_heat_cost = (
            electricity_price.astype(float) / delivered_heat_per_mwh_electric
        )
        return effective_electric_heat_cost <= (benchmark.astype(float) - safety_margin)

    @staticmethod
    def _fixed_da_position(
        fixed_positions: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        if "DA_position_MWh" in fixed_positions.columns:
            da_position = fixed_positions["DA_position_MWh"]
        elif "electricity_consumption_MWh" in fixed_positions.columns:
            da_position = fixed_positions["electricity_consumption_MWh"]
        else:
            raise ValueError(
                "IDC stage requires fixed day-ahead positions, but neither "
                "'DA_position_MWh' nor 'electricity_consumption_MWh' was found."
            )

        da_position = da_position.astype(float).reindex(index)
        if da_position.isna().any():
            raise ValueError("Fixed day-ahead positions are not aligned with forecast timestamps")
        return da_position.clip(lower=0.0)

    @staticmethod
    def _series_from_fixed_positions(
        fixed_positions: pd.DataFrame,
        column: str,
        index: pd.DatetimeIndex,
        default: float | None = 0.0,
    ) -> pd.Series | None:
        if column in fixed_positions.columns:
            series = fixed_positions[column].astype(float).reindex(index)
        elif default is None:
            return None
        else:
            series = pd.Series(float(default), index=index)
        if series.isna().any():
            raise ValueError(f"Fixed position column '{column}' is not aligned with forecasts")
        return series
