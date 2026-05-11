# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import warnings

import pandas as pd

from etes_market_model.config.case_config import CaseConfig
from etes_market_model.plants.steam_generation_plant import (
    DispatchSignals,
    IDCAdjustmentSignals,
    SteamGenerationPlant,
)
from etes_market_model.strategies.base_strategy import BaseStrategy

GAS_PRICE_SIGNAL = "natural_gas_price"
ELECTRICITY_PRICE_SAFETY_MARGIN_EUR_PER_MWH = 0.0
# TODO: Move IDC_MARGIN_EUR_PER_MWH to config.yaml once multi-country cases
# or sensitivity analyses are implemented.
IDC_MARGIN_EUR_PER_MWH = 10.0


class HybridETESGasStrategy(BaseStrategy):
    """Rule-based market wrapper around the Pyomo plant dispatch model."""

    def __init__(self, config: CaseConfig):
        self.config = config

    def required_forecast_columns(self) -> set[str]:
        required = {GAS_PRICE_SIGNAL}
        if "intraday_continuous" in self.config.enabled_markets:
            required.add(self.config.market_signal("intraday_continuous", "price"))
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
        # TODO: Add exogenous negative aFRR activation after DA/IDC headroom is known.
        return super().decide_afrr_energy(plant, forecasts, fixed_positions)

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
