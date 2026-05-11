# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from etes_market_model.config.case_config import CaseConfig
from etes_market_model.plants.steam_generation_plant import (
    DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL,
    DispatchSignals,
    SteamGenerationPlant,
)
from etes_market_model.strategies.base_strategy import BaseStrategy

GAS_PRICE_SIGNAL = "natural_gas_price"
CO2_PRICE_SIGNAL = "co2_price"
ELECTRICITY_PRICE_SAFETY_MARGIN_EUR_PER_MWH = 0.0
IDC_SELL_MARGIN_EUR_PER_MWH = 10.0
IDC_BUY_MARGIN_EUR_PER_MWH = 10.0


class HybridETESGasStrategy(BaseStrategy):
    """Rule-based market wrapper around the Pyomo plant dispatch model."""

    def __init__(self, config: CaseConfig):
        self.config = config

    def required_forecast_columns(self) -> set[str]:
        return {GAS_PRICE_SIGNAL, CO2_PRICE_SIGNAL}

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
            co2_price_col=CO2_PRICE_SIGNAL,
            gas_benchmark_eur_per_mwh_th=benchmark,
            charge_allowed=charge_allowed,
            co2_emission_factor_t_per_mwh_fuel=DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL,
        )
        return plant.solve_rolling(self.config, forecasts, signals)

    def decide_intraday_continuous(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
    ) -> pd.DataFrame:
        # TODO: Add IDC buy/sell logic using configured ID3 signal and fixed DA positions.
        return super().decide_intraday_continuous(plant, forecasts, fixed_positions)

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

        emission_factor = DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL
        gas_input_per_mwh_heat = 1.0 / plant.gas_boiler.efficiency
        benchmark = (
            forecasts[GAS_PRICE_SIGNAL].astype(float) * gas_input_per_mwh_heat
            + forecasts[CO2_PRICE_SIGNAL].astype(float) * emission_factor * gas_input_per_mwh_heat
        )
        benchmark.name = "gas_based_heat_benchmark_EUR_per_MWh_th"
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
