# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants.capabilities import PlantCapabilities
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant
from flexi_mod.strategies._benchmarks import (
    GAS_PRICE_SIGNAL,
    delivered_electricity_price,
    electricity_trading_benchmark,
    gas_based_heat_cost,
)
from flexi_mod.strategies._bids import (
    configured_afrr_energy_bid_margin,
)
from flexi_mod.strategies._charge_gate import (
    fixed_da_position,
    grid_charging_block,
    series_from_fixed_positions,
)
from flexi_mod.strategies._charges import (
    calculate_additional_charges_t,
    get_tax_rate,
)
from flexi_mod.strategies._clearing import (
    resolve_capacity_clearing_mechanism,
    resolve_energy_clearing_mechanism,
)
from flexi_mod.strategies.base_strategy import BaseStrategy
from flexi_mod.strategies.deciders.afrr_capacity import AFRRCapacityDecider
from flexi_mod.strategies.deciders.afrr_energy import (
    AFRREnergyDecider,
    prepare_afrr_down_energy_data,
)
from flexi_mod.strategies.deciders.day_ahead import DayAheadDecider
from flexi_mod.strategies.deciders.intraday_continuous import IntradayContinuousDecider

# TODO: Move IDC_MARGIN_EUR_PER_MWH to config.yaml once multi-country cases
# or sensitivity analyses are implemented.
IDC_MARGIN_EUR_PER_MWH = 0.0
AFRR_CAPACITY_MARGIN_EUR_PER_MW_H = 0.0


class HybridETESGasStrategy(BaseStrategy):
    """Operator strategy for electricity procurement and plant operation.

    The configured market classes describe market products and prepare market
    inputs. This strategy decides how the industrial operator acts on those
    inputs using the gas benchmark, margins and plant flexibility.
    """

    def __init__(self, config: CaseConfig):
        self.config = config
        self._capacity_clearing_mechanism = resolve_capacity_clearing_mechanism(config)
        self._energy_clearing_mechanism = resolve_energy_clearing_mechanism(config)
        self.afrr_energy_bid_margin_eur_per_mwh = configured_afrr_energy_bid_margin(config)
        self.afrr_energy_data_quality_summary = pd.DataFrame()
        self.afrr_capacity_block_summary = pd.DataFrame()
        self._afrr_down_energy_data_cache = {}

    # ─── Regulation-aware helpers ──────────────────────────────────

    @staticmethod
    def _get_tax_rate(plant: SteamGenerationPlant) -> float:
        """Read the multiplicative electricity tax rate from the plant's regulation."""

        return get_tax_rate(plant)

    def calculate_additional_charges_t(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
    ) -> pd.Series:
        """Return per-timestep additional electricity charges.

        - If the regulation declares a ``dynamic_charge_column``, that column must
          be present in the forecasts and its time series is used. A declared but
          missing column is a configuration error and raises — it is never
          silently replaced by the scalar, which would zero the per-MWh grid fee
          for countries whose charge lives entirely in that column (ES, FR).
        - Otherwise (no dynamic column declared, e.g. Germany): use the scalar
          marginal charge from the regulation.
        """

        return calculate_additional_charges_t(plant, forecasts)

    # ─── Core interface ────────────────────────────────────────────

    def required_forecast_columns(self) -> set[str]:
        required = {GAS_PRICE_SIGNAL}
        if "intraday_continuous" in self.config.enabled_markets:
            required.add(self.config.market_signal("intraday_continuous", "price"))
        if "afrr_energy" in self.config.enabled_markets:
            required.add(self.config.market_signal("afrr_energy", "price"))
            required.add(self.config.market_signal("afrr_energy", "system_activation"))
        return required

    def decide_day_ahead(
        self,
        plant: PlantCapabilities,
        forecasts: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        gas_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        return DayAheadDecider(self.config, plant, forecasts, gas_benchmark).decide(
            capacity_reservation, initial_soc_mwh, rolling
        )

    def decide_intraday_continuous(
        self,
        plant: PlantCapabilities,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant, gas_heat_benchmark
        )
        return IntradayContinuousDecider(
            self.config,
            plant,
            forecasts,
            gas_heat_benchmark,
            electricity_benchmark,
            IDC_MARGIN_EUR_PER_MWH,
        ).decide(fixed_positions, capacity_reservation, initial_soc_mwh, rolling)

    def decide_afrr_energy(
        self,
        plant: PlantCapabilities,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant, gas_heat_benchmark
        )
        decider = AFRREnergyDecider(
            self.config,
            plant,
            forecasts,
            gas_heat_benchmark,
            electricity_benchmark,
            self.afrr_energy_bid_margin_eur_per_mwh,
        )
        result = decider.decide(fixed_positions, capacity_reservation, initial_soc_mwh, rolling)
        self.afrr_energy_data_quality_summary = decider.quality_summary
        return result

    def decide_afrr_capacity(
        self,
        plant: PlantCapabilities,
        forecasts: pd.DataFrame,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_heat_benchmark,
        )
        decider = AFRRCapacityDecider(
            config=self.config,
            plant=plant,
            gas_heat_benchmark=gas_heat_benchmark,
            electricity_benchmark=electricity_benchmark,
            capacity_clearing_mechanism=self._capacity_clearing_mechanism,
            afrr_energy_bid_margin_eur_per_mwh=self.afrr_energy_bid_margin_eur_per_mwh,
            capacity_margin_eur_per_mw_h=AFRR_CAPACITY_MARGIN_EUR_PER_MW_H,
        )
        result = decider.decide(forecasts, initial_soc_mwh)
        self.afrr_capacity_block_summary = decider.block_summary
        if decider.quality_summary is not None:
            self.afrr_energy_data_quality_summary = decider.quality_summary
        return result

    def _prepare_afrr_down_energy_data(
        self,
        forecasts: pd.DataFrame,
        timestep_hours: float,
    ):
        cleaned = prepare_afrr_down_energy_data(self.config, forecasts, timestep_hours)
        self.afrr_energy_data_quality_summary = cleaned.quality_summary
        return cleaned

    def calculate_gas_based_heat_cost(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
    ) -> pd.Series:
        return gas_based_heat_cost(plant, forecasts)

    def calculate_electricity_trading_benchmark(
        self,
        plant: SteamGenerationPlant,
        gas_heat_benchmark: pd.Series,
    ) -> pd.Series:
        return electricity_trading_benchmark(plant, gas_heat_benchmark)

    @staticmethod
    def _delivered_electricity_price(
        market_price: pd.Series,
        tax_rate: float,
        additional_charges: pd.Series,
    ) -> pd.Series:
        return delivered_electricity_price(market_price, tax_rate, additional_charges)

    @staticmethod
    def _grid_charging_block(
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
    ) -> pd.Series:
        return grid_charging_block(plant, forecasts)

    @staticmethod
    def _fixed_da_position(
        fixed_positions: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        return fixed_da_position(fixed_positions, index)

    @staticmethod
    def _series_from_fixed_positions(
        fixed_positions: pd.DataFrame,
        column: str,
        index: pd.DatetimeIndex,
        default: float | None = 0.0,
    ) -> pd.Series | None:
        return series_from_fixed_positions(fixed_positions, column, index, default)
