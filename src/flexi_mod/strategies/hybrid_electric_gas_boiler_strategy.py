# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sequential electricity-market strategy for direct electric and gas boilers."""

from __future__ import annotations

import math
import warnings

import pandas as pd

from flexi_mod.markets.afrr_energy import AFRRDownEnergyMarket
from flexi_mod.markets.day_ahead import DayAheadMarket
from flexi_mod.markets.intraday_continuous import IntradayContinuousMarket
from flexi_mod.plants.steam_generation_plant import (
    DIRECT_ELECTRIC_GAS_BOILER_ROUTE,
    AFRRDownSignals,
    DispatchSignals,
    IDCAdjustmentSignals,
    SteamGenerationPlant,
)
from flexi_mod.strategies.hybrid_etes_gas_strategy import (
    ELECTRICITY_PRICE_SAFETY_MARGIN_EUR_PER_MWH,
    GAS_PRICE_SIGNAL,
    IDC_MARGIN_EUR_PER_MWH,
    HybridETESGasStrategy,
    _round_bid_down_to_increment,
    _validate_bid_rules,
    raw_electricity_bid_price,
)

CO2_PRICE_SIGNAL = "co2_price"


class HybridElectricGasBoilerStrategy(HybridETESGasStrategy):
    """Procure electricity for a direct electric boiler backed by gas.

    Day-ahead and intraday positions may replace gas heat immediately. After
    the intraday position is fixed, free aFRR-down energy is offered only
    against gas heat that is still scheduled in the same timestep. This route
    has no thermal storage and does not offer aFRR capacity.
    """

    def required_forecast_columns(self) -> set[str]:
        return super().required_forecast_columns() | {CO2_PRICE_SIGNAL}

    def decide_day_ahead(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        self._validate_direct_route(plant)
        self._reject_capacity_reservation(capacity_reservation)

        market = DayAheadMarket("day_ahead", self.config.market("day_ahead"))
        market_data = market.prepare_market_data(forecasts)
        price_col = market.signal_column("price")
        tax_rate = self._get_tax_rate(plant)
        additional_charges = self.calculate_additional_charges_t(plant, forecasts)
        gas_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_benchmark,
        )
        delivered_price = self._delivered_electricity_price(
            market_data["day_ahead_price_EUR_per_MWh"],
            tax_rate,
            additional_charges,
        )
        charge_allowed = (
            delivered_price <= electricity_benchmark - ELECTRICITY_PRICE_SAFETY_MARGIN_EUR_PER_MWH
        ) & ~self._grid_charging_block(plant, forecasts)

        signals = DispatchSignals(
            electricity_price_col=price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            gas_benchmark_eur_per_mwh_th=gas_benchmark,
            charge_allowed=charge_allowed,
            additional_electricity_charge_eur_per_mwh=additional_charges,
            tax_rate=tax_rate,
            co2_price_col=CO2_PRICE_SIGNAL,
            co2_emission_factor_t_per_mwh_fuel=self._gas_emission_factor(plant),
        )
        if rolling:
            return plant.solve_rolling(
                self.config,
                forecasts,
                signals,
                initial_soc_mwh=initial_soc_mwh,
            )
        return plant.solve_horizon(
            self.config,
            forecasts,
            signals,
            initial_soc_mwh=initial_soc_mwh,
        )

    def decide_intraday_continuous(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        self._validate_direct_route(plant)
        self._reject_capacity_reservation(capacity_reservation)

        market = IntradayContinuousMarket(
            "intraday_continuous",
            self.config.market("intraday_continuous"),
        )
        market_data = market.prepare_market_data(forecasts)
        idc_price_col = market.signal_column("price")
        da_price_col = self.config.market_signal("day_ahead", "price")
        if da_price_col not in forecasts.columns:
            forecasts = forecasts.copy()
            forecasts[da_price_col] = 0.0

        tax_rate = self._get_tax_rate(plant)
        additional_charges = self.calculate_additional_charges_t(plant, forecasts)
        da_position = self._fixed_da_position(fixed_positions, forecasts.index)
        delivered_idc_price = self._delivered_electricity_price(
            market_data["IDC_price_EUR_per_MWh"],
            tax_rate,
            additional_charges,
        )
        gas_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_benchmark,
        )

        missing_price = ~market_data["IDC_price_available"]
        if missing_price.any():
            warnings.warn(
                "IDC price contains missing values. IDC action is set to zero for "
                f"{int(missing_price.sum())} timestep(s).",
                stacklevel=2,
            )
        grid_block = self._grid_charging_block(plant, forecasts)
        buy_allowed = (
            (delivered_idc_price < electricity_benchmark - IDC_MARGIN_EUR_PER_MWH)
            & ~missing_price
            & ~grid_block
        )
        sell_allowed = (
            delivered_idc_price > electricity_benchmark + IDC_MARGIN_EUR_PER_MWH
        ) & ~missing_price
        if not market.buy_enabled:
            buy_allowed = pd.Series(False, index=forecasts.index)
        if not market.sell_enabled:
            sell_allowed = pd.Series(False, index=forecasts.index)

        timestep_hours = self.config.timestep_minutes / 60.0
        max_electricity_mwh = plant.electric_boiler.max_electricity_input_mw * timestep_hours
        buy_upper_bound = (
            (max_electricity_mwh - da_position)
            .clip(lower=0.0)
            .where(
                buy_allowed,
                0.0,
            )
        )
        sell_upper_bound = da_position.clip(lower=0.0).where(sell_allowed, 0.0)
        signals = IDCAdjustmentSignals(
            da_price_col=da_price_col,
            idc_price_col=idc_price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            da_position_mwh=da_position,
            idc_buy_upper_bound_mwh=buy_upper_bound,
            idc_sell_upper_bound_mwh=sell_upper_bound,
            gas_benchmark_eur_per_mwh_th=gas_benchmark,
            electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
            additional_electricity_charge_eur_per_mwh=additional_charges,
            tax_rate=tax_rate,
            co2_price_col=CO2_PRICE_SIGNAL,
            co2_emission_factor_t_per_mwh_fuel=self._gas_emission_factor(plant),
        )
        if rolling:
            return plant.solve_intraday_adjustment_rolling(
                self.config,
                forecasts,
                signals,
                initial_soc_mwh=initial_soc_mwh,
            )
        return plant.solve_intraday_adjustment_horizon(
            self.config,
            forecasts,
            signals,
            initial_soc_mwh=initial_soc_mwh,
        )

    def decide_afrr_energy(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        self._validate_direct_route(plant)
        self._reject_capacity_reservation(capacity_reservation)

        da_price_col = self.config.market_signal("day_ahead", "price")
        idc_price_col = self.config.market_signal("intraday_continuous", "price")
        forecasts = forecasts.copy()
        if da_price_col not in forecasts:
            forecasts[da_price_col] = 0.0
        if idc_price_col not in forecasts:
            forecasts[idc_price_col] = 0.0

        timestep_hours = self.config.timestep_minutes / 60.0
        market = AFRRDownEnergyMarket("afrr_energy", self.config.market("afrr_energy"))
        min_bid_mw = float(market.product_rules.get("min_bid_mw", 0.0))
        bid_increment_mw = float(market.product_rules.get("bid_increment_mw", 1.0))
        _validate_bid_rules("afrr_energy", min_bid_mw, bid_increment_mw)

        tax_rate = self._get_tax_rate(plant)
        additional_charges = self.calculate_additional_charges_t(plant, forecasts)
        clean_afrr = self._prepare_afrr_down_energy_data(
            forecasts,
            timestep_hours,
        ).frame
        da_position = self._required_fixed_series(
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
        gas_heat_after_idc = self._required_fixed_series(
            fixed_positions,
            "gas_heat_MWh",
            forecasts.index,
        )

        gas_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_benchmark,
        )
        delivered_bid_price = electricity_benchmark - self.afrr_energy_bid_margin_eur_per_mwh
        raw_bid_price = raw_electricity_bid_price(
            delivered_bid_price,
            tax_rate,
            additional_charges,
        )
        delivered_clearing_price = self._delivered_electricity_price(
            clean_afrr["afrr_energy_down_price_EUR_per_MWh"],
            tax_rate,
            additional_charges,
        )
        price_allowed = (
            clean_afrr["afrr_price_available"].astype(bool)
            & (delivered_clearing_price <= delivered_bid_price)
            & ~self._grid_charging_block(plant, forecasts)
        )

        max_electricity_mwh = plant.electric_boiler.max_electricity_input_mw * timestep_hours
        load_headroom_mwh = (max_electricity_mwh - final_planned).clip(lower=0.0)
        gas_replacement_mwh = gas_heat_after_idc.clip(lower=0.0) / plant.electric_boiler.efficiency
        feasible_bid_mwh = pd.concat(
            [load_headroom_mwh, gas_replacement_mwh],
            axis=1,
        ).min(axis=1)
        feasible_bid_mwh = feasible_bid_mwh.where(price_allowed, 0.0)
        compliant_bid_mwh = (
            feasible_bid_mwh.div(timestep_hours)
            .map(
                lambda feasible_mw: _round_bid_down_to_increment(
                    feasible_mw,
                    min_bid_mw=min_bid_mw,
                    bid_increment_mw=bid_increment_mw,
                )
            )
            .mul(timestep_hours)
        )
        activation_instruction = clean_afrr["afrr_system_activation_MWh"].where(
            price_allowed,
            0.0,
        )
        activated_mwh = pd.concat(
            [compliant_bid_mwh, activation_instruction],
            axis=1,
        ).min(axis=1)
        zero = pd.Series(0.0, index=forecasts.index)

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
            afrr_energy_bid_mwh=compliant_bid_mwh,
            afrr_energy_activated_mwh=activated_mwh,
            afrr_energy_bid_price=raw_bid_price,
            afrr_energy_delivered_bid_price=delivered_bid_price,
            afrr_energy_capacity_backed_bid_mwh=zero,
            afrr_energy_free_bid_mwh=compliant_bid_mwh,
            afrr_energy_capacity_backed_activated_mwh=zero,
            afrr_energy_free_activated_mwh=activated_mwh,
            afrr_headroom_binding=pd.Series(False, index=forecasts.index),
            afrr_curtailment_mwh=zero,
            gas_benchmark_eur_per_mwh_th=gas_benchmark,
            electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
            additional_electricity_charge_eur_per_mwh=additional_charges,
            tax_rate=tax_rate,
            co2_price_col=CO2_PRICE_SIGNAL,
            co2_emission_factor_t_per_mwh_fuel=self._gas_emission_factor(plant),
        )
        if rolling:
            return plant.solve_afrr_down_rolling(
                self.config,
                forecasts,
                signals,
                initial_soc_mwh=initial_soc_mwh,
            )
        return plant.solve_afrr_down_horizon(
            self.config,
            forecasts,
            signals,
            initial_soc_mwh=initial_soc_mwh,
        )

    def decide_afrr_capacity(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        del forecasts, initial_soc_mwh
        self._validate_direct_route(plant)
        raise ValueError(
            "Strategy 'hybrid_electric_gas_boiler' does not support aFRR capacity; "
            "disable markets.afrr_capacity"
        )

    def calculate_gas_based_heat_cost(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
    ) -> pd.Series:
        self._validate_direct_route(plant)
        gas_efficiency = float(plant.gas_boiler.efficiency)
        if gas_efficiency <= 0.0:
            raise ValueError("Gas-boiler efficiency must be positive")
        emission_factor = self._gas_emission_factor(plant)
        fuel_cost = forecasts[GAS_PRICE_SIGNAL].astype(float) + (
            forecasts[CO2_PRICE_SIGNAL].astype(float) * emission_factor
        )
        benchmark = fuel_cost / gas_efficiency
        benchmark.name = "gas_based_heat_benchmark_EUR_per_MWh_th"
        return benchmark

    def calculate_electricity_trading_benchmark(
        self,
        plant: SteamGenerationPlant,
        gas_heat_benchmark: pd.Series,
    ) -> pd.Series:
        self._validate_direct_route(plant)
        efficiency = float(plant.electric_boiler.efficiency)
        if efficiency <= 0.0:
            raise ValueError("Electric-boiler efficiency must be positive")
        benchmark = gas_heat_benchmark.astype(float) * efficiency
        benchmark.name = "electricity_trading_benchmark_EUR_per_MWh_el"
        return benchmark

    @staticmethod
    def _validate_direct_route(plant: SteamGenerationPlant) -> None:
        if plant.technology_route != DIRECT_ELECTRIC_GAS_BOILER_ROUTE:
            raise ValueError(
                "Strategy 'hybrid_electric_gas_boiler' requires the "
                "direct electric-boiler + gas-boiler route"
            )

    @staticmethod
    def _reject_capacity_reservation(
        capacity_reservation: pd.DataFrame | None,
    ) -> None:
        if capacity_reservation is not None and not capacity_reservation.empty:
            raise ValueError(
                "Strategy 'hybrid_electric_gas_boiler' does not support aFRR-capacity reservations"
            )

    @staticmethod
    def _gas_emission_factor(plant: SteamGenerationPlant) -> float:
        factor = float(plant.gas_emissions_factor_kg_per_mwh) / 1000.0
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError("gas_emissions_factor_kg_per_mwh must be finite and non-negative")
        return factor

    def _required_fixed_series(
        self,
        fixed_positions: pd.DataFrame,
        column: str,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        values = self._series_from_fixed_positions(
            fixed_positions,
            column,
            index,
            default=None,
        )
        if values is None:
            raise ValueError(f"aFRR energy requires fixed position column '{column}'")
        return values
