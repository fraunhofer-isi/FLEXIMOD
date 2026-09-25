# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Intraday-continuous market decider for the hybrid ETES + gas strategy.

Computes the IDC buy/sell bounds from the day-ahead position, the delivered
IDC price, and the electricity-trading benchmark, then asks the plant to
solve the rolling or horizon intraday adjustment. The gas and electricity
benchmarks are computed by the strategy (preserving the subclass override
hooks) and passed in; all other helpers are orchestrated here.
"""

from __future__ import annotations

import warnings

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.intraday_continuous import IntradayContinuousMarket
from flexi_mod.plants.capabilities import PlantCapabilities
from flexi_mod.plants.signals.idc_adjustment_signals import IDCAdjustmentSignals
from flexi_mod.strategies._benchmarks import GAS_PRICE_SIGNAL, delivered_electricity_price
from flexi_mod.strategies._capacity import capacity_signal_kwargs
from flexi_mod.strategies._charge_gate import fixed_da_position, grid_charging_block
from flexi_mod.strategies._charges import calculate_additional_charges_t, get_tax_rate


class IntradayContinuousDecider:
    """Build intraday-continuous adjustment signals and solve the plant schedule."""

    def __init__(
        self,
        config: CaseConfig,
        plant: PlantCapabilities,
        forecasts: pd.DataFrame,
        gas_benchmark: pd.Series,
        electricity_benchmark: pd.Series,
        idc_margin_eur_per_mwh: float,
    ):
        self.config = config
        self.plant = plant
        self.forecasts = forecasts
        self.gas_benchmark = gas_benchmark
        self.electricity_benchmark = electricity_benchmark
        self.idc_margin = idc_margin_eur_per_mwh

    def decide(
        self,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None,
        initial_soc_mwh: float | None,
        rolling: bool,
    ) -> pd.DataFrame:
        idc_market, idc_data, idc_price_col, da_price_col, forecasts = self._prepare_market_data()
        da_position = fixed_da_position(fixed_positions, forecasts.index)
        tax_rate = get_tax_rate(self.plant)
        additional_charges = calculate_additional_charges_t(self.plant, forecasts)
        delivered_idc_price = delivered_electricity_price(
            idc_data["IDC_price_EUR_per_MWh"], tax_rate, additional_charges
        )

        missing_price = ~idc_data["IDC_price_available"]
        self._warn_missing_prices(missing_price)
        buy_allowed, sell_allowed = self._compute_allowed(
            delivered_idc_price, missing_price, idc_market, forecasts
        )
        buy_bound, sell_bound = self._compute_bounds(
            buy_allowed, sell_allowed, da_position, forecasts
        )
        signals = self._build_signals(
            idc_price_col,
            da_price_col,
            da_position,
            buy_bound,
            sell_bound,
            tax_rate,
            additional_charges,
            capacity_reservation,
            forecasts,
        )
        return self._solve(signals, forecasts, initial_soc_mwh, rolling)

    def _prepare_market_data(
        self,
    ) -> tuple[IntradayContinuousMarket, pd.DataFrame, str, str, pd.DataFrame]:
        idc_market = IntradayContinuousMarket(
            "intraday_continuous", self.config.market("intraday_continuous")
        )
        idc_data = idc_market.prepare_market_data(self.forecasts)
        idc_price_col = idc_market.signal_column("price")
        da_price_col = self.config.market_signal("day_ahead", "price")
        forecasts = self.forecasts
        if da_price_col not in forecasts.columns:
            forecasts = forecasts.copy()
            forecasts[da_price_col] = 0.0
        return idc_market, idc_data, idc_price_col, da_price_col, forecasts

    def _warn_missing_prices(self, missing_price: pd.Series) -> None:
        if missing_price.any():
            warnings.warn(
                "IDC price contains missing values. IDC action is set to zero for "
                f"{int(missing_price.sum())} timestep(s).",
                stacklevel=2,
            )

    def _compute_allowed(
        self,
        delivered_idc_price: pd.Series,
        missing_price: pd.Series,
        idc_market: IntradayContinuousMarket,
        forecasts: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        grid_block = grid_charging_block(self.plant, forecasts)
        buy_allowed = (
            (delivered_idc_price < (self.electricity_benchmark - self.idc_margin))
            & ~missing_price
            & ~grid_block
        )
        sell_allowed = (
            delivered_idc_price > (self.electricity_benchmark + self.idc_margin)
        ) & ~missing_price
        if not idc_market.buy_enabled:
            buy_allowed = pd.Series(False, index=forecasts.index)
        if not idc_market.sell_enabled:
            sell_allowed = pd.Series(False, index=forecasts.index)
        return buy_allowed, sell_allowed

    def _compute_bounds(
        self,
        buy_allowed: pd.Series,
        sell_allowed: pd.Series,
        da_position: pd.Series,
        forecasts: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        timestep_hours = self.config.timestep_minutes / 60.0
        max_charge_mwh = self.plant.etes.max_power_charge_mw * timestep_hours
        buy_bound = pd.Series(0.0, index=forecasts.index)
        sell_bound = pd.Series(0.0, index=forecasts.index)
        buy_bound.loc[buy_allowed] = (max_charge_mwh - da_position.loc[buy_allowed]).clip(lower=0.0)
        sell_bound.loc[sell_allowed] = da_position.loc[sell_allowed].clip(lower=0.0)
        return buy_bound, sell_bound

    def _build_signals(
        self,
        idc_price_col: str,
        da_price_col: str,
        da_position: pd.Series,
        buy_bound: pd.Series,
        sell_bound: pd.Series,
        tax_rate: float,
        additional_charges: pd.Series,
        capacity_reservation: pd.DataFrame | None,
        forecasts: pd.DataFrame,
    ) -> IDCAdjustmentSignals:
        return IDCAdjustmentSignals(
            da_price_col=da_price_col,
            idc_price_col=idc_price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            da_position_mwh=da_position,
            idc_buy_upper_bound_mwh=buy_bound,
            idc_sell_upper_bound_mwh=sell_bound,
            gas_benchmark_eur_per_mwh_th=self.gas_benchmark,
            electricity_trading_benchmark_eur_per_mwh_el=self.electricity_benchmark,
            additional_electricity_charge_eur_per_mwh=additional_charges,
            tax_rate=tax_rate,
            **capacity_signal_kwargs(capacity_reservation, forecasts.index),
        )

    def _solve(
        self,
        signals: IDCAdjustmentSignals,
        forecasts: pd.DataFrame,
        initial_soc_mwh: float | None,
        rolling: bool,
    ) -> pd.DataFrame:
        if rolling:
            return self.plant.solve_intraday_adjustment_rolling(
                self.config,
                forecasts,
                signals,
                initial_soc_mwh=initial_soc_mwh,
            )
        return self.plant.solve_intraday_adjustment_horizon(
            self.config,
            forecasts,
            signals,
            initial_soc_mwh=initial_soc_mwh,
        )
