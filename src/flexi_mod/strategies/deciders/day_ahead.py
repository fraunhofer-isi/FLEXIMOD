# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Day-ahead market decider for the hybrid ETES + gas strategy.

Assembles the dispatch signals for the day-ahead market from the plant's
regulation, the gas-heat benchmark, and the charge gate, then asks the plant
to solve the rolling or horizon dispatch. The gas-heat benchmark is computed
by the strategy (preserving the subclass override hook) and passed in; all
other helpers are orchestrated here from the extracted ``_*.py`` modules.
"""

from __future__ import annotations

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.day_ahead import DayAheadMarket
from flexi_mod.plants.capabilities import PlantCapabilities
from flexi_mod.plants.signals.dispatch_signals import DispatchSignals
from flexi_mod.strategies._benchmarks import GAS_PRICE_SIGNAL, delivered_electricity_price
from flexi_mod.strategies._capacity import capacity_signal_kwargs
from flexi_mod.strategies._charge_gate import charge_gate, grid_charging_block
from flexi_mod.strategies._charges import calculate_additional_charges_t, get_tax_rate


class DayAheadDecider:
    """Build day-ahead dispatch signals and solve the plant schedule."""

    def __init__(
        self,
        config: CaseConfig,
        plant: PlantCapabilities,
        forecasts: pd.DataFrame,
        gas_benchmark: pd.Series,
    ):
        self.config = config
        self.plant = plant
        self.forecasts = forecasts
        self.gas_benchmark = gas_benchmark

    def decide(
        self,
        capacity_reservation: pd.DataFrame | None,
        initial_soc_mwh: float | None,
        rolling: bool,
    ) -> pd.DataFrame:
        market = DayAheadMarket("day_ahead", self.config.market("day_ahead"))
        market_data = market.prepare_market_data(self.forecasts)
        price_col = market.signal_column("price")

        tax_rate = get_tax_rate(self.plant)
        additional_charges = calculate_additional_charges_t(self.plant, self.forecasts)

        delivered_da_price = delivered_electricity_price(
            market_data["day_ahead_price_EUR_per_MWh"],
            tax_rate,
            additional_charges,
        )
        charge_allowed = charge_gate(
            plant=self.plant,
            electricity_price=delivered_da_price,
            benchmark=self.gas_benchmark,
        )
        charge_allowed = charge_allowed & ~grid_charging_block(self.plant, self.forecasts)

        signals = self._build_signals(
            price_col, tax_rate, additional_charges, charge_allowed, capacity_reservation
        )
        return self._solve(signals, initial_soc_mwh, rolling)

    def _build_signals(
        self,
        price_col: str,
        tax_rate: float,
        additional_charges: pd.Series,
        charge_allowed: pd.Series,
        capacity_reservation: pd.DataFrame | None,
    ) -> DispatchSignals:
        return DispatchSignals(
            electricity_price_col=price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            gas_benchmark_eur_per_mwh_th=self.gas_benchmark,
            charge_allowed=charge_allowed,
            additional_electricity_charge_eur_per_mwh=additional_charges,
            tax_rate=tax_rate,
            **capacity_signal_kwargs(capacity_reservation, self.forecasts.index),
        )

    def _solve(
        self,
        signals: DispatchSignals,
        initial_soc_mwh: float | None,
        rolling: bool,
    ) -> pd.DataFrame:
        if rolling:
            return self.plant.solve_rolling(
                self.config,
                self.forecasts,
                signals,
                initial_soc_mwh=initial_soc_mwh,
            )
        return self.plant.solve_horizon(
            self.config,
            self.forecasts,
            signals,
            initial_soc_mwh=initial_soc_mwh,
        )
