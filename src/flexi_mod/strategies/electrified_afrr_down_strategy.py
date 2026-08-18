# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""German DA + aFRR-down market strategy, shared across plant families.

Market orchestration only: which markets exist, how their data is prepared, and what the
regulatory calendar requires. What a plant consumes and how it dispatches lives behind
the plant object and two small hooks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_capacity import AFRRCapacityMarket
from flexi_mod.markets.afrr_energy import (
    AFRRDownEnergyMarket,
    AFRRDownMarketSignals,
    capacity_block_summary,
    duration_hours,
)
from flexi_mod.markets.day_ahead import DayAheadMarket

_AFRR_ENERGY_PRICE = "__afrr_energy_price_EUR_per_MWh"
_AFRR_SYSTEM_ACTIVATION = "__afrr_system_activation_MWh"
_AFRR_PRICE_AVAILABLE = "__afrr_price_available"
_AFRR_CAPACITY_BLOCK_ID = "__afrr_capacity_block_id"
_AFRR_CAPACITY_BLOCK_DURATION = "__afrr_capacity_block_duration_h"
_AFRR_CAPACITY_PRICE = "__afrr_capacity_price_EUR_per_MW_h"
_AFRR_CAPACITY_MISSING_PRICE = "__afrr_capacity_missing_price"

#: Markets an electrified aFRR-down case must enable, in this exact sequence.
REQUIRED_MARKET_SEQUENCE = ("afrr_capacity", "day_ahead", "afrr_energy")


class ElectrifiedAFRRDownStrategy:
    """Cross-market next-day DA + aFRR-capacity + aFRR-energy bidding, for any plant.

    Everything here is market orchestration: prepare the three markets, hand the plant
    a bid-ready forecast frame, and summarise what came back. What the plant consumes
    is reached through two hooks, so the same strategy drives steel and cement.
    """

    #: Forecast columns for the commodities this plant family buys. Needed before any
    #: plant object exists, since the runner resolves required columns up front.
    commodity_forecast_columns: tuple[str, ...] = ()

    def _build_signals(self, plant: Any, market: AFRRDownMarketSignals) -> Any:
        """Wrap the market half with this family's commodity columns."""
        raise NotImplementedError

    def __init__(self, config: CaseConfig):
        self.config = config
        self.afrr_energy_data_quality_summary = pd.DataFrame()
        self.afrr_capacity_block_summary = pd.DataFrame()
        self._validate_configuration()

    def required_forecast_columns(self) -> set[str]:
        return {
            self.config.market_signal("day_ahead", "price"),
            self.config.market_signal("afrr_capacity", "price"),
            self.config.market_signal("afrr_energy", "price"),
            self.config.market_signal("afrr_energy", "system_activation"),
            *self.commodity_forecast_columns,
        }

    def dispatch(
        self,
        plant: Any,
        forecasts: pd.DataFrame,
        progress_callback: Callable[[pd.Timestamp, pd.Timestamp], None] | None = None,
    ) -> pd.DataFrame:
        """Prepare the three markets, then let the plant roll its own horizon."""
        timestep_hours = self.config.timestep_minutes / 60.0
        day_ahead = DayAheadMarket("day_ahead", self.config.market("day_ahead"))
        capacity = AFRRCapacityMarket("afrr_capacity", self.config.market("afrr_capacity"))
        energy = AFRRDownEnergyMarket("afrr_energy", self.config.market("afrr_energy"))

        day_ahead.prepare_market_data(forecasts)
        capacity_data = capacity.prepare_market_data(forecasts, timestep_hours=timestep_hours)
        energy_data = energy.prepare_market_data(forecasts, timestep_hours=timestep_hours)
        self.afrr_energy_data_quality_summary = energy_data.quality_summary.copy()

        prepared = forecasts.copy()
        prepared[_AFRR_ENERGY_PRICE] = energy_data.frame["afrr_energy_down_price_EUR_per_MWh"]
        prepared[_AFRR_SYSTEM_ACTIVATION] = energy_data.frame["afrr_system_activation_MWh"]
        prepared[_AFRR_PRICE_AVAILABLE] = energy_data.frame["afrr_price_available"]
        prepared[_AFRR_CAPACITY_BLOCK_ID] = capacity_data.frame["afrr_capacity_block_id"]
        prepared[_AFRR_CAPACITY_BLOCK_DURATION] = capacity_data.frame["block_duration_h"]
        prepared[_AFRR_CAPACITY_PRICE] = capacity_data.frame["capacity_price_EUR_per_MW_h"]
        prepared[_AFRR_CAPACITY_MISSING_PRICE] = capacity_data.frame["missing_capacity_price_flag"]

        energy_rules = energy.product_rules
        capacity_rules = capacity.product_rules
        market = AFRRDownMarketSignals(
            da_price_col=day_ahead.signal_column("price"),
            afrr_energy_price_col=_AFRR_ENERGY_PRICE,
            afrr_system_activation_col=_AFRR_SYSTEM_ACTIVATION,
            afrr_price_available_col=_AFRR_PRICE_AVAILABLE,
            afrr_capacity_block_id_col=_AFRR_CAPACITY_BLOCK_ID,
            afrr_capacity_block_duration_col=_AFRR_CAPACITY_BLOCK_DURATION,
            afrr_capacity_price_col=_AFRR_CAPACITY_PRICE,
            afrr_capacity_missing_price_col=_AFRR_CAPACITY_MISSING_PRICE,
            additional_electricity_charge_eur_per_mwh=float(
                plant.additional_electricity_charge_eur_per_mwh
            ),
            afrr_energy_min_bid_mw=float(energy_rules.get("min_bid_mw", 1.0)),
            afrr_energy_bid_increment_mw=float(energy_rules.get("bid_increment_mw", 1.0)),
            afrr_capacity_min_bid_mw=float(capacity_rules.get("min_bid_mw", 1.0)),
            afrr_capacity_bid_increment_mw=float(capacity_rules.get("bid_increment_mw", 1.0)),
            afrr_capacity_product_duration_h=duration_hours(capacity.product_length),
        )
        signals = self._build_signals(plant, market)

        dispatch = plant.solve_afrr_down_rolling(
            self.config,
            prepared,
            signals,
            progress_callback=progress_callback,
        )
        self.afrr_capacity_block_summary = capacity_block_summary(dispatch)
        return dispatch

    def _validate_configuration(self) -> None:
        """Enforce the German aFRR-down + day-ahead calendar this strategy assumes."""
        case_name = self.config.strategy_name
        if self.config.country != "DE":
            raise ValueError(f"{case_name} requires country='DE'")
        if tuple(self.config.case.get("market_sequence", ())) != REQUIRED_MARKET_SEQUENCE:
            raise ValueError(
                f"{case_name} requires market_sequence == {list(REQUIRED_MARKET_SEQUENCE)}"
            )
        if not self.config.dispatch_setting("rolling_horizon_enabled", False):
            raise ValueError(f"{case_name} requires strategy.dispatch.rolling_horizon_enabled")
        for market_name in REQUIRED_MARKET_SEQUENCE:
            market = self.config.market(market_name)
            if not market.get("enabled", False):
                raise ValueError(f"{case_name} requires markets.{market_name}.enabled = true")
        for market_name in ("afrr_capacity", "afrr_energy"):
            direction = self.config.market(market_name).get("direction")
            if direction != "down":
                raise ValueError(f"{case_name} requires markets.{market_name}.direction = 'down'")
        capacity_gates = self.config.market("afrr_capacity")
        if "gate_open" not in capacity_gates or "gate_close" not in capacity_gates:
            raise ValueError(
                f"{case_name} requires markets.afrr_capacity.gate_open and .gate_close"
            )
        for market_name in ("day_ahead", "afrr_energy"):
            if "gate_close" not in self.config.market(market_name):
                raise ValueError(f"{case_name} requires markets.{market_name}.gate_close")
