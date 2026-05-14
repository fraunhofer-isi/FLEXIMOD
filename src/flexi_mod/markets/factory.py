# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Factory for configured market-rule objects."""

from __future__ import annotations

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_capacity import AFRRCapacityMarket
from flexi_mod.markets.afrr_energy import AFRRDownEnergyMarket, AFRRUpEnergyMarket
from flexi_mod.markets.base_market import BaseMarket, MarketConfigError
from flexi_mod.markets.day_ahead import DayAheadMarket
from flexi_mod.markets.intraday_continuous import IntradayContinuousMarket

MARKET_CLASSES: dict[str, type[BaseMarket]] = {
    "day_ahead": DayAheadMarket,
    "intraday_continuous": IntradayContinuousMarket,
    "afrr_energy": AFRRDownEnergyMarket,
    "afrr_capacity": AFRRCapacityMarket,
}


def build_market(config: CaseConfig, market_name: str) -> BaseMarket:
    if market_name == "afrr_energy":
        return _build_afrr_energy_market(config)

    try:
        market_class = MARKET_CLASSES[market_name]
    except KeyError as exc:
        raise MarketConfigError(f"No market class is registered for '{market_name}'") from exc

    market = market_class(name=market_name, config=config.market(market_name))
    market.validate_config(timestep_minutes=config.timestep_minutes)
    return market


def _build_afrr_energy_market(config: CaseConfig) -> BaseMarket:
    market_config = config.market("afrr_energy")
    direction = str(market_config.get("direction", "down")).lower()
    if direction in {"down", "negative"}:
        market: BaseMarket = AFRRDownEnergyMarket(name="afrr_energy", config=market_config)
    elif direction in {"up", "positive"}:
        market = AFRRUpEnergyMarket(name="afrr_energy", config=market_config)
    else:
        raise MarketConfigError(
            "afrr_energy.direction must be 'down'/'negative' or 'up'/'positive'"
        )
    market.validate_config(timestep_minutes=config.timestep_minutes)
    return market


def build_markets(config: CaseConfig) -> dict[str, BaseMarket]:
    return {
        market_name: build_market(config, market_name) for market_name in config.market_sequence
    }
