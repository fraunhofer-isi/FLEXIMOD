# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from etes_market_model.config.case_config import CaseConfig


def is_market_enabled(config: CaseConfig, market_name: str) -> bool:
    return bool(config.market(market_name).get("enabled", False))


def required_signals_for_enabled_markets(config: CaseConfig) -> set[str]:
    signals: set[str] = set()
    for market_name in config.enabled_markets:
        market = config.market(market_name)
        signals.update(str(column) for column in market.get("signals", {}).values())
    return signals
