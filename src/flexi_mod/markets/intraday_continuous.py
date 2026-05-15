# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Intraday continuous energy market rules and data preparation.

The intraday continuous market is modelled as an index/proxy price signal used
for incremental adjustments after fixed day-ahead positions. Order books,
liquidity, repeated trades and individual transactions are deliberately outside
the current model scope.
"""

from __future__ import annotations

import pandas as pd

from flexi_mod.markets.base_market import BaseMarket, MarketConfigError


class IntradayContinuousMarket(BaseMarket):
    """Configured intraday continuous energy adjustment market."""

    REQUIRED_SIGNALS = ("price",)

    @property
    def allowed_actions(self) -> dict[str, bool]:
        """Return configured IDC trading directions.

        Missing settings default to the current behaviour: both buy and sell
        adjustments are enabled. Setting both directions to false is a valid
        observe-only mode.
        """

        raw_actions = self.config.get("allowed_actions", {})
        if raw_actions is None:
            raw_actions = {}
        if not isinstance(raw_actions, dict):
            raise MarketConfigError(
                "Market 'intraday_continuous' allowed_actions must be a mapping "
                "with boolean buy and sell entries"
            )

        actions: dict[str, bool] = {}
        for action in ("buy", "sell"):
            value = raw_actions.get(action, True)
            if not isinstance(value, bool):
                raise MarketConfigError(
                    f"Market 'intraday_continuous' allowed_actions.{action} must be true or false"
                )
            actions[action] = value
        return actions

    @property
    def buy_enabled(self) -> bool:
        return self.allowed_actions["buy"]

    @property
    def sell_enabled(self) -> bool:
        return self.allowed_actions["sell"]

    def validate_config(self, timestep_minutes: int | None = None) -> None:
        super().validate_config(timestep_minutes=timestep_minutes)
        if self.enabled:
            _ = self.allowed_actions

    def prepare_market_data(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        self._require_forecast_columns(forecasts)
        price = self._numeric_signal(forecasts, "price")
        return pd.DataFrame(
            {
                "IDC_price_EUR_per_MWh": price,
                "IDC_price_available": ~price.isna(),
            },
            index=forecasts.index,
        )
