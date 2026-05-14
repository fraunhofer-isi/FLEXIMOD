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

from flexi_mod.markets.base_market import BaseMarket


class IntradayContinuousMarket(BaseMarket):
    """Configured intraday continuous energy adjustment market."""

    REQUIRED_SIGNALS = ("price",)

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
