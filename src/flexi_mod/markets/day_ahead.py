# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Day-ahead energy market rules and data preparation.

The day-ahead market trades electricity energy for the next delivery day. The
delivery products can be hourly or shorter depending on the configured
``product_resolution``. In FLEXIMOD, DA creates the first fixed electricity
procurement baseline; the operator strategy decides how much of that market to
use for a specific plant.
"""

from __future__ import annotations

import pandas as pd

from flexi_mod.markets.base_market import BaseMarket


class DayAheadMarket(BaseMarket):
    """Configured day-ahead energy market."""

    REQUIRED_SIGNALS = ("price",)

    def prepare_market_data(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        self._require_forecast_columns(forecasts)
        price = self._numeric_signal(forecasts, "price")
        return pd.DataFrame(
            {"day_ahead_price_EUR_per_MWh": price},
            index=forecasts.index,
        )
