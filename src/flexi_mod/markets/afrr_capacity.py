# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""aFRR capacity market placeholder.

The capacity product reserves balancing capacity before energy activation. It
will later cover product blocks, awarded reserve, capacity revenue and mandatory
energy availability. It is intentionally non-operative in the current MVP.
"""

from __future__ import annotations

import pandas as pd

from flexi_mod.markets.base_market import BaseMarket


class AFRRCapacityMarket(BaseMarket):
    """Configured but not yet implemented aFRR capacity market."""

    REQUIRED_SIGNALS = ("capacity_price",)

    def prepare_market_data(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        if self.enabled:
            raise NotImplementedError("aFRR capacity is not implemented yet")
        return pd.DataFrame(index=forecasts.index)
