# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""German DA and aFRR-down strategy for a cement clinker line.

The market side is shared with steel in
:mod:`flexi_mod.strategies.electrified_afrr_down_strategy`. All that lives here is what a
cement plant buys - no iron ore, no lime, and coal only when a stage actually burns it.
"""

from __future__ import annotations

from typing import Any

from flexi_mod.markets.afrr_energy import AFRRDownMarketSignals
from flexi_mod.plants.cement_plant import (
    CementAFRRDownCommoditySignals,
    CementAFRRDownSignals,
)
from flexi_mod.strategies.electrified_afrr_down_strategy import ElectrifiedAFRRDownStrategy

CO2_PRICE_SIGNAL = "co2_price"
NATURAL_GAS_PRICE_SIGNAL = "natural_gas_price"
HYDROGEN_PRICE_SIGNAL = "hydrogen_price"
BIOMASS_PRICE_SIGNAL = "biomass_price"


class ElectrifiedCementStrategy(ElectrifiedAFRRDownStrategy):
    """Cross-market next-day DA + aFRR-capacity + aFRR-energy strategy for clinker.

    Cement's downward flexibility is mostly its thermal store: the kiln line itself is
    pinned to the hourly clinker demand, so what the plant can offer is the room left in
    the store's charging power, plus stage auxiliaries.
    """

    commodity_forecast_columns = (
        CO2_PRICE_SIGNAL,
        NATURAL_GAS_PRICE_SIGNAL,
        HYDROGEN_PRICE_SIGNAL,
    )

    def _build_signals(self, plant: Any, market: AFRRDownMarketSignals) -> CementAFRRDownSignals:
        return CementAFRRDownSignals(
            market=market,
            commodities=CementAFRRDownCommoditySignals(
                co2_price_col=CO2_PRICE_SIGNAL,
                natural_gas_price_col=NATURAL_GAS_PRICE_SIGNAL,
                hydrogen_price_col=HYDROGEN_PRICE_SIGNAL,
                biomass_price_col=BIOMASS_PRICE_SIGNAL,
            ),
        )
