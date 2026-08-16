# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Perfect-foresight German DA and aFRR-down strategy for any steel production route.

The market side lives in :mod:`flexi_mod.strategies.electrified_afrr_down_strategy` and is
shared with other plant families. All that remains here is what a steel plant buys.
"""

from __future__ import annotations

from typing import Any

from flexi_mod.markets.afrr_energy import AFRRDownMarketSignals
from flexi_mod.plants.steel_plant import SteelAFRRDownSignals
from flexi_mod.strategies.electrified_afrr_down_strategy import ElectrifiedAFRRDownStrategy

IRON_ORE_PRICE_SIGNAL = "iron_ore_price"
LIME_PRICE_SIGNAL = "lime_price"
CO2_PRICE_SIGNAL = "co2_price"
NATURAL_GAS_PRICE_SIGNAL = "natural_gas_price"
HYDROGEN_PRICE_SIGNAL = "hydrogen_price"

__all__ = [
    "CO2_PRICE_SIGNAL",
    "HYDROGEN_PRICE_SIGNAL",
    "IRON_ORE_PRICE_SIGNAL",
    "LIME_PRICE_SIGNAL",
    "NATURAL_GAS_PRICE_SIGNAL",
    "ElectrifiedSteelStrategy",
]


class ElectrifiedSteelStrategy(ElectrifiedAFRRDownStrategy):
    """Cross-market next-day DA + aFRR-capacity + aFRR-energy strategy for steelmaking.

    Applies to any steel production route (BF-BOF, DRI+BOF, DRI+EAF; any fuel mix) that
    consumes electricity, not only hydrogen/electrolyser routes.
    """

    commodity_forecast_columns = (
        IRON_ORE_PRICE_SIGNAL,
        LIME_PRICE_SIGNAL,
        CO2_PRICE_SIGNAL,
        NATURAL_GAS_PRICE_SIGNAL,
        HYDROGEN_PRICE_SIGNAL,
    )

    def _build_signals(self, plant: Any, market: AFRRDownMarketSignals) -> SteelAFRRDownSignals:
        return SteelAFRRDownSignals(
            da_price_col=market.da_price_col,
            afrr_energy_price_col=market.afrr_energy_price_col,
            afrr_system_activation_col=market.afrr_system_activation_col,
            afrr_price_available_col=market.afrr_price_available_col,
            afrr_capacity_block_id_col=market.afrr_capacity_block_id_col,
            afrr_capacity_block_duration_col=market.afrr_capacity_block_duration_col,
            afrr_capacity_price_col=market.afrr_capacity_price_col,
            afrr_capacity_missing_price_col=market.afrr_capacity_missing_price_col,
            iron_ore_price_col=IRON_ORE_PRICE_SIGNAL,
            lime_price_col=LIME_PRICE_SIGNAL,
            co2_price_col=CO2_PRICE_SIGNAL,
            natural_gas_price_col=NATURAL_GAS_PRICE_SIGNAL,
            hydrogen_price_col=HYDROGEN_PRICE_SIGNAL,
            additional_electricity_charge_eur_per_mwh=(
                market.additional_electricity_charge_eur_per_mwh
            ),
            afrr_energy_min_bid_mw=market.afrr_energy_min_bid_mw,
            afrr_energy_bid_increment_mw=market.afrr_energy_bid_increment_mw,
            afrr_capacity_min_bid_mw=market.afrr_capacity_min_bid_mw,
            afrr_capacity_bid_increment_mw=market.afrr_capacity_bid_increment_mw,
            afrr_capacity_product_duration_h=market.afrr_capacity_product_duration_h,
        )
