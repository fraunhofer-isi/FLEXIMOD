# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Hybrid ETES strategy with pay-as-cleared aFRR-capacity settlement."""

from __future__ import annotations

from flexi_mod.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy


class PayAsClearedCapacityHybridETESGasStrategy(HybridETESGasStrategy):
    """Settle awarded aFRR capacity at the exogenous marginal clearing price."""

    @property
    def capacity_pricing_rule(self) -> str:
        return "pay_as_cleared"

    def capacity_settlement_price(
        self,
        capacity_bid_price_eur_per_mw_h: float,
        clearing_price_eur_per_mw_h: float,
    ) -> float:
        del capacity_bid_price_eur_per_mw_h
        return clearing_price_eur_per_mw_h

    def capacity_bid_price(
        self,
        minimum_acceptable_price_eur_per_mw_h: float,
        market_reference_price_eur_per_mw_h: float,
    ) -> float:
        del market_reference_price_eur_per_mw_h
        return minimum_acceptable_price_eur_per_mw_h
