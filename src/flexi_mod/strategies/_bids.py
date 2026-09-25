# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bid-price and bid-rule helpers shared by steam-plant market strategies."""

from __future__ import annotations

import math

import pandas as pd

from flexi_mod.config.case_config import CaseConfig

AFRR_ENERGY_BID_MARGIN_EUR_PER_MWH = 0.0
AFRR_ENERGY_BID_MARGIN_SETTING = "afrr_energy_bid_margin_eur_per_mwh"


def configured_afrr_energy_bid_margin(config: CaseConfig) -> float:
    """Return the required aFRR-energy saving margin in EUR/MWh_el.

    The margin is a positive deduction from the plant's delivered-electricity
    strike price. A zero default preserves the historical break-even bid.
    """

    raw_value = config.dispatch_setting(
        AFRR_ENERGY_BID_MARGIN_SETTING,
        AFRR_ENERGY_BID_MARGIN_EUR_PER_MWH,
    )
    if isinstance(raw_value, bool):
        raise ValueError(
            f"strategy.dispatch.{AFRR_ENERGY_BID_MARGIN_SETTING} must be a finite "
            "non-negative number"
        )
    try:
        margin = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"strategy.dispatch.{AFRR_ENERGY_BID_MARGIN_SETTING} must be a finite "
            "non-negative number"
        ) from exc
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(
            f"strategy.dispatch.{AFRR_ENERGY_BID_MARGIN_SETTING} must be a finite "
            "non-negative number"
        )
    return margin


def raw_electricity_bid_price(
    delivered_strike_price: pd.Series,
    tax_rate: float,
    additional_charges: pd.Series,
) -> pd.Series:
    """Convert a delivered-price ceiling into the submitted market bid price."""

    tax_multiplier = 1.0 + float(tax_rate)
    if not math.isfinite(tax_multiplier) or tax_multiplier <= 0.0:
        raise ValueError("Electricity tax rate must be finite and greater than -1")
    raw_bid = delivered_strike_price.astype(float) / tax_multiplier - additional_charges.astype(
        float
    )
    raw_bid.name = "afrr_energy_bid_price_EUR_per_MWh"
    return raw_bid


def _validate_bid_rules(market_name: str, min_bid_mw: float, bid_increment_mw: float) -> None:
    if not math.isfinite(min_bid_mw):
        raise ValueError(f"{market_name}.product_rules.min_bid_mw must be finite")
    if not math.isfinite(bid_increment_mw):
        raise ValueError(f"{market_name}.product_rules.bid_increment_mw must be finite")
    if min_bid_mw < 0:
        raise ValueError(f"{market_name}.product_rules.min_bid_mw cannot be negative")
    if bid_increment_mw <= 0:
        raise ValueError(f"{market_name}.product_rules.bid_increment_mw must be positive")


def _round_bid_down_to_increment(
    feasible_bid_mw: float,
    min_bid_mw: float,
    bid_increment_mw: float,
) -> float:
    """Return the largest market-compliant bid not exceeding physical capability."""

    if feasible_bid_mw < min_bid_mw:
        return 0.0
    rounded = math.floor((feasible_bid_mw + 1e-12) / bid_increment_mw) * bid_increment_mw
    if rounded < min_bid_mw:
        return 0.0
    return float(rounded)
