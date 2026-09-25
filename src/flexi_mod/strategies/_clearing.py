# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""aFRR clearing-mechanism resolution and capacity pricing helpers.

The clearing mechanism (``pay_as_bid`` vs ``pay_as_cleared``) is a property of
each market, configured under ``markets.<market>.clearing_mechanism``. These
helpers read the mechanism from a :class:`~flexi_mod.config.case_config.CaseConfig`
and compute the capacity bid, settlement, and pricing-rule values that the
strategy reports in its outputs.
"""

from __future__ import annotations

from flexi_mod.config.case_config import CaseConfig

# aFRR clearing mechanisms. ``pay_as_bid`` pays each awarded bid its own
# submitted price; ``pay_as_cleared`` pays every awarded bid the marginal
# clearing price.
_VALID_CLEARING_MECHANISMS = ("pay_as_bid", "pay_as_cleared")
_AFRR_CAPACITY_MARKET = "afrr_capacity"
_AFRR_ENERGY_MARKET = "afrr_energy"
# Legacy strategy name that selected pay-as-cleared capacity before the
# clearing mechanism moved onto the market blocks. Kept for backward compatibility.
_PAY_AS_CLEARED_STRATEGY_NAME = "hybrid_etes_gas_pay_as_cleared_capacity"


def market_clearing_mechanism(config: CaseConfig, market_name: str) -> str | None:
    """Return ``markets.<market_name>.clearing_mechanism`` (lowercased) or None."""

    markets = config.case.get("markets") or {}
    market_cfg = markets.get(market_name) or {}
    value = market_cfg.get("clearing_mechanism")
    return None if value is None else str(value).strip().lower()


def resolve_capacity_clearing_mechanism(config: CaseConfig) -> str:
    """Determine the aFRR-capacity clearing mechanism from configuration.

    Resolution order (first match wins):
    1. ``markets.afrr_capacity.clearing_mechanism`` — the preferred location,
       since the clearing rule is a property of the market.
    2. Legacy ``strategy.clearing_mechanism`` field.
    3. Legacy strategy name ``hybrid_etes_gas_pay_as_cleared_capacity``.
    4. Default ``"pay_as_bid"``.
    """

    mechanism = market_clearing_mechanism(config, _AFRR_CAPACITY_MARKET)
    if mechanism is None:
        strategy_cfg = config.case.get("strategy") or {}
        legacy_field = strategy_cfg.get("clearing_mechanism")
        if legacy_field is not None:
            mechanism = str(legacy_field).strip().lower()
        elif config.strategy_name == _PAY_AS_CLEARED_STRATEGY_NAME:
            mechanism = "pay_as_cleared"
        else:
            mechanism = "pay_as_bid"
    if mechanism not in _VALID_CLEARING_MECHANISMS:
        options = ", ".join(_VALID_CLEARING_MECHANISMS)
        raise ValueError(
            f"Unknown afrr_capacity clearing_mechanism '{mechanism}'. Valid options: {options}."
        )
    return mechanism


def resolve_energy_clearing_mechanism(config: CaseConfig) -> str:
    """Determine the aFRR-energy clearing mechanism from configuration.

    aFRR energy is settled at the market clearing price, so ``pay_as_cleared``
    is the only mechanism implemented today. The field is accepted under
    ``markets.afrr_energy.clearing_mechanism`` for symmetry with capacity;
    a ``pay_as_bid`` value raises a clear error rather than silently being
    ignored. Defaults to ``"pay_as_cleared"``.
    """

    mechanism = market_clearing_mechanism(config, _AFRR_ENERGY_MARKET)
    if mechanism is None:
        return "pay_as_cleared"
    if mechanism not in _VALID_CLEARING_MECHANISMS:
        options = ", ".join(_VALID_CLEARING_MECHANISMS)
        raise ValueError(
            f"Unknown afrr_energy clearing_mechanism '{mechanism}'. Valid options: {options}."
        )
    if mechanism != "pay_as_cleared":
        raise ValueError(
            "afrr_energy clearing_mechanism 'pay_as_bid' is not implemented; "
            "aFRR energy is currently settled pay_as_cleared. "
            "Set clearing_mechanism: pay_as_cleared or omit the field."
        )
    return mechanism


def capacity_pricing_rule(mechanism: str) -> str:
    """Return the aFRR-capacity clearing mechanism in force for this case.

    Selected from configuration (``markets.afrr_capacity.clearing_mechanism``,
    the legacy ``strategy.clearing_mechanism`` field, or the legacy
    ``hybrid_etes_gas_pay_as_cleared_capacity`` strategy name) and defaulting
    to ``"pay_as_bid"``. Reported in outputs as ``afrr_capacity_pricing_rule``.
    """

    return mechanism


def capacity_settlement_price(
    mechanism: str,
    capacity_bid_price_eur_per_mw_h: float,
    clearing_price_eur_per_mw_h: float,
) -> float:
    """Return the awarded-capacity settlement price in EUR/MW/h.

    - ``pay_as_bid``: the operator is paid its own submitted bid price.
    - ``pay_as_cleared``: every awarded bid is paid the marginal clearing
      price, independent of the submitted bid.
    """

    if mechanism == "pay_as_cleared":
        return clearing_price_eur_per_mw_h
    return capacity_bid_price_eur_per_mw_h


def capacity_bid_price(
    mechanism: str,
    minimum_acceptable_price_eur_per_mw_h: float,
    market_reference_price_eur_per_mw_h: float,
) -> float:
    """Return the submitted bid price for the capacity product in EUR/MW/h.

    - ``pay_as_bid``: the configured capacity-price signal is interpreted as
      the submitted and awarded bid price, preserving prior behaviour.
    - ``pay_as_cleared``: the operator bids its true reservation price (the
      minimum acceptable price, i.e. opportunity cost plus capacity margin)
      and is awarded whenever the clearing price covers it.
    """

    if mechanism == "pay_as_cleared":
        return minimum_acceptable_price_eur_per_mw_h
    return market_reference_price_eur_per_mw_h
