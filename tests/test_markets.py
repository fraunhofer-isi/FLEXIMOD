# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd
import pytest

from flexi_mod.markets.afrr_capacity import AFRRCapacityMarket
from flexi_mod.markets.afrr_energy import AFRRDownEnergyMarket, AFRRUpEnergyMarket
from flexi_mod.markets.base_market import MarketConfigError, product_resolution_to_minutes
from flexi_mod.markets.day_ahead import DayAheadMarket
from flexi_mod.markets.intraday_continuous import IntradayContinuousMarket


def test_product_resolution_parser() -> None:
    assert product_resolution_to_minutes("15min") == 15
    assert product_resolution_to_minutes("1h") == 60


def test_base_market_required_signal_validation() -> None:
    market = DayAheadMarket(
        name="day_ahead",
        config={"enabled": True, "signals": {}},
    )

    with pytest.raises(MarketConfigError, match="price"):
        market.validate_config()


def test_day_ahead_market_prepares_price_series() -> None:
    forecasts = _forecast_frame({"DE_DA_price": [50.0, 55.0]})
    market = DayAheadMarket(
        name="day_ahead",
        config={"enabled": True, "product_resolution": "1h", "signals": {"price": "DE_DA_price"}},
    )

    data = market.prepare_market_data(forecasts)

    assert data["day_ahead_price_EUR_per_MWh"].tolist() == [50.0, 55.0]


def test_intraday_market_prepares_price_availability() -> None:
    forecasts = _forecast_frame({"DE_ID3_price": [45.0, None]})
    market = IntradayContinuousMarket(
        name="intraday_continuous",
        config={
            "enabled": True,
            "product_resolution": "15min",
            "signals": {"price": "DE_ID3_price"},
        },
    )

    data = market.prepare_market_data(forecasts)

    assert data["IDC_price_EUR_per_MWh"].iloc[0] == pytest.approx(45.0)
    assert data["IDC_price_available"].tolist() == [True, False]


def test_intraday_allowed_actions_default_to_buy_and_sell() -> None:
    market = IntradayContinuousMarket(
        name="intraday_continuous",
        config={
            "enabled": True,
            "product_resolution": "15min",
            "signals": {"price": "DE_ID3_price"},
        },
    )

    market.validate_config(timestep_minutes=15)

    assert market.buy_enabled is True
    assert market.sell_enabled is True


def test_intraday_allowed_actions_can_disable_each_direction() -> None:
    market = IntradayContinuousMarket(
        name="intraday_continuous",
        config={
            "enabled": True,
            "product_resolution": "15min",
            "allowed_actions": {"buy": False, "sell": True},
            "signals": {"price": "DE_ID3_price"},
        },
    )

    market.validate_config(timestep_minutes=15)

    assert market.buy_enabled is False
    assert market.sell_enabled is True


def test_intraday_allowed_actions_reject_non_boolean_values() -> None:
    market = IntradayContinuousMarket(
        name="intraday_continuous",
        config={
            "enabled": True,
            "product_resolution": "15min",
            "allowed_actions": {"buy": "yes", "sell": True},
            "signals": {"price": "DE_ID3_price"},
        },
    )

    with pytest.raises(MarketConfigError, match="allowed_actions.buy"):
        market.validate_config(timestep_minutes=15)


def test_afrr_down_market_prepares_cleaned_activation_data() -> None:
    forecasts = _forecast_frame(
        {
            "aFRR_energy_down_price": [0.0, 20.0],
            "aFRR_energy_down_quantity": [2.0, -1.0],
        }
    )
    market = AFRRDownEnergyMarket(
        name="afrr_energy",
        config={
            "enabled": True,
            "product_resolution": "15min",
            "product_rules": {"validity_period_minutes": 15},
            "signals": {
                "price": "aFRR_energy_down_price",
                "system_activation": "aFRR_energy_down_quantity",
            },
            "interpretation": {"activation_unit": "MW"},
        },
    )

    cleaned = market.prepare_market_data(forecasts, timestep_hours=0.25)

    assert cleaned.frame["afrr_system_activation_MWh"].tolist() == [0.5, 0.25]
    assert cleaned.quality_summary["aFRR_down_negative_quantity_rows"].iloc[0] == 1
    assert cleaned.quality_summary["aFRR_down_price_zero_with_activation_rows"].iloc[0] == 1


def test_afrr_down_market_validates_product_period() -> None:
    market = AFRRDownEnergyMarket(
        name="afrr_energy",
        config={
            "enabled": True,
            "product_rules": {"validity_period_minutes": 30},
            "signals": {
                "price": "aFRR_energy_down_price",
                "system_activation": "aFRR_energy_down_quantity",
            },
            "interpretation": {"activation_unit": "MW"},
        },
    )

    with pytest.raises(MarketConfigError, match="validity_period_minutes"):
        market.validate_config(timestep_minutes=15)


def test_afrr_capacity_market_generates_blocks() -> None:
    forecasts = _forecast_frame({"aFRR_capacity_down_price": [5.0] * 16})
    market = AFRRCapacityMarket(
        name="afrr_capacity",
        config={
            "enabled": True,
            "direction": "down",
            "product_length": "4h",
            "price_unit": "EUR_per_MW_per_h",
            "signals": {"price": "aFRR_capacity_down_price"},
        },
    )

    data = market.prepare_market_data(forecasts, timestep_hours=0.25)

    assert data.frame["afrr_capacity_block_id"].nunique() == 1
    assert data.block_summary["block_duration_h"].iloc[0] == pytest.approx(4.0)
    assert data.block_summary["capacity_price_EUR_per_MW_h"].iloc[0] == pytest.approx(5.0)


def test_afrr_capacity_market_warns_on_inconsistent_block_prices() -> None:
    forecasts = _forecast_frame({"aFRR_capacity_down_price": [5.0] * 15 + [6.0]})
    market = AFRRCapacityMarket(
        name="afrr_capacity",
        config={
            "enabled": True,
            "direction": "down",
            "product_length": "4h",
            "price_unit": "EUR_per_MW_per_h",
            "signals": {"price": "aFRR_capacity_down_price"},
        },
    )

    with pytest.warns(UserWarning, match="prices differ inside block"):
        data = market.prepare_market_data(forecasts, timestep_hours=0.25)

    assert data.block_summary["price_inconsistency_flag"].iloc[0]


def test_afrr_capacity_market_missing_price_blocks_bid_data() -> None:
    forecasts = _forecast_frame({"aFRR_capacity_down_price": [None] * 16})
    market = AFRRCapacityMarket(
        name="afrr_capacity",
        config={
            "enabled": True,
            "direction": "down",
            "product_length": "4h",
            "price_unit": "EUR_per_MW_per_h",
            "signals": {"price": "aFRR_capacity_down_price"},
        },
    )

    data = market.prepare_market_data(forecasts, timestep_hours=0.25)

    assert data.block_summary["missing_capacity_price_flag"].iloc[0]
    assert data.block_summary["capacity_price_EUR_per_MW_h"].iloc[0] == pytest.approx(0.0)


def test_afrr_up_energy_placeholder_is_non_operational() -> None:
    forecasts = _forecast_frame(
        {
            "aFRR_energy_up_price": [25.0],
            "aFRR_energy_up_quantity": [0.0],
        }
    )
    market = AFRRUpEnergyMarket(
        name="afrr_energy",
        config={
            "enabled": False,
            "signals": {
                "price": "aFRR_energy_up_price",
                "system_activation": "aFRR_energy_up_quantity",
            },
        },
    )

    data = market.prepare_market_data(forecasts)

    assert data.empty
    assert data.index.equals(forecasts.index)


def _forecast_frame(columns: dict[str, list[float | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        columns,
        index=pd.date_range("2025-01-01", periods=len(next(iter(columns.values()))), freq="15min"),
    )
