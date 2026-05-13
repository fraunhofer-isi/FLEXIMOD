# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd
import pytest

from etes_market_model.markets.afrr_energy import clean_afrr_down_data


def test_afrr_down_cleaning_treats_zero_price_as_valid() -> None:
    forecasts = _forecast_frame(prices=[0.0], quantities=[2.0])

    cleaned = clean_afrr_down_data(
        forecasts,
        price_col="aFRR_energy_down_price",
        quantity_col="aFRR_energy_down_quantity",
        activation_unit="MW",
        timestep_hours=0.25,
    )

    assert cleaned.frame["afrr_down_system_activation_MWh_clean"].iloc[0] == pytest.approx(0.5)
    assert cleaned.quality_summary["aFRR_down_price_zero_with_activation_rows"].iloc[0] == 1


def test_afrr_down_cleaning_skips_activation_without_price() -> None:
    forecasts = _forecast_frame(prices=[None], quantities=[2.0])

    with pytest.warns(UserWarning, match="aFRR down price contains missing values"):
        cleaned = clean_afrr_down_data(
            forecasts,
            price_col="aFRR_energy_down_price",
            quantity_col="aFRR_energy_down_quantity",
            activation_unit="MW",
            timestep_hours=0.25,
        )

    assert cleaned.frame["afrr_down_system_activation_MWh_clean"].iloc[0] == pytest.approx(0.0)
    assert cleaned.frame["afrr_energy_down_price_clean"].iloc[0] == pytest.approx(0.0)
    assert cleaned.frame["afrr_data_quality_flag"].iloc[0] == "activation_without_price"
    assert cleaned.quality_summary["aFRR_down_skipped_activation_MWh_due_to_missing_price"].iloc[
        0
    ] == pytest.approx(0.5)


def test_afrr_down_cleaning_missing_quantity_gives_zero_activation() -> None:
    forecasts = _forecast_frame(prices=[10.0], quantities=[None])

    with pytest.warns(UserWarning, match="aFRR down system activation contains missing values"):
        cleaned = clean_afrr_down_data(
            forecasts,
            price_col="aFRR_energy_down_price",
            quantity_col="aFRR_energy_down_quantity",
            activation_unit="MW",
            timestep_hours=0.25,
        )

    assert cleaned.frame["afrr_down_system_activation_MWh_clean"].iloc[0] == pytest.approx(0.0)
    assert cleaned.frame["afrr_data_quality_flag"].iloc[0] == "no_activation_or_missing_quantity"


def test_afrr_down_cleaning_negative_quantity_is_abs_and_flagged() -> None:
    forecasts = _forecast_frame(prices=[10.0], quantities=[-2.0])

    cleaned = clean_afrr_down_data(
        forecasts,
        price_col="aFRR_energy_down_price",
        quantity_col="aFRR_energy_down_quantity",
        activation_unit="MW",
        timestep_hours=0.25,
    )

    assert cleaned.frame["afrr_down_system_activation_MWh_clean"].iloc[0] == pytest.approx(0.5)
    assert cleaned.frame["afrr_data_quality_flag"].iloc[0] == "negative_quantity_converted"
    assert cleaned.quality_summary["aFRR_down_negative_quantity_rows"].iloc[0] == 1


def test_afrr_down_cleaning_mwh_unit_uses_quantity_directly() -> None:
    forecasts = _forecast_frame(prices=[10.0], quantities=[2.0])

    cleaned = clean_afrr_down_data(
        forecasts,
        price_col="aFRR_energy_down_price",
        quantity_col="aFRR_energy_down_quantity",
        activation_unit="MWh",
        timestep_hours=0.25,
    )

    assert cleaned.frame["afrr_down_system_activation_MWh_clean"].iloc[0] == pytest.approx(2.0)


def _forecast_frame(prices: list[float | None], quantities: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "aFRR_energy_down_price": prices,
            "aFRR_energy_down_quantity": quantities,
        },
        index=pd.date_range("2025-01-01", periods=len(prices), freq="15min"),
    )
