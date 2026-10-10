# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from flexi_mod.modeling.validation import (
    numeric_forecast,
    validate_required_forecasts,
)


def test_required_forecast_check_reports_context_and_missing_columns() -> None:
    forecasts = pd.DataFrame({"demand": [1.0]})

    with pytest.raises(
        ValueError,
        match="Steel plant 'steel_1'.*forecasts_df.csv.*electricity_price",
    ):
        validate_required_forecasts(
            forecasts,
            {"demand", "electricity_price"},
            "Steel plant 'steel_1'",
        )


def test_numeric_forecast_rejects_non_numeric_and_negative_demand() -> None:
    forecasts = pd.DataFrame({"demand": [1.0, "not-a-number"]})

    with pytest.raises(ValueError, match="forecast 'demand'.*finite and numeric"):
        numeric_forecast(
            forecasts,
            "demand",
            "Steel plant 'steel_1'",
            require_non_negative=True,
        )

    forecasts["demand"] = [1.0, -1.0]
    with pytest.raises(ValueError, match="forecast 'demand'.*cannot be negative"):
        numeric_forecast(
            forecasts,
            "demand",
            "Steel plant 'steel_1'",
            require_non_negative=True,
        )


def test_numeric_forecast_can_preserve_an_explicit_missing_value_rule() -> None:
    forecasts = pd.DataFrame({"idc_price": [20.0, None]})

    values = numeric_forecast(
        forecasts,
        "idc_price",
        "Steam plant 'steam_1'",
        allow_missing_values=True,
    )

    assert values.iloc[0] == pytest.approx(20.0)
    assert pd.isna(values.iloc[1])
