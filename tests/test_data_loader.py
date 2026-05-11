# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd

from etes_market_model.config.case_config import CaseConfig
from etes_market_model.data.data_loader import DataLoader
from etes_market_model.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DE"


def test_forecasts_are_loaded_and_filtered() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    strategy = HybridETESGasStrategy(config)
    loader = DataLoader(config, input_dir=CASE_DIR)
    plants = loader.load_plants()
    required_columns = loader.required_forecast_columns(
        plants,
        extra_required_columns=strategy.required_forecast_columns(),
    )
    forecasts = loader.load_forecasts(required_columns)

    assert isinstance(forecasts.index, pd.DatetimeIndex)
    assert forecasts.index.min() == pd.Timestamp("2025-01-01 00:00")
    assert forecasts.index.max() == pd.Timestamp("2025-01-07 23:45")
    assert len(forecasts) == 7 * 24 * 4
    assert {"plant_1_heat_demand", "DE_DA_price", "natural_gas_price", "co2_price"}.issubset(
        forecasts.columns
    )
