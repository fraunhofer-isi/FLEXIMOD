# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from etes_market_model.config.case_config import CaseConfig
from etes_market_model.data.data_loader import DataLoader, DataValidationError
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


def test_idc_enabled_does_not_resample_price_grid(tmp_path: Path) -> None:
    case_dir = tmp_path / "idc_resolution_case"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        """
case:
  name: idc_resolution_case
  country: DE
  timestep_minutes: 15
  simulation_start: "2025-01-01 00:00"
  simulation_end: "2025-01-01 00:45"
strategy:
  name: hybrid_etes_gas
  dispatch:
    dispatch_method: pyomo
solver:
  name: highs
  fallback_solvers: []
  tee: false
market_sequence:
  - day_ahead
  - intraday_continuous
markets:
  day_ahead:
    enabled: true
    signals:
      price: DE_DA_price
  intraday_continuous:
    enabled: true
    signals:
      price: DE_ID3_price
      volume: DE_ID3_volume
""".strip(),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01 00:00", periods=2, freq="1h"),
            "plant_1_heat_demand": [2.0, 2.0],
            "DE_DA_price": [50.0, 55.0],
            "DE_ID3_price": [40.0, 45.0],
            "natural_gas_price": [80.0, 80.0],
        }
    ).to_csv(case_dir / "forecasts_df.csv", index=False)

    config = CaseConfig.from_case_dir(case_dir)
    loader = DataLoader(config, input_dir=case_dir)

    with pytest.raises(DataValidationError, match="Intraday continuous is enabled"):
        loader.load_forecasts(required_columns={"DE_ID3_price"})
