# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest
from conftest import case_config_text

from flexi_mod.config.case_config import CaseConfig, ConfigError
from flexi_mod.simulation.simulation_runner import _decision_windows

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DA_ID_buy"


def test_config_can_be_loaded() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    assert config.case_name == "hybrid_ETES_DA_ID_buy"
    assert config.study_case == "hybrid_ETES_DA_ID_buy"
    assert config.strategy_name == "hybrid_etes_gas"
    assert config.output_folder_name == "hybrid_ETES_DA_ID_buy_hybrid_etes_gas"
    assert config.timestep_minutes == 15
    assert config.market_sequence == [
        "afrr_capacity",
        "day_ahead",
        "intraday_continuous",
        "afrr_energy",
    ]


def test_required_sections_exist() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    assert "cases" in config.raw
    for section in ["strategy", "solver", "market_sequence", "markets"]:
        assert section in config.case
    assert "paths" not in config.raw
    assert "outputs" not in config.raw
    assert "benchmark" not in config.case["strategy"]
    assert config.dispatch_setting("dispatch_method") == "pyomo"


def test_multiple_cases_require_explicit_study_case(tmp_path: Path) -> None:
    case_dir = tmp_path / "multi_case"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        """
cases:
  case_a:
    name: case_a
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
    markets:
      day_ahead:
        enabled: true
        signals:
          price: DE_DA_price
  case_b:
    name: case_b
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
    markets:
      day_ahead:
        enabled: true
        signals:
          price: DE_DA_price
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="multiple study cases"):
        CaseConfig.from_case_dir(case_dir)

    config = CaseConfig.from_case_dir(case_dir, study_case="case_b")
    assert config.case_name == "case_b"
    assert config.study_case == "case_b"


def test_old_top_level_config_format_is_rejected(tmp_path: Path) -> None:
    case_dir = tmp_path / "old_format"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        """
case:
  name: old_format
strategy:
  name: hybrid_etes_gas
market_sequence: []
markets: {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="top-level 'cases:' mapping"):
        CaseConfig.from_case_dir(case_dir)


def test_enabled_markets_have_required_signals() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    assert "day_ahead" in config.enabled_markets
    assert config.market_signal("day_ahead", "price") == "DE_DA_price"
    assert config.market_signal("intraday_continuous", "price") == "DE_ID3_price"
    assert config.market_signal("afrr_energy", "price") == "aFRR_energy_down_price"
    assert config.market_signal("afrr_energy", "system_activation") == "aFRR_energy_down_quantity"
    assert config.market_signal("afrr_capacity", "price") == "aFRR_capacity_down_price"
    assert config.market("afrr_capacity")["gate_open"] == {
        "day_relation": "D-7",
        "time": "10:00",
    }
    assert config.market("afrr_capacity")["gate_close"] == {
        "day_relation": "D-1",
        "time": "09:00",
    }


def test_afrr_validity_period_must_match_timestep(tmp_path: Path) -> None:
    case_dir = tmp_path / "bad_afrr_validity"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        case_config_text(
            """
case:
  name: bad_afrr_validity
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
  - afrr_energy
markets:
  day_ahead:
    enabled: true
    signals:
      price: DE_DA_price
  afrr_energy:
    enabled: true
    direction: down
    signals:
      price: aFRR_energy_down_price
      system_activation: aFRR_energy_down_quantity
    product_rules:
      validity_period_minutes: 30
""".strip()
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="validity_period_minutes"):
        CaseConfig.from_case_dir(case_dir)


def test_enabled_afrr_capacity_must_be_before_day_ahead(tmp_path: Path) -> None:
    case_dir = tmp_path / "bad_capacity_order"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        case_config_text(
            """
case:
  name: bad_capacity_order
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
  - afrr_energy
  - afrr_capacity
markets:
  day_ahead:
    enabled: true
    signals:
      price: DE_DA_price
  afrr_energy:
    enabled: true
    direction: down
    signals:
      price: aFRR_energy_down_price
      system_activation: aFRR_energy_down_quantity
    product_rules:
      validity_period_minutes: 15
  afrr_capacity:
    enabled: true
    direction: down
    product_length: "4h"
    price_unit: "EUR_per_MW_per_h"
    signals:
      price: aFRR_capacity_down_price
""".strip()
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="before day_ahead"):
        CaseConfig.from_case_dir(case_dir)


def test_enabled_afrr_capacity_allows_disabled_afrr_energy(tmp_path: Path) -> None:
    case_dir = tmp_path / "capacity_without_energy"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        case_config_text(
            """
case:
  name: capacity_without_energy
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
  - afrr_capacity
  - day_ahead
  - afrr_energy
markets:
  afrr_capacity:
    enabled: true
    direction: down
    product_length: "4h"
    price_unit: "EUR_per_MW_per_h"
    signals:
      price: aFRR_capacity_down_price
  day_ahead:
    enabled: true
    signals:
      price: DE_DA_price
  afrr_energy:
    enabled: false
    direction: down
    signals:
      price: aFRR_energy_down_price
      system_activation: aFRR_energy_down_quantity
""".strip()
        ),
        encoding="utf-8",
    )

    config = CaseConfig.from_case_dir(case_dir)
    assert "afrr_capacity" in config.enabled_markets
    assert "afrr_energy" not in config.enabled_markets


def test_additional_charges_flag_must_be_boolean(tmp_path: Path) -> None:
    case_dir = tmp_path / "bad_additional_charges"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        case_config_text(
            """
case:
  name: bad_additional_charges
  country: DE
  timestep_minutes: 15
  simulation_start: "2025-01-01 00:00"
  simulation_end: "2025-01-01 00:45"
  additional_charges: "yes"
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
markets:
  day_ahead:
    enabled: true
    signals:
      price: DE_DA_price
""".strip()
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="additional_charges"):
        CaseConfig.from_case_dir(case_dir)


def test_rolling_step_defines_market_commit_window(tmp_path: Path) -> None:
    case_dir = tmp_path / "window_case"
    case_dir.mkdir()
    _write_window_config(case_dir / "config.yaml", dispatch_horizon_hours=24)
    config = CaseConfig.from_case_dir(case_dir)
    forecasts = pd.DataFrame(index=pd.date_range("2025-01-01 00:00", periods=96 * 3, freq="15min"))

    windows = _decision_windows(config, forecasts)

    assert len(windows) == 3
    assert all(len(window.commit_index) == 96 for window in windows)


def test_larger_dispatch_horizon_keeps_daily_commit_windows(tmp_path: Path) -> None:
    case_dir = tmp_path / "two_day_window_case"
    case_dir.mkdir()
    _write_window_config(
        case_dir / "config.yaml",
        dispatch_horizon_hours=48,
        rolling_step_hours=24,
    )
    config = CaseConfig.from_case_dir(case_dir)
    forecasts = pd.DataFrame(index=pd.date_range("2025-01-01 00:00", periods=96 * 4, freq="15min"))

    windows = _decision_windows(config, forecasts)

    assert len(windows) == 4
    assert all(len(window.commit_index) == 96 for window in windows)
    assert len(windows[0].forecasts) == 192


def _write_window_config(
    path: Path,
    dispatch_horizon_hours: int,
    rolling_step_hours: int | None = None,
) -> None:
    rolling_step_line = (
        f"    rolling_step_hours: {rolling_step_hours}\n" if rolling_step_hours is not None else ""
    )
    path.write_text(
        case_config_text(
            f"""
case:
  name: window_case
  country: DE
  timestep_minutes: 15
  simulation_start: "2025-01-01 00:00"
  simulation_end: "2025-01-04 23:45"
strategy:
  name: hybrid_etes_gas
  dispatch:
    dispatch_method: pyomo
    rolling_horizon_enabled: true
    dispatch_horizon_hours: {dispatch_horizon_hours}
{rolling_step_line.rstrip()}
solver:
  name: highs
  fallback_solvers: []
  tee: false
market_sequence:
  - day_ahead
markets:
  day_ahead:
    enabled: true
    signals:
      price: DE_DA_price
""".strip()
        ),
        encoding="utf-8",
    )
