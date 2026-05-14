# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest

from flexi_mod.config.case_config import CaseConfig, ConfigError

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DE"


def test_config_can_be_loaded() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    assert config.case_name == "hybrid_ETES_DE"
    assert config.timestep_minutes == 15


def test_required_sections_exist() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    for section in ["case", "strategy", "solver", "market_sequence", "markets"]:
        assert section in config.raw
    assert "paths" not in config.raw
    assert "outputs" not in config.raw
    assert "benchmark" not in config.raw["strategy"]
    assert config.dispatch_setting("dispatch_method") == "pyomo"


def test_enabled_markets_have_required_signals() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    assert config.enabled_markets == ["day_ahead", "intraday_continuous", "afrr_energy"]
    assert config.market_signal("day_ahead", "price") == "DE_DA_price"
    assert config.market_signal("intraday_continuous", "price") == "DE_ID3_price"
    assert config.market_signal("afrr_energy", "price") == "aFRR_energy_down_price"
    assert config.market_signal("afrr_energy", "system_activation") == "aFRR_energy_down_quantity"


def test_afrr_validity_period_must_match_timestep(tmp_path: Path) -> None:
    case_dir = tmp_path / "bad_afrr_validity"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
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
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="validity_period_minutes"):
        CaseConfig.from_case_dir(case_dir)
