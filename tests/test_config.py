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
    assert config.market_sequence == [
        "afrr_capacity",
        "day_ahead",
        "intraday_continuous",
        "afrr_energy",
    ]


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


def test_enabled_afrr_capacity_must_be_before_day_ahead(tmp_path: Path) -> None:
    case_dir = tmp_path / "bad_capacity_order"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
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
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="before day_ahead"):
        CaseConfig.from_case_dir(case_dir)


def test_enabled_afrr_capacity_requires_enabled_afrr_energy(tmp_path: Path) -> None:
    case_dir = tmp_path / "capacity_without_energy"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
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
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="afrr_energy.enabled=true"):
        CaseConfig.from_case_dir(case_dir)


def test_additional_charges_flag_must_be_boolean(tmp_path: Path) -> None:
    case_dir = tmp_path / "bad_additional_charges"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
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
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="case.additional_charges"):
        CaseConfig.from_case_dir(case_dir)
