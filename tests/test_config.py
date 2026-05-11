# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

from etes_market_model.config.case_config import CaseConfig

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
    assert config.enabled_markets == ["day_ahead"]
    assert config.market_signal("day_ahead", "price") == "DE_DA_price"
