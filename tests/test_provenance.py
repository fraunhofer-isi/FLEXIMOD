# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from conftest import case_config_text

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.simulation.provenance import ProvenanceCollector

SIM_START = "2025-01-01 00:00"
SIM_END = "2025-01-01 01:00"


def test_build_emits_expected_top_level_sections(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    metadata = collector.build(
        plants=_plants(),
        forecasts=_forecasts(),
        started_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        duration_seconds=42,
    )

    assert metadata["schema_version"] == ProvenanceCollector.SCHEMA_VERSION
    required_sections = {"schema_version", "run", "code", "inputs", "parameters", "environment"}
    assert required_sections <= set(metadata)


def test_run_section_uses_utc_iso_timestamp(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    metadata = collector.build(
        plants=_plants(),
        forecasts=_forecasts(),
        started_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        duration_seconds=90,
    )

    assert metadata["run"] == {
        "id": "2025-01-01T12:00:00Z",
        "started_at": "2025-01-01T12:00:00Z",
        "duration_seconds": 90,
    }


def test_parameters_include_forecast_prices(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    metadata = collector.build(_plants(), _forecasts(), _now(), 1.0)

    forecasts_params = metadata["parameters"]["forecasts"]
    assert forecasts_params["natural_gas_price"] == 80.0
    assert forecasts_params["co2_price"] == 0.0


def test_parameters_list_plant_technology_and_specs(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    metadata = collector.build(_plants(), _forecasts(), _now(), 1.0)

    plants_params = metadata["parameters"]["plants"]
    assert len(plants_params) == 2
    thermal_storage = plants_params[0]
    assert thermal_storage["name"] == "plant_1"
    assert thermal_storage["technology"] == "thermal_storage"
    assert thermal_storage["max_capacity"] == 4.0
    assert thermal_storage["max_power_charge"] == 7.0
    boiler = plants_params[1]
    assert boiler["technology"] == "boiler"
    assert boiler["efficiency"] == 0.9
    assert "max_capacity" not in boiler


def test_parameters_include_enabled_markets(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    metadata = collector.build(_plants(), _forecasts(), _now(), 1.0)

    markets = metadata["parameters"]["markets"]
    assert markets["enabled"] == ["day_ahead", "intraday_continuous"]
    assert markets["sequence"] == ["day_ahead", "intraday_continuous"]


def test_parameters_include_simulation_strategy_and_solver(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    metadata = collector.build(_plants(), _forecasts(), _now(), 1.0)
    parameters = metadata["parameters"]

    assert parameters["simulation"]["case_name"] == "provenance_case"
    assert parameters["simulation"]["country"] == "DE"
    assert parameters["strategy"]["name"] == "hybrid_etes_gas"
    assert parameters["solver"]["name"] == "highs"


def test_inputs_list_data_files(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    metadata = collector.build(_plants(), _forecasts(), _now(), 1.0)

    names = [entry["name"] for entry in metadata["inputs"]]
    assert {"forecasts", "plants", "config"} <= set(names)
    forecasts_entry = next(entry for entry in metadata["inputs"] if entry["name"] == "forecasts")
    assert forecasts_entry["rows"] == 5
    plants_entry = next(entry for entry in metadata["inputs"] if entry["name"] == "plants")
    assert plants_entry["rows"] == 2


def test_inputs_include_additional_charges_when_enabled(tmp_path: Path) -> None:
    collector = _collector(tmp_path, additional_charges=True)

    metadata = collector.build(_plants(), _forecasts(), _now(), 1.0)

    names = [entry["name"] for entry in metadata["inputs"]]
    assert "additional_charges" in names


def test_to_json_round_trips(tmp_path: Path) -> None:
    collector = _collector(tmp_path)

    metadata = collector.build(_plants(), _forecasts(), _now(), 1.0)
    loaded = json.loads(ProvenanceCollector.to_json(metadata))

    assert loaded == metadata


def _collector(tmp_path: Path, additional_charges: bool = False) -> ProvenanceCollector:
    case_dir = _write_case(tmp_path, additional_charges=additional_charges)
    config = CaseConfig.from_case_dir(case_dir)
    loader = DataLoader(config, input_dir=case_dir)
    return ProvenanceCollector(config, loader, tmp_path)


def _write_case(tmp_path: Path, additional_charges: bool = False) -> Path:
    case_dir = tmp_path / "provenance_case"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        case_config_text(_case_config_text(additional_charges)),
        encoding="utf-8",
    )
    _plants().to_csv(case_dir / "plants.csv", index=False)
    _forecasts().to_csv(case_dir / "forecasts_df.csv", index=False)
    if additional_charges:
        (case_dir / "additional_charges.csv").write_text(
            "component,unit,plant_1\n"
            "Network consumption price,EUR/MWh,6.9\n"
            "Metering and operation,EUR/MWh,2.0\n",
            encoding="utf-8",
        )
    return case_dir


def _case_config_text(additional_charges: bool) -> str:
    return f"""
case:
  name: provenance_case
  country: DE
  timestep_minutes: 15
  simulation_start: "{SIM_START}"
  simulation_end: "{SIM_END}"
  additional_charges: {"true" if additional_charges else "false"}
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
""".strip()


def _plants() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "plant_1",
                "unit_type": "steam_plant",
                "technology": "thermal_storage",
                "demand": "plant_1_heat_demand",
                "max_power": 7.0,
                "efficiency": 0.0,
                "max_capacity": 4.0,
                "min_capacity": 0.0,
                "max_power_charge": 7.0,
                "max_power_discharge": 7.0,
            },
            {
                "name": "plant_1",
                "unit_type": "steam_plant",
                "technology": "boiler",
                "demand": "",
                "max_power": 5.0,
                "efficiency": 0.9,
                "max_capacity": None,
                "min_capacity": None,
                "max_power_charge": None,
                "max_power_discharge": None,
            },
        ]
    )


def _forecasts() -> pd.DataFrame:
    index = pd.date_range(SIM_START, SIM_END, freq="15min")
    return pd.DataFrame(
        {
            "datetime": [timestamp.strftime("%d.%m.%Y %H:%M") for timestamp in index],
            "plant_1_heat_demand": [2.0] * len(index),
            "DE_DA_price": [50.0] * len(index),
            "DE_ID3_price": [45.0] * len(index),
            "DE_ID3_volume": [0.0] * len(index),
            "natural_gas_price": [80.0] * len(index),
            "co2_price": [0.0] * len(index),
        }
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)
