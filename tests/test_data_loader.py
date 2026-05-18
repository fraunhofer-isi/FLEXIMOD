# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader, DataValidationError
from flexi_mod.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy


def test_forecasts_are_loaded_and_filtered(tmp_path: Path) -> None:
    case_dir = _write_loader_case(tmp_path)
    config = CaseConfig.from_case_dir(case_dir)
    strategy = HybridETESGasStrategy(config)
    loader = DataLoader(config, input_dir=case_dir)
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
    assert {
        "plant_1_heat_demand",
        "DE_DA_price",
        "DE_ID3_price",
        "natural_gas_price",
        "co2_price",
        "aFRR_energy_down_price",
        "aFRR_energy_down_quantity",
    }.issubset(forecasts.columns)


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
  simulation_end: "2025-01-01 01:00"
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

    with pytest.raises(DataValidationError, match="Intraday continuous, aFRR energy"):
        loader.load_forecasts(required_columns={"DE_ID3_price"})


def test_additional_charges_are_loaded_only_when_enabled(tmp_path: Path) -> None:
    case_dir = _write_loader_case(tmp_path)
    config_path = case_dir / "config.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'simulation_end: "2025-01-07 23:45"',
            'simulation_end: "2025-01-07 23:45"\n  additional_charges: true',
        ),
        encoding="utf-8",
    )
    (case_dir / "additional_charges.csv").write_text(
        "\n".join(
            [
                "component,unit,plant_1",
                "Network consumption price,EUR/MWh,6.9",
                "Metering and operation,EUR/MWh,2.0",
                "Concession fees,EUR/MWh,1.1",
                "Surcharges and levies,EUR/MWh,1.6",
                "Electricity tax,EUR/MWh,0.5",
            ]
        ),
        encoding="utf-8",
    )

    config = CaseConfig.from_case_dir(case_dir)
    loader = DataLoader(config, input_dir=case_dir)
    plants = loader.load_plants()

    assert loader.load_additional_charges(plants)["plant_1"] == pytest.approx(12.1)


def test_day_ahead_only_ignores_dst_irregularities_outside_case_range(tmp_path: Path) -> None:
    case_dir = _write_day_ahead_only_case(tmp_path)
    index = pd.date_range("2025-01-01 00:00", "2025-04-01 00:00", freq="15min")
    index = index.difference(pd.date_range("2025-03-30 02:00", "2025-03-30 02:45", freq="15min"))
    pd.DataFrame(
        {
            "datetime": [timestamp.strftime("%d.%m.%Y %H:%M") for timestamp in index],
            "plant_1_heat_demand": [2.0] * len(index),
            "DE_DA_price": [50.0] * len(index),
            "natural_gas_price": [80.0] * len(index),
        }
    ).to_csv(case_dir / "forecasts_df.csv", index=False)

    config = CaseConfig.from_case_dir(case_dir)
    loader = DataLoader(config, input_dir=case_dir)

    forecasts = loader.load_forecasts(
        required_columns={"plant_1_heat_demand", "DE_DA_price", "natural_gas_price"}
    )

    assert forecasts.index.min() == pd.Timestamp("2025-01-01 00:00")
    assert forecasts.index.max() == pd.Timestamp("2025-01-02 23:45")
    assert len(forecasts) == 2 * 24 * 4


def test_forecast_resolution_accepts_spring_dst_clock_change(tmp_path: Path) -> None:
    case_dir = _write_dst_loader_case(
        tmp_path,
        simulation_start="2025-03-30 00:00",
        simulation_end="2025-03-30 23:45",
    )
    index = pd.date_range(
        "2025-03-30 00:00",
        "2025-03-30 23:45",
        freq="15min",
        tz="Europe/Berlin",
    ).tz_localize(None)
    _write_forecast_rows(case_dir, index)

    config = CaseConfig.from_case_dir(case_dir)
    loader = DataLoader(config, input_dir=case_dir)

    forecasts = loader.load_forecasts(required_columns={"DE_DA_price", "plant_1_heat_demand"})

    assert forecasts.index.min() == pd.Timestamp("2025-03-30 00:00", tz="Europe/Berlin")
    assert forecasts.index.max() == pd.Timestamp("2025-03-30 23:45", tz="Europe/Berlin")
    assert len(forecasts) == 92
    assert not any(timestamp.hour == 2 for timestamp in forecasts.index)


def test_forecast_resolution_accepts_autumn_dst_duplicate_hour(tmp_path: Path) -> None:
    case_dir = _write_dst_loader_case(
        tmp_path,
        simulation_start="2025-10-26 00:00",
        simulation_end="2025-10-26 23:45",
    )
    index = pd.date_range(
        "2025-10-26 00:00",
        "2025-10-26 23:45",
        freq="15min",
        tz="Europe/Berlin",
    ).tz_localize(None)
    _write_forecast_rows(case_dir, index)

    config = CaseConfig.from_case_dir(case_dir)
    loader = DataLoader(config, input_dir=case_dir)

    forecasts = loader.load_forecasts(required_columns={"DE_DA_price", "plant_1_heat_demand"})

    assert len(forecasts) == 100
    assert forecasts.index.min() == pd.Timestamp("2025-10-26 00:00", tz="Europe/Berlin")
    assert forecasts.index.max() == pd.Timestamp("2025-10-26 23:45", tz="Europe/Berlin")
    assert forecasts.index.is_unique


def test_forecast_resolution_still_rejects_real_non_dst_gap(tmp_path: Path) -> None:
    case_dir = _write_dst_loader_case(
        tmp_path,
        simulation_start="2025-01-01 00:00",
        simulation_end="2025-01-01 01:00",
    )
    index = pd.DatetimeIndex(
        [
            "2025-01-01 00:00",
            "2025-01-01 00:15",
            "2025-01-01 00:45",
            "2025-01-01 01:00",
        ]
    )
    _write_forecast_rows(case_dir, index)

    config = CaseConfig.from_case_dir(case_dir)
    loader = DataLoader(config, input_dir=case_dir)

    with pytest.raises(DataValidationError, match="regular time grid"):
        loader.load_forecasts(required_columns={"DE_DA_price", "plant_1_heat_demand"})


def _write_loader_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "loader_case"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        """
case:
  name: loader_case
  country: DE
  timestep_minutes: 15
  simulation_start: "2025-01-01 00:00"
  simulation_end: "2025-01-07 23:45"
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
  - afrr_energy
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
  afrr_energy:
    enabled: true
    direction: down
    product_resolution: "15min"
    product_rules:
      min_bid_mw: 1.0
      bid_increment_mw: 1.0
      validity_period_minutes: 15
    signals:
      price: aFRR_energy_down_price
      system_activation: aFRR_energy_down_quantity
    interpretation:
      activation_unit: MW
""".strip(),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "name": "plant_1",
                "unit_type": "steam_plant",
                "technology": "thermal_storage",
                "demand": "plant_1_heat_demand",
                "max_capacity": 4.0,
                "min_capacity": 0.0,
                "max_power_charge": 7.0,
                "max_power_discharge": 7.0,
                "initial_soc": 0.0,
                "efficiency_charge": 0.92,
                "efficiency_discharge": 0.92,
                "storage_loss_rate": 0.0,
            },
            {
                "name": "plant_1",
                "unit_type": "steam_plant",
                "technology": "boiler",
                "fuel_type": "natural_gas",
                "max_power": 5.0,
                "min_power": 0.0,
                "efficiency": 0.9,
            },
        ]
    ).to_csv(case_dir / "plants.csv", index=False)
    index = pd.date_range("2025-01-01 00:00", "2025-01-07 23:45", freq="15min")
    pd.DataFrame(
        {
            "datetime": [timestamp.strftime("%d.%m.%Y %H:%M") for timestamp in index],
            "plant_1_heat_demand": [2.0] * len(index),
            "DE_DA_price": [50.0] * len(index),
            "DE_ID3_price": [45.0] * len(index),
            "natural_gas_price": [80.0] * len(index),
            "co2_price": [0.0] * len(index),
            "aFRR_energy_down_price": [20.0] * len(index),
            "aFRR_energy_down_quantity": [0.0] * len(index),
        }
    ).to_csv(case_dir / "forecasts_df.csv", index=False)
    return case_dir


def _write_day_ahead_only_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "da_only_loader_case"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        """
case:
  name: da_only_loader_case
  country: DE
  timestep_minutes: 15
  simulation_start: "2025-01-01 00:00"
  simulation_end: "2025-01-02 23:45"
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
  - afrr_energy
markets:
  day_ahead:
    enabled: true
    signals:
      price: DE_DA_price
  intraday_continuous:
    enabled: false
    signals:
      price: DE_ID3_price
  afrr_energy:
    enabled: false
    signals:
      price: aFRR_energy_down_price
      system_activation: aFRR_energy_down_quantity
""".strip(),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "name": "plant_1",
                "unit_type": "steam_plant",
                "technology": "thermal_storage",
                "demand": "plant_1_heat_demand",
                "max_capacity": 4.0,
                "min_capacity": 0.0,
                "max_power_charge": 7.0,
                "max_power_discharge": 7.0,
                "initial_soc": 0.0,
                "efficiency_charge": 0.92,
                "efficiency_discharge": 0.92,
                "storage_loss_rate": 0.0,
            },
            {
                "name": "plant_1",
                "unit_type": "steam_plant",
                "technology": "boiler",
                "fuel_type": "natural_gas",
                "max_power": 5.0,
                "min_power": 0.0,
                "efficiency": 0.9,
            },
        ]
    ).to_csv(case_dir / "plants.csv", index=False)
    return case_dir


def _write_dst_loader_case(
    tmp_path: Path,
    simulation_start: str,
    simulation_end: str,
) -> Path:
    case_dir = tmp_path / "dst_loader_case"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        f"""
case:
  name: dst_loader_case
  country: DE
  timestep_minutes: 15
  simulation_start: "{simulation_start}"
  simulation_end: "{simulation_end}"
  timezone: "Europe/Berlin"
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
    return case_dir


def _write_forecast_rows(case_dir: Path, index: pd.DatetimeIndex) -> None:
    pd.DataFrame(
        {
            "datetime": [timestamp.strftime("%d.%m.%Y %H:%M") for timestamp in index],
            "plant_1_heat_demand": [2.0] * len(index),
            "DE_DA_price": [50.0] * len(index),
            "natural_gas_price": [80.0] * len(index),
        }
    ).to_csv(case_dir / "forecasts_df.csv", index=False)
