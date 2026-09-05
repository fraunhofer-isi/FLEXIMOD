# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.plants.building import Building
from flexi_mod.plants.technologies import ChargingStation, ElectricVehicle

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DA_ID_buy"


def _depot_rows(directionality: str = "bidirectional") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "bangkok_depot",
                "unit_type": "bus_depot",
                "technology": "electric_vehicle",
                "node": "grid_node",
                "demand": "depot_demand",
                "battery_capacity_mwh": 0.3,
                "fleet_size": 2,
                "max_power_charge": 0.2,
                "max_power_discharge": 0.2,
                "min_soc": 0.2,
                "max_soc": 1.0,
                "initial_soc": 0.7,
                "terminal_soc": 0.7,
                "efficiency_charge": 0.95,
                "efficiency_discharge": 0.95,
                "power_flow_directionality": directionality,
                "availability_column": "bus_availability",
                "trip_energy_column": "bus_trip_energy",
            },
            {
                "name": "bangkok_depot",
                "unit_type": "bus_depot",
                "technology": "charging_station",
                "node": "grid_node",
                "demand": "depot_demand",
                "charger_count": 2,
                "max_power_charge": 0.15,
                "max_power_discharge": 0.15,
                "power_flow_directionality": directionality,
            },
        ]
    )


def test_building_constructs_fleet_and_chargers_from_rows() -> None:
    building = Building.from_rows("bangkok_depot", _depot_rows())

    assert isinstance(building.electric_vehicle, ElectricVehicle)
    assert isinstance(building.charging_station, ChargingStation)
    assert building.electric_vehicle.total_capacity_mwh == pytest.approx(0.6)
    assert building.charging_station.total_max_power_charge_mw == pytest.approx(0.3)
    assert building.v2g_enabled
    assert building.required_forecast_columns() == {
        "depot_demand",
        "bus_availability",
        "bus_trip_energy",
    }


def test_bidirectional_depot_uses_low_price_energy_for_v2g() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("bangkok_depot", _depot_rows())
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0] * 8,
            "bus_availability": [1.0] * 8,
            "bus_trip_energy": [0.0] * 8,
            "import_price": [10.0] * 4 + [200.0] * 4,
            "export_price": [10.0] * 4 + [200.0] * 4,
        },
        index=index,
    )

    result = building.solve_horizon(
        config,
        forecasts,
        "import_price",
        export_price_column="export_price",
    )

    assert result["bus_charge_MWh"].iloc[:4].sum() > 0.0
    assert result["grid_export_MWh"].iloc[4:].sum() > 0.0
    assert result["bus_discharge_MWh"].to_numpy() == pytest.approx(
        result["grid_export_MWh"].to_numpy()
    )
    assert result["bus_soc_MWh"].iloc[-1] == pytest.approx(0.42)
    balance = result["grid_import_MWh"] + result["bus_discharge_MWh"]
    uses = result["building_demand_MWh"] + result["bus_charge_MWh"] + result["grid_export_MWh"]
    assert balance.to_numpy() == pytest.approx(uses.to_numpy())


def test_unidirectional_depot_cannot_export() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("bangkok_depot", _depot_rows("unidirectional"))
    index = pd.date_range("2025-01-01", periods=4, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0] * 4,
            "bus_availability": [1.0] * 4,
            "bus_trip_energy": [0.0] * 4,
            "price": [10.0, 10.0, 200.0, 200.0],
        },
        index=index,
    )

    result = building.solve_horizon(config, forecasts, "price")

    assert result["bus_discharge_MWh"].sum() == pytest.approx(0.0)
    assert result["grid_export_MWh"].sum() == pytest.approx(0.0)


def test_bus_trip_uses_battery_energy_and_blocks_grid_exchange_while_away() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("bangkok_depot", _depot_rows())
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0] * 8,
            "bus_availability": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "bus_trip_energy": [0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
            "price": [50.0] * 8,
        },
        index=index,
    )

    result = building.solve_horizon(config, forecasts, "price")

    assert result["bus_trip_energy_MWh"].sum() == pytest.approx(0.1)
    assert result["bus_charge_MWh"].iloc[2:4].sum() == pytest.approx(0.0)
    assert result["bus_discharge_MWh"].iloc[2:4].sum() == pytest.approx(0.0)
    assert result["bus_soc_MWh"].iloc[2] < result["bus_soc_MWh"].iloc[1]


def test_data_loader_resolves_bus_depot_profiles() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    required = DataLoader(config).required_forecast_columns(_depot_rows())

    assert {"depot_demand", "bus_availability", "bus_trip_energy"}.issubset(required)
