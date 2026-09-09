# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.plants.building import Building
from flexi_mod.plants.technologies import ChargingStation, ElectricVehicle, PVPlant

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DA_ID_buy"


def _depot_rows(directionality: str = "bidirectional") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "depot_building",
                "unit_type": "building",
                "technology": "electric_vehicle",
                "component_name": "bus_1",
                "node": "grid_node",
                "demand": "depot_demand",
                "battery_capacity_mwh": 0.3,
                "max_power_charge": 0.2,
                "max_power_discharge": 0.2,
                "min_soc": 0.2,
                "max_soc": 1.0,
                "initial_soc": 0.7,
                "terminal_soc": 0.7,
                "efficiency_charge": 0.95,
                "efficiency_discharge": 0.95,
                "mileage_mwh_per_km": 0.001,
                "power_flow_directionality": directionality,
                "availability_column": "bus_1_availability",
                "trip_distance_column": "bus_1_trip_distance_km",
            },
            {
                "name": "depot_building",
                "unit_type": "building",
                "technology": "charging_station",
                "component_name": "charger_1",
                "node": "grid_node",
                "demand": "depot_demand",
                "max_power_charge": 0.15,
                "max_power_discharge": 0.15,
                "power_flow_directionality": directionality,
            },
        ]
    )


def _rows_with_pv(directionality: str = "bidirectional") -> pd.DataFrame:
    rows = _depot_rows(directionality)
    pv_row = {
        "name": "depot_building",
        "unit_type": "building",
        "technology": "pv_plant",
        "component_name": "pv_1",
        "node": "grid_node",
        "demand": "depot_demand",
        "max_power": 0.1,
        "generation_column": "pv_1_generation_mw",
    }
    return pd.concat([rows, pd.DataFrame([pv_row])], ignore_index=True)


def test_building_constructs_bus_and_charger_from_rows() -> None:
    building = Building.from_rows("depot_building", _depot_rows())

    assert isinstance(building.electric_vehicle, ElectricVehicle)
    assert isinstance(building.charging_station, ChargingStation)
    assert set(building.electric_vehicles) == {"bus_1"}
    assert set(building.charging_stations) == {"charger_1"}
    assert building.pv_plants == {}
    assert building.electric_vehicle.total_capacity_mwh == pytest.approx(0.3)
    assert building.charging_station.total_max_power_charge_mw == pytest.approx(0.15)
    assert building.v2g_enabled
    assert building.required_forecast_columns() == {
        "depot_demand",
        "bus_1_availability",
        "bus_1_trip_distance_km",
    }


def test_bidirectional_depot_uses_low_price_energy_for_v2g() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("depot_building", _depot_rows())
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0] * 8,
            "bus_1_availability": [1.0] * 8,
            "bus_1_trip_distance_km": [0.0] * 8,
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
    assert result["bus_1_soc_MWh"].iloc[-1] == pytest.approx(0.21)
    balance = result["grid_import_MWh"] + result["bus_discharge_MWh"]
    uses = result["building_demand_MWh"] + result["bus_charge_MWh"] + result["grid_export_MWh"]
    assert balance.to_numpy() == pytest.approx(uses.to_numpy())


def test_unidirectional_depot_cannot_export() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("depot_building", _depot_rows("unidirectional"))
    index = pd.date_range("2025-01-01", periods=4, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0] * 4,
            "bus_1_availability": [1.0] * 4,
            "bus_1_trip_distance_km": [0.0] * 4,
            "price": [10.0, 10.0, 200.0, 200.0],
        },
        index=index,
    )

    result = building.solve_horizon(config, forecasts, "price")

    assert result["bus_discharge_MWh"].sum() == pytest.approx(0.0)
    assert result["grid_export_MWh"].sum() == pytest.approx(0.0)


def test_grid_support_moves_charging_within_its_cost_budget() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0] * 8,
            "bus_1_availability": [1.0] * 7 + [0.0],
            "bus_1_trip_distance_km": [0.0] * 7 + [80.0],
            "price": [10.0] * 4 + [20.0] * 4,
            "grid_weight": [1.0] * 4 + [0.0] * 4,
        },
        index=index,
    )
    building = Building.from_rows("depot_building", _depot_rows())

    cost_config = CaseConfig.from_case_dir(CASE_DIR)
    cost_config.case["strategy"]["dispatch"].update(
        {"vehicle_discharge_enabled": False, "grid_export_limit_mw": 0.0}
    )
    least_cost = building.solve_horizon(cost_config, forecasts, "price")

    support_config = CaseConfig.from_case_dir(CASE_DIR)
    support_config.case["strategy"]["dispatch"].update(
        {
            "dispatch_objective": "min_congestion_with_cost_budget",
            "grid_congestion_weight_column": "grid_weight",
            "grid_support_cost_budget_fraction_per_horizon": 1.1,
            "vehicle_discharge_enabled": False,
            "grid_export_limit_mw": 0.0,
        }
    )
    grid_support = building.solve_horizon(support_config, forecasts, "price")

    least_cost_weighted_import = (
        least_cost["grid_import_MWh"] * forecasts["grid_weight"]
    ).sum()
    support_weighted_import = (
        grid_support["grid_import_MWh"] * forecasts["grid_weight"]
    ).sum()
    assert support_weighted_import < least_cost_weighted_import
    assert grid_support["energy_cost"].sum() <= least_cost["energy_cost"].sum() * 2.1 + 1e-6


def test_renewable_objective_moves_charging_to_high_availability() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0] * 8,
            "bus_1_availability": [1.0] * 7 + [0.0],
            "bus_1_trip_distance_km": [0.0] * 7 + [80.0],
            "price": [10.0] * 4 + [20.0] * 4,
            "renewable_weight": [0.0] * 4 + [1.0] * 4,
        },
        index=index,
    )
    building = Building.from_rows("depot_building", _depot_rows())

    cost_config = CaseConfig.from_case_dir(CASE_DIR)
    cost_config.case["strategy"]["dispatch"].update(
        {"vehicle_discharge_enabled": False, "grid_export_limit_mw": 0.0}
    )
    least_cost = building.solve_horizon(cost_config, forecasts, "price")

    renewable_config = CaseConfig.from_case_dir(CASE_DIR)
    renewable_config.case["strategy"]["dispatch"].update(
        {
            "dispatch_objective": "min_renewable_misalignment_with_cost_budget",
            "renewable_availability_weight_column": "renewable_weight",
            "renewable_alignment_cost_budget_fraction_per_horizon": 1.1,
            "vehicle_discharge_enabled": False,
            "grid_export_limit_mw": 0.0,
        }
    )
    renewable_aligned = building.solve_horizon(renewable_config, forecasts, "price")

    least_cost_aligned_charge = (
        least_cost["bus_charge_MWh"] * forecasts["renewable_weight"]
    ).sum()
    renewable_aligned_charge = (
        renewable_aligned["bus_charge_MWh"] * forecasts["renewable_weight"]
    ).sum()
    assert renewable_aligned_charge > least_cost_aligned_charge
    assert (
        renewable_aligned["energy_cost"].sum()
        <= least_cost["energy_cost"].sum() * 2.1 + 1e-6
    )


def test_max_grid_support_respects_export_limits_and_mobility() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.001] * 8,
            "bus_1_availability": [1.0] * 8,
            "bus_1_trip_distance_km": [0.0] * 8,
            "price": [100.0] * 8,
            "grid_weight": [1.0] * 4 + [0.0] * 4,
        },
        index=index,
    )
    building = Building.from_rows("depot_building", _depot_rows())

    results = {}
    for label, export_limit in {"none": 0.0, "policy": 0.005, "technical": 0.45}.items():
        config = CaseConfig.from_case_dir(CASE_DIR)
        config.case["strategy"]["dispatch"].update(
            {
                "dispatch_objective": "max_grid_support",
                "grid_congestion_weight_column": "grid_weight",
                "vehicle_discharge_enabled": True,
                "grid_export_limit_mw": export_limit,
            }
        )
        results[label] = building.solve_horizon(config, forecasts, "price")

    assert results["none"]["grid_export_MW"].max() == pytest.approx(0.0)
    assert results["policy"]["grid_export_MW"].max() <= 0.005 + 1e-7
    assert results["technical"]["grid_export_MW"].max() <= 0.45 + 1e-7
    assert results["technical"]["grid_export_MWh"].sum() > results["policy"][
        "grid_export_MWh"
    ].sum()
    for result in results.values():
        assert result["unmet_trip_energy_MWh"].sum() == pytest.approx(0.0)
        assert result["bus_1_trip_energy_MWh"].sum() == pytest.approx(0.0)
        assert result["bus_1_soc_MWh"].iloc[-1] >= 0.21 - 1e-7


def test_max_renewable_shifting_tracks_energy_before_discharge() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.001] * 8,
            "bus_1_availability": [1.0] * 8,
            "bus_1_trip_distance_km": [0.0] * 8,
            "price": [100.0] * 8,
            "renewable_weight": [1.0] * 4 + [0.0] * 4,
        },
        index=index,
    )
    building = Building.from_rows("depot_building", _depot_rows())
    config = CaseConfig.from_case_dir(CASE_DIR)
    config.case["strategy"]["dispatch"].update(
        {
            "dispatch_objective": "max_renewable_shifting",
            "renewable_availability_weight_column": "renewable_weight",
            "vehicle_discharge_enabled": True,
            "grid_export_limit_mw": 0.45,
        }
    )

    result = building.solve_horizon(config, forecasts, "price")

    assert result["renewable_equivalent_charge_MWh"].iloc[:4].sum() > 0.0
    assert result["renewable_equivalent_discharge_MWh"].iloc[:4].sum() == pytest.approx(0.0)
    assert result["renewable_equivalent_discharge_MWh"].iloc[4:].sum() > 0.0
    assert result["renewable_deficit_weighted_discharge_MWh"].sum() > 0.0
    assert result["renewable_equivalent_grid_export_MWh"].sum() > 0.0
    assert result["renewable_equivalent_soc_MWh"].between(
        0.0, result["bus_soc_MWh"] + 1e-8
    ).all()


def test_renewable_initial_soc_cannot_exceed_total_soc() -> None:
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0],
            "bus_1_availability": [1.0],
            "bus_1_trip_distance_km": [0.0],
            "price": [100.0],
            "renewable_weight": [1.0],
        },
        index=pd.date_range("2025-01-01", periods=1, freq="15min"),
    )
    building = Building.from_rows("depot_building", _depot_rows())
    config = CaseConfig.from_case_dir(CASE_DIR)
    config.case["strategy"]["dispatch"].update(
        {
            "dispatch_objective": "max_renewable_shifting",
            "renewable_availability_weight_column": "renewable_weight",
        }
    )

    with pytest.raises(ValueError, match="Renewable initial SOC"):
        building.build_model(
            config,
            forecasts,
            "price",
            initial_renewable_soc_mwh={"bus_1": 0.25},
        )


def test_monthly_demand_charge_uses_highest_fifteen_minute_import() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    config.case["strategy"]["dispatch"].update(
        {
            "currency": "THB",
            "demand_charge_per_kw_month": 210.0,
            "vehicle_discharge_enabled": False,
            "grid_export_limit_mw": 0.0,
        }
    )
    building = Building.from_rows("depot_building", _depot_rows())
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.1, 0.2],
            "bus_1_availability": [1.0, 1.0],
            "bus_1_trip_distance_km": [0.0, 0.0],
            "price": [1000.0, 1000.0],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="15min"),
    )

    result = building.solve_horizon(
        config,
        forecasts,
        "price",
        initial_monthly_peak_mw={"2025-01": 0.15},
    )

    assert result["grid_import_MW"].to_numpy() == pytest.approx([0.1, 0.2])
    assert result["billing_peak_MW"].to_numpy() == pytest.approx([0.15, 0.2])
    assert result["demand_charge_cost"].sum() == pytest.approx(10_500.0)
    assert result["total_cost"].sum() == pytest.approx(result["energy_cost"].sum() + 10_500.0)
    assert result["currency"].unique().tolist() == ["THB"]


def test_bus_trip_uses_battery_energy_and_blocks_grid_exchange_while_away() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("depot_building", _depot_rows())
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0] * 8,
            "bus_1_availability": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "bus_1_trip_distance_km": [0.0, 0.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0],
            "price": [50.0] * 8,
        },
        index=index,
    )

    result = building.solve_horizon(config, forecasts, "price")

    assert result["bus_1_trip_distance_km"].sum() == pytest.approx(100.0)
    assert result["bus_1_trip_energy_MWh"].sum() == pytest.approx(0.1)
    assert result["bus_1_charge_MWh"].iloc[2:4].sum() == pytest.approx(0.0)
    assert result["bus_1_discharge_MWh"].iloc[2:4].sum() == pytest.approx(0.0)
    assert result["bus_1_soc_MWh"].iloc[2] < result["bus_1_soc_MWh"].iloc[1]


def test_data_loader_resolves_bus_depot_profiles() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    required = DataLoader(config).required_forecast_columns(_depot_rows())
    required_with_pv = DataLoader(config).required_forecast_columns(_rows_with_pv())

    assert {"depot_demand", "bus_1_availability", "bus_1_trip_distance_km"}.issubset(required)
    assert "pv_1_generation_mw" in required_with_pv


def test_optional_pv_supplies_building_and_exports_surplus() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("depot_building", _rows_with_pv())
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.05] * 4,
            "bus_1_availability": [1.0] * 4,
            "bus_1_trip_distance_km": [0.0] * 4,
            "pv_1_generation_mw": [0.0, 0.05, 0.1, 0.0],
            "price": [50.0] * 4,
        },
        index=pd.date_range("2025-01-01", periods=4, freq="15min"),
    )

    result = building.solve_horizon(config, forecasts, "price")

    assert isinstance(building.pv_plants["pv_1"], PVPlant)
    assert "pv_1_generation_mw" in building.required_forecast_columns()
    assert result["pv_available_generation_MWh"].sum() == pytest.approx(0.0375)
    assert result["pv_generation_MWh"].sum() == pytest.approx(0.0375)
    assert result["pv_curtailment_MWh"].sum() == pytest.approx(0.0)
    assert result["grid_import_MWh"].sum() == pytest.approx(0.025)
    assert result["grid_export_MWh"].sum() == pytest.approx(0.0125)

    supply = result["grid_import_MWh"] + result["bus_discharge_MWh"] + result["pv_generation_MWh"]
    use = result["building_demand_MWh"] + result["bus_charge_MWh"] + result["grid_export_MWh"]
    assert supply.to_numpy() == pytest.approx(use.to_numpy())


def test_pv_profile_cannot_exceed_installed_power() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("depot_building", _rows_with_pv())
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0],
            "bus_1_availability": [1.0],
            "bus_1_trip_distance_km": [0.0],
            "pv_1_generation_mw": [0.11],
            "price": [50.0],
        },
        index=pd.date_range("2025-01-01", periods=1, freq="15min"),
    )

    with pytest.raises(ValueError, match="PV generation cannot exceed 0.1"):
        building.build_model(config, forecasts, "price")


def test_multiple_bus_rows_keep_individual_availability_and_soc() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    rows = _depot_rows()
    bus_2 = rows.iloc[0].copy()
    bus_2["component_name"] = "bus_2"
    bus_2["availability_column"] = "bus_2_availability"
    bus_2["trip_distance_column"] = "bus_2_trip_distance_km"
    charger_2 = rows.iloc[1].copy()
    charger_2["component_name"] = "charger_2"
    rows = pd.concat([rows, pd.DataFrame([bus_2, charger_2])], ignore_index=True)
    building = Building.from_rows("depot_building", rows)

    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0] * 8,
            "bus_1_availability": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "bus_1_trip_distance_km": [0.0, 0.0, 30.0, 30.0, 0.0, 0.0, 0.0, 0.0],
            "bus_2_availability": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "bus_2_trip_distance_km": [0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 0.0, 0.0],
            "price": [50.0] * 8,
        },
        index=index,
    )

    result = building.solve_horizon(
        config,
        forecasts,
        "price",
        initial_soc_mwh={"bus_1": 0.21, "bus_2": 0.21},
    )

    assert set(building.electric_vehicles) == {"bus_1", "bus_2"}
    assert set(building.charging_stations) == {"charger_1", "charger_2"}
    assert result["bus_1_trip_distance_km"].sum() == pytest.approx(60.0)
    assert result["bus_2_trip_distance_km"].sum() == pytest.approx(40.0)
    assert result["bus_1_charge_MWh"].iloc[2:4].sum() == pytest.approx(0.0)
    assert result["bus_2_charge_MWh"].iloc[4:6].sum() == pytest.approx(0.0)
    assert result["bus_1_soc_MWh"].iloc[2] < result["bus_1_soc_MWh"].iloc[1]
    assert result["bus_2_soc_MWh"].iloc[4] < result["bus_2_soc_MWh"].iloc[3]
    assert result["bus_soc_MWh"].to_numpy() == pytest.approx(
        (result["bus_1_soc_MWh"] + result["bus_2_soc_MWh"]).to_numpy()
    )


def test_rolling_soc_handoff_tolerates_floating_point_error_at_bound() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("depot_building", _depot_rows())
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0] * 8,
            "bus_1_availability": [1.0] * 8,
            "bus_1_trip_distance_km": [0.0] * 8,
            "price": [50.0] * 8,
        },
        index=index,
    )

    result = building.solve_horizon(
        config,
        forecasts,
        "price",
        initial_soc_mwh={"bus_1": 0.300000001},
    )

    assert result["bus_1_soc_MWh"].max() <= 0.3 + 1e-9


def test_bus_availability_must_be_binary() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("depot_building", _depot_rows())
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0],
            "bus_1_availability": [0.5],
            "bus_1_trip_distance_km": [0.0],
            "price": [50.0],
        },
        index=pd.date_range("2025-01-01", periods=1, freq="15min"),
    )

    with pytest.raises(ValueError, match="only 0 or 1"):
        building.build_model(config, forecasts, "price")


def test_connected_bus_cannot_have_trip_distance() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    building = Building.from_rows("depot_building", _depot_rows())
    forecasts = pd.DataFrame(
        {
            "depot_demand": [0.0],
            "bus_1_availability": [1.0],
            "bus_1_trip_distance_km": [1.0],
            "price": [50.0],
        },
        index=pd.date_range("2025-01-01", periods=1, freq="15min"),
    )

    with pytest.raises(ValueError, match="cannot travel while availability is 1"):
        building.build_model(config, forecasts, "price")
