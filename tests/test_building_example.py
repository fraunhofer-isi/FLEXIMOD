# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.plants import Building
from flexi_mod.simulation.run_case import resolve_example_paths
from flexi_mod.simulation.simulation_runner import OutputOptions, SimulationRunner
from flexi_mod.strategies import build_strategy
from flexi_mod.strategies.building_strategy import BuildingStrategy

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "building_v2g_example"
V1G_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "building_v1g_baseline"
V2B_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "building_v2b_no_export"


def test_building_example_optimizes_with_rolling_horizon(
    tmp_path: Path,
) -> None:
    paths = resolve_example_paths("building_v2g_example")
    assert paths["case_dir"] == EXAMPLE_DIR

    config = CaseConfig.from_case_dir(EXAMPLE_DIR, study_case="building_v2g_example")
    assert config.timestep_minutes == 15
    assert config.dispatch_setting("currency") == "THB"
    assert config.dispatch_setting("demand_charge_per_kw_month") == 210
    assert config.dispatch_setting("grid_export_limit_mw") == pytest.approx(0.005)
    loader = DataLoader(config, input_dir=EXAMPLE_DIR)
    plants = loader.load_plants()
    forecasts = loader.load_forecasts()
    assert len(forecasts) == 96
    assert set(forecasts["tou_period"]) == {"off_peak", "on_peak"}
    assert set(forecasts["fuel_adjustment_thb_per_kwh"]) == {0.1623}
    assert set(forecasts["electricity_import_price_thb_per_mwh"]) == {2799.2, 4492.0}
    assert set(forecasts["v2g_export_price_current_thb_per_mwh"]) == {2200.0}
    assert set(forecasts["regional_grid_load_source_year"]) == {2024}
    assert set(forecasts["regional_grid_load_profile"]) == {"september_monday_mean"}
    assert forecasts["regional_grid_load_mw"].max() == pytest.approx(9637.58)
    assert forecasts["grid_congestion_weight"].max() == pytest.approx(0.129955)
    building = Building.from_plants_dataframe(plants)[0]
    bus_names = {"bus_route_7k", "bus_route_79", "bus_route_101"}
    charger_names = {"charger_1", "charger_2", "charger_3"}
    assert set(building.electric_vehicles) == bus_names
    assert set(building.charging_stations) == charger_names
    for vehicle in building.electric_vehicles.values():
        assert vehicle.battery_capacity_mwh == pytest.approx(0.3)
        assert vehicle.mileage_mwh_per_km == pytest.approx(0.00125)
        assert vehicle.min_soc_fraction == pytest.approx(0.0)
    for station in building.charging_stations.values():
        assert station.max_power_charge_mw == pytest.approx(0.15)
        assert station.max_power_discharge_mw == pytest.approx(0.15)
    strategy = build_strategy(config.strategy_name, config)
    assert isinstance(strategy, BuildingStrategy)

    runner = SimulationRunner(
        case_dir=EXAMPLE_DIR,
        input_dir=EXAMPLE_DIR,
        output_dir=tmp_path / "output",
        study_case="building_v2g_example",
        output_options=OutputOptions(
            create_plots=False,
        ),
    )
    outputs = runner.run()
    result = pd.read_csv(outputs["dispatch_results"], parse_dates=["datetime"])
    market = pd.read_csv(outputs["market_ledger"])
    storage = pd.read_csv(outputs["storage_cost_ledger"])
    summary = pd.read_csv(outputs["summary_indicators"])

    assert len(result) == 96
    assert result["rolling_window"].nunique() == 4
    assert market["grid_export_MWh"].sum() == pytest.approx(result["grid_export_MWh"].sum())
    assert market["regional_grid_load_mw"].to_numpy() == pytest.approx(
        result["regional_grid_load_mw"].to_numpy()
    )
    route_distances = {
        "bus_route_7k": 240.0,
        "bus_route_79": 280.0,
        "bus_route_101": 288.0,
    }
    for bus_name, expected_distance in route_distances.items():
        assert storage[f"{bus_name}_soc_MWh"].iloc[-1] == pytest.approx(
            result[f"{bus_name}_soc_MWh"].iloc[-1]
        )
        assert storage[f"{bus_name}_trip_distance_km"].sum() == pytest.approx(expected_distance)
        assert set(result[f"{bus_name}_availability_fraction"].unique()) <= {0.0, 1.0}
        unavailable = result[f"{bus_name}_availability_fraction"] == 0.0
        assert result.loc[unavailable, f"{bus_name}_charge_MWh"].sum() == 0.0

    assert result["bus_trip_distance_km"].sum() == pytest.approx(808.0)
    assert result["bus_trip_energy_MWh"].sum() == pytest.approx(1.01)
    assert result["pv_generation_MWh"].sum() == pytest.approx(0.0)
    assert summary["total_pv_generation_MWh"].iloc[0] == pytest.approx(0.0)
    assert summary["pv_utilisation_fraction"].iloc[0] == pytest.approx(0.0)
    assert summary["pv_self_consumption_MWh"].iloc[0] == pytest.approx(0.0)
    assert summary["pv_self_consumption_fraction"].iloc[0] == pytest.approx(0.0)
    assert result.loc[result["bus_availability_fraction"] == 0.0, "bus_charge_MWh"].sum() == 0
    assert result["grid_export_MWh"].sum() == pytest.approx(0.0)
    assert result["bus_discharge_MWh"].sum() > 0.0
    assert result["grid_export_MW"].max() <= 0.005 + 1e-9
    assert result["bus_soc_fraction"].iloc[-1] == pytest.approx(0.8)
    assert summary["currency"].iloc[0] == "THB"
    assert summary["total_energy_cost"].iloc[0] == pytest.approx(result["energy_cost"].sum())
    assert summary["total_demand_charge_cost"].iloc[0] == pytest.approx(
        result["demand_charge_cost"].sum()
    )
    assert summary["total_cost"].iloc[0] == pytest.approx(result["total_cost"].sum())
    assert summary["peak_grid_import_MW"].iloc[0] == pytest.approx(result["grid_import_MW"].max())
    assert summary["peak_regional_grid_load_MW"].iloc[0] == pytest.approx(9637.58)
    assert summary["congestion_weighted_grid_import_MWh"].iloc[0] == pytest.approx(
        (result["grid_import_MWh"] * result["grid_congestion_weight"]).sum()
    )
    assert summary["total_demand_charge_cost"].iloc[0] == pytest.approx(
        summary["peak_grid_import_MW"].iloc[0] * 1000.0 * 210.0
    )
    assert set(result["electricity_import_price_THB_per_MWh"]) == {2799.2, 4492.0}
    assert set(result["electricity_export_price_THB_per_MWh"]) == {2200.0}

    # SOC remains continuous when the implemented state is passed to the next window.
    for bus_name, vehicle in building.electric_vehicles.items():
        previous_soc = vehicle.initial_soc_mwh
        for _, row in result.iterrows():
            expected_soc = (
                previous_soc
                + row[f"{bus_name}_charge_MWh"] * vehicle.efficiency_charge
                - row[f"{bus_name}_discharge_MWh"] / vehicle.efficiency_discharge
                - row[f"{bus_name}_trip_energy_MWh"]
            )
            assert row[f"{bus_name}_soc_MWh"] == pytest.approx(expected_soc)
            previous_soc = row[f"{bus_name}_soc_MWh"]


def test_building_cases_enforce_v1g_v2b_and_v2g_boundaries(tmp_path: Path) -> None:
    def run_case(case_dir: Path, study_case: str) -> tuple[pd.DataFrame, pd.Series]:
        runner = SimulationRunner(
            case_dir=case_dir,
            input_dir=case_dir,
            output_dir=tmp_path / study_case,
            study_case=study_case,
            output_options=OutputOptions(
                save_market_ledger=False,
                save_storage_cost_ledger=False,
                create_plots=False,
            ),
        )
        outputs = runner.run()
        dispatch = pd.read_csv(outputs["dispatch_results"])
        summary = pd.read_csv(outputs["summary_indicators"]).iloc[0]
        return dispatch, summary

    assert resolve_example_paths("building_v1g_baseline")["case_dir"] == V1G_EXAMPLE_DIR
    assert resolve_example_paths("building_v2b_no_export")["case_dir"] == V2B_EXAMPLE_DIR
    assert CaseConfig.from_case_dir(V1G_EXAMPLE_DIR).study_case == "building_v1g_baseline"
    assert CaseConfig.from_case_dir(V2B_EXAMPLE_DIR).study_case == "building_v2b_no_export"

    baseline, baseline_summary = run_case(V1G_EXAMPLE_DIR, "building_v1g_baseline")
    v2b, v2b_summary = run_case(V2B_EXAMPLE_DIR, "building_v2b_no_export")
    below_break_even, _ = run_case(EXAMPLE_DIR, "building_v2g_plus_3500")
    above_break_even, _ = run_case(EXAMPLE_DIR, "building_v2g_plus_4000")

    assert baseline["bus_discharge_MWh"].sum() == pytest.approx(0.0)
    assert baseline["grid_export_MWh"].sum() == pytest.approx(0.0)
    assert v2b["bus_discharge_MWh"].sum() > 0.0
    assert v2b["grid_export_MWh"].sum() == pytest.approx(0.0)
    assert v2b_summary["peak_grid_import_MW"] < baseline_summary["peak_grid_import_MW"]
    assert below_break_even["grid_export_MWh"].sum() == pytest.approx(0.0)
    assert above_break_even["grid_export_MWh"].sum() > 0.0
    assert above_break_even["grid_export_MW"].max() <= 0.005 + 1e-9
