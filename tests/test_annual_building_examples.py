# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd

from flexi_mod.config.case_config import CaseConfig, available_study_cases
from flexi_mod.simulation.run_case import resolve_example_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNUAL_EXAMPLES = {
    "building_v1g_annual": "building_v1g_annual",
    "building_v2b_cost_annual": "building_v2b_cost_annual",
    "building_v2b_grid_support_annual": "building_v2b_grid_support_max_no_export",
    "building_v2b_pv_self_consumption_annual": "building_v2b_pv_self_consumption",
    "building_v2b_renewable_alignment_annual": "building_v2b_renewable_shifting_max_no_export",
    "building_v2g_current_tariff_annual": "building_v2g_current_tariff_annual",
    "building_v2g_tariff_sweep_annual": "building_v2g_sweep_current",
}


def test_annual_examples_are_registered_and_configured() -> None:
    for example, study_case in ANNUAL_EXAMPLES.items():
        paths = resolve_example_paths(example)
        case_dir = PROJECT_ROOT / "data" / "input" / example
        assert paths["case_dir"] == case_dir
        assert paths["study_case"] == study_case
        config = CaseConfig.from_case_dir(case_dir, study_case=study_case)
        assert config.timestep_minutes == 15
        assert config.simulation_start == "2024-01-01 00:00"
        assert config.simulation_end == "2024-12-31 23:45"
        assert (case_dir / "plants.csv").exists()
        assert (case_dir / "forecasts_df.csv").exists()


def test_annual_grid_support_and_tariff_sensitivity_cases_exist() -> None:
    grid_cases = available_study_cases(
        PROJECT_ROOT / "data" / "input" / "building_v2b_grid_support_annual" / "config.yaml"
    )
    tariff_cases = available_study_cases(
        PROJECT_ROOT / "data" / "input" / "building_v2g_tariff_sweep_annual" / "config.yaml"
    )
    renewable_cases = available_study_cases(
        PROJECT_ROOT
        / "data"
        / "input"
        / "building_v2b_renewable_alignment_annual"
        / "config.yaml"
    )
    assert len(grid_cases) == 3
    assert len(renewable_cases) == 3
    assert set(grid_cases) == {
        "building_v2b_grid_support_max_no_export",
        "building_v2g_grid_support_max_export_450kw",
        "building_v2g_grid_support_max_export_5kw",
    }
    assert set(renewable_cases) == {
        "building_v2b_renewable_shifting_max_no_export",
        "building_v2g_renewable_shifting_max_export_450kw",
        "building_v2g_renewable_shifting_max_export_5kw",
    }
    assert len(tariff_cases) == 11


def test_renewable_forecast_is_complete_and_bounded() -> None:
    path = (
        PROJECT_ROOT
        / "data"
        / "input"
        / "building_v2b_renewable_alignment_annual"
        / "forecasts_df.csv"
    )
    forecast = pd.read_csv(path)

    assert len(forecast) == 35_136
    assert forecast["renewable_weather_year"].eq(2024).all()
    assert forecast["renewable_capacity_basis_year"].eq(2024).all()
    assert forecast["renewable_availability_weight"].between(0.0, 1.0).all()
    assert forecast["solar_pv_capacity_factor"].between(0.0, 1.0).all()
    assert forecast["wind_capacity_factor"].between(0.0, 1.0).all()
    assert forecast["national_system_demand_mw"].gt(0.0).all()
    assert forecast["modelled_vre_potential_share_of_demand"].between(0.0, 1.0).all()
    assert set(forecast["renewable_rich"]) == {0, 1}


def test_pv_self_consumption_forecast_matches_roof_assessment() -> None:
    case_dir = (
        PROJECT_ROOT
        / "data"
        / "input"
        / "building_v2b_pv_self_consumption_annual"
    )
    forecast = pd.read_csv(case_dir / "forecasts_df.csv")
    plants = pd.read_csv(case_dir / "plants.csv")
    pv = plants.loc[plants["technology"] == "pv_plant"].iloc[0]
    mobile_components = plants[
        plants["technology"].isin(["electric_vehicle", "charging_station"])
    ]

    assert len(forecast) == 35_136
    assert forecast["pv_roof_count"].eq(7).all()
    assert forecast["pv_roof_gross_area_m2"].round(1).eq(3_923.0).all()
    assert forecast["pv_roof_usable_fraction"].eq(0.65).all()
    assert forecast["pv_roof_usable_area_m2"].round(3).eq(2_549.950).all()
    assert forecast["pv_dc_capacity_mwp"].round(6).eq(0.509990).all()
    assert forecast["pv_structural_survey_completed"].eq(0).all()
    assert set(mobile_components["power_flow_directionality"]) == {"bidirectional"}
    assert forecast["pv_1_generation_mw"].between(0.0, float(pv["max_power"])).all()
    assert forecast["pv_generation_conservative_mw"].le(
        forecast["pv_generation_base_mw"]
    ).all()
    assert forecast["pv_generation_base_mw"].le(
        forecast["pv_generation_optimistic_mw"]
    ).all()


def test_annual_forecast_covers_leap_year_and_keeps_extreme_loads() -> None:
    path = PROJECT_ROOT / "data" / "input" / "building_v1g_annual" / "forecasts_df.csv"
    forecast = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")

    assert len(forecast) == 35_136
    assert forecast.index[0] == pd.Timestamp("2024-01-01 00:00")
    assert forecast.index[-1] == pd.Timestamp("2024-12-31 23:45")
    assert forecast.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    assert set(forecast["regional_grid_load_state"]) == {
        "normal",
        "elevated",
        "stressed",
    }
    assert forecast["grid_stress_threshold"].eq(0.8).all()
    assert forecast.loc[
        forecast["regional_grid_load_state"] == "normal", "grid_congestion_weight"
    ].eq(0.0).all()
    assert forecast.loc[
        forecast["regional_grid_load_state"] == "elevated", "grid_congestion_weight"
    ].between(0.0, 0.8, inclusive="neither").all()
    assert forecast.loc[
        forecast["regional_grid_load_state"] == "stressed", "grid_congestion_weight"
    ].between(0.8, 1.0).all()
    assert forecast["regional_grid_load_source_year"].eq(2024).all()
    assert forecast["regional_grid_load_profile"].eq("historical_2024").all()

    availability_columns = [
        column for column in forecast if column.endswith("_availability")
    ]
    assert set(forecast[availability_columns].stack().unique()) == {0.0, 1.0}

    route_columns = [column for column in forecast if "trip_distance_km" in column]
    first_day = forecast.iloc[:96][route_columns].reset_index(drop=True)
    second_day = forecast.iloc[96:192][route_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(first_day, second_day)


def test_annual_tou_calendar_distinguishes_holidays_and_weekdays() -> None:
    path = PROJECT_ROOT / "data" / "input" / "building_v1g_annual" / "forecasts_df.csv"
    forecast = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")

    assert forecast.loc["2024-01-01 12:00", "tou_period"] == "off_peak"
    assert forecast.loc["2024-01-02 12:00", "tou_period"] == "on_peak"
    assert forecast.loc["2024-01-06 12:00", "tou_period"] == "off_peak"
    assert forecast.loc["2024-01-02 08:45", "tou_period"] == "off_peak"
    assert forecast.loc["2024-01-02 09:00", "tou_period"] == "on_peak"
    assert forecast.loc["2024-01-02 22:00", "tou_period"] == "off_peak"
