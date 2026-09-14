# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression coverage for the targeted I08 emergency V2G cases."""

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.simulation.simulation_runner import OutputOptions, SimulationRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("case_name", "start", "end", "duration_hours"),
    [
        (
            "building_v2g_emergency_backup_4h_prepared",
            "2024-04-26 13:00:00",
            "2024-04-26 16:45:00",
            4.0,
        ),
        (
            "building_v2g_emergency_backup_8h_prepared",
            "2024-04-29 09:30:00",
            "2024-04-29 17:15:00",
            8.0,
        ),
    ],
)
def test_i08_events_are_reproducible_and_qualified(
    case_name: str, start: str, end: str, duration_hours: float
) -> None:
    forecast = pd.read_csv(
        PROJECT_ROOT / "data" / "input" / case_name / "forecasts_df.csv",
        parse_dates=["datetime"],
    )
    outage = forecast.loc[forecast["synthetic_outage_event"].eq(1)]

    assert outage["datetime"].iloc[0] == pd.Timestamp(start)
    assert outage["datetime"].iloc[-1] == pd.Timestamp(end)
    assert len(outage) * 0.25 == pytest.approx(duration_hours)
    assert outage["grid_congestion_weight"].ge(0.9).all()
    assert outage["grid_connection_available"].eq(0).all()
    assert forecast.loc[forecast["synthetic_outage_event"].eq(0), "grid_connection_available"].eq(1).all()


def test_i08_islands_the_depot_and_reports_emergency_transfer(tmp_path: Path) -> None:
    case_name = "building_v2g_emergency_backup_4h_prepared"
    case_dir = PROJECT_ROOT / "data" / "input" / case_name
    runner = SimulationRunner(
        case_dir=case_dir,
        study_case=case_name,
        output_dir=tmp_path,
        output_options=OutputOptions(create_plots=False),
    )
    outputs = runner.run()
    dispatch = pd.read_csv(outputs["dispatch_results"], parse_dates=["datetime"])
    summary = pd.read_csv(outputs["summary_indicators"])
    outage = dispatch["synthetic_outage_event"].eq(1)

    assert dispatch.loc[outage, "grid_import_MWh"].sum() == pytest.approx(0.0)
    assert dispatch.loc[outage, "grid_export_MWh"].sum() == pytest.approx(0.0)
    assert dispatch.loc[outage, "bus_discharge_MWh"].sum() > 0.0
    assert dispatch.loc[outage, "bus_discharge_MWh"].max() <= 0.45 * 0.25 + 1e-8
    assert summary.loc[0, "synthetic_outage_duration_hours"] == pytest.approx(4.0)
    assert summary.loc[0, "emergency_v2g_transfer_MWh"] == pytest.approx(
        dispatch.loc[outage, "bus_discharge_MWh"].sum()
    )
    assert summary.loc[0, "grid_import_during_synthetic_outage_MWh"] == pytest.approx(0.0)
    assert summary.loc[0, "grid_export_during_synthetic_outage_MWh"] == pytest.approx(0.0)


def test_i08_soc_bases_match_the_declared_pre_outage_conditions() -> None:
    prepared = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "input"
        / "building_v2g_emergency_backup_4h_prepared"
        / "plants.csv"
    )
    inherited = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "input"
        / "building_v2g_emergency_backup_4h_baseline"
        / "plants.csv"
    )
    dispatch = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "output"
        / "building_use_case_analysis"
        / "v1g_baseline"
        / "dispatch_results.csv",
        parse_dates=["datetime"],
    )
    dispatch["datetime"] = pd.DatetimeIndex(dispatch["datetime"]).tz_localize(None)
    prior = dispatch.loc[dispatch["datetime"].eq(pd.Timestamp("2024-04-26 12:45:00"))].iloc[0]
    buses = prepared[prepared["technology"].eq("electric_vehicle")]

    assert buses["initial_soc"].eq(0.8).all()
    for _, bus in inherited[inherited["technology"].eq("electric_vehicle")].iterrows():
        expected = prior[f"{bus['component_name']}_soc_MWh"] / bus["battery_capacity_mwh"]
        assert bus["initial_soc"] == pytest.approx(expected)
