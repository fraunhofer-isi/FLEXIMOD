from __future__ import annotations

import numpy as np
import pandas as pd

from flexi_mod.analysis.building_case_analysis import build_service_cost_gap


def _comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "total_cost_THB": [1000.0, 1100.0, 900.0, 1200.0],
            "grid_support_during_stress_MWh": [0.0, 2.0, 0.0, 0.0],
            "maximum_grid_support_MW": [0.0, 0.2, 0.0, 0.0],
            "renewable_deficit_support_MWh": [0.0, 0.0, 0.0, 0.0],
            "grid_export_MWh": [0.0, 0.0, 0.0, 4.0],
            "peak_grid_export_MW": [0.0, 0.0, 0.0, 0.1],
        },
        index=[
            "V1G baseline",
            "I03 maximum power-system peak-demand import reduction without export",
            "PV-only operating reference",
            "I07c rooftop PV operation at viable export tariff",
        ],
    )


def _dispatch(energy_cost: float, outage: list[int]) -> pd.DataFrame:
    index = pd.date_range("2024-04-26 12:00", periods=4, freq="15min", tz="Asia/Bangkok")
    return pd.DataFrame(
        {
            "energy_cost_THB": [energy_cost] * 4,
            "bus_discharge_MWh": [0.025, 0.025, 0.0, 0.0],
            "traction_grid_import_MWh": [0.0, 0.0, 0.03, 0.04],
            "grid_import_MWh": [0.0, 0.0, 0.03, 0.04],
            "grid_export_MWh": [0.0, 0.0, 0.0, 0.0],
            "synthetic_outage_event": outage,
        },
        index=index,
    )


def test_service_cost_gap_uses_relevant_baselines_and_safe_denominators() -> None:
    result = build_service_cost_gap(_comparison(), {})

    grid = result.loc["I03 maximum power-system peak-demand import reduction without export"]
    assert grid["comparison_basis"] == "V1G baseline"
    assert grid["cost_gap_THB"] == 100.0
    assert grid["cost_gap_THB_per_service_kWh"] == 0.05
    assert grid["cost_gap_THB_per_kW_year"] == 0.5

    pv = result.loc["I07c rooftop PV operation at viable export tariff"]
    assert pv["comparison_basis"] == "PV-only operating reference"
    assert pv["cost_gap_THB"] == 300.0
    assert pv["cost_gap_THB_per_service_kWh"] == 0.075

    # Annual rows do not present an emergency-event capacity price.
    assert "cost_gap_THB_per_kW_event" not in result.columns


def test_emergency_cost_gap_checks_islanding_and_separates_peak_exposure() -> None:
    comparison = _comparison()
    baseline = _dispatch(10.0, [0, 0, 0, 0])
    emergency = _dispatch(12.0, [1, 1, 0, 0])
    baseline["traction_grid_import_MWh"] = [0.0, 0.0, 0.02, 0.02]

    result = build_service_cost_gap(
        comparison,
        {"V1G baseline": baseline, "I08a-P emergency V2G 4 h prepared SOC": emergency},
    )
    row = result.loc["I08a-P emergency V2G 4 h prepared SOC"]

    assert row["comparison_basis"] == "matched V1G event-and-recovery window"
    assert row["islanding_check"] == "pass"
    assert row["cost_gap_THB"] == 8.0
    assert row["service_energy_MWh"] == 0.05
    assert row["cost_gap_THB_per_service_kWh"] == 0.16
    assert row["potential_monthly_demand_charge_exposure_THB"] > 0.0
