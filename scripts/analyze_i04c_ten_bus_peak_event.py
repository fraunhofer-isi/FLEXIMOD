#!/usr/bin/env python3
"""Summarise the illustrative ten-bus 6 May peak-event sensitivity.

This is an event-capability sensitivity, selected to answer how a larger
route-constrained fleet could compare with a nominal 1.5 MW / 3 MWh stationary
battery. It does not represent an annual service commitment, an approved
connection capacity or a verified current BMTA fleet allocation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "output" / "building_use_case_analysis"
RESULT_PATH = OUTPUT_ROOT / "ten_bus_peak_event_sensitivity_summary.csv"
EVENT_DAY = pd.Timestamp("2024-05-06").date()
CASES = {
    "Ten-bus V1G reference": "ten_bus_peak_event_v1g_reference",
    "Ten-bus peak response: managed charging": "ten_bus_peak_event_charging_only",
    "Ten-bus V2G technical potential (1.5 MW)": "ten_bus_peak_event_v2g_technical",
}


def aggregate_bus_columns(frame: pd.DataFrame, suffix: str) -> pd.Series:
    columns = [
        column
        for column in frame.columns
        if column.startswith("bus_route_") and column.endswith(suffix)
    ]
    return frame[columns].sum(axis=1)


def load_case(label: str, directory: str) -> tuple[dict[str, object], pd.DataFrame]:
    summary = pd.read_csv(OUTPUT_ROOT / directory / "summary_indicators.csv").iloc[0]
    dispatch = pd.read_csv(OUTPUT_ROOT / directory / "dispatch_results.csv")
    dispatch["datetime"] = pd.to_datetime(dispatch["datetime"])
    event_day = dispatch["datetime"].dt.date.eq(EVENT_DAY)
    selected_peak = event_day & dispatch["regional_grid_load_state"].eq("stressed")
    final_soc_columns = [
        column
        for column in summary.index
        if column.startswith("final_bus_route_") and column.endswith("_soc_fraction")
    ]

    row = {
        "case": label,
        "trip_energy_MWh_over_event_window": summary["total_bus_trip_energy_MWh"],
        "unmet_trip_energy_MWh": summary["total_unmet_trip_energy_MWh"],
        "minimum_final_soc_fraction": summary[final_soc_columns].min(),
        "selected_peak_net_import_MWh": dispatch.loc[selected_peak, "net_grid_import_MWh"].sum(),
        "selected_peak_bus_charge_MWh": aggregate_bus_columns(
            dispatch.loc[selected_peak], "_charge_MWh"
        ).sum(),
        "selected_peak_bus_discharge_MWh": aggregate_bus_columns(
            dispatch.loc[selected_peak], "_discharge_MWh"
        ).sum(),
        "selected_peak_grid_export_MWh": dispatch.loc[selected_peak, "grid_export_MWh"].sum(),
        "max_connected_buses_in_selected_peak_period": aggregate_bus_columns(
            dispatch.loc[selected_peak], "_availability_fraction"
        ).max(),
        "mean_connected_buses_in_selected_peak_period": aggregate_bus_columns(
            dispatch.loc[selected_peak], "_availability_fraction"
        ).mean(),
    }
    return row, dispatch.loc[selected_peak].copy()


def main() -> None:
    rows: list[dict[str, object]] = []
    selected_dispatch: dict[str, pd.DataFrame] = {}
    for label, directory in CASES.items():
        row, dispatch = load_case(label, directory)
        rows.append(row)
        selected_dispatch[label] = dispatch
    result = pd.DataFrame(rows)
    reference_label = "Ten-bus V1G reference"
    reference = result.loc[result["case"].eq(reference_label)].iloc[0]
    reference_import_mw = (
        selected_dispatch[reference_label]["grid_import_MW"]
        - selected_dispatch[reference_label]["grid_export_MW"]
    )
    result["selected_peak_import_reduction_vs_v1g_MWh"] = (
        reference["selected_peak_net_import_MWh"] - result["selected_peak_net_import_MWh"]
    )
    max_reduction = []
    for label in result["case"]:
        candidate = selected_dispatch[label]
        candidate_import_mw = candidate["grid_import_MW"] - candidate["grid_export_MW"]
        max_reduction.append((reference_import_mw - candidate_import_mw).max())
    result["maximum_15_minute_import_reduction_vs_v1g_MW"] = max_reduction
    managed_relief = result.loc[
        result["case"].eq("Ten-bus peak response: managed charging"),
        "selected_peak_import_reduction_vs_v1g_MWh",
    ].iloc[0]
    result["incremental_relief_beyond_managed_charging_MWh"] = (
        result["selected_peak_import_reduction_vs_v1g_MWh"] - managed_relief
    )
    result.loc[0:1, "incremental_relief_beyond_managed_charging_MWh"] = 0.0
    result["nominal_bidirectional_power_MW"] = [0.0, 0.0, 1.5]
    result["nominal_battery_energy_MWh"] = 3.0
    result.to_csv(RESULT_PATH, index=False, float_format="%.9f")

    if (result["unmet_trip_energy_MWh"] > 1e-8).any():
        raise ValueError("The sensitivity has unmet trip energy; do not report it.")
    if (result["minimum_final_soc_fraction"] < 0.8 - 1e-8).any():
        raise ValueError("A bus misses its terminal SOC constraint; do not report it.")
    print(result.to_string(index=False))
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
