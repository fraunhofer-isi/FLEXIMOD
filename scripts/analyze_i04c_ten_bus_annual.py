#!/usr/bin/env python3
"""Build annual I04c metrics for the three- versus ten-bus comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "output" / "building_use_case_analysis"
RESULT_PATH = OUTPUT_ROOT / "ten_bus_annual_peak_demand_summary.csv"

RESULTS = {
    "V1G managed charging (3 buses)": "v1g_baseline",
    "Technical V2G (3 buses)": "grid_support_export_450kw",
    "V1G managed charging (10 buses)": "ten_bus_v1g_reference",
    "Technical V2G (10 buses)": "ten_bus_v2g_technical_annual",
}


def bus_total(frame: pd.DataFrame, suffix: str, mask: pd.Series) -> pd.Series:
    columns = [
        column
        for column in frame.columns
        if column.startswith("bus_route_") and column.endswith(suffix)
    ]
    if not columns:
        # The completed three-bus outputs include aggregated fields as well as
        # individual fields. The latter remain preferred for both fleet sizes.
        aggregate = "bus_charge_MWh" if suffix == "_charge_MWh" else "bus_discharge_MWh"
        return frame.loc[mask, aggregate]
    return frame.loc[mask, columns].sum(axis=1)


def load_result(label: str, directory: str) -> tuple[dict[str, object], pd.Series]:
    summary = pd.read_csv(OUTPUT_ROOT / directory / "summary_indicators.csv").iloc[0]
    dispatch = pd.read_csv(OUTPUT_ROOT / directory / "dispatch_results.csv", parse_dates=["datetime"])
    peak = dispatch["regional_grid_load_state"].eq("stressed")
    charge = bus_total(dispatch, "_charge_MWh", peak)
    discharge = bus_total(dispatch, "_discharge_MWh", peak)
    net_import = dispatch.loc[peak, "net_grid_import_MWh"]
    final_soc_columns = [
        column
        for column in summary.index
        if column.startswith("final_bus_route_") and column.endswith("_soc_fraction")
    ]
    return {
        "case": label,
        "peak_period_bus_charge_MWh": charge.sum(),
        "peak_period_bus_discharge_MWh": discharge.sum(),
        "peak_period_net_depot_import_MWh": net_import.sum(),
        "unmet_trip_energy_MWh": summary["total_unmet_trip_energy_MWh"],
        "minimum_final_soc_fraction": summary[final_soc_columns].min(),
    }, charge


def main() -> None:
    rows: list[dict[str, object]] = []
    peak_charge: dict[str, pd.Series] = {}
    for label, directory in RESULTS.items():
        row, charge = load_result(label, directory)
        rows.append(row)
        peak_charge[label] = charge
    raw = pd.DataFrame(rows).set_index("case")
    v1g_3 = raw.loc["V1G managed charging (3 buses)"]
    v2g_3 = raw.loc["Technical V2G (3 buses)"]
    v1g_10 = raw.loc["V1G managed charging (10 buses)"]
    v2g_10 = raw.loc["Technical V2G (10 buses)"]
    report = pd.DataFrame(
        {
            "case": [
                "V1G managed charging (3 buses)",
                "Technical V2G (3 buses)",
                "Technical V2G (10 buses)",
            ],
            "bus_energy_delivered_in_peak_periods_MWh_year": [
                0.0,
                v2g_3["peak_period_bus_discharge_MWh"],
                v2g_10["peak_period_bus_discharge_MWh"],
            ],
            "charging_shifted_out_of_peak_periods_MWh_year": [
                0.0,
                v1g_3["peak_period_bus_charge_MWh"] - v2g_3["peak_period_bus_charge_MWh"],
                v1g_10["peak_period_bus_charge_MWh"] - v2g_10["peak_period_bus_charge_MWh"],
            ],
            "maximum_charging_avoided_in_peak_periods_kW": [
                0.0,
                (peak_charge["V1G managed charging (3 buses)"] - peak_charge["Technical V2G (3 buses)"]).max() / 0.25 * 1000,
                (peak_charge["V1G managed charging (10 buses)"] - peak_charge["Technical V2G (10 buses)"]).max() / 0.25 * 1000,
            ],
            "net_depot_import_reduction_in_peak_periods_MWh_year": [
                0.0,
                v1g_3["peak_period_net_depot_import_MWh"] - v2g_3["peak_period_net_depot_import_MWh"],
                v1g_10["peak_period_net_depot_import_MWh"] - v2g_10["peak_period_net_depot_import_MWh"],
            ],
        }
    )
    report.to_csv(RESULT_PATH, index=False, float_format="%.9f")
    if (raw["unmet_trip_energy_MWh"] > 1e-8).any():
        raise ValueError("Unmet trip energy in one of the annual I04c cases.")
    if (raw["minimum_final_soc_fraction"] < 0.8 - 1e-8).any():
        raise ValueError("Terminal SOC violation in one of the annual I04c cases.")
    print(report.to_string(index=False))
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
