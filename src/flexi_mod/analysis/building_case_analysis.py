"""Derived operating-service metrics for the electric-bus depot study.

The helpers deliberately calculate operating compensation indicators only.
They do not infer investment costs, market-clearing prices, or regulatory
approval for a service.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


DEMAND_CHARGE_THB_PER_KW_MONTH = 210.0
TIME_STEP_HOURS = 0.25


def result_availability(
    cases: Mapping[str, Mapping[str, object]], output_root: str | object
) -> pd.DataFrame:
    """Return an availability table without treating missing outputs as zero."""
    root = Path(output_root)
    rows = []
    for label, spec in cases.items():
        output = str(spec["output"])
        path = root / output
        summary = path / "summary_indicators.csv"
        dispatch = path / "dispatch_results.csv"
        rows.append(
            {
                "scenario": label,
                "output_folder": output,
                "summary_available": summary.exists(),
                "dispatch_available": dispatch.exists(),
                "status": "complete"
                if summary.exists() and dispatch.exists()
                else "pending / unavailable",
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def _safe_unit_cost(cost_gap: float, service_mwh: float, peak_mw: float, suffix: str) -> dict[str, float]:
    """Return unit-cost values only when a measurable service was delivered."""
    if not np.isfinite(service_mwh) or service_mwh <= 1e-9:
        return {
            "cost_gap_THB_per_service_kWh": np.nan,
            f"cost_gap_THB_per_kW_{suffix}": np.nan,
        }
    return {
        "cost_gap_THB_per_service_kWh": cost_gap / (service_mwh * 1000.0),
        f"cost_gap_THB_per_kW_{suffix}": cost_gap / (peak_mw * 1000.0)
        if np.isfinite(peak_mw) and peak_mw > 1e-9
        else np.nan,
    }


def _annual_service_row(
    comparison: pd.DataFrame,
    label: str,
    baseline_label: str,
    service: str,
    energy_column: str,
    power_column: str,
) -> dict[str, object] | None:
    if label not in comparison.index or baseline_label not in comparison.index:
        return None
    case = comparison.loc[label]
    baseline = comparison.loc[baseline_label]
    service_mwh = float(case.get(energy_column, np.nan))
    peak_mw = float(case.get(power_column, np.nan))
    cost_gap = float(case["total_cost_THB"] - baseline["total_cost_THB"])
    row: dict[str, object] = {
        "scenario": label,
        "service_type": service,
        "comparison_basis": baseline_label,
        "time_basis": "annual",
        "cost_gap_THB": cost_gap,
        "service_energy_MWh": service_mwh,
        "peak_service_kW": peak_mw * 1000.0,
        "unit_cost_status": "operator saving" if cost_gap < 0 else "compensation gap",
        "potential_monthly_demand_charge_exposure_THB": np.nan,
    }
    row.update(_safe_unit_cost(cost_gap, service_mwh, peak_mw, "year"))
    return row


def _pv_high_load_service_row(
    comparison: pd.DataFrame,
    dispatch_by_case: Mapping[str, pd.DataFrame],
    label: str,
) -> dict[str, object] | None:
    """Measure PV high-load relief incrementally against PV-only operation."""
    baseline_label = "PV-only operating reference"
    if (
        label not in comparison.index
        or baseline_label not in comparison.index
        or label not in dispatch_by_case
        or baseline_label not in dispatch_by_case
    ):
        return None

    case_dispatch = dispatch_by_case[label]
    baseline_dispatch = dispatch_by_case[baseline_label]
    common_index = baseline_dispatch.index.intersection(case_dispatch.index)
    if common_index.empty:
        return None

    def net_import(frame: pd.DataFrame) -> pd.Series:
        if "traction_net_grid_import_MWh" in frame:
            return frame["traction_net_grid_import_MWh"]
        return frame["grid_import_MWh"] - frame["grid_export_MWh"]

    case_net = net_import(case_dispatch).reindex(common_index)
    baseline_net = net_import(baseline_dispatch).reindex(common_index)
    relief = baseline_net - case_net
    high_load = (
        case_dispatch.loc[common_index, "grid_congestion_weight"]
        >= case_dispatch.loc[common_index, "grid_stress_threshold"]
    )
    service_mwh = float(relief.loc[high_load].clip(lower=0.0).sum())
    peak_mw = float(relief.loc[high_load].clip(lower=0.0).max() / TIME_STEP_HOURS)
    cost_gap = float(
        comparison.loc[label, "total_cost_THB"]
        - comparison.loc[baseline_label, "total_cost_THB"]
    )
    row: dict[str, object] = {
        "scenario": label,
        "service_type": "PV high-load demand management",
        "comparison_basis": baseline_label,
        "time_basis": "annual",
        "cost_gap_THB": cost_gap,
        "service_energy_MWh": service_mwh,
        "peak_service_kW": peak_mw * 1000.0,
        "unit_cost_status": "operator saving" if cost_gap < 0 else "compensation gap",
        "potential_monthly_demand_charge_exposure_THB": np.nan,
    }
    row.update(_safe_unit_cost(cost_gap, service_mwh, peak_mw, "year"))
    return row


def _emergency_service_row(label: str, event: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, object]:
    """Calculate an event cost relative to normal V1G over matching timestamps."""
    outage = event.get("synthetic_outage_event", pd.Series(0, index=event.index)).astype(bool)
    baseline_slice = baseline.reindex(event.index)
    if baseline_slice.isna().all(axis=None):
        raise ValueError(f"V1G dispatch does not cover the {label} emergency event")
    service_mwh = float(event.loc[outage, "bus_discharge_MWh"].sum())
    peak_mw = float((event.loc[outage, "bus_discharge_MWh"] / TIME_STEP_HOURS).max()) if outage.any() else np.nan
    energy_gap = float(event["energy_cost_THB"].sum() - baseline_slice["energy_cost_THB"].sum())
    recovery_peak_mw = float(event.loc[~outage, "traction_grid_import_MWh"].max() / TIME_STEP_HOURS)
    normal_peak_mw = float(baseline_slice["traction_grid_import_MWh"].max() / TIME_STEP_HOURS)
    potential_peak_charge = max(0.0, recovery_peak_mw - normal_peak_mw) * 1000.0 * DEMAND_CHARGE_THB_PER_KW_MONTH
    islanded_import = float(event.loc[outage, "grid_import_MWh"].sum())
    islanded_export = float(event.loc[outage, "grid_export_MWh"].sum())
    row: dict[str, object] = {
        "scenario": label,
        "service_type": "regional-high-load-correlated emergency transfer",
        "comparison_basis": "matched V1G event-and-recovery window",
        "time_basis": "event",
        "cost_gap_THB": energy_gap,
        "service_energy_MWh": service_mwh,
        "peak_service_kW": peak_mw * 1000.0,
        "unit_cost_status": "operator saving" if energy_gap < 0 else "recovery-energy compensation gap",
        "potential_monthly_demand_charge_exposure_THB": potential_peak_charge,
        "outage_grid_import_MWh": islanded_import,
        "outage_grid_export_MWh": islanded_export,
        "islanding_check": "pass" if abs(islanded_import) < 1e-9 and abs(islanded_export) < 1e-9 else "FAIL",
    }
    row.update(_safe_unit_cost(energy_gap, service_mwh, peak_mw, "event"))
    return row


def build_service_cost_gap(
    comparison: pd.DataFrame, dispatch_by_case: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Build non-revenue service compensation indicators from completed results."""
    annual_specs = (
        ("I03 maximum regional high-load import relief without export", "regional high-load import relief", "grid_support_during_stress_MWh", "maximum_grid_support_MW"),
        ("I04a maximum regional high-load import relief at 5 kW policy limit", "regional high-load import relief", "grid_support_during_stress_MWh", "maximum_grid_support_MW"),
        ("I04b maximum regional high-load import relief at 450 kW technical sensitivity", "regional high-load import relief", "grid_support_during_stress_MWh", "maximum_grid_support_MW"),
        ("I05 maximum renewable-deficit support without export", "renewable-equivalent proxy service", "renewable_deficit_support_MWh", "maximum_grid_support_MW"),
        ("I06a maximum renewable-deficit support at 5 kW policy limit", "renewable-equivalent proxy service", "renewable_deficit_support_MWh", "maximum_grid_support_MW"),
        ("I06b maximum renewable-deficit support at 450 kW technical sensitivity", "renewable-equivalent proxy service", "renewable_deficit_support_MWh", "maximum_grid_support_MW"),
        ("I07b rooftop PV operation at current export policy", "rooftop-PV export", "grid_export_MWh", "peak_grid_export_MW"),
        ("I07c rooftop PV operation at viable export tariff", "rooftop-PV export", "grid_export_MWh", "peak_grid_export_MW"),
    )
    rows: list[dict[str, object]] = []
    for label, service, energy_column, power_column in annual_specs:
        baseline = "PV-only operating reference" if label.startswith("I07") else "V1G baseline"
        row = _annual_service_row(comparison, label, baseline, service, energy_column, power_column)
        if row is not None:
            rows.append(row)
    pv_high_load_row = _pv_high_load_service_row(
        comparison,
        dispatch_by_case,
        "I07d PV with high-load demand management",
    )
    if pv_high_load_row is not None:
        rows.append(pv_high_load_row)
    v1g = dispatch_by_case.get("V1G baseline")
    if v1g is not None:
        for label, event in dispatch_by_case.items():
            if label.startswith("I08"):
                rows.append(_emergency_service_row(label, event, v1g))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("scenario")


def stakeholder_matrix() -> pd.DataFrame:
    """Return a model-bound stakeholder allocation matrix for the Thailand study."""
    rows = [
        ("Fleet / public-bus operator", "Mobility-compliant charging; any operating saving", "SOC readiness, recovery energy and peak-demand exposure", "Energy-service THB/kWh; availability THB/kW-year or THB/kW-event"),
        ("Depot owner, PV owner and charging-service provider", "PV utilisation, controlled export and service revenue opportunity", "Charger controls, metering, protection and settlement interface", "PV-V2G service cost-gap metric"),
        ("PEA or MEA distribution utility", "Potential local regional-high-load-period net-import relief", "Connection approval, local-network validation and operating limits", "Verified high-load import-relief THB/kWh and availability THB/kW-year"),
        ("EGAT system operator", "Potential system-support signal response", "Dispatch coordination; no claimed avoided transmission investment", "Only after a defined system-service product"),
        ("ERC, EPPO and Ministry of Energy", "Evidence for pilot tariff, settlement and consumer-protection design", "Interconnection, measurement, allocation of value and consumer safeguards", "Use cost gaps as procurement evidence, not tariff recommendations"),
        ("Passengers, transport authorities and wider society", "Mobility continuity, resilience insight and lower traction-boundary emissions", "No external critical-load or social value monetised in this model", "Public value remains unpriced in this operational analysis"),
    ]
    return pd.DataFrame(rows, columns=["stakeholder", "benefit", "responsibility / exposure", "decision-relevant metric"]).set_index("stakeholder")
