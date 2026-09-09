# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Create the shared full-year, 15-minute building forecast for 2024."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_regional_grid_load import (
    EXPECTED_MD5,
    SOURCE_YEAR,
    _check_source_file,
    _clean_hourly_load,
)

TIMESTEP = "15min"
CURRENT_EXPORT_PAYMENT_THB_PER_MWH = 2200.0
FUEL_ADJUSTMENT_THB_PER_KWH = 0.1623
VAT_RATE = 0.07
PEAK_ENERGY_CHARGE_THB_PER_KWH = 4.3297
OFF_PEAK_ENERGY_CHARGE_THB_PER_KWH = 2.6369
GRID_STRESS_THRESHOLD = 0.8

# Only the actual holiday dates are listed. Substitution holidays are not TOU holidays.
TOU_HOLIDAYS_2024 = {
    "2024-01-01",
    "2024-02-24",
    "2024-04-06",
    "2024-04-13",
    "2024-04-14",
    "2024-04-15",
    "2024-05-01",
    "2024-05-04",
    "2024-05-22",
    "2024-06-03",
    "2024-07-20",
    "2024-07-21",
    "2024-07-28",
    "2024-08-12",
    "2024-10-13",
    "2024-10-23",
    "2024-12-05",
    "2024-12-10",
    "2024-12-31",
}


def prepare_annual_forecast(source_file: Path, daily_template_file: Path) -> pd.DataFrame:
    """Combine chronological grid load, repeated routes, and tariff assumptions."""

    _check_source_file(source_file)
    raw = pd.read_csv(source_file, encoding="utf-8-sig")
    hourly_load = _clean_hourly_load(raw)
    index = pd.date_range("2024-01-01", "2024-12-31 23:45", freq=TIMESTEP)

    annual = _repeat_daily_template(daily_template_file, index)
    annual = annual.join(_regional_grid_columns(hourly_load, raw, index))
    annual = annual.join(_tariff_columns(index))
    return annual


def _repeat_daily_template(template_file: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    template = pd.read_csv(template_file, index_col="datetime", parse_dates=True)
    profile_columns = [
        "building_1_electricity_demand",
        "bus_route_7k_availability",
        "bus_route_7k_trip_distance_km",
        "bus_route_79_availability",
        "bus_route_79_trip_distance_km",
        "bus_route_101_availability",
        "bus_route_101_trip_distance_km",
    ]
    if len(template) != 96:
        raise ValueError("The daily route template must contain 96 quarter-hour rows")
    missing = set(profile_columns) - set(template.columns)
    if missing:
        raise ValueError("Daily template is missing columns: " + ", ".join(sorted(missing)))
    values = np.tile(template[profile_columns].to_numpy(), (len(index) // 96, 1))
    return pd.DataFrame(values, index=index, columns=profile_columns)


def _regional_grid_columns(
    hourly_load: pd.Series,
    raw: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    next_midnight = _next_midnight_load(raw)
    points = pd.concat(
        [hourly_load, pd.Series([next_midnight], index=[pd.Timestamp("2025-01-01")])]
    )
    load = points.reindex(points.index.union(index)).interpolate(method="time").reindex(index)
    p90 = float(hourly_load.quantile(0.90))
    p99 = float(hourly_load.quantile(0.99))
    peak = float(hourly_load.max())
    weights = np.clip((load - p90) / (p99 - p90), 0.0, 1.0)
    state = np.where(
        weights >= GRID_STRESS_THRESHOLD,
        "stressed",
        np.where(weights > 0.0, "elevated", "normal"),
    )
    return pd.DataFrame(
        {
            "regional_grid_load_mw": load,
            "regional_grid_load_fraction": load / peak,
            "grid_congestion_weight": weights,
            "regional_grid_load_state": state,
            "grid_stress_threshold": GRID_STRESS_THRESHOLD,
            "regional_grid_load_source_year": SOURCE_YEAR,
            "regional_grid_load_profile": "historical_2024",
            "regional_grid_load_interpolated": (index.minute != 0).astype(int),
        },
        index=index,
    )


def _next_midnight_load(raw: pd.DataFrame) -> float:
    timestamp = pd.to_datetime(raw["datetime"], dayfirst=True).dt.floor("h")
    values = pd.to_numeric(raw.loc[timestamp == "2025-01-01 00:00", "metropolitan_demand"])
    if values.empty:
        raise ValueError("The source file does not contain the next midnight endpoint")
    return float(values.iloc[-1])


def _tariff_columns(index: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.Series(index.strftime("%Y-%m-%d"), index=index)
    holiday = dates.isin(TOU_HOLIDAYS_2024)
    off_peak = (index.weekday >= 5) | holiday | (index.hour < 9) | (index.hour >= 22)
    base = np.where(off_peak, OFF_PEAK_ENERGY_CHARGE_THB_PER_KWH, PEAK_ENERGY_CHARGE_THB_PER_KWH)
    import_price = (base + FUEL_ADJUSTMENT_THB_PER_KWH) * 1000.0
    frame = pd.DataFrame(
        {
            "tou_period": np.where(off_peak, "off_peak", "on_peak"),
            "is_public_holiday": holiday.astype(int),
            "base_energy_charge_thb_per_kwh": base,
            "fuel_adjustment_thb_per_kwh": FUEL_ADJUSTMENT_THB_PER_KWH,
            "vat_rate": VAT_RATE,
            "electricity_import_price_thb_per_mwh": import_price,
            "electricity_import_price_with_vat_thb_per_mwh": import_price * (1 + VAT_RATE),
            "v2g_export_price_current_thb_per_mwh": CURRENT_EXPORT_PAYMENT_THB_PER_MWH,
            "tariff_assumption_year": 2026,
        },
        index=index,
    )
    for increment in range(500, 5001, 500):
        frame[f"v2g_export_price_plus_{increment}_thb_per_mwh"] = (
            CURRENT_EXPORT_PAYMENT_THB_PER_MWH + increment
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_file", type=Path)
    parser.add_argument("daily_template_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    forecast = prepare_annual_forecast(args.source_file, args.daily_template_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    forecast.rename_axis("datetime").to_csv(args.output_file)
    print(
        f"Created {args.output_file} with {len(forecast)} rows "
        f"using source checksum {EXPECTED_MD5}"
    )


if __name__ == "__main__":
    main()
