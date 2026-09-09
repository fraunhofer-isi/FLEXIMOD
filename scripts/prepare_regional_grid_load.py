# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Create a 15-minute regional grid-load profile from the 2024 Zenodo file."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_YEAR = 2024
EXPECTED_MD5 = "5c1f09d6641ad6addb6f282f981e808a"


def prepare_profile(source_file: Path, target_date: str) -> pd.DataFrame:
    """Return the mean September-Monday profile at 15-minute resolution."""

    _check_source_file(source_file)
    raw = pd.read_csv(source_file, encoding="utf-8-sig")
    load = _clean_hourly_load(raw)

    september_mondays = sorted(
        set(load.loc[(load.index.month == 9) & (load.index.weekday == 0)].index.normalize())
    )
    hourly_values = [
        np.mean([load.loc[day + pd.Timedelta(hours=hour)] for day in september_mondays])
        for hour in range(24)
    ]
    next_midnight = np.mean([load.loc[day + pd.Timedelta(days=1)] for day in september_mondays])

    source_points = np.asarray([*hourly_values, next_midnight], dtype=float)
    quarter_hours = np.arange(96, dtype=float) / 4.0
    interpolated_load = np.interp(quarter_hours, np.arange(25), source_points)

    p90 = float(load.quantile(0.90))
    p99 = float(load.quantile(0.99))
    annual_peak = float(load.max())
    congestion_weight = np.clip((interpolated_load - p90) / (p99 - p90), 0.0, 1.0)

    index = pd.date_range(target_date, periods=96, freq="15min")
    return pd.DataFrame(
        {
            "regional_grid_load_mw": interpolated_load,
            "regional_grid_load_fraction": interpolated_load / annual_peak,
            "grid_congestion_weight": congestion_weight,
            "regional_grid_load_source_year": SOURCE_YEAR,
            "regional_grid_load_profile": "september_monday_mean",
            "regional_grid_load_interpolated": (index.minute != 0).astype(int),
        },
        index=index,
    )


def _check_source_file(source_file: Path) -> None:
    if not source_file.exists():
        raise FileNotFoundError(source_file)
    checksum = hashlib.md5(source_file.read_bytes(), usedforsecurity=False).hexdigest()
    if checksum != EXPECTED_MD5:
        raise ValueError(f"Unexpected source-file checksum {checksum}; expected Zenodo version 1.0")


def _clean_hourly_load(raw: pd.DataFrame) -> pd.Series:
    required = {"datetime", "metropolitan_demand"}
    missing_columns = required - set(raw.columns)
    if missing_columns:
        raise ValueError("Missing source columns: " + ", ".join(sorted(missing_columns)))

    frame = raw.loc[:, ["datetime", "metropolitan_demand"]].copy()
    frame["source_datetime"] = pd.to_datetime(frame["datetime"], dayfirst=True)
    frame["hour"] = frame["source_datetime"].dt.floor("h")
    frame["exact_hour"] = (frame["source_datetime"].dt.minute == 0).astype(int)
    frame = frame.loc[
        (frame["hour"] >= f"{SOURCE_YEAR}-01-01") & (frame["hour"] < f"{SOURCE_YEAR + 1}-01-01")
    ]
    frame = frame.sort_values(["hour", "exact_hour"]).drop_duplicates("hour", keep="last")

    expected_index = pd.date_range(f"{SOURCE_YEAR}-01-01", f"{SOURCE_YEAR}-12-31 23:00", freq="h")
    load = frame.set_index("hour")["metropolitan_demand"].reindex(expected_index)
    if load.isna().sum() != 3:
        raise ValueError("Expected exactly three missing hourly observations in the 2024 file")
    return load.interpolate(method="time")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_file", type=Path, help="Downloaded system_2024.csv")
    parser.add_argument("output_file", type=Path, help="Destination CSV")
    parser.add_argument("--target-date", default="2026-09-07")
    args = parser.parse_args()

    profile = prepare_profile(args.source_file, args.target_date)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    profile.rename_axis("datetime").to_csv(args.output_file)
    print(f"Created {args.output_file} with {len(profile)} quarter-hour rows")


if __name__ == "__main__":
    main()
