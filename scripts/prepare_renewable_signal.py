# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add a 2024 weather-driven PV and wind availability proxy to a forecast."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEM_FILE_MD5 = "5c1f09d6641ad6addb6f282f981e808a"
SYSTEM_DEMAND_COLUMNS = [
    "north_demand",
    "south_demand",
    "metropolitan_demand",
    "central_demand",
    "northeast_demand",
]

PV_CAPACITY_2024_MW = 3383.0
WIND_CAPACITY_2024_MW = 1544.0
PV_CAPACITY_2025_MW = 6837.0
WIND_CAPACITY_2025_MW = 1544.0

PV_SYSTEM_LOSS = 0.10
PV_NOCT_C = 45.0
PV_TEMPERATURE_COEFFICIENT = -0.004
WIND_SHEAR_EXPONENT = 1.0 / 7.0
WIND_HUB_HEIGHT_M = 100.0
WIND_CUT_IN_MPS = 3.0
WIND_RATED_MPS = 12.0
WIND_CUT_OUT_MPS = 25.0


def prepare_signal(
    weather_file: Path,
    system_file: Path,
    base_forecast_file: Path,
) -> pd.DataFrame:
    """Return the annual forecast with renewable proxy columns appended."""

    weather, weather_metadata = _load_weather(weather_file)
    demand = _load_national_demand(system_file)
    base = pd.read_csv(base_forecast_file, index_col="datetime", parse_dates=True)

    pv_cf = _pv_capacity_factor(weather)
    wind_cf = _wind_capacity_factor(weather["WS50M"])
    availability_2024 = _weighted_availability(
        pv_cf,
        wind_cf,
        PV_CAPACITY_2024_MW,
        WIND_CAPACITY_2024_MW,
    )
    availability_2025 = _weighted_availability(
        pv_cf,
        wind_cf,
        PV_CAPACITY_2025_MW,
        WIND_CAPACITY_2025_MW,
    )
    potential_mw = pv_cf * PV_CAPACITY_2024_MW + wind_cf * WIND_CAPACITY_2024_MW
    rich_threshold = float(availability_2024.quantile(0.80))

    hourly = pd.DataFrame(
        {
            "solar_pv_capacity_factor": pv_cf,
            "wind_capacity_factor": wind_cf,
            "renewable_availability_weight": availability_2024,
            "renewable_availability_weight_2025_capacity": availability_2025,
            "modelled_vre_potential_mw": potential_mw,
            "national_system_demand_mw": demand,
            "modelled_vre_potential_share_of_demand": potential_mw / demand,
            "renewable_rich_threshold": rich_threshold,
            "renewable_rich": (availability_2024 >= rich_threshold).astype(int),
        }
    )

    if hourly.isna().any().any():
        raise ValueError("Renewable proxy contains missing hourly values")
    if not hourly["renewable_availability_weight"].between(0.0, 1.0).all():
        raise ValueError("Renewable availability must remain between 0 and 1")

    lookup_index = base.index.floor("h")
    for column in hourly:
        base[column] = hourly[column].reindex(lookup_index).to_numpy()
    base["renewable_weather_source"] = "NASA_POWER"
    base["renewable_weather_year"] = 2024
    base["renewable_weather_scope"] = "local_weather_national_capacity_mix"
    base["renewable_capacity_basis_year"] = 2024
    base["renewable_weather_api_version"] = weather_metadata["api_version"]
    return base


def _load_weather(weather_file: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    with weather_file.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    parameters = payload["properties"]["parameter"]
    required = {"ALLSKY_SFC_SW_DWN", "T2M", "WS50M"}
    missing = required - set(parameters)
    if missing:
        raise ValueError("Weather file is missing parameters: " + ", ".join(sorted(missing)))

    frame = pd.DataFrame(parameters, dtype=float)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H", utc=True)
    frame.index = frame.index.tz_convert("Asia/Bangkok").tz_localize(None)
    frame = frame.loc["2024-01-01 00:00":"2024-12-31 23:00"]
    expected = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    frame = frame.reindex(expected)
    frame = frame.replace(float(payload["header"]["fill_value"]), np.nan)
    if frame.isna().any().any():
        raise ValueError("Weather file does not provide a complete local 2024 calendar year")
    return frame, {"api_version": str(payload["header"]["api"]["version"])}


def _load_national_demand(system_file: Path) -> pd.Series:
    checksum = hashlib.md5(system_file.read_bytes(), usedforsecurity=False).hexdigest()
    if checksum != SYSTEM_FILE_MD5:
        raise ValueError("Unexpected power-system source-file checksum")

    raw = pd.read_csv(system_file, encoding="utf-8-sig")
    missing = set(["datetime", *SYSTEM_DEMAND_COLUMNS]) - set(raw.columns)
    if missing:
        raise ValueError("Power-system file is missing columns: " + ", ".join(sorted(missing)))
    raw["source_datetime"] = pd.to_datetime(raw["datetime"], dayfirst=True)
    raw["hour"] = raw["source_datetime"].dt.floor("h")
    raw["exact_hour"] = (raw["source_datetime"].dt.minute == 0).astype(int)
    raw = raw.loc[(raw["hour"] >= "2024-01-01") & (raw["hour"] < "2025-01-01")]
    raw = raw.sort_values(["hour", "exact_hour"]).drop_duplicates("hour", keep="last")

    expected = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    regional = raw.set_index("hour")[SYSTEM_DEMAND_COLUMNS].reindex(expected)
    regional = regional.interpolate(method="time", limit_direction="both")
    if regional.isna().any().any() or (regional < 0.0).any().any():
        raise ValueError("National demand must be complete and non-negative")
    return regional.sum(axis=1).rename("national_system_demand_mw")


def _pv_capacity_factor(weather: pd.DataFrame) -> pd.Series:
    irradiance_w_per_m2 = weather["ALLSKY_SFC_SW_DWN"]
    cell_temperature = weather["T2M"] + (PV_NOCT_C - 20.0) / 800.0 * irradiance_w_per_m2
    temperature_factor = 1.0 + PV_TEMPERATURE_COEFFICIENT * (cell_temperature - 25.0)
    capacity_factor = (
        irradiance_w_per_m2 / 1000.0 * temperature_factor * (1.0 - PV_SYSTEM_LOSS)
    )
    return capacity_factor.clip(0.0, 1.0).rename("solar_pv_capacity_factor")


def _wind_capacity_factor(wind_speed_50m: pd.Series) -> pd.Series:
    speed = wind_speed_50m * (WIND_HUB_HEIGHT_M / 50.0) ** WIND_SHEAR_EXPONENT
    cubic_section = (speed**3 - WIND_CUT_IN_MPS**3) / (
        WIND_RATED_MPS**3 - WIND_CUT_IN_MPS**3
    )
    capacity_factor = np.where(
        speed < WIND_CUT_IN_MPS,
        0.0,
        np.where(
            speed < WIND_RATED_MPS,
            cubic_section,
            np.where(speed <= WIND_CUT_OUT_MPS, 1.0, 0.0),
        ),
    )
    return pd.Series(capacity_factor, index=wind_speed_50m.index, name="wind_capacity_factor")


def _weighted_availability(
    pv_cf: pd.Series,
    wind_cf: pd.Series,
    pv_capacity_mw: float,
    wind_capacity_mw: float,
) -> pd.Series:
    total_capacity = pv_capacity_mw + wind_capacity_mw
    return (
        (pv_capacity_mw * pv_cf + wind_capacity_mw * wind_cf) / total_capacity
    ).rename("renewable_availability_weight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weather_file", type=Path, help="NASA POWER hourly JSON")
    parser.add_argument("system_file", type=Path, help="2024 power-system CSV")
    parser.add_argument("base_forecast_file", type=Path, help="Annual building forecast CSV")
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    forecast = prepare_signal(args.weather_file, args.system_file, args.base_forecast_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    forecast.rename_axis("datetime").to_csv(args.output_file)
    print(f"Created {args.output_file} with {len(forecast)} rows")


if __name__ == "__main__":
    main()
