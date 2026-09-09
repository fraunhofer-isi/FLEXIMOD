# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Create the annual rooftop-PV forecast from measured roof polygons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

EARTH_RADIUS_M = 6_378_137.0
USABLE_ROOF_FRACTIONS = {
    "conservative": 0.50,
    "base": 0.65,
    "optimistic": 0.80,
}
MODULE_DENSITY_KWP_PER_M2 = 0.20
DC_TO_AC_RATIO = 1.20
INVERTER_EFFICIENCY = 0.96


def prepare_pv_forecast(base_forecast_file: Path, roof_geojson_file: Path) -> pd.DataFrame:
    """Return the annual forecast with rooftop-PV generation columns."""

    forecast = pd.read_csv(base_forecast_file, index_col="datetime", parse_dates=True)
    required = {"solar_pv_capacity_factor", "renewable_weather_year"}
    missing = required - set(forecast.columns)
    if missing:
        raise ValueError("Base forecast is missing columns: " + ", ".join(sorted(missing)))
    if len(forecast) != 35_136 or not forecast["renewable_weather_year"].eq(2024).all():
        raise ValueError("PV case requires a complete 15-minute 2024 forecast")

    gross_area_m2, roof_count = _load_roof_area(roof_geojson_file)
    capacity_factors = forecast["solar_pv_capacity_factor"].astype(float)
    if not capacity_factors.between(0.0, 1.0).all():
        raise ValueError("Solar PV capacity factor must remain between zero and one")

    for scenario, usable_fraction in USABLE_ROOF_FRACTIONS.items():
        usable_area_m2 = gross_area_m2 * usable_fraction
        dc_capacity_mwp = usable_area_m2 * MODULE_DENSITY_KWP_PER_M2 / 1000.0
        ac_capacity_mw = dc_capacity_mwp / DC_TO_AC_RATIO
        generation = (dc_capacity_mwp * capacity_factors * INVERTER_EFFICIENCY).clip(
            upper=ac_capacity_mw
        )
        forecast[f"pv_generation_{scenario}_mw"] = generation

    base_usable_area = gross_area_m2 * USABLE_ROOF_FRACTIONS["base"]
    base_dc_capacity = base_usable_area * MODULE_DENSITY_KWP_PER_M2 / 1000.0
    forecast["pv_1_generation_mw"] = forecast["pv_generation_base_mw"]
    forecast["pv_roof_count"] = roof_count
    forecast["pv_roof_gross_area_m2"] = gross_area_m2
    forecast["pv_roof_usable_fraction"] = USABLE_ROOF_FRACTIONS["base"]
    forecast["pv_roof_usable_area_m2"] = base_usable_area
    forecast["pv_module_density_kwp_per_m2"] = MODULE_DENSITY_KWP_PER_M2
    forecast["pv_dc_capacity_mwp"] = base_dc_capacity
    forecast["pv_dc_to_ac_ratio"] = DC_TO_AC_RATIO
    forecast["pv_ac_capacity_mw"] = base_dc_capacity / DC_TO_AC_RATIO
    forecast["pv_inverter_efficiency"] = INVERTER_EFFICIENCY
    forecast["pv_roof_area_assessment"] = "desktop_satellite_digitisation"
    forecast["pv_structural_survey_completed"] = 0
    return forecast


def _load_roof_area(roof_geojson_file: Path) -> tuple[float, int]:
    with roof_geojson_file.open(encoding="utf-8") as stream:
        geojson = json.load(stream)
    features = geojson.get("features", [])
    if not features:
        raise ValueError("Roof GeoJSON does not contain any polygons")

    gross_area_m2 = 0.0
    for feature in features:
        coordinates = feature["geometry"]["coordinates"][0]
        calculated_area = _polygon_area_m2(coordinates)
        recorded_area = float(feature["properties"]["gross_area_m2"])
        if not math.isclose(calculated_area, recorded_area, rel_tol=0.01):
            raise ValueError(
                f"Roof {feature['properties']['roof_id']} area does not match its geometry"
            )
        gross_area_m2 += recorded_area
    return gross_area_m2, len(features)


def _polygon_area_m2(coordinates: list[list[float]]) -> float:
    mean_latitude = sum(point[1] for point in coordinates[:-1]) / (len(coordinates) - 1)
    projected = [
        (
            math.radians(longitude) * EARTH_RADIUS_M * math.cos(math.radians(mean_latitude)),
            math.radians(latitude) * EARTH_RADIUS_M,
        )
        for longitude, latitude in coordinates
    ]
    cross_products = (
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(projected, projected[1:], strict=False)
    )
    return abs(sum(cross_products)) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_forecast_file", type=Path)
    parser.add_argument("roof_geojson_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    forecast = prepare_pv_forecast(args.base_forecast_file, args.roof_geojson_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    forecast.rename_axis("datetime").to_csv(args.output_file)
    print(f"Created {args.output_file} with {len(forecast)} rows")


if __name__ == "__main__":
    main()
