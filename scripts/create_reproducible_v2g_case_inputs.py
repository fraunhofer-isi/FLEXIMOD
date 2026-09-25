"""Create immutable input directories for the two reported reference results.

The 3.15 THB/kWh export-onset result and the PV + unidirectional reference
were previously produced from temporary or subsequently changed configurations.
This helper makes dedicated copies of their source data and adds only the
case-specific physical/forecast fields needed for reproducible reruns.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def copy_case_inputs(source_name: str, target_name: str) -> Path:
    source = ROOT / "data" / "input" / source_name
    target = ROOT / "data" / "input" / target_name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing input directory: {target}")
    shutil.copytree(source, target)
    return target


def main() -> None:
    onset_dir = copy_case_inputs(
        "building_v2g_tariff_sweep_annual",
        "building_v2g_export_onset_3150_annual",
    )
    onset_forecast = onset_dir / "forecasts_df.csv"
    forecast = pd.read_csv(onset_forecast)
    forecast["v2g_export_price_3150_thb_per_mwh"] = 3150.0
    forecast.to_csv(onset_forecast, index=False)

    reference_dir = copy_case_inputs(
        "building_v2b_pv_self_consumption_annual",
        "building_i07_reference_annual",
    )
    reference_plants = reference_dir / "plants.csv"
    plants = pd.read_csv(reference_plants)
    flexible_assets = plants["technology"].isin(["electric_vehicle", "charging_station"])
    plants.loc[flexible_assets, "max_power_discharge"] = 0.0
    plants.loc[flexible_assets, "power_flow_directionality"] = "unidirectional"
    plants.to_csv(reference_plants, index=False)


if __name__ == "__main__":
    main()
