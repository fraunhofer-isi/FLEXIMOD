"""Promote the completed, equivalent 3.15 THB/kWh I02 annual output.

The original annual run was created from a temporary directory.  Its input
data and effective dispatch configuration are validated against the permanent
case before copying the four output tables into the canonical output folder.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INPUT = ROOT / "data" / "input" / "building_v2g_tariff_sweep_annual"
CANONICAL_INPUT = ROOT / "data" / "input" / "building_v2g_export_onset_3150_annual"
SOURCE_OUTPUT = ROOT / "data" / "output" / "building_use_case_analysis" / "i02_threshold_3150"
CANONICAL_OUTPUT = ROOT / "data" / "output" / "building_use_case_analysis" / "i02_export_onset_3150"
OUTPUT_FILES = (
    "dispatch_results.csv",
    "market_ledger.csv",
    "storage_cost_ledger.csv",
    "summary_indicators.csv",
)


def main() -> None:
    source_forecast = pd.read_csv(SOURCE_INPUT / "forecasts_df.csv")
    canonical_forecast = pd.read_csv(CANONICAL_INPUT / "forecasts_df.csv")
    canonical_price_column = "v2g_export_price_3150_thb_per_mwh"
    if canonical_price_column not in canonical_forecast:
        raise ValueError("Canonical 3.15 THB/kWh export-price column is missing")
    pd.testing.assert_frame_equal(
        canonical_forecast.drop(columns=canonical_price_column), source_forecast,
        check_dtype=False,
    )
    if not canonical_forecast[canonical_price_column].eq(3150.0).all():
        raise ValueError("Canonical export-price signal is not constantly 3150 THB/MWh")
    pd.testing.assert_frame_equal(
        pd.read_csv(CANONICAL_INPUT / "plants.csv"),
        pd.read_csv(SOURCE_INPUT / "plants.csv"),
        check_dtype=False,
    )

    source_config = yaml.safe_load((SOURCE_INPUT / "config.yaml").read_text(encoding="utf-8"))
    source_case = source_config["cases"]["building_v2g_sweep_current"]
    canonical_config = yaml.safe_load((CANONICAL_INPUT / "config.yaml").read_text(encoding="utf-8"))
    canonical_case = canonical_config["cases"]["building_v2g_export_onset_3150"]
    for field in ("country", "timestep_minutes", "simulation_start", "simulation_end", "timezone", "additional_charges", "strategy", "market_sequence"):
        if canonical_case[field] != source_case[field]:
            raise ValueError(f"Canonical case differs from the archived run in {field}")
    if canonical_case["markets"]["day_ahead"]["signals"]["price"] != "electricity_import_price_thb_per_mwh":
        raise ValueError("Canonical import-price signal differs from archived run")
    if canonical_case["markets"]["day_ahead"]["signals"]["export_price"] != canonical_price_column:
        raise ValueError("Canonical export-price signal is not the dedicated 3.15 THB/kWh column")

    missing = [name for name in OUTPUT_FILES if not (SOURCE_OUTPUT / name).exists()]
    if missing:
        raise FileNotFoundError("Archived completed result is missing: " + ", ".join(missing))
    CANONICAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    existing = [path for path in CANONICAL_OUTPUT.iterdir() if path.is_file()]
    if existing:
        raise FileExistsError("Refusing to overwrite canonical output files")
    for name in OUTPUT_FILES:
        shutil.copy2(SOURCE_OUTPUT / name, CANONICAL_OUTPUT / name)

    (CANONICAL_OUTPUT / "provenance.md").write_text(
        "# I02b 3.15 THB/kWh export-onset output provenance\n\n"
        "This canonical output promotes the completed `i02_threshold_3150` annual run. "
        "Before copying, `scripts/promote_i02_export_onset_result.py` verified identical "
        "plants, the same 2024 forecast values, a constant 3150 THB/MWh export-price signal, "
        "and the same dispatch strategy, 48-hour horizon, 24-hour commit step, and 5 kW export limit.\n",
        encoding="utf-8",
    )
    print("Promoted and validated I02b 3.15 THB/kWh output.")


if __name__ == "__main__":
    main()
