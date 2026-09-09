#!/usr/bin/env python3
"""Allocate 15-min electricity demand (aviation/maritime) to day-ahead vs aFRR
energy markets, based on which market is cheaper at each timestep.

This script is standalone (not part of the flexi_mod package). It reads:
  - data/input/archive/electricity_15min/*.csv.gz
      15-min electricity demand per location, per scenario, spanning 2028-2050.
  - data/input/cement_inputs/<scenario>_<year>_R1/forecasts_df.csv
      National DE day-ahead (DE_DA_price) and aFRR energy down
      (aFRR_energy_down_price) price series at 15-min resolution. Prices are
      identical across cement regions / steel technologies for a given
      scenario+year, so any single region ("R1") is a valid representative
      source.

For each demand timestep, the entire demand value at that timestep is
allocated to whichever market (day-ahead vs aFRR energy) has the lower price
at that same timestep (no load-shifting across time; ties go to day-ahead).

Only scenarios that have a matching price scenario, and only years for which
cement/steel price inputs exist (2030, 2035, 2040, 2045), are processed. The
two worst-case risk-averse demand scenarios have no price analogue and are
dropped, as are all other years.

Output is written to data/input/archive/electricity_15min_price_allocated/,
mirroring the input filenames, with each original location column split into
two columns: "<location> -- day_ahead" and "<location> -- afrr_energy".
"""
from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "data" / "input" / "archive" / "electricity_15min"
OUTPUT_DIR = REPO_ROOT / "data" / "input" / "archive" / "electricity_15min_price_allocated"
CEMENT_INPUTS_DIR = REPO_ROOT / "data" / "input" / "cement_inputs"

PRICE_YEARS = [2030, 2035, 2040, 2045]

# Demand-file scenario labels -> price-case scenario prefixes.
SCENARIO_MAP = {
    "aktuellepolitiken-risk-neutral": "aktuellepolitiken",
    "fokush2-risk-neutral": "fokusH2",
    "fokusstrom-risk-neutral": "fokusstrom",
    "hohenachfrage-risk-neutral": "hohenachfrage",
    "nachfrageniedrig-risk-neutral": "niedrigenachfrage",
    "technologiemix-risk-neutral": "technologiemix",
    # worstcasemean-risk-averse / worstcasetailrisk-risk-averse: no price
    # analogue exists, so they are intentionally omitted here.
}

DAY_AHEAD_SUFFIX = " -- day_ahead"
AFRR_SUFFIX = " -- afrr_energy"


def load_price_series(scenario_prefix: str, year: int) -> pd.DataFrame:
    """Load DE_DA_price and aFRR_energy_down_price for one scenario+year.

    Prices are national and identical across cement regions for a given
    scenario+year, so the first available region directory is used.
    """
    candidates = sorted(CEMENT_INPUTS_DIR.glob(f"{scenario_prefix}_{year}_R*"))
    if not candidates:
        raise FileNotFoundError(f"No cement input case found for {scenario_prefix}_{year}_R*")
    forecasts_path = candidates[0] / "forecasts_df.csv"
    prices = pd.read_csv(
        forecasts_path,
        usecols=["datetime", "DE_DA_price", "aFRR_energy_down_price"],
        parse_dates=["datetime"],
    )
    prices = prices.rename(columns={"datetime": "timestamp"})
    return prices.set_index("timestamp")[["DE_DA_price", "aFRR_energy_down_price"]]


def build_price_cache() -> dict[tuple[str, int], pd.DataFrame]:
    cache: dict[tuple[str, int], pd.DataFrame] = {}
    for scenario_prefix in set(SCENARIO_MAP.values()):
        for year in PRICE_YEARS:
            cache[(scenario_prefix, year)] = load_price_series(scenario_prefix, year)
    return cache


def allocate_file(input_path: Path, output_path: Path, price_cache: dict) -> None:
    df = pd.read_csv(input_path, compression="gzip")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year

    location_cols = [c for c in df.columns if c not in ("scenario", "timestamp", "year")]

    df = df[df["scenario"].isin(SCENARIO_MAP) & df["year"].isin(PRICE_YEARS)]
    if df.empty:
        print(f"  (no matching scenario/year rows in {input_path.name}, skipping)")
        return

    allocated_chunks = []
    for (demand_scenario, year), group in df.groupby(["scenario", "year"], sort=False):
        scenario_prefix = SCENARIO_MAP[demand_scenario]
        prices = price_cache[(scenario_prefix, year)]

        group = group.set_index("timestamp")
        merged = group.join(prices, how="left")

        # Missing price (e.g. Dec 31 in leap years, which price inputs omit)
        # defaults to day-ahead, since that market is always assumed available.
        da_cheaper = (
            merged["DE_DA_price"].isna()
            | (merged["DE_DA_price"] <= merged["aFRR_energy_down_price"])
        )

        out = pd.DataFrame(index=merged.index)
        out["scenario"] = merged["scenario"]
        for loc in location_cols:
            demand = merged[loc]
            out[loc + DAY_AHEAD_SUFFIX] = demand.where(da_cheaper, 0.0)
            out[loc + AFRR_SUFFIX] = demand.where(~da_cheaper, 0.0)

        allocated_chunks.append(out.reset_index())

    result = pd.concat(allocated_chunks, ignore_index=True)
    result = result.sort_values(["scenario", "timestamp"]).reset_index(drop=True)

    ordered_cols = ["scenario", "timestamp"]
    for loc in location_cols:
        ordered_cols.append(loc + DAY_AHEAD_SUFFIX)
        ordered_cols.append(loc + AFRR_SUFFIX)
    result = result[ordered_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", newline="") as f:
        result.to_csv(f, index=False)


def main() -> None:
    print("Building price cache from cement_inputs (DE_DA_price / aFRR_energy_down_price)...")
    price_cache = build_price_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(INPUT_DIR.glob("*.csv.gz")):
        output_path = OUTPUT_DIR / input_path.name
        print(f"Allocating {input_path.name} -> {output_path.relative_to(REPO_ROOT)}")
        allocate_file(input_path, output_path, price_cache)

        license_path = input_path.with_suffix(input_path.suffix + ".license")
        if license_path.exists():
            shutil.copy(license_path, output_path.with_suffix(output_path.suffix + ".license"))

    print(f"Done. Output written to {OUTPUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
