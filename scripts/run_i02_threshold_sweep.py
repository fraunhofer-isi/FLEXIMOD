#!/usr/bin/env python3
"""Run targeted annual I02 export-price cases in parallel."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "input" / "building_v2g_tariff_sweep_annual"
OUTPUT = ROOT / "data" / "output" / "building_use_case_analysis"
TARGETS = {"3000": 3000.0, "3100": 3100.0, "3150": 3150.0}


def prepare_case(work: Path, increment: str, export_price: float) -> Path:
    case_dir = work / f"case_{increment}"
    case_dir.mkdir()
    for name in ("plants.csv", "forecasts_df.csv"):
        shutil.copy2(SOURCE / name, case_dir / name)

    config = yaml.safe_load((SOURCE / "config.yaml").read_text(encoding="utf-8"))
    base = dict(config["cases"]["building_v2g_sweep_current"])
    base["name"] = f"building_v2g_sweep_plus_{increment}"
    base["description"] = f"Targeted annual I02 export payment: {export_price / 1000:.3f} THB/kWh"
    base["markets"] = {
        "day_ahead": {
            "enabled": True,
            "product_resolution": "15min",
            "signals": {
                "price": "electricity_import_price_thb_per_mwh",
                "export_price": f"target_export_price_{increment}_thb_per_mwh",
            },
        }
    }
    config["cases"] = {base["name"]: base}
    (case_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    forecast_path = case_dir / "forecasts_df.csv"
    forecasts = pd.read_csv(forecast_path)
    forecasts[f"target_export_price_{increment}_thb_per_mwh"] = export_price
    forecasts.to_csv(forecast_path, index=False)
    return case_dir


def run_one(item: tuple[str, float], work: Path) -> tuple[str, int]:
    increment, export_price = item
    case_dir = prepare_case(work, increment, export_price)
    output_dir = OUTPUT / f"i02_threshold_{increment}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "logs" / f"i02_threshold_{increment}.log"
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m", "flexi_mod.simulation.run_case",
        "--case", str(case_dir),
        "--study-case", f"building_v2g_sweep_plus_{increment}",
        "--output-dir", str(output_dir),
        "--no-plots",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return increment, result.returncode


def main() -> None:
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fleximod_i02_threshold_") as temp:
        work = Path(temp)
        with ThreadPoolExecutor(max_workers=len(TARGETS)) as pool:
            futures = [pool.submit(run_one, item, work) for item in TARGETS.items()]
            for future in as_completed(futures):
                increment, code = future.result()
                print(f"{'DONE' if code == 0 else 'FAILED'} I02 target +{increment} THB/MWh", flush=True)
                if code:
                    raise SystemExit(code)


if __name__ == "__main__":
    main()
