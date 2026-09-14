#!/usr/bin/env python3
"""Run all configured electric-bus building use cases concurrently."""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "data" / "input"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "output" / "building_use_case_analysis"


@dataclass(frozen=True)
class Job:
    label: str
    input_dir: Path
    study_case: str
    output_dir: Path


def annual_jobs(output_root: Path) -> list[Job]:
    specs = [
        ("V1G baseline", "building_v1g_annual", "building_v1g_annual", "v1g_baseline"),
        ("I02 least-cost V2G export", "building_v2b_cost_annual", "building_v2b_cost_annual", "v2b_cost"),
        ("I03 regional high-load import relief without export", "building_v2b_grid_support_annual", "building_v2b_grid_support_max_no_export", "grid_support_no_export"),
        ("I04 regional high-load import relief at 5 kW", "building_v2b_grid_support_annual", "building_v2g_grid_support_max_export_5kw", "grid_support_export_5kw"),
        ("I04 regional high-load import relief at 450 kW", "building_v2b_grid_support_annual", "building_v2g_grid_support_max_export_450kw", "grid_support_export_450kw"),
        ("I05 renewable-deficit support no export", "building_v2b_renewable_alignment_annual", "building_v2b_renewable_shifting_max_no_export", "renewable_shifting_no_export"),
        ("I06 renewable-deficit support with 5 kW policy export limit", "building_v2b_renewable_alignment_annual", "building_v2g_renewable_shifting_max_export_5kw", "renewable_shifting_export_5kw"),
        ("I06 renewable-deficit support with 450 kW technical export sensitivity", "building_v2b_renewable_alignment_annual", "building_v2g_renewable_shifting_max_export_450kw", "renewable_shifting_export_450kw"),
        ("I07b PV-assisted V2G current policy", "building_v2b_pv_self_consumption_annual", "building_v2b_pv_self_consumption", "v2g_pv_current_policy"),
        ("I07c PV-assisted V2G viable export tariff", "building_v2b_pv_self_consumption_annual", "building_v2g_pv_export_viable_tariff", "v2g_pv_viable_tariff"),
        ("I07d PV-assisted V2G maximum technical export", "building_v2b_pv_self_consumption_annual", "building_v2g_pv_max_technical_export", "v2g_pv_max_technical_export"),
        ("I07e PV-assisted V2G regional high-load support", "building_v2b_pv_self_consumption_annual", "building_v2g_pv_grid_stress_policy_export", "v2g_pv_grid_stress_policy_export"),
        ("I08a-P emergency V2G 4 h prepared SOC", "building_v2g_emergency_backup_4h_prepared", "building_v2g_emergency_backup_4h_prepared", "i08_4h_prepared"),
        ("I08a-B emergency V2G 4 h inherited SOC", "building_v2g_emergency_backup_4h_baseline", "building_v2g_emergency_backup_4h_baseline", "i08_4h_baseline"),
        ("I08b-P emergency V2G 8 h prepared SOC", "building_v2g_emergency_backup_8h_prepared", "building_v2g_emergency_backup_8h_prepared", "i08_8h_prepared"),
        ("I08b-B emergency V2G 8 h inherited SOC", "building_v2g_emergency_backup_8h_baseline", "building_v2g_emergency_backup_8h_baseline", "i08_8h_baseline"),
        ("I02 current tariff duplicate", "building_v2g_current_tariff_annual", "building_v2g_current_tariff_annual", "v2g_current_export"),
    ]
    jobs = [
        Job(label, INPUT_ROOT / case_dir, study_case, output_root / output_name)
        for label, case_dir, study_case, output_name in specs
    ]
    tariff_dir = INPUT_ROOT / "building_v2g_tariff_sweep_annual"
    config = yaml.safe_load((tariff_dir / "config.yaml").read_text(encoding="utf-8"))
    for study_case in config["cases"]:
        increment = "0" if study_case.endswith("_current") else study_case.rsplit("_", 1)[-1]
        jobs.append(
            Job(
                f"I10 tariff sweep {study_case}",
                tariff_dir,
                study_case,
                output_root / f"tariff_plus_{increment}",
            )
        )
    return jobs


def diagnostic_jobs(output_root: Path) -> list[Job]:
    jobs = [
        Job("diagnostic V1G", INPUT_ROOT / "building_v1g_baseline", "building_v1g_baseline", output_root / "diagnostics" / "building_v1g_baseline"),
        Job("diagnostic no-export V2X", INPUT_ROOT / "building_v2b_no_export", "building_v2b_no_export", output_root / "diagnostics" / "building_v2b_no_export"),
    ]
    example_dir = INPUT_ROOT / "building_v2g_example"
    config = yaml.safe_load((example_dir / "config.yaml").read_text(encoding="utf-8"))
    for study_case in config["cases"]:
        jobs.append(
            Job(
                f"diagnostic {study_case}",
                example_dir,
                study_case,
                output_root / "diagnostics" / f"building_v2g_example__{study_case}",
            )
        )
    return jobs


def run_job(job: Job, log_root: Path) -> tuple[str, int]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{job.output_dir.name}.log"
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "flexi_mod.simulation.run_case",
        "--case",
        str(job.input_dir),
        "--study-case",
        job.study_case,
        "--output-dir",
        str(job.output_dir),
        "--no-plots",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return job.label, result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = annual_jobs(output_root) + diagnostic_jobs(output_root)
    print(f"Launching {len(jobs)} jobs with {args.workers} workers")
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job, log_root): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                label, returncode = future.result()
            except Exception as exc:  # pragma: no cover - operational wrapper
                print(f"FAILED {job.label}: {exc}", flush=True)
                failures.append(job.label)
                continue
            if returncode:
                print(f"FAILED {label} (exit {returncode})", flush=True)
                failures.append(label)
            else:
                print(f"DONE {label}", flush=True)
    print(f"Completed {len(jobs) - len(failures)}/{len(jobs)} jobs")
    if failures:
        print("Failures:")
        print("\n".join(f"- {label}" for label in failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
