# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rerun every generated steel case and refresh compressed result artifacts.

The normal CLI writes uncompressed CSV files below ``data/output``.  The steel
analysis notebooks instead consume ``data/output/steel_results/<case>/`` where
each artifact is Zstandard-compressed.  This driver keeps the two conventions
consistent without exposing half-written output: a case runs in a unique staging
directory, its CSV artifacts are compressed, and each ``.zst`` result replaces
the prior artifact atomically only after the simulation succeeds.

The generated catalogue contains 432 cases: six scenario families, four years,
and eighteen technology routes.  It intentionally excludes the
``steel_plant_DE`` template input.

Examples:

    .venv/bin/python scripts/run_steel_cases_to_results.py --workers 4
    .venv/bin/python scripts/run_steel_cases_to_results.py \\
        --case aktuellepolitiken_2030_bf_bof_coal_external
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import zstandard

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

INPUT_ROOT = REPO_ROOT / "data" / "input" / "steel_inputs"
RESULTS_ROOT = REPO_ROOT / "data" / "output" / "steel_results"
STAGING_ROOT = REPO_ROOT / "data" / "output" / ".steel_results_staging"


@dataclass(frozen=True)
class CaseRunResult:
    """Serializable result returned by one isolated simulation worker."""

    case_name: str
    elapsed_seconds: float
    artifacts: tuple[str, ...]


def _configure_worker_environment() -> None:
    """Avoid thread-pool oversubscription across case-level workers."""

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["FLEXIMOD_HIGHS_THREADS"] = "1"


def _compress_and_publish(staging_dir: Path, result_dir: Path) -> tuple[str, ...]:
    """Atomically replace compressed artifacts for one completed simulation."""

    raw_artifacts = sorted(staging_dir.glob("*.csv"))
    if not raw_artifacts:
        raise RuntimeError(f"Simulation produced no CSV artifacts in {staging_dir}")

    result_dir.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=3, threads=1)
    published: list[str] = []
    for raw_path in raw_artifacts:
        result_path = result_dir / f"{raw_path.name}.zst"
        temporary_path = result_path.with_suffix(result_path.suffix + f".{uuid.uuid4().hex}.tmp")
        try:
            with raw_path.open("rb") as source_handle, temporary_path.open("wb") as target_handle:
                compressor.copy_stream(source_handle, target_handle)
            os.replace(temporary_path, result_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        published.append(result_path.name)

    expected = set(published)
    stale = sorted(path for path in result_dir.glob("*.csv.zst") if path.name not in expected)
    for path in stale:
        path.unlink()
    return tuple(published)


def _run_case(
    case_name: str,
    results_root: str,
    staging_root: str,
    run_token: str,
    assumed_grid_tier: str,
) -> CaseRunResult:
    """Run, compress, and publish one case in a dedicated process."""

    _configure_worker_environment()
    from flexi_mod.simulation.simulation_runner import OutputOptions, SimulationRunner

    case_dir = INPUT_ROOT / case_name
    staging_dir = Path(staging_root) / run_token / case_name
    output_options = OutputOptions(
        save_dispatch_results=True,
        save_market_ledger=True,
        save_storage_cost_ledger=True,
        save_summary_indicators=True,
        create_plots=False,
    )
    started = time.monotonic()
    with warnings.catch_warnings():
        # Missing aFRR activation values are deliberately represented as zero
        # activation by the market layer and are recorded in each quality summary.
        warnings.simplefilter("ignore")
        SimulationRunner(
            case_dir=case_dir,
            input_dir=case_dir,
            output_dir=staging_dir,
            study_case=case_name,
            output_options=output_options,
            assumed_grid_tier=assumed_grid_tier,
        ).run()
    result_dir = Path(results_root) / _output_folder_name(case_name)
    artifacts = _compress_and_publish(staging_dir, result_dir)
    shutil.rmtree(staging_dir)
    return CaseRunResult(
        case_name=case_name,
        elapsed_seconds=time.monotonic() - started,
        artifacts=artifacts,
    )


def _output_folder_name(case_name: str) -> str:
    """Match CaseConfig.output_folder_name without reloading configuration in parent code."""

    if "hybrid_hydrogen_natural_gas_electrolyser" in case_name:
        strategy = "electrified_steel"
    else:
        strategy = "electrified_steel_rule_based"
    return f"{case_name}_{strategy}"


def _available_cases() -> tuple[str, ...]:
    from flexi_mod.simulation.run_case import GENERATED_EXAMPLE_NAMES

    missing = [name for name in GENERATED_EXAMPLE_NAMES if not (INPUT_ROOT / name).is_dir()]
    if missing:
        raise FileNotFoundError("Generated steel input directory is missing: " + ", ".join(missing))
    return GENERATED_EXAMPLE_NAMES


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only this generated steel case. Repeat to select several cases.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=4,
        help="Independent simulation processes to run concurrently (default: 4).",
    )
    parser.add_argument(
        "--assumed-grid-tier",
        choices=["high", "low"],
        default="high",
        help="Grid-fee full-load-hour tier passed to every case (default: high).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root directory for compressed per-case results.",
    )
    args = parser.parse_args()

    available = _available_cases()
    selected = tuple(args.cases) if args.cases else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        parser.error("Unknown generated steel case(s): " + ", ".join(unknown))

    run_token = time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"
    worker_count = min(args.workers, len(selected))
    print(
        f"Starting {len(selected)} steel simulation(s) with {worker_count} worker(s); "
        f"results: {args.results_root}",
        flush=True,
    )
    failures: list[tuple[str, str]] = []
    completed = 0
    run_started = time.monotonic()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_case,
                case_name,
                str(args.results_root),
                str(STAGING_ROOT),
                run_token,
                args.assumed_grid_tier,
            ): case_name
            for case_name in selected
        }
        for future in as_completed(futures):
            case_name = futures[future]
            completed += 1
            try:
                result = future.result()
            except Exception:
                failures.append((case_name, traceback.format_exc()))
                print(f"[{completed}/{len(selected)}] FAILED {case_name}", flush=True)
            else:
                print(
                    f"[{completed}/{len(selected)}] Completed {result.case_name} "
                    f"({_format_duration(result.elapsed_seconds)}; "
                    f"{len(result.artifacts)} artifact(s))",
                    flush=True,
                )

    if failures:
        print(f"Completed with {len(failures)} failed case(s):", file=sys.stderr)
        for case_name, error in failures:
            print(f"\n--- {case_name} ---\n{error}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"All {len(selected)} steel simulations completed in "
        f"{_format_duration(time.monotonic() - run_started)}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
