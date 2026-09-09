# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command-line entry point for running configured FLEXIMOD cases."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

#: One small file per live worker, overwritten (not appended) on every progress
#: update, so a separate dashboard process can render an in-place live table
#: instead of everyone tailing an ever-scrolling shared log.
WORKER_STATUS_DIR = PROJECT_ROOT / "data" / "output" / ".worker_status"

# Explicit catalogue of the generated input folders. These names are expanded into
# ``available_examples`` below rather than discovered from the file system, so the
# runner registry remains visible and reproducible in version control.
_GENERATED_STUDY_FAMILIES = (
    "aktuellepolitiken",
    "fokusH2",
    "fokusstrom",
    "hohenachfrage",
    "niedrigenachfrage",
    "technologiemix",
)
_GENERATED_YEARS = (
    "2030",
    "2035",
    "2040",
    "2045",
)

_GENERATED_ROUTE_VARIANTS = (
    "bf_bof_coal_external",
    "bf_bof_hybrid_hydrogen_natural_gas_electrolyser",
    "bf_bof_hybrid_hydrogen_natural_gas_external",
    "bf_bof_hydrogen_electrolyser",
    "bf_bof_hydrogen_external",
    "bf_bof_natural_gas_external",
    "dri_bof_coal_external",
    "dri_bof_hybrid_hydrogen_natural_gas_electrolyser",
    "dri_bof_hybrid_hydrogen_natural_gas_external",
    "dri_bof_hydrogen_electrolyser",
    "dri_bof_hydrogen_external",
    "dri_bof_natural_gas_external",
    "dri_eaf_coal_external",
    "dri_eaf_hybrid_hydrogen_natural_gas_electrolyser",
    "dri_eaf_hybrid_hydrogen_natural_gas_external",
    "dri_eaf_hydrogen_electrolyser",
    "dri_eaf_hydrogen_external",
    "dri_eaf_natural_gas_external",
)
GENERATED_EXAMPLE_NAMES = tuple(
    f"{family}_{year}_{variant}"
    for family in _GENERATED_STUDY_FAMILIES
    for year in _GENERATED_YEARS
    for variant in _GENERATED_ROUTE_VARIANTS
)

# Cement plant cases: same family/year axes as steel, but each route is an
# anonymised code (R1..R16, with a/b sub-variants) rather than a descriptive
# technology name, and every case uses the single "electrified_cement"
# strategy regardless of route.
_GENERATED_CEMENT_STUDY_FAMILIES = (
    "aktuellepolitiken",
    "fokusH2",
    "fokusstrom",
    "hohenachfrage",
    "niedrigenachfrage",
    "technologiemix",
)
_GENERATED_CEMENT_ROUTE_VARIANTS = (
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "R8",
    "R9",
    "R10",
    "R11a",
    "R11b",
    "R12a",
    "R12b",
    "R13",
    "R14",
    "R15a",
    "R15b",
    "R16",
)
GENERATED_CEMENT_EXAMPLE_NAMES = tuple(
    f"{family}_{year}_{variant}"
    for family in _GENERATED_CEMENT_STUDY_FAMILIES
    for year in _GENERATED_YEARS
    for variant in _GENERATED_CEMENT_ROUTE_VARIANTS
)


available_examples: dict[str, dict[str, str]] = {
    **{
        name: {
            "scenario": name,
            "study_case": name,
        }
        for name in GENERATED_EXAMPLE_NAMES
    },
    **{
        name: {
            "scenario": name,
            "study_case": name,
        }
        for name in GENERATED_CEMENT_EXAMPLE_NAMES
    },
}


# Select the example to run when ``examples_to_run`` is empty.
example = "aktuellepolitiken_2030_dri_eaf_hybrid_hydrogen_natural_gas_external"

# Run every currently uncommented generated case. Order is preserved.
# Add names here if another case should be skipped temporarily.
excluded_examples_from_run: set[str] = set()
examples_to_run: list[str] = [
    name
    for name in (*GENERATED_EXAMPLE_NAMES, *GENERATED_CEMENT_EXAMPLE_NAMES)
    if name not in excluded_examples_from_run
]


def selected_examples() -> tuple[str, ...]:
    """Validate and return the user-selected sequence of examples to run."""
    unknown = [name for name in examples_to_run if name not in available_examples]
    if unknown:
        options = ", ".join(sorted(available_examples))
        raise ValueError(
            f"Unknown selected example(s): {', '.join(unknown)}. Available examples: {options}"
        )
    return tuple(examples_to_run)


def resolve_example_paths(example: str) -> dict[str, Path | str]:
    if example not in available_examples:
        options = ", ".join(sorted(available_examples))
        raise ValueError(f"Unknown example '{example}'. Available examples: {options}")

    settings = available_examples[example]
    scenario = settings["scenario"]
    study_case = settings["study_case"]

    input_dir = PROJECT_ROOT / "data" / "input" / scenario
    if not input_dir.exists():
        input_dir = PROJECT_ROOT / "data" / "input" / "cement_inputs" / scenario
    if not input_dir.exists():
        input_dir = PROJECT_ROOT / "data" / "input" / "steel_inputs" / scenario

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory for example '{example}' does not exist: {input_dir}"
        )

    return {
        "case_dir": input_dir,
        "input_dir": input_dir,
        "study_case": study_case,
    }


def build_runner_settings(
    args: argparse.Namespace,
    selected_example: str | None = None,
) -> dict[str, Any]:
    from flexi_mod.simulation.simulation_runner import OutputOptions

    if args.case:
        case_dir = Path(args.case).resolve()
        paths = {
            "case_dir": case_dir,
            "input_dir": case_dir,
            "study_case": args.study_case,
        }
    else:
        paths = resolve_example_paths(selected_example or args.example)
        if args.study_case:
            paths["study_case"] = args.study_case

    if args.output_dir:
        paths["output_dir"] = Path(args.output_dir).resolve()
    else:
        paths["output_dir"] = None

    defaults = _default_output_options()
    output_options = OutputOptions(
        save_dispatch_results=defaults.save_dispatch_results and not args.skip_dispatch_results,
        save_market_ledger=defaults.save_market_ledger and not args.skip_market_ledger,
        save_storage_cost_ledger=defaults.save_storage_cost_ledger
        and not args.skip_storage_cost_ledger,
        save_summary_indicators=defaults.save_summary_indicators
        and not args.skip_summary_indicators,
        create_plots=defaults.create_plots and not args.no_plots,
    )

    return {
        **paths,
        "plants_file": args.plants_file,
        "forecasts_file": args.forecasts_file,
        "output_options": output_options,
        "assumed_grid_tier": args.assumed_grid_tier,
    }


def _default_output_options() -> Any:
    from flexi_mod.simulation.simulation_runner import OutputOptions

    return OutputOptions(
        save_dispatch_results=True,
        save_market_ledger=True,
        save_storage_cost_ledger=True,
        save_summary_indicators=True,
        create_plots=True,
    )


def main() -> None:
    from flexi_mod.simulation.cli_logging import CliLogger

    parser = argparse.ArgumentParser(description="Run a FLEXIMOD case.")
    parser.add_argument(
        "--example",
        default=example,
        choices=sorted(available_examples),
        help="Named example from the runner registry. Defaults to the module-level example.",
    )
    parser.add_argument(
        "--case",
        help="Optional direct path to an input directory containing config.yaml.",
    )
    parser.add_argument(
        "--study-case",
        "--case-name",
        dest="study_case",
        help="Study-case key inside config.yaml cases: mapping.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to data/output/<case_name>_<strategy_name>.",
    )
    parser.add_argument("--plants-file", default="plants.csv")
    parser.add_argument("--forecasts-file", default="forecasts_df.csv")
    parser.add_argument(
        "--assumed-grid-tier",
        choices=["high", "low"],
        default="high",
        help=(
            "Full-load-hour tier assumed for the per-MWh grid energy charge in the dispatch "
            "strike price. Defaults to 'high' (option 1); use '--assumed-grid-tier low' to "
            "select the low tier. The bill is corrected ex-post if the realized tier differs."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-dispatch-results", action="store_true")
    parser.add_argument("--skip-market-ledger", action="store_true")
    parser.add_argument("--skip-storage-cost-ledger", action="store_true")
    parser.add_argument("--skip-summary-indicators", action="store_true")
    parser.add_argument(
        "--plant",
        choices=["all", "steel", "cement"],
        default="all",
        help=(
            "Restrict a batch run (examples_to_run) to one generated catalogue. "
            "Defaults to 'all' (steel + cement)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=os.cpu_count() or 1,
        help=(
            "Number of selected examples to run concurrently in separate processes. "
            "Defaults to the machine's CPU count; use --workers 1 for the original "
            "sequential, single-process execution."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed created file paths.",
    )
    args = parser.parse_args()
    logger = CliLogger(verbose=args.verbose)

    if args.case:
        _run_one_case(args, logger)
    elif examples_to_run:
        _run_selected_examples(args, logger)
    else:
        _run_one_case(args, logger)


def _run_selected_examples(args: argparse.Namespace, logger: Any) -> None:
    """Run the selected examples, filtered by --plant, and report failures at the end."""
    selected = selected_examples()
    if args.case or args.study_case or args.output_dir:
        raise ValueError(
            "examples_to_run cannot be combined with --case, --study-case, or --output-dir."
        )

    if args.plant == "steel":
        selected = tuple(name for name in selected if name in set(GENERATED_EXAMPLE_NAMES))
    elif args.plant == "cement":
        selected = tuple(
            name for name in selected if name in set(GENERATED_CEMENT_EXAMPLE_NAMES)
        )
    if not selected:
        logger.info(f"No selected examples match --plant {args.plant}.")
        return

    worker_count = min(args.workers, len(selected))
    if worker_count == 1:
        _run_selected_examples_sequential(args, logger, selected)
        return

    logger.info(
        f"Parallel run started: {len(selected)} selected example(s), "
        f"{worker_count} worker processes."
    )
    failures: list[tuple[str, Exception]] = []
    completed = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = {
            executor.submit(_run_example_worker, args, selected_example): selected_example
            for selected_example in selected
        }
        try:
            for future in as_completed(futures):
                selected_example = futures[future]
                completed += 1
                try:
                    elapsed_seconds = future.result()
                except Exception as exc:
                    failures.append((selected_example, exc))
                    logger.error(
                        f"[{completed}/{len(selected)}] Example '{selected_example}' "
                        f"failed: {exc}"
                    )
                else:
                    logger.success(
                        f"[{completed}/{len(selected)}] Case completed: {selected_example} "
                        f"({_format_duration(elapsed_seconds)})"
                    )
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise

    if failures:
        failed_names = ", ".join(name for name, _ in failures)
        raise RuntimeError(
            f"Parallel run completed with {len(failures)} failed example(s): {failed_names}"
        )
    logger.success(f"Parallel run completed: {len(selected)} examples succeeded.")


def _run_selected_examples_sequential(
    args: argparse.Namespace,
    logger: Any,
    selected: tuple[str, ...],
) -> None:
    """Keep the original in-process execution available for debugging."""

    logger.info(f"Sequential run started: {len(selected)} selected example(s).")
    failures: list[tuple[str, Exception]] = []
    for number, selected_example in enumerate(selected, start=1):
        logger.info(f"\nSelected example {number}/{len(selected)}: {selected_example}")
        try:
            _run_one_case(args, logger, selected_example=selected_example)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failures.append((selected_example, exc))
            logger.error(f"Example '{selected_example}' failed: {exc}")

    if failures:
        failed_names = ", ".join(name for name, _ in failures)
        raise RuntimeError(
            f"Sequential run completed with {len(failures)} failed example(s): {failed_names}"
        )
    logger.success(f"Sequential run completed: {len(selected)} examples succeeded.")


def _run_example_worker(args: argparse.Namespace, selected_example: str) -> float:
    """Run one example in an isolated process and return elapsed wall-clock seconds."""

    # Each process owns one solver. Prevent numerical libraries and HiGHS from
    # creating their own large thread pools on top of the case-level process pool.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["FLEXIMOD_HIGHS_THREADS"] = "1"

    from flexi_mod.simulation.cli_logging import CliLogger

    # "Simulating 2030-03-14 for P100..._R3 (140/11648 windows, 11508 remaining)"
    progress_pattern = re.compile(r"Simulating (\S+(?: to \S+)?) for (\S+) \((\d+)/(\d+) windows")

    class QuietWorkerLogger(CliLogger):
        """Suppress per-window log spam; report progress to a per-worker status file.

        With 25+ workers each processing thousands of windows (32 plants x ~365
        windows per case), printing every window would flood the shared log, and
        printing nothing (the original design) left the run invisible between
        "Started" and completion. Instead, each update overwrites this worker's own
        small status file, which `scripts/cement_dashboard.py` polls to render a
        live, in-place table -- no log scrolling required.
        """

        def __init__(self, verbose: bool = False, *, case_name: str = "", pid: int = 0) -> None:
            super().__init__(verbose=verbose)
            self._progress_count = 0
            self._case_name = case_name
            self._status_path = WORKER_STATUS_DIR / f"{pid}.status"

        def info(self, message: str) -> None:
            del message

        def detail(self, message: str) -> None:
            del message

        def notice(self, message: str) -> None:
            del message

        def success(self, message: str) -> None:
            del message

        def error(self, message: str) -> None:
            del message

        def progress(self, message: str) -> None:
            self._progress_count += 1
            match = progress_pattern.search(message)
            if match is not None:
                date_label, plant_name, current, total = match.groups()
                self._write_status(plant_name, current, total, date_label)
            if self._progress_count % 20 == 1:
                print(
                    f"[{time.strftime('%H:%M:%S')}] (pid {os.getpid()}) {message}",
                    flush=True,
                )

        def _write_status(self, plant_name: str, current: str, total: str, date_label: str) -> None:
            payload = (
                f"{self._case_name}\t{plant_name}\t{current}\t{total}\t"
                f"{date_label}\t{time.strftime('%H:%M:%S')}\n"
            )
            tmp_path = self._status_path.with_suffix(".tmp")
            tmp_path.write_text(payload)
            tmp_path.replace(self._status_path)

        @contextmanager
        def capture_warnings(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

    WORKER_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    logger = QuietWorkerLogger(verbose=False, case_name=selected_example, pid=pid)
    logger._write_status("-", "0", "?", "-")

    started = time.monotonic()
    print(f"[{time.strftime('%H:%M:%S')}] Started (pid {pid}): {selected_example}", flush=True)
    try:
        _run_one_case(args, logger, selected_example=selected_example)
    except SystemExit as exc:
        raise RuntimeError(f"worker exited with status {exc.code}") from exc
    return time.monotonic() - started


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


def _run_one_case(
    args: argparse.Namespace,
    logger: Any,
    selected_example: str | None = None,
) -> None:
    """Run one named or directly specified case using the normal CLI reporting."""
    from flexi_mod.config.case_config import CaseConfig
    from flexi_mod.simulation.cli_logging import output_summary, print_verbose_outputs
    from flexi_mod.simulation.simulation_runner import SimulationRunner

    settings = build_runner_settings(args, selected_example=selected_example)
    config = CaseConfig.from_case_dir(settings["case_dir"], study_case=settings["study_case"])
    if settings["output_dir"] is None:
        settings["output_dir"] = PROJECT_ROOT / "data" / "output" / config.output_folder_name
    logger.info(f"Case started: {config.case_name}")
    logger.info(f"Study case: {config.study_case}")
    logger.info(f"Strategy: {config.strategy_name}")
    logger.info(
        "Simulation: "
        f"{config.simulation_start} to {config.simulation_end}, "
        f"{config.timestep_minutes} min"
    )
    logger.info(f"Enabled markets: {_enabled_market_summary(config)}")
    logger.info(f"Solver: {config.solver_name}")
    logger.info(f"Output folder: {settings['output_dir']}")

    _report_additional_charges(logger, config, settings)
    _report_intraday_mode(logger, config)

    if settings["assumed_grid_tier"] is None:
        settings["assumed_grid_tier"] = _prompt_grid_tier(config, settings) or "high"
        # Reuse the selected tier for the remainder of a sequential run, so a
        # user selecting several tariffed cases is prompted only once.
        if selected_example is not None:
            args.assumed_grid_tier = settings["assumed_grid_tier"]

    runner = SimulationRunner(**settings, progress_callback=logger.progress)
    with logger.capture_warnings():
        outputs = runner.run()

    logger.success(f"Case completed: {output_summary(outputs)} saved.")
    print_verbose_outputs(logger, outputs)


def _prompt_grid_tier(config: Any, settings: dict[str, Any]) -> str | None:
    """Interactively ask which full-load-hour tier to assume, if the tariff is tiered."""
    from flexi_mod.data.data_loader import DataLoader
    from flexi_mod.regulations import build_grid_fee_regulation

    if not config.additional_charges_enabled:
        return None

    loader = DataLoader(
        config,
        input_dir=settings["input_dir"],
        plants_file=settings["plants_file"],
        forecasts_file=settings["forecasts_file"],
    )
    try:
        plants = loader.load_plants()
        charges = loader.load_additional_charges(plants)
    except Exception:
        return None

    if not charges:
        return None

    plant_charges = next(iter(charges.values()))
    try:
        reg = build_grid_fee_regulation(config.country, plant_charges)
    except Exception:
        return None

    options = reg.tier_prompt_options()
    if not options:
        return None

    print()
    print("Tiered grid energy charge detected in additional_charges.csv:")
    for i, opt in enumerate(options, start=1):
        print(f"  [{i}] {opt['key']:5s}  {opt['label']:15s}  {opt['rate']}")
    print()
    while True:
        raw = input(f"Which tier to assume for this run? [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            chosen = options[int(raw) - 1]
            print(f"Using tier: {chosen['key']} ({chosen['label']}, {chosen['rate']})")
            print()
            return chosen["key"]
        print(f"  Please enter a number between 1 and {len(options)}.")


def _report_additional_charges(
    logger: Any,
    config: Any,
    settings: dict[str, Any],
) -> None:
    from flexi_mod.data.data_loader import DataLoader
    from flexi_mod.simulation.cli_logging import (
        additional_charges_message,
        missing_additional_charges_message,
    )

    loader = DataLoader(
        config,
        input_dir=settings["input_dir"],
        plants_file=settings["plants_file"],
        forecasts_file=settings["forecasts_file"],
    )
    plants = loader.load_plants()
    try:
        charges = loader.load_additional_charges(plants)
    except FileNotFoundError as exc:
        if config.additional_charges_enabled:
            logger.error(missing_additional_charges_message(loader.additional_charges_path))
            raise SystemExit(1) from exc
        raise
    logger.info(additional_charges_message(config.additional_charges_enabled, charges))


def _report_intraday_mode(logger: Any, config: Any) -> None:
    if "intraday_continuous" not in config.market_sequence:
        return
    market = config.market("intraday_continuous")
    if not market.get("enabled", False):
        return
    allowed = market.get("allowed_actions", {})
    buy = bool(allowed.get("buy", True))
    sell = bool(allowed.get("sell", True))
    if buy and sell:
        mode = "buy and sell/reduction"
    elif buy:
        mode = "buy-only"
    elif sell:
        mode = "sell/reduction-only"
    else:
        mode = "observe-only"
    logger.info(f"Intraday mode: {mode}.")


def _enabled_market_summary(config: Any) -> str:
    enabled = [market for market in config.market_sequence if market in config.enabled_markets]
    return ", ".join(enabled) if enabled else "none"


if __name__ == "__main__":
    main()
