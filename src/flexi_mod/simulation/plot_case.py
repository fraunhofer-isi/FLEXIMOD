# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command-line entry point for plotting FLEXIMOD case outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    from flexi_mod.config.case_config import CaseConfig
    from flexi_mod.simulation.cli_logging import CliLogger
    from flexi_mod.simulation.run_case import available_examples
    from flexi_mod.visualisation.analytics import RESULT_FILES
    from flexi_mod.visualisation.plots import create_all_plots_from_output

    parser = argparse.ArgumentParser(description="Create FlexIMOD report plots for one case.")
    parser.add_argument(
        "--case",
        help="Path to a case input folder containing config.yaml.",
    )
    parser.add_argument(
        "--example",
        default="hybrid_etes_de",
        choices=sorted(available_examples),
        help="Named example used when --case is not provided.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "both"],
        help="Figure output format.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    parser.add_argument(
        "--sample-day",
        help="Optional sample day for the detailed operation plot, e.g. 2025-01-03.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed plot file paths.",
    )
    args = parser.parse_args()
    logger = CliLogger(verbose=args.verbose)

    case_dir = _resolve_case_dir(args.case, args.example)
    config = CaseConfig.from_case_dir(case_dir)
    output_dir = config.project_root / "data" / "output" / config.case_name
    plot_dir = output_dir / "plots"

    logger.info(f"Plot creation started for case {config.case_name}")
    logger.info(f"Output folder: {output_dir}")
    logger.info(f"Result tables found: {_available_result_tables(output_dir, RESULT_FILES)}")
    with logger.capture_warnings():
        created = create_all_plots_from_output(
            output_dir=output_dir,
            file_format=args.format,
            show=args.show,
            sample_day=args.sample_day,
        )
    logger.success(f"Plots created: {len(created)} file(s).")
    logger.info(f"Plot folder: {plot_dir}")
    if args.verbose:
        logger.detail("Created plot files:")
        for path in created:
            logger.detail(f"  {path}")


def _resolve_case_dir(case: str | None, example: str) -> Path:
    from flexi_mod.simulation.run_case import resolve_example_paths

    if case:
        return Path(case).resolve()
    return resolve_example_paths(example)["case_dir"]


def _available_result_tables(output_dir: Path, result_files: dict[str, str]) -> str:
    names = [
        name
        for name, filename in result_files.items()
        if name != "summary_indicators" and (output_dir / filename).exists()
    ]
    if (output_dir / result_files["summary_indicators"]).exists():
        names.append("summary_indicators")
    return ", ".join(names) if names else "none"


if __name__ == "__main__":
    main()
