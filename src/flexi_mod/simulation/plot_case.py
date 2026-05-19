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
        "--study-case",
        "--case-name",
        dest="study_case",
        help="Study-case key inside config.yaml cases: mapping.",
    )
    parser.add_argument(
        "--example",
        default="hybrid_ETES_ID_buy",
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
        "--output-dir",
        help="Optional output directory. Defaults to data/output/<case_name>_<strategy_name>.",
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

    paths = _resolve_case_paths(args.case, args.example, args.study_case)
    case_dir = paths["case_dir"]
    config = CaseConfig.from_case_dir(case_dir, study_case=paths["study_case"])
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else PROJECT_ROOT / "data" / "output" / config.output_folder_name
    )
    plot_dir = output_dir / "plots"

    logger.info(f"Plot creation started for case {config.case_name}")
    logger.info(f"Study case: {config.study_case}")
    logger.info(f"Strategy: {config.strategy_name}")
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


def _resolve_case_paths(
    case: str | None,
    example: str,
    study_case: str | None,
) -> dict[str, Path | str | None]:
    from flexi_mod.simulation.run_case import resolve_example_paths

    if case:
        case_dir = Path(case).resolve()
        return {
            "case_dir": case_dir,
            "study_case": study_case,
        }
    paths = resolve_example_paths(example)
    if study_case:
        paths["study_case"] = study_case
    return {
        "case_dir": paths["case_dir"],
        "study_case": paths["study_case"],
    }


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
