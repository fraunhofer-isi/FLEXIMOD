# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in [SRC_DIR, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_case import available_examples, resolve_example_paths

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.visualisation.plots import create_all_plots_from_output


def main() -> None:
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
    args = parser.parse_args()

    case_dir = _resolve_case_dir(args.case, args.example)
    config = CaseConfig.from_case_dir(case_dir)
    output_dir = config.project_root / "data" / "output" / config.case_name
    plot_dir = output_dir / "plots"

    created = create_all_plots_from_output(
        output_dir=output_dir,
        file_format=args.format,
        show=args.show,
        sample_day=args.sample_day,
    )
    print(f"Case: {config.case_name}")
    print(f"Output directory: {output_dir}")
    print(f"Created {len(created)} plot file(s) in {plot_dir}")
    for path in created:
        print(f"  {path}")


def _resolve_case_dir(case: str | None, example: str) -> Path:
    if case:
        return Path(case).resolve()
    return resolve_example_paths(example)["case_dir"]


if __name__ == "__main__":
    main()
