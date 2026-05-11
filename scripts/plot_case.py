# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in [SRC_DIR, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_case import available_examples, resolve_example_paths

from etes_market_model.visualisation.plots import create_case_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Create plots for a FLEXIMOD case.")
    parser.add_argument(
        "--example",
        default="hybrid_etes_de",
        choices=sorted(available_examples),
        help="Named example from the runner registry.",
    )
    parser.add_argument(
        "--case",
        help="Optional direct path to an input directory containing config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to data/output/<scenario>.",
    )
    args = parser.parse_args()

    if args.case:
        case_dir = Path(args.case).resolve()
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else PROJECT_ROOT / "data" / "output" / case_dir.name
        )
    else:
        paths = resolve_example_paths(args.example)
        output_dir = Path(args.output_dir).resolve() if args.output_dir else paths["output_dir"]

    dispatch_path = output_dir / "dispatch_results.csv"
    summary_path = output_dir / "summary_indicators.csv"
    dispatch = pd.read_csv(dispatch_path, parse_dates=["datetime"]).set_index("datetime")
    summary = pd.read_csv(summary_path)
    paths = create_case_plots(dispatch, summary, output_dir)
    print(f"Created {len(paths)} plot files in {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
