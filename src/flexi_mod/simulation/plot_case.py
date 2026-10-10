# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command-line entry point for building the interactive dashboard of case outputs."""

from __future__ import annotations

import argparse
import sys
import webbrowser
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


# Optional: set a case output folder here to build its dashboard when the script is run
# without arguments (for example with the editor's play button). If left as None, the most
# recently written result folder under data/output is used.
DEFAULT_OUTPUT_DIR: str | None = None


def main() -> None:
    from flexi_mod.simulation.cli_logging import CliLogger
    from flexi_mod.visualisation.dashboard.comparison import (
        COMPARISON_FILENAME,
        write_comparison_dashboard,
    )
    from flexi_mod.visualisation.dashboard.data import load_case
    from flexi_mod.visualisation.dashboard.static_html import (
        DASHBOARD_FILENAME,
        write_case_dashboard,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Build the interactive FLEXIMOD dashboard (a self-contained HTML file) from the "
            "output of one case, or a comparison page for many cases."
        )
    )
    parser.add_argument(
        "--output-dir",
        help="Case output folder with dispatch_results.csv. Defaults to the folder of --case.",
    )
    parser.add_argument(
        "--input-dir",
        help=(
            "Case input folder (plants.csv, additional_charges.csv) for the system-setup tab. "
            "Found automatically from the output folder name or --case when omitted."
        ),
    )
    parser.add_argument(
        "--case",
        help="Path to a case input folder containing config.yaml (locates the output folder).",
    )
    parser.add_argument(
        "--study-case",
        "--case-name",
        dest="study_case",
        help="Study-case key inside config.yaml cases: mapping.",
    )
    parser.add_argument("--example", help="Named example from the run_case registry.")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="FOLDER",
        help="Build a comparison page for the cases found in these folders instead.",
    )
    parser.add_argument(
        "--max-plants",
        type=int,
        default=12,
        help="Plants shown individually (largest electricity users); all are aggregated.",
    )
    parser.add_argument(
        "--plotly-js",
        choices=["embed", "cdn"],
        default="embed",
        help="Embed plotly.js (works offline, ~5 MB) or load it from the CDN (smaller file).",
    )
    parser.add_argument("--start", help="First day to show, e.g. 2025-01-01.")
    parser.add_argument("--end", help="Last day to show, e.g. 2025-01-31.")
    parser.add_argument(
        "--resolution",
        default="auto",
        choices=["auto", "native", "1h", "6h", "1D", "1W"],
        help="Time resolution of the charts. 'auto' keeps each chart readable.",
    )
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser.")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the browser when the script is run without arguments.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print the written file path.")
    no_arguments = len(sys.argv) == 1
    args = parser.parse_args()
    logger = CliLogger(verbose=args.verbose)
    if no_arguments:
        # Play-button use: pick a sensible default and show the result.
        args.output_dir = DEFAULT_OUTPUT_DIR or str(_latest_output_dir())
        args.open = True

    if args.compare:
        target = Path(args.output_dir) if args.output_dir else Path(args.compare[0])
        path = (target / COMPARISON_FILENAME) if target.is_dir() else target
        logger.info(f"Comparison started for {', '.join(args.compare)}")
        with logger.capture_warnings():
            written = write_comparison_dashboard(args.compare, path, plotly_js=args.plotly_js)
        logger.success(f"Comparison dashboard created: {written}")
    else:
        output_dir = _resolve_output_dir(args)
        logger.info(f"Dashboard creation started for {output_dir}")
        input_dir = args.input_dir or (args.case if args.case else None)
        if input_dir is None and args.example:
            from flexi_mod.simulation.run_case import resolve_example_paths

            input_dir = resolve_example_paths(args.example)["case_dir"]
        with logger.capture_warnings():
            case = load_case(output_dir, input_dir=input_dir)
            written = write_case_dashboard(
                case,
                output_dir / DASHBOARD_FILENAME,
                max_plants=args.max_plants,
                plotly_js=args.plotly_js,
                start=args.start,
                end=args.end,
                resolution=args.resolution,
            )
        logger.success(f"Dashboard created: {written}")

    if args.open and not args.no_open:
        webbrowser.open(written.resolve().as_uri())


def _latest_output_dir() -> Path:
    """Return the most recently written case output folder below data/output."""

    from flexi_mod.visualisation.dashboard.data import discover_cases

    folders = discover_cases(PROJECT_ROOT / "data" / "output")
    if not folders:
        raise SystemExit(
            "No case results found under data/output. Run a case first with run_case.py, "
            "or pass --output-dir."
        )

    def written(folder: Path) -> float:
        files = [*folder.glob("dispatch_results.csv*")]
        return max(file.stat().st_mtime for file in files)

    return max(folders, key=written)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()

    from flexi_mod.config.case_config import CaseConfig
    from flexi_mod.simulation.run_case import resolve_example_paths

    if args.case:
        case_dir, study_case = Path(args.case).resolve(), args.study_case
    elif args.example:
        paths = resolve_example_paths(args.example)
        case_dir = paths["case_dir"]
        study_case = args.study_case or paths["study_case"]
    else:
        raise SystemExit("Give --output-dir, --case or --example (or --compare FOLDER ...).")
    config = CaseConfig.from_case_dir(case_dir, study_case=study_case)
    return PROJECT_ROOT / "data" / "output" / config.output_folder_name


if __name__ == "__main__":
    main()
