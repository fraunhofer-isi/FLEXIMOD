# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flexi_mod.simulation.simulation_runner import OutputOptions, SimulationRunner

available_examples: dict[str, dict[str, str]] = {
    "hybrid_etes_de": {
        "scenario": "hybrid_ETES_DE",
        "study_case": "base",
    },
    # Add future examples here once their input folders exist, for example:
    # "hybrid_etes_de_high_gas": {
    #     "scenario": "hybrid_ETES_DE",
    #     "study_case": "high_gas_price",
    # },
}


# Select the example to run from the available examples above.
example = "hybrid_etes_de"


default_output_options = OutputOptions(
    save_dispatch_results=True,
    save_market_ledger=True,
    save_storage_cost_ledger=True,
    save_summary_indicators=True,
    create_plots=True,
)


def resolve_example_paths(example: str) -> dict[str, Path]:
    if example not in available_examples:
        options = ", ".join(sorted(available_examples))
        raise ValueError(f"Unknown example '{example}'. Available examples: {options}")

    settings = available_examples[example]
    scenario = settings["scenario"]
    study_case = settings["study_case"]

    scenario_input_dir = PROJECT_ROOT / "data" / "input" / scenario
    if study_case == "base":
        input_dir = scenario_input_dir
    else:
        input_dir = scenario_input_dir / study_case

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory for example '{example}' does not exist: {input_dir}"
        )

    scenario_output_dir = PROJECT_ROOT / "data" / "output" / scenario
    output_dir = scenario_output_dir if study_case == "base" else scenario_output_dir / study_case

    return {
        "case_dir": input_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
    }


def build_runner_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.case:
        case_dir = Path(args.case).resolve()
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else PROJECT_ROOT / "data" / "output" / case_dir.name
        )
        paths = {
            "case_dir": case_dir,
            "input_dir": case_dir,
            "output_dir": output_dir,
        }
    else:
        paths = resolve_example_paths(args.example)
        if args.output_dir:
            paths["output_dir"] = Path(args.output_dir).resolve()

    output_options = OutputOptions(
        save_dispatch_results=default_output_options.save_dispatch_results
        and not args.skip_dispatch_results,
        save_market_ledger=default_output_options.save_market_ledger
        and not args.skip_market_ledger,
        save_storage_cost_ledger=default_output_options.save_storage_cost_ledger
        and not args.skip_storage_cost_ledger,
        save_summary_indicators=default_output_options.save_summary_indicators
        and not args.skip_summary_indicators,
        create_plots=default_output_options.create_plots and not args.no_plots,
    )

    return {
        **paths,
        "plants_file": args.plants_file,
        "forecasts_file": args.forecasts_file,
        "output_options": output_options,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a FLEXIMOD case.")
    parser.add_argument(
        "--example",
        default=example,
        choices=sorted(available_examples),
        help=(
            "Named example from the runner registry. Defaults to the script-level example variable."
        ),
    )
    parser.add_argument(
        "--case",
        help="Optional direct path to an input directory containing config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to data/output/<scenario>.",
    )
    parser.add_argument("--plants-file", default="plants.csv")
    parser.add_argument("--forecasts-file", default="forecasts_df.csv")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-dispatch-results", action="store_true")
    parser.add_argument("--skip-market-ledger", action="store_true")
    parser.add_argument("--skip-storage-cost-ledger", action="store_true")
    parser.add_argument("--skip-summary-indicators", action="store_true")
    args = parser.parse_args()

    settings = build_runner_settings(args)
    runner = SimulationRunner(**settings)
    outputs = runner.run()

    print(f"Input directory: {settings['input_dir']}")
    print(f"Output directory: {settings['output_dir']}")
    print("Created outputs:")
    for name, path in outputs.items():
        if isinstance(path, list):
            print(f"  {name}: {len(path)} files")
        else:
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
