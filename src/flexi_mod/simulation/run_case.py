# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command-line entry point for running configured FLEXIMOD cases."""

from __future__ import annotations

import argparse
import sys
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
    from flexi_mod.simulation.simulation_runner import OutputOptions

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
    from flexi_mod.config.case_config import CaseConfig
    from flexi_mod.simulation.cli_logging import (
        CliLogger,
        output_summary,
        print_verbose_outputs,
    )
    from flexi_mod.simulation.simulation_runner import SimulationRunner

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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed created file paths.",
    )
    args = parser.parse_args()
    logger = CliLogger(verbose=args.verbose)

    settings = build_runner_settings(args)
    config = CaseConfig.from_case_dir(settings["case_dir"])
    logger.info(f"Case started: {config.case_name}")
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

    runner = SimulationRunner(**settings, progress_callback=logger.progress)
    with logger.capture_warnings():
        outputs = runner.run()

    logger.success(f"Case completed: {output_summary(outputs)} saved.")
    print_verbose_outputs(logger, outputs)


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
