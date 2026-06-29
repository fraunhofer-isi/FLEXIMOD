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
    "hybrid_ETES_DA": {
        "scenario": "hybrid_ETES_DA",
        "study_case": "hybrid_ETES_DA",
    },
    "hybrid_ETES_DA_ID_buy": {
        "scenario": "hybrid_ETES_DA_ID_buy",
        "study_case": "hybrid_ETES_DA_ID_buy",
    },
    "hybrid_ETES_DA_ID_buy_sell": {
        "scenario": "hybrid_ETES_DA_ID_buy_sell",
        "study_case": "hybrid_ETES_DA_ID_buy_sell",
    },
    "hybrid_ETES_DA_ID_aFRR_energy": {
        "scenario": "hybrid_ETES_DA_ID_aFRR_energy",
        "study_case": "hybrid_ETES_DA_ID_aFRR_energy",
    },
    "hybrid_ETES_DA_ID_aFRR_energy_capacity": {
        "scenario": "hybrid_ETES_DA_ID_aFRR_energy_capacity",
        "study_case": "hybrid_ETES_DA_ID_aFRR_energy_capacity",
    },
}

# Select the example to run from the available examples above.
example = "hybrid_ETES_DA"


def resolve_example_paths(example: str) -> dict[str, Path | str]:
    if example not in available_examples:
        options = ", ".join(sorted(available_examples))
        raise ValueError(f"Unknown example '{example}'. Available examples: {options}")

    settings = available_examples[example]
    scenario = settings["scenario"]
    study_case = settings["study_case"]

    input_dir = PROJECT_ROOT / "data" / "input" / scenario

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory for example '{example}' does not exist: {input_dir}"
        )

    return {
        "case_dir": input_dir,
        "input_dir": input_dir,
        "study_case": study_case,
    }


def build_runner_settings(args: argparse.Namespace) -> dict[str, Any]:
    from flexi_mod.simulation.simulation_runner import OutputOptions

    if args.case:
        case_dir = Path(args.case).resolve()
        paths = {
            "case_dir": case_dir,
            "input_dir": case_dir,
            "study_case": args.study_case,
        }
    else:
        paths = resolve_example_paths(args.example)
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
        default=None,
        help=(
            "Full-load-hour tier assumed for the per-MWh grid energy charge in the dispatch "
            "strike price. If omitted and tiered rates are present in additional_charges.csv, "
            "you will be prompted interactively. The bill is corrected ex-post if the realized "
            "tier differs."
        ),
    )
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
