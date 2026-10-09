# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rerun every generated cement case and publish compressed result artifacts.

This reuses the isolated, atomic case runner used for steel.  The cement
catalogue has 456 generated cases (six scenario families, four years, and
nineteen routes), including the R6 hybrid-strategy cases.

Example:

    .venv/bin/python scripts/run_cement_cases_to_results.py --workers 24
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

runner = importlib.import_module("run_steel_cases_to_results")


REPO_ROOT = Path(__file__).resolve().parents[1]
runner.INPUT_ROOT = REPO_ROOT / "data" / "input" / "cement_inputs"
runner.RESULTS_ROOT = REPO_ROOT / "data" / "output" / "cement_results"
runner.STAGING_ROOT = REPO_ROOT / "data" / "output" / ".cement_results_staging"


def _available_cases() -> tuple[str, ...]:
    from flexi_mod.simulation.run_case import GENERATED_CEMENT_EXAMPLE_NAMES

    missing = [
        name for name in GENERATED_CEMENT_EXAMPLE_NAMES if not (runner.INPUT_ROOT / name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Generated cement input directory is missing: " + ", ".join(missing)
        )
    return GENERATED_CEMENT_EXAMPLE_NAMES


def _output_folder_name(case_name: str) -> str:
    """Read the configured strategy so output names match the model exactly."""

    config_path = runner.INPUT_ROOT / case_name / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    case_config = config["cases"][case_name]
    strategy = case_config["strategy"]["name"]
    return f"{case_name}_{strategy}"


runner._available_cases = _available_cases
runner._output_folder_name = _output_folder_name
runner.__doc__ = __doc__

if __name__ == "__main__":
    runner.main()
