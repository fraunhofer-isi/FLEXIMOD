# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small modeller-facing logging helpers for command-line entry points."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class CliLogger:
    """Print concise, user-facing CLI messages with optional details."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def info(self, message: str) -> None:
        print(message)

    def detail(self, message: str) -> None:
        if self.verbose:
            print(message)

    def notice(self, message: str) -> None:
        print(f"Notice: {message}")

    def success(self, message: str) -> None:
        print(message)

    def error(self, message: str) -> None:
        print(f"Error: {message}")

    def progress(self, message: str) -> None:
        if self.verbose or _show_progress_by_default(message):
            print(message)

    @contextmanager
    def capture_warnings(self) -> Iterator[None]:
        """Capture warnings and print them without Python file/function traces."""

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            yield
        self.report_warnings(records)

    def report_warnings(self, records: list[warnings.WarningMessage]) -> None:
        seen: set[str] = set()
        for record in records:
            message = _friendly_warning(str(record.message))
            if message in seen:
                continue
            seen.add(message)
            self.notice(message)


def output_summary(outputs: dict[str, Any]) -> str:
    csv_outputs = [name for name, path in outputs.items() if not isinstance(path, list)]
    plot_count = _plot_count(outputs)
    parts = []
    if csv_outputs:
        parts.append(f"{len(csv_outputs)} table file(s)")
    if plot_count:
        parts.append(f"{plot_count} plot file(s)")
    return ", ".join(parts) if parts else "no files"


def print_verbose_outputs(logger: CliLogger, outputs: dict[str, Any]) -> None:
    if not logger.verbose:
        return
    logger.detail("Created files:")
    for name, path in outputs.items():
        if isinstance(path, list):
            logger.detail(f"  {name}:")
            for item in path:
                logger.detail(f"    {Path(item)}")
        else:
            logger.detail(f"  {name}: {Path(path)}")


def additional_charges_message(enabled: bool, charges: dict[str, Any] | None = None) -> str:
    if not enabled:
        return "Additional charges: disabled; market prices are used directly."
    charges = charges or {}
    if not charges:
        return "Additional charges: enabled; no plant-specific charge rows were loaded."
    plant_parts = [
        f"{plant_name} = {len(frame)} tariff components"
        for plant_name, frame in sorted(charges.items())
    ]
    return "Additional charges: enabled; " + "; ".join(plant_parts) + "."


def missing_additional_charges_message(path: Path) -> str:
    return (
        "Additional charges are enabled, but additional_charges.csv was not found at "
        f"{path}. Add the file or set cases.<case_name>.additional_charges: false."
    )


def _plot_count(outputs: dict[str, Any]) -> int:
    plots = outputs.get("plots", [])
    return len(plots) if isinstance(plots, list) else 0


def _friendly_warning(message: str) -> str:
    replacements = {
        "No IDC sell/reduction volumes found. Skipping IDC sell source and compensation plot.": (
            "IDC sell source plot skipped because there are no IDC sell/reduction volumes."
        ),
    }
    if message.startswith("aFRR down system activation contains missing values."):
        return (
            "aFRR down system activation contains missing values in one or more decision "
            "windows. Missing activation values are set to zero; see "
            "afrr_energy_data_quality_summary.csv for counts."
        )
    if message.startswith("aFRR down price contains missing values."):
        return (
            "aFRR down price contains missing values in one or more decision windows. "
            "Bids and activations are set to zero for those timesteps; see "
            "afrr_energy_data_quality_summary.csv for counts."
        )
    if message.startswith("IDC price contains missing values."):
        return (
            "IDC price contains missing values in one or more decision windows. IDC action "
            "is set to zero for those timesteps."
        )
    return replacements.get(message, message)


def _show_progress_by_default(message: str) -> bool:
    default_prefixes = (
        "Loading input data",
        "Input data loaded",
        "Market calendar:",
        "Simulating ",
        "Plot creation started",
        "Plots created",
        "Outputs saved",
        "Notice:",
    )
    return message.startswith(default_prefixes)
