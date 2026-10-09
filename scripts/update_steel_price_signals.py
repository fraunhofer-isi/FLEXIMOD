# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Replace generated industrial-case market signals from archived scenario price data.

Each generated case below ``data/input/steel_inputs`` (or a compatible input root
passed with ``--input-root``) is named
``<scenario>_<year>_<route>``.  Its ``forecasts_df.csv`` contains plant-specific
demand and commodity signals in addition to the four electricity-market signals.
This utility maps the case prefix to
``data/input/archive/price_signal/<scenario>_<year>.csv`` and replaces only:

* ``DE_DA_price``;
* ``aFRR_capacity_down_price``;
* ``aFRR_energy_down_price``; and
* ``aFRR_energy_down_quantity``.

Rows are joined on timestamps, rather than their ordinal position.  Archived
price files start one day before the modelled period, so a positional replacement
would shift every signal by 24 hours.  The first defined exception is 2040
steel: those leap-year archive files contain 364 contiguous days (1 January--29
December), while their 364-day modelled periods are 2 January--30 December.
The script detects that exact one-day offset and uses ordinal alignment for
those files. Cement's 2040 runs include 30 December while the archive ends on
29 December; the established ASSUME converter convention is used there: repeat
the final available day's 15-minute signal profile. The template case
``steel_plant_DE`` is not a generated scenario-year case and is intentionally
excluded.

Run a non-mutating audit first:

    .venv/bin/python scripts/update_steel_price_signals.py --dry-run
    .venv/bin/python scripts/update_steel_price_signals.py \\
        --input-root data/input/cement_inputs --dry-run

Then apply and verify the updates:

    .venv/bin/python scripts/update_steel_price_signals.py
    .venv/bin/python scripts/update_steel_price_signals.py --check
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
STEEL_INPUT_ROOT = REPO_ROOT / "data" / "input" / "steel_inputs"
PRICE_SIGNAL_ROOT = REPO_ROOT / "data" / "input" / "archive" / "price_signal"

SOURCE_TIME_COLUMN = "time"
SOURCE_TO_FORECAST_COLUMN = {
    "DA_price_EUR_per_MWh": "DE_DA_price",
    "aFRR_neg_capacity_price_EUR_per_MW_VWAP": "aFRR_capacity_down_price",
    "aFRR_neg_energy_price_EUR_per_MWh": "aFRR_energy_down_price",
    "aFRR_neg_energy_quantity_MW": "aFRR_energy_down_quantity",
}
FORECAST_TIME_COLUMN = "datetime"
TEMPLATE_CASE_NAME = "steel_plant_DE"


@dataclass(frozen=True)
class PriceSignalSource:
    """One archived scenario-year signal file, indexed by timestamp."""

    name: str
    path: Path
    values_by_time: dict[str, dict[str, str]]
    ordered_timestamps: tuple[str, ...]
    ordered_values: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class UpdatePlan:
    """A fully validated replacement plan for one generated steel input."""

    forecast_path: Path
    source: PriceSignalSource
    row_count: int
    use_positional_alignment: bool
    trailing_repeat_steps: int
    simulation_start: str
    simulation_end: str


def _load_sources(price_signal_root: Path) -> dict[str, PriceSignalSource]:
    required_columns = {SOURCE_TIME_COLUMN, *SOURCE_TO_FORECAST_COLUMN}
    sources: dict[str, PriceSignalSource] = {}
    for path in sorted(price_signal_root.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Price-signal file has no header: {path}")
            missing = required_columns - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Price-signal file {path} is missing column(s): {', '.join(sorted(missing))}"
                )

            values_by_time: dict[str, dict[str, str]] = {}
            ordered_timestamps: list[str] = []
            ordered_values: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                timestamp = _source_timestamp_key(row[SOURCE_TIME_COLUMN], path, line_number)
                if timestamp in values_by_time:
                    raise ValueError(
                        f"Price-signal file {path} has duplicate timestamp {timestamp}"
                    )
                values = {
                    target: row[source] for source, target in SOURCE_TO_FORECAST_COLUMN.items()
                }
                values_by_time[timestamp] = values
                ordered_timestamps.append(timestamp)
                ordered_values.append(values)

        if not values_by_time:
            raise ValueError(f"Price-signal file is empty: {path}")
        sources[path.stem] = PriceSignalSource(
            name=path.stem,
            path=path,
            values_by_time=values_by_time,
            ordered_timestamps=tuple(ordered_timestamps),
            ordered_values=tuple(ordered_values),
        )

    if not sources:
        raise FileNotFoundError(f"No CSV price-signal files found in {price_signal_root}")
    return sources


def _source_timestamp_key(value: str, path: Path, line_number: int) -> str:
    """Validate and return an archived ISO timestamp without scalar pandas parsing."""

    text = value.strip()
    if len(text) != 19 or text[4] != "-" or text[7] != "-" or text[10] != " ":
        raise ValueError(f"Invalid ISO timestamp {value!r} at {path}:{line_number}")
    try:
        year, month, day = (int(part) for part in text[:10].split("-"))
        hour, minute, second = (int(part) for part in text[11:].split(":"))
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp {value!r} at {path}:{line_number}") from exc
    if not (1 <= month <= 12 and 1 <= day <= 31 and hour <= 23 and minute <= 59 and second <= 59):
        raise ValueError(f"Invalid timestamp {value!r} at {path}:{line_number}")
    return text


def _forecast_timestamp_key(value: str, path: Path, line_number: int) -> str:
    """Convert a generated forecast timestamp to the archived ISO key.

    This deliberately uses simple string handling rather than a scalar pandas call:
    processing 432 files means more than fifteen million timestamps, and scalar
    timestamp inference would make this otherwise I/O-bound update unnecessarily
    slow.
    """

    text = value.strip()
    if len(text) == 19 and text[4] == "-" and text[7] == "-" and text[10] == " ":
        return _source_timestamp_key(text, path, line_number)
    try:
        date_text, time_text = text.split(maxsplit=1)
        month_text, day_text, year_text = date_text.split("/")
        hour_text, minute_text = time_text.split(":")[:2]
        year = int(year_text)
        month = int(month_text)
        day = int(day_text)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError as exc:
        raise ValueError(f"Invalid forecast timestamp {value!r} at {path}:{line_number}") from exc
    if not (1 <= month <= 12 and 1 <= day <= 31 and hour <= 23 and minute <= 59):
        raise ValueError(f"Invalid forecast timestamp {value!r} at {path}:{line_number}")
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:00"


def _source_for_case(case_name: str, sources: dict[str, PriceSignalSource]) -> PriceSignalSource:
    matches = [name for name in sources if case_name.startswith(f"{name}_")]
    if len(matches) != 1:
        raise ValueError(
            f"Could not map generated case {case_name!r} to exactly one scenario-year "
            f"price-signal file; matches: {matches}"
        )
    return sources[matches[0]]


def _is_exact_one_day_offset(forecast_timestamps: list[str], source: PriceSignalSource) -> bool:
    """Return whether the full source and forecast horizons differ by one day.

    This is a deliberately narrow fallback for the 2040 leap-year series.  It
    cannot silently activate for a partial, gappy, or differently sized source.
    """

    if len(forecast_timestamps) != len(source.ordered_timestamps):
        return False
    try:
        forecast_start = datetime.fromisoformat(forecast_timestamps[0])
        forecast_end = datetime.fromisoformat(forecast_timestamps[-1])
        source_start = datetime.fromisoformat(source.ordered_timestamps[0])
        source_end = datetime.fromisoformat(source.ordered_timestamps[-1])
    except ValueError:
        return False
    return forecast_start - source_start == timedelta(
        days=1
    ) and forecast_end - source_end == timedelta(days=1)


def _trailing_repeat_steps(forecast_timestamps: list[str], source: PriceSignalSource) -> int:
    """Return safe tail-padding length when only one final day is unavailable."""

    source_count = len(source.ordered_timestamps)
    trailing_steps = len(forecast_timestamps) - source_count
    if trailing_steps <= 0 or source_count == 0:
        return 0
    if tuple(forecast_timestamps[:source_count]) != source.ordered_timestamps:
        return 0
    try:
        forecast_end = datetime.fromisoformat(forecast_timestamps[-1])
        source_end = datetime.fromisoformat(source.ordered_timestamps[-1])
    except ValueError:
        return 0
    if forecast_end - source_end != timedelta(days=1):
        return 0
    return trailing_steps


def _generated_forecasts(input_root: Path) -> Iterable[Path]:
    for path in sorted(input_root.glob("*/forecasts_df.csv")):
        if path.parent.name != TEMPLATE_CASE_NAME:
            yield path


def _simulation_window(forecast_path: Path) -> tuple[str, str]:
    """Return the inclusive simulation window in the archive timestamp format."""

    case_name = forecast_path.parent.name
    config_path = forecast_path.parent / "config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        case = config["cases"][case_name]
        start = datetime.fromisoformat(str(case["simulation_start"]))
        end = datetime.fromisoformat(str(case["simulation_end"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not read simulation window from {config_path}") from exc
    return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")


def _validate_update_plans(
    forecasts: Iterable[Path], sources: dict[str, PriceSignalSource]
) -> list[UpdatePlan]:
    plans: list[UpdatePlan] = []
    required_columns = {FORECAST_TIME_COLUMN, *SOURCE_TO_FORECAST_COLUMN.values()}

    for forecast_path in forecasts:
        source = _source_for_case(forecast_path.parent.name, sources)
        simulation_start, simulation_end = _simulation_window(forecast_path)
        with forecast_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Forecast file has no header: {forecast_path}")
            missing = required_columns - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Forecast file {forecast_path} is missing column(s): "
                    + ", ".join(sorted(missing))
                )

            seen: set[str] = set()
            active_timestamps: list[str] = []
            for line_number, row in enumerate(reader, start=2):
                timestamp = _forecast_timestamp_key(
                    row[FORECAST_TIME_COLUMN], forecast_path, line_number
                )
                if timestamp in seen:
                    raise ValueError(
                        f"Forecast file {forecast_path} has duplicate timestamp {timestamp}"
                    )
                seen.add(timestamp)
                if simulation_start <= timestamp <= simulation_end:
                    active_timestamps.append(timestamp)

        if not active_timestamps:
            raise ValueError(f"Forecast file has no rows in its simulation window: {forecast_path}")
        timestamps_match = all(
            timestamp in source.values_by_time for timestamp in active_timestamps
        )
        use_positional_alignment = not timestamps_match and _is_exact_one_day_offset(
            active_timestamps, source
        )
        trailing_repeat_steps = (
            0
            if timestamps_match or use_positional_alignment
            else _trailing_repeat_steps(active_timestamps, source)
        )
        if not timestamps_match and not use_positional_alignment and not trailing_repeat_steps:
            missing_timestamp = next(
                timestamp
                for timestamp in active_timestamps
                if timestamp not in source.values_by_time
            )
            raise ValueError(
                f"Forecast timestamp {missing_timestamp} in {forecast_path} is absent from "
                f"{source.path}, and no safe positional fallback applies"
            )
        plans.append(
            UpdatePlan(
                forecast_path=forecast_path,
                source=source,
                row_count=len(active_timestamps),
                use_positional_alignment=use_positional_alignment,
                trailing_repeat_steps=trailing_repeat_steps,
                simulation_start=simulation_start,
                simulation_end=simulation_end,
            )
        )

    if not plans:
        raise FileNotFoundError(f"No generated forecasts found under {STEEL_INPUT_ROOT}")
    return plans


def _rewrite_forecast(plan: UpdatePlan) -> None:
    temporary_path = plan.forecast_path.with_suffix(".csv.tmp")
    try:
        with (
            plan.forecast_path.open(encoding="utf-8-sig", newline="") as source_handle,
            temporary_path.open("w", encoding="utf-8", newline="") as target_handle,
        ):
            reader = csv.DictReader(source_handle)
            assert reader.fieldnames is not None  # checked before any writes occur
            writer = csv.DictWriter(target_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            active_position = 0
            for row_number, row in enumerate(reader, start=2):
                timestamp = _forecast_timestamp_key(
                    row[FORECAST_TIME_COLUMN], plan.forecast_path, row_number
                )
                if plan.simulation_start <= timestamp <= plan.simulation_end:
                    replacements = _replacements_for_row(plan, row, active_position)
                    row.update(replacements)
                    active_position += 1
                writer.writerow(row)
        os.replace(temporary_path, plan.forecast_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _check_updated_plans(plans: Iterable[UpdatePlan]) -> None:
    for plan in plans:
        with plan.forecast_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None
            active_position = 0
            for row_number, row in enumerate(reader, start=2):
                timestamp = _forecast_timestamp_key(
                    row[FORECAST_TIME_COLUMN], plan.forecast_path, row_number
                )
                if plan.simulation_start <= timestamp <= plan.simulation_end:
                    expected = _replacements_for_row(plan, row, active_position)
                    observed = {
                        column: row[column] for column in SOURCE_TO_FORECAST_COLUMN.values()
                    }
                    if observed != expected:
                        raise ValueError(
                            f"Signal mismatch at {plan.forecast_path}:row {row_number}: "
                            f"expected {expected}, found {observed}"
                        )
                    active_position += 1


def _replacements_for_row(
    plan: UpdatePlan, row: dict[str, str], active_position: int
) -> dict[str, str]:
    if plan.use_positional_alignment:
        return plan.source.ordered_values[active_position]
    if plan.trailing_repeat_steps and active_position >= len(plan.source.ordered_values):
        return plan.source.ordered_values[active_position - plan.trailing_repeat_steps]
    timestamp = _forecast_timestamp_key(
        row[FORECAST_TIME_COLUMN], plan.forecast_path, active_position + 2
    )
    return plan.source.values_by_time[timestamp]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate mappings and report the planned updates without modifying forecasts.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that every target signal already matches its archived source.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=STEEL_INPUT_ROOT,
        help="Generated case root to update (default: data/input/steel_inputs).",
    )
    args = parser.parse_args()
    if args.dry_run and args.check:
        parser.error("--dry-run and --check cannot be used together")

    sources = _load_sources(PRICE_SIGNAL_ROOT)
    plans = _validate_update_plans(_generated_forecasts(args.input_root), sources)
    source_counts: dict[str, int] = {}
    offset_cases: list[str] = []
    for plan in plans:
        source_counts[plan.source.name] = source_counts.get(plan.source.name, 0) + 1
        if plan.use_positional_alignment:
            offset_cases.append(plan.forecast_path.parent.name)

    if args.check:
        _check_updated_plans(plans)
        print(
            f"Verified {len(plans)} generated forecasts in {args.input_root} against "
            f"{len(sources)} price files."
        )
        return

    if args.dry_run:
        print(
            f"Validated {len(plans)} generated forecasts in {args.input_root} against "
            f"{len(sources)} price files."
        )
        for name, count in sorted(source_counts.items()):
            print(f"  {name}: {count} forecasts")
        if offset_cases:
            print(f"  positional 2040 alignment: {len(offset_cases)} forecasts")
        padded_cases = [plan for plan in plans if plan.trailing_repeat_steps]
        if padded_cases:
            print(
                f"  repeated final 2040 day: {len(padded_cases)} forecasts "
                f"({padded_cases[0].trailing_repeat_steps} timestep(s) each)"
            )
        return

    for position, plan in enumerate(plans, start=1):
        _rewrite_forecast(plan)
        print(f"[{position}/{len(plans)}] Updated {plan.forecast_path.parent.name}")
    _check_updated_plans(plans)
    print(f"Updated and verified {len(plans)} generated forecasts in {args.input_root}.")


if __name__ == "__main__":
    main()
