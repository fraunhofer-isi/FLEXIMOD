# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure datetime-axis helpers for parsing, slicing and validating time grids.

These utilities are shared by data loaders that consume CSV files with a
datetime column. They are timezone-aware (DST handling) and validate that a
time grid is regular. None of them depend on a case configuration, so they can
be reused by any loader (case forecasts, market signals, ...).
"""

from __future__ import annotations

import pandas as pd


class DataValidationError(ValueError):
    """Raised when input data does not satisfy the configured case requirements."""


def find_datetime_column(frame: pd.DataFrame, source_name: str = "input file") -> str:
    """Return the name of the datetime-like column in ``frame``."""

    for candidate in ["datetime", "timestamp", "time"]:
        if candidate in frame.columns:
            return candidate
    raise DataValidationError(f"{source_name} must contain a datetime column")


def parse_datetime_column(series: pd.Series, column_name: str, source_name: str) -> pd.Series:
    """Parse a datetime column, accepting common European/ISO formats."""

    if series.astype(str).str.match(r"\s*\d{1,2}\.\d{1,2}\.\d{4}").any():
        try:
            return pd.to_datetime(series, errors="raise", format="mixed", dayfirst=True)
        except (TypeError, ValueError) as dotted_error:
            raise DataValidationError(
                f"Could not parse datetime column '{column_name}' in {source_name}. "
                "Use a consistent datetime format such as YYYY-MM-DD HH:MM or DD.MM.YYYY HH:MM."
            ) from dotted_error

    try:
        return pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError):
        try:
            return pd.to_datetime(series, errors="raise", format="mixed", dayfirst=True)
        except (TypeError, ValueError) as second_error:
            raise DataValidationError(
                f"Could not parse datetime column '{column_name}' in {source_name}. "
                "Use a consistent datetime format such as YYYY-MM-DD HH:MM or DD.MM.YYYY HH:MM."
            ) from second_error


def case_datetime_index(series: pd.Series, timezone: str | None = None) -> pd.DatetimeIndex:
    """Build a timezone-aware ``DatetimeIndex`` from a naive datetime series."""

    index = pd.DatetimeIndex(series)
    if timezone is None or index.tz is not None:
        return index
    try:
        return index.tz_localize(timezone, ambiguous="infer", nonexistent="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Could not interpret timestamps in timezone '{timezone}'. "
            "For spring DST changes, remove nonexistent local timestamps. For autumn "
            "DST changes, include both duplicated local timestamps in chronological order "
            "or provide timezone-aware timestamps."
        ) from exc


def infer_step_minutes(index: pd.DatetimeIndex, timezone: str | None = None) -> int:
    """Infer the regular step (minutes) of a time grid; raise if irregular."""

    validation_index = elapsed_time_index(index, timezone)
    diffs = validation_index.to_series().diff().dropna().dt.total_seconds().div(60)
    unique = sorted(set(int(value) for value in diffs))
    if len(unique) != 1:
        raise DataValidationError(
            "input must use a regular time grid; observed steps are "
            + ", ".join(str(value) for value in unique)
            + " minutes"
        )
    return unique[0]


def expected_period_count(
    start: pd.Timestamp,
    end: pd.Timestamp,
    timestep_minutes: int,
    timezone: str | None = None,
) -> int:
    """Number of grid points between ``start`` and ``end`` at the given step."""

    if timezone is None:
        elapsed_minutes = (end - start).total_seconds() / 60
    else:
        start_utc = local_timestamp_to_utc(start, timezone, label="simulation_start")
        end_utc = local_timestamp_to_utc(end, timezone, label="simulation_end")
        elapsed_minutes = (end_utc - start_utc).total_seconds() / 60
    return int(elapsed_minutes / timestep_minutes) + 1


def case_date_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    timestep_minutes: int,
    timezone: str | None = None,
) -> pd.DatetimeIndex:
    """Build a regular datetime grid for a simulation range."""

    if timezone is None:
        return pd.date_range(start=start, end=end, freq=f"{timestep_minutes}min")
    try:
        return pd.date_range(
            start=start,
            end=end,
            freq=f"{timestep_minutes}min",
            tz=timezone,
        )
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Could not create a {timestep_minutes}-minute datetime grid in timezone "
            f"'{timezone}'. Check timezone, simulation_start and simulation_end."
        ) from exc


def elapsed_time_index(index: pd.DatetimeIndex, timezone: str | None) -> pd.DatetimeIndex:
    """Return timestamps on an elapsed-time axis for robust DST-aware grid checks."""

    if timezone is None:
        return index
    if index.tz is not None:
        return index.tz_convert("UTC")
    try:
        return index.tz_localize(timezone, ambiguous="infer", nonexistent="raise").tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Could not validate timestamps in timezone '{timezone}'. "
            "If the data cover the autumn DST clock change, use timezone-aware "
            "timestamps or make duplicated local times distinguishable."
        ) from exc


def timestamp_for_index(
    timestamp: pd.Timestamp,
    index: pd.DatetimeIndex,
    label: str,
) -> pd.Timestamp:
    """Align a configured timestamp to the timezone of an existing index."""

    if index.tz is None:
        return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    try:
        if timestamp.tzinfo is not None:
            return timestamp.tz_convert(index.tz)
        return timestamp.tz_localize(index.tz, ambiguous="raise", nonexistent="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Configured {label}={timestamp} cannot be interpreted unambiguously in "
            f"timezone '{index.tz}'. Choose an existing, unambiguous local timestamp "
            "or provide timezone-aware timestamps."
        ) from exc


def local_timestamp_to_utc(
    timestamp: pd.Timestamp,
    timezone: str,
    label: str,
) -> pd.Timestamp:
    """Convert a naive local timestamp to UTC, failing on ambiguous DST times."""

    if timestamp.tzinfo is not None:
        return timestamp.tz_convert("UTC")
    try:
        return timestamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise").tz_convert(
            "UTC"
        )
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Configured {label}={timestamp} cannot be interpreted unambiguously in "
            f"timezone '{timezone}'. Choose an existing, unambiguous local timestamp "
            "or provide timezone-aware timestamps."
        ) from exc


def slice_time_range(
    frame: pd.DataFrame,
    simulation_start: str,
    simulation_end: str,
    timezone: str | None = None,
    source_name: str = "input file",
) -> pd.DataFrame:
    """Slice a datetime-indexed frame to ``[simulation_start, simulation_end]``."""

    start = timestamp_for_index(
        pd.Timestamp(simulation_start),
        frame.index,
        "simulation_start",
    )
    end = timestamp_for_index(
        pd.Timestamp(simulation_end),
        frame.index,
        "simulation_end",
    )
    filtered = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
    if filtered.empty:
        raise DataValidationError(
            f"{source_name} has no rows in configured simulation range {start} to {end}"
        )
    return filtered
