# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convert ASSUME ``market_meta.csv`` outputs to a FLEXIMOD ``forecasts_df.csv``.

The script expands ASSUME product intervals into a regular 15-minute time series
and maps the market ontology used in ASSUME to the FLEXIMOD signal names used by
the German steel strategy:

* ``EOM.price`` -> ``DE_DA_price``
* ``CRM_capacity_neg.price`` -> ``aFRR_capacity_down_price``
* ``CRM_energy_neg.price`` -> ``aFRR_energy_down_price``
* ``CRM_energy_neg.demand_volume`` -> ``aFRR_energy_down_quantity``

If a ``fuel_prices_df.csv`` file exists next to ``market_meta.csv``, it is also
expanded from hourly values to the same 15-minute index. When ASSUME market
timestamps and fuel-price timestamps use different years, the fuel prices are
aligned by calendar month/day/time:

* ``hard coal for industry`` -> ``coal_price``
* ``natural gas for industry`` -> ``natural_gas_price``
* ``solid biomass for industry`` -> ``biomass_price``
* ``naphtha for industry`` -> ``naphtha_price``
* ``co2`` -> ``co2_price``
* ``hydrogen`` -> ``hydrogen_price``

Steel-demand columns for the 9 physical plants are also merged in, sourced from
the scenario-family demand-series workbook in the "steel plant database" folder
(``AktuellePolitiken_demand_series_15min.xlsx``, ``HoheNachfrage_demand_series_15min.xlsx``,
``Niedrigenachfrage_demand_series_15min.xlsx``, or
``FokusH2_FokusStrom_Technologiemix_load_profile_15min_timestep_demand.xlsx`` --
one sheet per year). The scenario family and year are inferred from the input
folder name (``<scenario>_<year>/market_meta.csv``) unless overridden via
``--demand-scenario``/``--demand-year``. Columns are copied verbatim under their
existing bare physical-ID headers (e.g. ``P100000120423``) so
``generate_steel_cases.py`` picks them up unchanged.

For the default ``activation_unit="MW"``, ``demand_volume`` is repeated as a
power value in every 15-minute row. FLEXIMOD then converts it internally to MWh
when the case configuration declares ``afrr_energy.interpretation.activation_unit:
MW``. Use ``activation_unit="MWh"`` only when the receiving config expects
per-timestep energy.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DEFAULT_INPUT_PATH = Path(
    r"C:\Users\khm\ownCloud (2)\Dropbox\Ph.D\My publications\Journal paper\4\Data"
    r"\Assume\Output\fokusstrom_2045\market_meta.csv"
)

# Matches generate_steel_cases.py's EXTERNAL_OUTPUT_DIR -- the base folder containing one
# '<scenario>_<year>/market_meta.csv' subfolder per ASSUME run, used by --all.
DEFAULT_ASSUME_OUTPUT_DIR = Path(
    r"C:\Users\khm\ownCloud (2)\Dropbox\Ph.D\My publications\Journal paper\4\Data\Assume\Output"
)

OUTPUT_COLUMNS = [
    "DE_DA_price",
    "aFRR_capacity_down_price",
    "aFRR_energy_down_price",
    "aFRR_energy_down_quantity",
]

FUEL_PRICE_COLUMNS = {
    "hard coal for industry": "coal_price",
    "natural gas for industry": "natural_gas_price",
    "solid biomass for industry": "biomass_price",
    "naphtha for industry": "naphtha_price",
    "co2": "co2_price",
    "hydrogen": "hydrogen_price",
}

DEMAND_SERIES_DIR = Path(
    r"C:\Users\khm\ownCloud (2)\Dropbox\Ph.D\My publications\Journal paper\4\Data"
    r"\steel plant database"
)
DEMAND_SERIES_BY_SCENARIO = {
    "aktuellepolitiken": DEMAND_SERIES_DIR / "AktuellePolitiken_demand_series_15min.xlsx",
    "hohenachfrage": DEMAND_SERIES_DIR / "HoheNachfrage_demand_series_15min.xlsx",
    "niedrigenachfrage": DEMAND_SERIES_DIR / "Niedrigenachfrage_demand_series_15min.xlsx",
    "fokush2": (
        DEMAND_SERIES_DIR
        / "FokusH2_FokusStrom_Technologiemix_load_profile_15min_timestep_demand.xlsx"
    ),
    "fokusstrom": (
        DEMAND_SERIES_DIR
        / "FokusH2_FokusStrom_Technologiemix_load_profile_15min_timestep_demand.xlsx"
    ),
    "technologiemix": (
        DEMAND_SERIES_DIR
        / "FokusH2_FokusStrom_Technologiemix_load_profile_15min_timestep_demand.xlsx"
    ),
}


@dataclass(frozen=True)
class ConversionSummary:
    """Compact metadata returned by the converter for logging/tests."""

    rows_written: int
    start: pd.Timestamp
    end: pd.Timestamp
    output_path: Path
    simulation: str | None


def convert_assume_market_meta_to_forecasts(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path | None = None,
    *,
    timestep_minutes: int = 15,
    simulation: str | None = None,
    activation_unit: Literal["MW", "MWh"] = "MW",
    fuel_prices_path: str | Path | None = None,
    include_fuel_prices: bool = True,
    include_demand: bool = True,
    demand_scenario: str | None = None,
    demand_year: str | None = None,
    demand_workbook_path: str | Path | None = None,
    pad_to_year_end: bool = False,
    allow_missing: bool = False,
) -> ConversionSummary:
    """Create a FLEXIMOD forecast CSV from an ASSUME ``market_meta.csv`` file."""

    source = Path(input_path)
    destination = (
        Path(output_path) if output_path is not None else source.with_name("forecasts_df.csv")
    )
    if timestep_minutes <= 0:
        raise ValueError("timestep_minutes must be positive")
    if activation_unit not in {"MW", "MWh"}:
        raise ValueError("activation_unit must be 'MW' or 'MWh'")

    market_meta = pd.read_csv(source, skipinitialspace=True)
    market_meta = _prepare_market_meta(market_meta, simulation=simulation)
    frequency = pd.Timedelta(minutes=timestep_minutes)
    index = _output_index(market_meta, frequency)

    forecasts = pd.DataFrame(index=index)
    forecasts["DE_DA_price"] = _expand_interval_series(
        market_meta,
        market_id="EOM",
        value_column="price",
        output_index=index,
        frequency=frequency,
    )
    forecasts["aFRR_capacity_down_price"] = _expand_interval_series(
        market_meta,
        market_id="CRM_capacity_neg",
        value_column="price",
        output_index=index,
        frequency=frequency,
    )
    forecasts["aFRR_energy_down_price"] = _expand_interval_series(
        market_meta,
        market_id="CRM_energy_neg",
        value_column="price",
        output_index=index,
        frequency=frequency,
    )
    forecasts["aFRR_energy_down_quantity"] = _expand_interval_series(
        market_meta,
        market_id="CRM_energy_neg",
        value_column="demand_volume",
        output_index=index,
        frequency=frequency,
        value_transform=(
            (lambda values: values * (timestep_minutes / 60.0))
            if activation_unit == "MWh"
            else None
        ),
    )
    if include_fuel_prices:
        fuel_source = (
            Path(fuel_prices_path) if fuel_prices_path is not None else _default_fuel_path(source)
        )
        if fuel_source.exists():
            fuel_prices = _expand_fuel_prices(
                fuel_source,
                output_index=index,
                frequency=frequency,
            )
            forecasts = forecasts.join(fuel_prices)

    if include_demand:
        year_name = demand_year
        if year_name is None:
            _, year_name = _infer_scenario_year(source)
        if demand_workbook_path is not None:
            workbook = Path(demand_workbook_path)
        else:
            scenario_name = demand_scenario
            if scenario_name is None:
                scenario_name, _ = _infer_scenario_year(source)
            workbook = demand_series_workbook_for_scenario(scenario_name)
        if not workbook.exists():
            raise FileNotFoundError(f"Demand-series workbook not found: {workbook}")
        demand = _expand_demand_series(workbook, year=year_name, output_index=index)
        overlap = set(demand.columns) & set(forecasts.columns)
        if overlap:
            raise ValueError(f"Demand column(s) collide with market column(s): {sorted(overlap)}")
        forecasts = forecasts.join(demand)

    if pad_to_year_end:
        pad_year = demand_year
        if pad_year is None:
            _, pad_year = _infer_scenario_year(source)
        forecasts, days_added = _pad_to_year_end(forecasts, year=pad_year, frequency=frequency)
        if days_added:
            print(
                f"Warning: {source.parent.name} horizon fell {days_added} day(s) short of "
                f"{pad_year}-12-30 23:45; padded by repeating the last available day."
            )
        index = forecasts.index

    if not allow_missing:
        missing = forecasts.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            details = ", ".join(f"{column}={count}" for column, count in missing.items())
            raise ValueError(f"Missing output values after interval expansion: {details}")

    output = forecasts.reset_index(names="datetime")
    output.to_csv(destination, index=False)
    return ConversionSummary(
        rows_written=len(output),
        start=pd.Timestamp(index[0]),
        end=pd.Timestamp(index[-1]),
        output_path=destination,
        simulation=simulation,
    )


def convert_all_scenario_years(
    *,
    assume_output_dir: str | Path = DEFAULT_ASSUME_OUTPUT_DIR,
    timestep_minutes: int = 15,
    activation_unit: Literal["MW", "MWh"] = "MW",
    include_fuel_prices: bool = True,
    include_demand: bool = True,
    pad_to_year_end: bool = True,
    allow_missing: bool = False,
) -> list[ConversionSummary]:
    """Convert every '<scenario>_<year>/market_meta.csv' found under assume_output_dir.

    Scenario, year, and (when include_demand) the demand-series workbook are inferred
    per folder, exactly as convert_assume_market_meta_to_forecasts() does for a single
    file -- there is no per-folder override here. A folder that is missing
    market_meta.csv, or that fails conversion (unknown scenario family, missing demand
    workbook, ambiguous ASSUME simulation, etc.), is skipped with a reason rather than
    aborting the whole batch, since folders are independent. Defaults to padding short
    horizons out to each year's Dec 30 23:45 (see _pad_to_year_end) since this is the
    primary entry point for the leap-year-affected ASSUME runs that need it.
    """
    base = Path(assume_output_dir)
    if not base.exists():
        raise SystemExit(f"ASSUME output directory not found: {base}")

    summaries: list[ConversionSummary] = []
    skipped: list[str] = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        market_meta_path = folder / "market_meta.csv"
        if not market_meta_path.exists():
            skipped.append(f"{folder.name}: no market_meta.csv")
            continue
        try:
            summary = convert_assume_market_meta_to_forecasts(
                market_meta_path,
                timestep_minutes=timestep_minutes,
                activation_unit=activation_unit,
                include_fuel_prices=include_fuel_prices,
                include_demand=include_demand,
                pad_to_year_end=pad_to_year_end,
                allow_missing=allow_missing,
            )
        except (ValueError, FileNotFoundError) as exc:
            skipped.append(f"{folder.name}: {exc}")
            continue
        summaries.append(summary)
        print(f"{folder.name}: wrote {summary.rows_written} rows -> {summary.output_path}")

    print(f"\nConverted {len(summaries)} scenario-year folder(s).")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for item in skipped:
            print(f"  - {item}")
    return summaries


def _prepare_market_meta(df: pd.DataFrame, *, simulation: str | None) -> pd.DataFrame:
    required = {
        "market_id",
        "price",
        "demand_volume",
        "product_start",
        "product_end",
        "time",
        "simulation",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError("market_meta.csv is missing column(s): " + ", ".join(sorted(missing)))

    selected = df.copy()
    if simulation is None:
        simulations = sorted(str(value) for value in selected["simulation"].dropna().unique())
        if len(simulations) > 1:
            raise ValueError(
                "market_meta.csv contains multiple simulations; pass --simulation. "
                f"Found: {', '.join(simulations[:10])}"
            )
        simulation = simulations[0] if simulations else None
    if simulation is not None:
        selected = selected[selected["simulation"].astype(str) == str(simulation)].copy()
    if selected.empty:
        raise ValueError(f"No rows found for simulation {simulation!r}")

    for column in ["product_start", "product_end", "time"]:
        selected[column] = pd.to_datetime(selected[column], errors="coerce")
        if selected[column].isna().any():
            bad = int(selected[column].isna().sum())
            raise ValueError(f"Column '{column}' contains {bad} unparsable timestamp(s)")

    invalid_duration = selected["product_end"] <= selected["product_start"]
    if invalid_duration.any():
        raise ValueError(f"{int(invalid_duration.sum())} row(s) have product_end <= product_start")

    selected["market_id"] = selected["market_id"].astype(str)
    for market_id in ["EOM", "CRM_capacity_neg", "CRM_energy_neg"]:
        if market_id not in set(selected["market_id"]):
            raise ValueError(f"Required market_id '{market_id}' is missing from market_meta.csv")

    for column in ["price", "demand_volume"]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    return selected


def _output_index(df: pd.DataFrame, frequency: pd.Timedelta) -> pd.DatetimeIndex:
    selected = df[df["market_id"].isin(["EOM", "CRM_capacity_neg", "CRM_energy_neg"])]
    start = pd.Timestamp(selected["product_start"].min())
    end = pd.Timestamp(selected["product_end"].max())
    if start >= end:
        raise ValueError("Cannot create output index because product date range is empty")
    return pd.date_range(start=start, end=end - frequency, freq=frequency)


def _expand_interval_series(
    df: pd.DataFrame,
    *,
    market_id: str,
    value_column: str,
    output_index: pd.DatetimeIndex,
    frequency: pd.Timedelta,
    value_transform=None,
) -> pd.Series:
    market = df[df["market_id"] == market_id].copy()
    if market.empty:
        raise ValueError(f"No rows found for market_id '{market_id}'")
    if market[value_column].isna().any():
        bad = int(market[value_column].isna().sum())
        raise ValueError(f"{market_id}.{value_column} contains {bad} non-numeric value(s)")

    expanded_parts = []
    for row in market.itertuples(index=False):
        product_start = pd.Timestamp(row.product_start)
        product_end = pd.Timestamp(row.product_end)
        duration = product_end - product_start
        steps_float = duration / frequency
        steps = int(round(steps_float))
        if abs(steps_float - steps) > 1e-9 or steps <= 0:
            raise ValueError(
                f"{market_id} product {product_start} -> {product_end} does not align "
                f"with {frequency}"
            )
        timestamps = pd.date_range(
            start=product_start,
            periods=steps,
            freq=frequency,
        )
        value = float(getattr(row, value_column))
        values = pd.Series(value, index=timestamps, dtype=float)
        if value_transform is not None:
            values = value_transform(values)
        expanded_parts.append(values)

    expanded = pd.concat(expanded_parts).sort_index()
    duplicated = expanded.index.duplicated(keep=False)
    if duplicated.any():
        duplicates = expanded.loc[duplicated]
        conflicting = duplicates.groupby(level=0).nunique().gt(1)
        conflicting = conflicting[conflicting]
        if not conflicting.empty:
            examples = ", ".join(str(ts) for ts in conflicting.index[:5])
            raise ValueError(f"{market_id}.{value_column} has conflicting overlaps at: {examples}")
        expanded = expanded[~expanded.index.duplicated(keep="first")]
    return expanded.reindex(output_index)


def _default_fuel_path(market_meta_path: Path) -> Path:
    return market_meta_path.with_name("fuel_prices_df.csv")


def demand_series_workbook_for_scenario(scenario: str) -> Path:
    """Return the configured demand-series workbook for a scenario family."""
    scenario_key = scenario.strip().casefold()
    try:
        return DEMAND_SERIES_BY_SCENARIO[scenario_key]
    except KeyError as exc:
        supported = ", ".join(sorted(DEMAND_SERIES_BY_SCENARIO))
        raise ValueError(
            f"No demand-series workbook is configured for scenario family '{scenario}'. "
            f"Supported families: {supported}"
        ) from exc


_BASE_CASE_SUFFIX_RE = re.compile(r"_base_case_\d+$")


def _infer_scenario_year(source: Path) -> tuple[str, str]:
    """Infer (scenario, year) from a '<scenario>_<year>[_base_case_<year>]/market_meta.csv' path.

    ASSUME output folders carry a trailing '_base_case_<year>' suffix (the year repeated); it
    is stripped before splitting so 'aktuellepolitiken_2030_base_case_2030' still resolves to
    scenario='aktuellepolitiken', year='2030'.
    """
    folder_name = _BASE_CASE_SUFFIX_RE.sub("", source.parent.name)
    scenario, separator, year = folder_name.rpartition("_")
    if not separator or not year.isdigit():
        raise ValueError(
            f"Cannot infer scenario/year from folder '{source.parent.name}'; "
            "pass --demand-scenario and --demand-year explicitly."
        )
    return scenario, year


def _pad_to_year_end(
    forecasts: pd.DataFrame,
    *,
    year: str,
    frequency: pd.Timedelta,
) -> tuple[pd.DataFrame, int]:
    """Pad a short simulation horizon to '<year>-12-30 23:45' by repeating the last day.

    Some ASSUME runs produce a fixed-duration horizon that falls short of the calendar
    window FLEXIMOD cases need -- observed for leap years, where a fixed 364-day run
    starting Jan 1 lands on Dec 29 instead of Dec 30 (a normal year's day 364 from Jan 1
    *is* Dec 30, so non-leap years are unaffected). This repeats the last complete day's
    actual values forward -- not interpolated or synthetic -- one full day at a time,
    until the target end is reached. No-op (0 days added) if already at or past target.
    Returns (padded_frame, days_added).
    """
    target_end = pd.Timestamp(f"{year}-12-30 23:45:00")
    current_end = forecasts.index.max()
    if current_end >= target_end:
        return forecasts, 0

    steps_per_day = int(round(pd.Timedelta(days=1) / frequency))
    missing_steps = int(round((target_end - current_end) / frequency))
    # Repeat block is capped at whatever's actually available -- shorter than a full day
    # only for tiny/synthetic inputs; real ASSUME horizons always exceed one day.
    available_steps = min(steps_per_day, len(forecasts))
    last_day_values = forecasts.iloc[-available_steps:].to_numpy()
    reps = -(-missing_steps // available_steps)  # ceil division
    tiled = np.tile(last_day_values, (reps, 1))[:missing_steps]
    new_index = pd.date_range(start=current_end + frequency, periods=missing_steps, freq=frequency)
    padding = pd.DataFrame(tiled, index=new_index, columns=forecasts.columns)
    padded = pd.concat([forecasts, padding])
    days_added = int(round(missing_steps / steps_per_day))
    return padded, days_added


def _expand_demand_series(
    workbook: Path,
    *,
    year: str,
    output_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name=year, engine="openpyxl")
    if "datetime" not in frame.columns:
        raise ValueError(f"{workbook.name} sheet '{year}' is missing a 'datetime' column")
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    if frame["datetime"].isna().any():
        bad = int(frame["datetime"].isna().sum())
        raise ValueError(f"{workbook.name} sheet '{year}'.datetime has {bad} unparsable timestamp(s)")
    frame = frame.set_index("datetime").sort_index()
    if frame.index.duplicated().any():
        bad = int(frame.index.duplicated().sum())
        raise ValueError(f"{workbook.name} sheet '{year}' has {bad} duplicate datetime row(s)")
    demand_columns = [column for column in frame.columns if column != "datetime"]
    if not demand_columns:
        raise ValueError(f"{workbook.name} sheet '{year}' has no plant demand columns")
    return frame[demand_columns].reindex(output_index)


def _expand_fuel_prices(
    fuel_prices_path: Path,
    *,
    output_index: pd.DatetimeIndex,
    frequency: pd.Timedelta,
) -> pd.DataFrame:
    fuel_prices = pd.read_csv(fuel_prices_path, skipinitialspace=True)
    missing = {"datetime", *FUEL_PRICE_COLUMNS} - set(fuel_prices.columns)
    if missing:
        raise ValueError(
            f"{fuel_prices_path.name} is missing column(s): " + ", ".join(sorted(missing))
        )

    fuel_prices = fuel_prices.copy()
    fuel_prices["datetime"] = pd.to_datetime(fuel_prices["datetime"], errors="coerce")
    if fuel_prices["datetime"].isna().any():
        bad = int(fuel_prices["datetime"].isna().sum())
        raise ValueError(f"{fuel_prices_path.name}.datetime has {bad} unparsable timestamp(s)")
    fuel_prices = fuel_prices.sort_values("datetime")
    if fuel_prices["datetime"].duplicated().any():
        bad = int(fuel_prices["datetime"].duplicated().sum())
        raise ValueError(f"{fuel_prices_path.name}.datetime has {bad} duplicate timestamp(s)")

    output = pd.DataFrame(index=output_index)
    for source_column, output_column in FUEL_PRICE_COLUMNS.items():
        values = pd.to_numeric(fuel_prices[source_column], errors="coerce")
        if values.isna().any():
            bad = int(values.isna().sum())
            raise ValueError(f"{fuel_prices_path.name}.{source_column} has {bad} invalid value(s)")
        output[output_column] = _expand_step_series(
            pd.Series(values.to_numpy(dtype=float), index=fuel_prices["datetime"]),
            output_index=output_index,
            frequency=frequency,
        )
    return output


def _expand_step_series(
    series: pd.Series,
    *,
    output_index: pd.DatetimeIndex,
    frequency: pd.Timedelta,
) -> pd.Series:
    if series.empty:
        raise ValueError("Cannot expand an empty hourly time series")
    timestamps = pd.DatetimeIndex(series.index)
    if len(timestamps) > 1:
        source_step = timestamps.to_series().diff().dropna().mode().iloc[0]
    else:
        source_step = pd.Timedelta(hours=1)
    if source_step <= pd.Timedelta(0):
        raise ValueError("Fuel-price datetime step must be positive")

    expanded_parts = []
    for timestamp, value in series.items():
        steps_float = source_step / frequency
        steps = int(round(steps_float))
        if abs(steps_float - steps) > 1e-9 or steps <= 0:
            raise ValueError(f"Fuel-price step {source_step} does not align with {frequency}")
        expanded_parts.append(
            pd.Series(
                float(value),
                index=pd.date_range(start=timestamp, periods=steps, freq=frequency),
                dtype=float,
            )
        )
    expanded = pd.concat(expanded_parts).sort_index()
    expanded = expanded[~expanded.index.duplicated(keep="first")]
    aligned = expanded.reindex(output_index)
    if aligned.isna().all():
        calendar_aligned = _align_by_calendar(expanded, output_index)
        if calendar_aligned.notna().any():
            return calendar_aligned
    return aligned


def _align_by_calendar(
    expanded: pd.Series,
    output_index: pd.DatetimeIndex,
) -> pd.Series:
    source_years = pd.DatetimeIndex(expanded.index).year.unique()
    output_years = output_index.year.unique()
    if len(source_years) != 1 or len(output_years) != 1 or source_years[0] == output_years[0]:
        return pd.Series(index=output_index, dtype=float)

    source_year = int(source_years[0])
    mapped_index = [_same_calendar_timestamp(timestamp, source_year) for timestamp in output_index]
    values = expanded.reindex(pd.DatetimeIndex(mapped_index)).to_numpy(dtype=float)
    return pd.Series(values, index=output_index, dtype=float)


def _same_calendar_timestamp(timestamp: pd.Timestamp, year: int) -> pd.Timestamp:
    try:
        return timestamp.replace(year=year)
    except ValueError:
        if timestamp.month == 2 and timestamp.day == 29:
            return timestamp.replace(year=year, month=2, day=28)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create FLEXIMOD forecasts_df.csv from ASSUME market_meta.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to a single ASSUME market_meta.csv. If omitted, every "
            "'<scenario>_<year>/market_meta.csv' under --assume-output-dir is converted "
            "instead (same as passing --all)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output forecasts_df.csv path. Default: next to input file.",
    )
    parser.add_argument(
        "--timestep-minutes",
        type=int,
        default=15,
        help="Output resolution in minutes. Default: 15.",
    )
    parser.add_argument(
        "--simulation",
        default=None,
        help="ASSUME simulation name if market_meta.csv contains multiple simulations.",
    )
    parser.add_argument(
        "--activation-unit",
        choices=["MW", "MWh"],
        default="MW",
        help=(
            "Unit to write for aFRR_energy_down_quantity. Use MW for FLEXIMOD configs "
            "with afrr_energy.interpretation.activation_unit: MW. Default: MW."
        ),
    )
    parser.add_argument(
        "--fuel-prices",
        type=Path,
        default=None,
        help=(
            "Optional fuel_prices_df.csv path. Default: use fuel_prices_df.csv next to "
            "market_meta.csv if present."
        ),
    )
    parser.add_argument(
        "--no-fuel-prices",
        action="store_true",
        help="Do not add fuel-price columns even if fuel_prices_df.csv is present.",
    )
    parser.add_argument(
        "--demand-scenario",
        default=None,
        help=(
            "Scenario family for demand-series lookup (e.g. 'fokusH2'). Default: inferred "
            "from the input folder name '<scenario>_<year>/market_meta.csv'."
        ),
    )
    parser.add_argument(
        "--demand-year",
        default=None,
        help="Demand-series sheet/year (e.g. '2030'). Default: inferred from the input folder name.",
    )
    parser.add_argument(
        "--demand-workbook",
        type=Path,
        default=None,
        help="Optional explicit demand-series workbook path, overriding --demand-scenario lookup.",
    )
    parser.add_argument(
        "--no-demand",
        action="store_true",
        help="Do not add steel-demand columns.",
    )
    parser.add_argument(
        "--no-pad-short-years",
        action="store_true",
        help=(
            "Do not pad a short simulation horizon out to Dec 30 23:45 (default: pad by "
            "repeating the last available day -- affects leap years whose ASSUME run used "
            "a fixed non-leap-year duration)."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write the CSV even if required output values are missing.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Convert every '<scenario>_<year>/market_meta.csv' under --assume-output-dir. "
            "This is the default whenever --input is omitted; pass it explicitly only for "
            "clarity in scripts. Scenario/year/demand-workbook are inferred per folder; "
            "--output/--simulation/--fuel-prices/--demand-* are not used in this mode."
        ),
    )
    parser.add_argument(
        "--assume-output-dir",
        type=Path,
        default=DEFAULT_ASSUME_OUTPUT_DIR,
        help=f"Base directory for --all. Default: {DEFAULT_ASSUME_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    run_all = args.all or args.input is None
    if run_all:
        per_file_only = {
            "--output": args.output is not None,
            "--simulation": args.simulation is not None,
            "--fuel-prices": args.fuel_prices is not None,
            "--demand-scenario": args.demand_scenario is not None,
            "--demand-year": args.demand_year is not None,
            "--demand-workbook": args.demand_workbook is not None,
        }
        conflicting = [flag for flag, used in per_file_only.items() if used]
        if conflicting:
            raise SystemExit(
                "--all (or omitting --input) cannot be combined with "
                + ", ".join(conflicting)
                + " (per-folder only)."
            )
        convert_all_scenario_years(
            assume_output_dir=args.assume_output_dir,
            timestep_minutes=args.timestep_minutes,
            activation_unit=args.activation_unit,
            include_fuel_prices=not args.no_fuel_prices,
            include_demand=not args.no_demand,
            pad_to_year_end=not args.no_pad_short_years,
            allow_missing=args.allow_missing,
        )
        return

    summary = convert_assume_market_meta_to_forecasts(
        input_path=args.input,
        output_path=args.output,
        timestep_minutes=args.timestep_minutes,
        simulation=args.simulation,
        activation_unit=args.activation_unit,
        fuel_prices_path=args.fuel_prices,
        include_fuel_prices=not args.no_fuel_prices,
        include_demand=not args.no_demand,
        demand_scenario=args.demand_scenario,
        demand_year=args.demand_year,
        demand_workbook_path=args.demand_workbook,
        pad_to_year_end=not args.no_pad_short_years,
        allow_missing=args.allow_missing,
    )
    print(
        "Wrote "
        f"{summary.rows_written} rows from {summary.start} to {summary.end} "
        f"-> {summary.output_path}"
    )


if __name__ == "__main__":
    main()
