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

* ``hard coal`` -> ``coal_price``
* ``natural gas`` -> ``natural_gas_price``
* ``co2`` -> ``co2_price``
* ``hydrogen`` -> ``hydrogen_price``

For the default ``activation_unit="MW"``, ``demand_volume`` is repeated as a
power value in every 15-minute row. FLEXIMOD then converts it internally to MWh
when the case configuration declares ``afrr_energy.interpretation.activation_unit:
MW``. Use ``activation_unit="MWh"`` only when the receiving config expects
per-timestep energy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

DEFAULT_INPUT_PATH = Path(
    r"C:\Users\khm\ownCloud\Dropbox\Ph.D\My publications\Journal paper\4\Data"
    r"\Assume\Output\fokusstrom_2045\market_meta.csv"
)

OUTPUT_COLUMNS = [
    "DE_DA_price",
    "aFRR_capacity_down_price",
    "aFRR_energy_down_price",
    "aFRR_energy_down_quantity",
]

FUEL_PRICE_COLUMNS = {
    "hard coal": "coal_price",
    "natural gas": "natural_gas_price",
    "co2": "co2_price",
    "hydrogen": "hydrogen_price",
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
        default=DEFAULT_INPUT_PATH,
        help=f"Path to ASSUME market_meta.csv. Default: {DEFAULT_INPUT_PATH}",
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
        "--allow-missing",
        action="store_true",
        help="Write the CSV even if required output values are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = convert_assume_market_meta_to_forecasts(
        input_path=args.input,
        output_path=args.output,
        timestep_minutes=args.timestep_minutes,
        simulation=args.simulation,
        activation_unit=args.activation_unit,
        fuel_prices_path=args.fuel_prices,
        include_fuel_prices=not args.no_fuel_prices,
        allow_missing=args.allow_missing,
    )
    print(
        "Wrote "
        f"{summary.rows_written} rows from {summary.start} to {summary.end} "
        f"-> {summary.output_path}"
    )


if __name__ == "__main__":
    main()
