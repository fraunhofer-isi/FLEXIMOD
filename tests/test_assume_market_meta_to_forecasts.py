# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from scripts.create_forecasts_from_assume_market_meta import (
    convert_assume_market_meta_to_forecasts,
)


def test_assume_market_meta_to_forecasts_expands_products_to_15_min() -> None:
    input_path, output_path = _temporary_csv_paths()
    try:
        _market_meta().to_csv(input_path, index=False)
        summary = convert_assume_market_meta_to_forecasts(input_path, output_path)
        forecasts = pd.read_csv(output_path)
        assert summary.rows_written == 8
        assert list(forecasts.columns) == [
            "datetime",
            "DE_DA_price",
            "aFRR_capacity_down_price",
            "aFRR_energy_down_price",
            "aFRR_energy_down_quantity",
        ]
        assert forecasts["DE_DA_price"].tolist() == pytest.approx([10.0] * 4 + [20.0] * 4)
        assert forecasts["aFRR_capacity_down_price"].tolist() == pytest.approx([5.0] * 8)
        assert forecasts["aFRR_energy_down_price"].tolist() == pytest.approx([1.0] * 4 + [2.0] * 4)
        assert forecasts["aFRR_energy_down_quantity"].tolist() == pytest.approx(
            [100.0] * 4 + [200.0] * 4
        )
    finally:
        _remove_if_exists(input_path)
        _remove_if_exists(output_path)


def test_assume_market_meta_can_write_activation_as_per_timestep_mwh() -> None:
    input_path, output_path = _temporary_csv_paths()
    try:
        _market_meta().to_csv(input_path, index=False)
        convert_assume_market_meta_to_forecasts(
            input_path,
            output_path,
            activation_unit="MWh",
        )
        forecasts = pd.read_csv(output_path)
        assert forecasts["aFRR_energy_down_quantity"].tolist() == pytest.approx(
            [25.0] * 4 + [50.0] * 4
        )
    finally:
        _remove_if_exists(input_path)
        _remove_if_exists(output_path)


def test_assume_market_meta_includes_hourly_fuel_prices_when_present() -> None:
    input_path, output_path = _temporary_csv_paths()
    fuel_prices_path = input_path.with_name("fuel_prices_df.csv")
    try:
        _market_meta().to_csv(input_path, index=False)
        _fuel_prices().to_csv(fuel_prices_path, index=False)

        convert_assume_market_meta_to_forecasts(input_path, output_path)

        forecasts = pd.read_csv(output_path)
        assert forecasts["coal_price"].tolist() == pytest.approx([30.0] * 4 + [40.0] * 4)
        assert forecasts["natural_gas_price"].tolist() == pytest.approx([50.0] * 4 + [60.0] * 4)
        assert forecasts["co2_price"].tolist() == pytest.approx([70.0] * 4 + [80.0] * 4)
        assert forecasts["hydrogen_price"].tolist() == pytest.approx([90.0] * 4 + [100.0] * 4)
    finally:
        _remove_if_exists(input_path)
        _remove_if_exists(output_path)
        _remove_if_exists(fuel_prices_path)


def test_assume_market_meta_can_skip_present_fuel_prices() -> None:
    input_path, output_path = _temporary_csv_paths()
    fuel_prices_path = input_path.with_name("fuel_prices_df.csv")
    try:
        _market_meta().to_csv(input_path, index=False)
        _fuel_prices().to_csv(fuel_prices_path, index=False)

        convert_assume_market_meta_to_forecasts(
            input_path,
            output_path,
            include_fuel_prices=False,
        )

        forecasts = pd.read_csv(output_path)
        assert "coal_price" not in forecasts.columns
        assert "natural_gas_price" not in forecasts.columns
        assert "co2_price" not in forecasts.columns
        assert "hydrogen_price" not in forecasts.columns
    finally:
        _remove_if_exists(input_path)
        _remove_if_exists(output_path)
        _remove_if_exists(fuel_prices_path)


def test_assume_market_meta_aligns_fuel_prices_by_calendar_when_years_differ() -> None:
    input_path, output_path = _temporary_csv_paths()
    fuel_prices_path = input_path.with_name("fuel_prices_df.csv")
    try:
        _market_meta(year=2040).to_csv(input_path, index=False)
        _fuel_prices(year=2045).to_csv(fuel_prices_path, index=False)

        convert_assume_market_meta_to_forecasts(input_path, output_path)

        forecasts = pd.read_csv(output_path)
        assert forecasts["coal_price"].tolist() == pytest.approx([30.0] * 4 + [40.0] * 4)
        assert forecasts["natural_gas_price"].tolist() == pytest.approx([50.0] * 4 + [60.0] * 4)
        assert forecasts["co2_price"].tolist() == pytest.approx([70.0] * 4 + [80.0] * 4)
        assert forecasts["hydrogen_price"].tolist() == pytest.approx([90.0] * 4 + [100.0] * 4)
    finally:
        _remove_if_exists(input_path)
        _remove_if_exists(output_path)
        _remove_if_exists(fuel_prices_path)


def test_assume_market_meta_uses_previous_day_for_missing_fuel_leap_day() -> None:
    input_path, output_path = _temporary_csv_paths()
    fuel_prices_path = input_path.with_name("fuel_prices_df.csv")
    try:
        _market_meta(
            year=2040,
            first_hour="02-29 00:00:00",
            second_hour="02-29 01:00:00",
        ).to_csv(input_path, index=False)
        _fuel_prices(
            year=2045,
            first_hour="02-28 00:00:00",
            second_hour="02-28 01:00:00",
        ).to_csv(fuel_prices_path, index=False)

        convert_assume_market_meta_to_forecasts(input_path, output_path)

        forecasts = pd.read_csv(output_path)
        assert forecasts["coal_price"].tolist() == pytest.approx([30.0] * 4 + [40.0] * 4)
    finally:
        _remove_if_exists(input_path)
        _remove_if_exists(output_path)
        _remove_if_exists(fuel_prices_path)


def _market_meta(
    *,
    year: int = 2030,
    first_hour: str = "01-01 00:00:00",
    second_hour: str = "01-01 01:00:00",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    simulation = "demo"
    for hour, eom_price, energy_price, demand_volume in [
        (first_hour, 10.0, 1.0, 100.0),
        (second_hour, 20.0, 2.0, 200.0),
    ]:
        start = pd.Timestamp(f"{year}-{hour}")
        end = start + pd.Timedelta(hours=1)
        rows.append(
            _row(
                market_id="EOM",
                product_start=start,
                product_end=end,
                price=eom_price,
                demand_volume=0.0,
                simulation=simulation,
            )
        )
        rows.append(
            _row(
                market_id="CRM_energy_neg",
                product_start=start,
                product_end=end,
                price=energy_price,
                demand_volume=demand_volume,
                simulation=simulation,
            )
        )
    rows.append(
        _row(
            market_id="CRM_capacity_neg",
            product_start=pd.Timestamp(f"{year}-{first_hour}"),
            product_end=pd.Timestamp(f"{year}-{second_hour}") + pd.Timedelta(hours=1),
            price=5.0,
            demand_volume=0.0,
            simulation=simulation,
        )
    )
    return pd.DataFrame(rows)


def _fuel_prices(
    *,
    year: int = 2030,
    first_hour: str = "01-01 00:00:00",
    second_hour: str = "01-01 01:00:00",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp(f"{year}-{first_hour}"),
                pd.Timestamp(f"{year}-{second_hour}"),
            ],
            "hard coal": [30.0, 40.0],
            "natural gas": [50.0, 60.0],
            "co2": [70.0, 80.0],
            "hydrogen": [90.0, 100.0],
        }
    )


def _row(
    *,
    market_id: str,
    product_start: pd.Timestamp,
    product_end: pd.Timestamp,
    price: float,
    demand_volume: float,
    simulation: str,
) -> dict[str, object]:
    return {
        "market_id": market_id,
        "product_start": product_start,
        "product_end": product_end,
        "time": product_start,
        "price": price,
        "demand_volume": demand_volume,
        "simulation": simulation,
    }


def _temporary_csv_paths() -> tuple[Path, Path]:
    token = uuid4().hex
    return (
        Path(f"assume_market_meta_test_input_{token}.csv"),
        Path(f"assume_market_meta_test_output_{token}.csv"),
    )


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()
