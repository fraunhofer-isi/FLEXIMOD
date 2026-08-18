# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from scripts.create_forecasts_from_assume_market_meta import (
    convert_all_scenario_years,
    convert_assume_market_meta_to_forecasts,
    demand_series_workbook_for_scenario,
)


def test_assume_market_meta_to_forecasts_expands_products_to_15_min() -> None:
    input_path, output_path = _temporary_csv_paths()
    try:
        _market_meta().to_csv(input_path, index=False)
        summary = convert_assume_market_meta_to_forecasts(
            input_path, output_path, include_demand=False
        )
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
            include_demand=False,
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

        convert_assume_market_meta_to_forecasts(input_path, output_path, include_demand=False)

        forecasts = pd.read_csv(output_path)
        assert forecasts["coal_price"].tolist() == pytest.approx([30.0] * 4 + [40.0] * 4)
        assert forecasts["natural_gas_price"].tolist() == pytest.approx([50.0] * 4 + [60.0] * 4)
        assert forecasts["biomass_price"].tolist() == pytest.approx([35.0] * 4 + [45.0] * 4)
        assert forecasts["naphtha_price"].tolist() == pytest.approx([55.0] * 4 + [65.0] * 4)
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
            include_demand=False,
        )

        forecasts = pd.read_csv(output_path)
        assert "coal_price" not in forecasts.columns
        assert "natural_gas_price" not in forecasts.columns
        assert "biomass_price" not in forecasts.columns
        assert "naphtha_price" not in forecasts.columns
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

        convert_assume_market_meta_to_forecasts(input_path, output_path, include_demand=False)

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

        convert_assume_market_meta_to_forecasts(input_path, output_path, include_demand=False)

        forecasts = pd.read_csv(output_path)
        assert forecasts["coal_price"].tolist() == pytest.approx([30.0] * 4 + [40.0] * 4)
    finally:
        _remove_if_exists(input_path)
        _remove_if_exists(output_path)
        _remove_if_exists(fuel_prices_path)


def test_assume_market_meta_merges_demand_columns_from_explicit_workbook(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "market_meta.csv"
    output_path = tmp_path / "forecasts_df.csv"
    workbook_path = tmp_path / "demand_series.xlsx"
    _market_meta().to_csv(input_path, index=False)
    _demand_workbook(workbook_path, year="2030", frame=_demand_series())

    convert_assume_market_meta_to_forecasts(
        input_path,
        output_path,
        demand_workbook_path=workbook_path,
        demand_year="2030",
    )

    forecasts = pd.read_csv(output_path)
    assert forecasts["P1"].tolist() == pytest.approx([11.0] * 4 + [12.0] * 4)
    assert forecasts["P2"].tolist() == pytest.approx([21.0] * 4 + [22.0] * 4)


def test_assume_market_meta_infers_demand_year_from_input_folder_name(
    tmp_path: Path,
) -> None:
    scenario_dir = tmp_path / "myfamily_2031"
    scenario_dir.mkdir()
    input_path = scenario_dir / "market_meta.csv"
    output_path = scenario_dir / "forecasts_df.csv"
    workbook_path = tmp_path / "demand_series.xlsx"
    _market_meta(year=2031).to_csv(input_path, index=False)
    _demand_workbook(workbook_path, year="2031", frame=_demand_series(year=2031))

    convert_assume_market_meta_to_forecasts(
        input_path,
        output_path,
        demand_workbook_path=workbook_path,
    )

    forecasts = pd.read_csv(output_path)
    assert forecasts["P1"].tolist() == pytest.approx([11.0] * 4 + [12.0] * 4)


def test_assume_market_meta_can_skip_demand_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "market_meta.csv"
    output_path = tmp_path / "forecasts_df.csv"
    _market_meta().to_csv(input_path, index=False)

    convert_assume_market_meta_to_forecasts(input_path, output_path, include_demand=False)

    forecasts = pd.read_csv(output_path)
    assert list(forecasts.columns) == [
        "datetime",
        "DE_DA_price",
        "aFRR_capacity_down_price",
        "aFRR_energy_down_price",
        "aFRR_energy_down_quantity",
    ]


def test_assume_market_meta_demand_column_collision_raises_clear_error(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "market_meta.csv"
    output_path = tmp_path / "forecasts_df.csv"
    workbook_path = tmp_path / "demand_series.xlsx"
    _market_meta().to_csv(input_path, index=False)
    colliding = _demand_series(columns={"DE_DA_price": [11.0] * 4 + [12.0] * 4})
    _demand_workbook(workbook_path, year="2030", frame=colliding)

    with pytest.raises(ValueError, match="collide"):
        convert_assume_market_meta_to_forecasts(
            input_path,
            output_path,
            demand_workbook_path=workbook_path,
            demand_year="2030",
        )


def test_demand_series_workbook_for_scenario_unknown_family_has_no_silent_fallback() -> None:
    with pytest.raises(ValueError, match="No demand-series workbook.*unknown"):
        demand_series_workbook_for_scenario("unknown")


def test_assume_market_meta_pads_short_leap_year_horizon_by_repeating_last_day(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "market_meta.csv"
    output_path = tmp_path / "forecasts_df.csv"
    _market_meta_multi_day(start="2044-12-28 00:00:00", days=2).to_csv(input_path, index=False)

    convert_assume_market_meta_to_forecasts(
        input_path,
        output_path,
        include_demand=False,
        pad_to_year_end=True,
        demand_year="2044",
    )

    forecasts = pd.read_csv(output_path, parse_dates=["datetime"])
    assert forecasts["datetime"].max() == pd.Timestamp("2044-12-30 23:45:00")
    last_real_day = forecasts.loc[
        forecasts["datetime"].dt.date == pd.Timestamp("2044-12-29").date(), "DE_DA_price"
    ].tolist()
    padded_day = forecasts.loc[
        forecasts["datetime"].dt.date == pd.Timestamp("2044-12-30").date(), "DE_DA_price"
    ].tolist()
    assert padded_day == last_real_day


def test_assume_market_meta_does_not_pad_by_default(tmp_path: Path) -> None:
    input_path = tmp_path / "market_meta.csv"
    output_path = tmp_path / "forecasts_df.csv"
    _market_meta_multi_day(start="2044-12-28 00:00:00", days=2).to_csv(input_path, index=False)

    convert_assume_market_meta_to_forecasts(input_path, output_path, include_demand=False)

    forecasts = pd.read_csv(output_path, parse_dates=["datetime"])
    assert forecasts["datetime"].max() == pd.Timestamp("2044-12-29 23:45:00")


def _market_meta_multi_day(*, start: str, days: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    simulation = "demo"
    start_ts = pd.Timestamp(start)
    hours = days * 24
    for h in range(hours):
        hour_start = start_ts + pd.Timedelta(hours=h)
        hour_end = hour_start + pd.Timedelta(hours=1)
        rows.append(
            _row(
                market_id="EOM",
                product_start=hour_start,
                product_end=hour_end,
                price=10.0 + h,
                demand_volume=0.0,
                simulation=simulation,
            )
        )
        rows.append(
            _row(
                market_id="CRM_energy_neg",
                product_start=hour_start,
                product_end=hour_end,
                price=1.0,
                demand_volume=100.0,
                simulation=simulation,
            )
        )
    rows.append(
        _row(
            market_id="CRM_capacity_neg",
            product_start=start_ts,
            product_end=start_ts + pd.Timedelta(hours=hours),
            price=5.0,
            demand_volume=0.0,
            simulation=simulation,
        )
    )
    return pd.DataFrame(rows)


def test_assume_market_meta_infers_scenario_and_year_through_base_case_suffix(
    tmp_path: Path,
) -> None:
    scenario_dir = tmp_path / "myfamily_2031_base_case_2031"
    scenario_dir.mkdir()
    input_path = scenario_dir / "market_meta.csv"
    output_path = scenario_dir / "forecasts_df.csv"
    workbook_path = tmp_path / "demand_series.xlsx"
    _market_meta(year=2031).to_csv(input_path, index=False)
    _demand_workbook(workbook_path, year="2031", frame=_demand_series(year=2031))

    convert_assume_market_meta_to_forecasts(
        input_path,
        output_path,
        demand_workbook_path=workbook_path,
    )

    forecasts = pd.read_csv(output_path)
    assert forecasts["P1"].tolist() == pytest.approx([11.0] * 4 + [12.0] * 4)


def test_convert_all_scenario_years_processes_folders_and_skips_missing_market_meta(
    tmp_path: Path,
) -> None:
    (tmp_path / "fokusH2_2030").mkdir()
    _market_meta().to_csv(tmp_path / "fokusH2_2030" / "market_meta.csv", index=False)
    (tmp_path / "emptyfamily_2031").mkdir()  # no market_meta.csv -> should be skipped

    summaries = convert_all_scenario_years(
        assume_output_dir=tmp_path, include_demand=False, pad_to_year_end=False
    )

    assert len(summaries) == 1
    forecasts = pd.read_csv(summaries[0].output_path)
    assert list(forecasts.columns) == [
        "datetime",
        "DE_DA_price",
        "aFRR_capacity_down_price",
        "aFRR_energy_down_price",
        "aFRR_energy_down_quantity",
    ]
    assert not (tmp_path / "emptyfamily_2031" / "forecasts_df.csv").exists()


def test_convert_all_scenario_years_continues_after_a_folder_fails(tmp_path: Path) -> None:
    (tmp_path / "fokusH2_2030").mkdir()
    _market_meta().to_csv(tmp_path / "fokusH2_2030" / "market_meta.csv", index=False)
    (tmp_path / "broken_2031").mkdir()
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(
        tmp_path / "broken_2031" / "market_meta.csv", index=False
    )

    summaries = convert_all_scenario_years(
        assume_output_dir=tmp_path, include_demand=False, pad_to_year_end=False
    )

    assert len(summaries) == 1
    assert summaries[0].output_path == tmp_path / "fokusH2_2030" / "forecasts_df.csv"


def test_convert_all_scenario_years_missing_base_directory_raises_clear_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="ASSUME output directory not found"):
        convert_all_scenario_years(assume_output_dir=tmp_path / "does_not_exist")


def _demand_workbook(path: Path, *, year: str, frame: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=year, index=False)


def _demand_series(
    *,
    year: int = 2030,
    start_hour: str = "01-01 00:00:00",
    columns: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    index = pd.date_range(start=pd.Timestamp(f"{year}-{start_hour}"), periods=8, freq="15min")
    columns = columns or {"P1": [11.0] * 4 + [12.0] * 4, "P2": [21.0] * 4 + [22.0] * 4}
    data: dict[str, object] = {"datetime": index}
    data.update(columns)
    return pd.DataFrame(data)


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
            "hard coal for industry": [30.0, 40.0],
            "natural gas for industry": [50.0, 60.0],
            "solid biomass for industry": [35.0, 45.0],
            "naphtha for industry": [55.0, 65.0],
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
