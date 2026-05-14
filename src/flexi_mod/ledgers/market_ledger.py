# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

BASE_COLUMNS = [
    "datetime",
    "plant_name",
]

DAY_AHEAD_COLUMNS = [
    "DA_position_MWh",
    "DA_price",
]

INTRADAY_COLUMNS = [
    "IDC_buy_MWh",
    "IDC_sell_MWh",
    "IDC_price",
    "final_planned_electricity_MWh",
]

AFRR_ENERGY_COLUMNS = [
    "afrr_energy_bid_MW",
    "afrr_energy_bid_MWh",
    "afrr_energy_activated_MWh",
    "afrr_energy_price",
    "afrr_system_activation_MWh",
]

OPERATION_COLUMNS = [
    "actual_electricity_consumption_MWh",
    "gas_heat_MWh",
    "etes_charge_MWh",
    "etes_discharge_MWh",
    "etes_soc_MWh",
]

MARKET_LEDGER_COLUMNS = (
    BASE_COLUMNS + DAY_AHEAD_COLUMNS + INTRADAY_COLUMNS + AFRR_ENERGY_COLUMNS + OPERATION_COLUMNS
)

ZERO_COLUMNS = [
    "DA_position_MWh",
    "IDC_buy_MWh",
    "IDC_sell_MWh",
    "final_planned_electricity_MWh",
    "afrr_energy_bid_MW",
    "afrr_energy_bid_MWh",
    "afrr_energy_activated_MWh",
    "afrr_system_activation_MWh",
    "actual_electricity_consumption_MWh",
    "gas_heat_MWh",
    "etes_charge_MWh",
    "etes_discharge_MWh",
    "etes_soc_MWh",
]


class MarketLedger:
    """Modeller-facing market positions and plant operation by time step."""

    def __init__(self) -> None:
        self.rows = pd.DataFrame(columns=MARKET_LEDGER_COLUMNS)

    def initialise(self, datetimes: Iterable[pd.Timestamp], plant_names: Iterable[str]) -> None:
        records = [
            _default_record(timestamp, plant_name)
            for timestamp in datetimes
            for plant_name in plant_names
        ]
        self.rows = pd.DataFrame(records, columns=MARKET_LEDGER_COLUMNS)
        self._coerce_numeric_columns()

    def add_or_update_da_positions(
        self,
        plant_name: str,
        datetimes: Iterable[pd.Timestamp],
        da_position_mwh: Iterable[float],
        da_price: Iterable[float],
    ) -> None:
        datetimes = list(datetimes)
        if self.rows.empty:
            self.initialise(datetimes, [plant_name])

        indexed = self.rows.set_index(["datetime", "plant_name"])
        for timestamp, position, price in zip(datetimes, da_position_mwh, da_price, strict=False):
            key = (pd.Timestamp(timestamp), plant_name)
            _ensure_row(indexed, key, timestamp, plant_name)
            indexed.loc[key, "DA_position_MWh"] = float(position)
            indexed.loc[key, "DA_price"] = float(price)
            indexed.loc[key, "final_planned_electricity_MWh"] = float(position)
        self._replace_rows(indexed)

    def update_from_dispatch_results(self, dispatch_results: pd.DataFrame) -> None:
        if dispatch_results.empty:
            return

        if self.rows.empty:
            self.initialise(
                dispatch_results.index.unique(),
                dispatch_results["plant_name"].unique(),
            )

        indexed = self.rows.set_index(["datetime", "plant_name"])
        for timestamp, row in dispatch_results.iterrows():
            plant_name = str(row["plant_name"])
            key = (pd.Timestamp(timestamp), plant_name)
            _ensure_row(indexed, key, timestamp, plant_name)

            values = _market_values(row)
            _validate_electricity_balance(values)
            _set_day_ahead_fields(indexed, key, row, values)
            _set_intraday_fields(indexed, key, row, values)
            _set_afrr_energy_fields(indexed, key, row, values)
            _set_operation_fields(indexed, key, row, values)

        self._replace_rows(indexed)

    def to_dataframe(self) -> pd.DataFrame:
        return self.rows.copy()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows.to_csv(path, index=False)
        return path

    def _replace_rows(self, indexed: pd.DataFrame) -> None:
        self.rows = indexed.reset_index()[MARKET_LEDGER_COLUMNS]
        self.rows = self.rows.sort_values(["plant_name", "datetime"]).reset_index(drop=True)
        self._coerce_numeric_columns()

    def _coerce_numeric_columns(self) -> None:
        for column in MARKET_LEDGER_COLUMNS:
            if column not in BASE_COLUMNS and column in self.rows.columns:
                self.rows[column] = pd.to_numeric(self.rows[column], errors="coerce")


def _market_values(row: pd.Series) -> dict[str, float]:
    da_position = _row_float(row, "DA_position_MWh", row["electricity_consumption_MWh"])
    idc_buy = _row_float(row, "IDC_buy_MWh", 0.0)
    idc_sell = _row_float(row, "IDC_sell_MWh", 0.0)
    final_planned = _row_float(
        row,
        "final_planned_electricity_MWh",
        da_position + idc_buy - idc_sell,
    )
    afrr_activation = _row_float(row, "afrr_energy_activated_MWh", 0.0)
    actual_electricity = _row_float(
        row,
        "actual_electricity_consumption_MWh",
        row["electricity_consumption_MWh"],
    )
    return {
        "da_position": da_position,
        "idc_buy": idc_buy,
        "idc_sell": idc_sell,
        "final_planned": final_planned,
        "afrr_activation": afrr_activation,
        "actual_electricity": actual_electricity,
    }


def _validate_electricity_balance(values: dict[str, float]) -> None:
    expected_actual = values["final_planned"] + values["afrr_activation"]
    if abs(values["actual_electricity"] - expected_actual) > 1e-6:
        raise ValueError(
            "actual_electricity_consumption_MWh must match "
            "final_planned_electricity_MWh plus afrr_energy_activated_MWh"
        )


def _set_day_ahead_fields(
    indexed: pd.DataFrame,
    key: tuple[pd.Timestamp, str],
    row: pd.Series,
    values: dict[str, float],
) -> None:
    indexed.loc[key, "DA_position_MWh"] = values["da_position"]
    indexed.loc[key, "DA_price"] = _row_float(row, "day_ahead_price_EUR_per_MWh", pd.NA)


def _set_intraday_fields(
    indexed: pd.DataFrame,
    key: tuple[pd.Timestamp, str],
    row: pd.Series,
    values: dict[str, float],
) -> None:
    indexed.loc[key, "IDC_buy_MWh"] = values["idc_buy"]
    indexed.loc[key, "IDC_sell_MWh"] = values["idc_sell"]
    indexed.loc[key, "IDC_price"] = _row_float(row, "IDC_price_EUR_per_MWh", pd.NA)
    indexed.loc[key, "final_planned_electricity_MWh"] = values["final_planned"]


def _set_afrr_energy_fields(
    indexed: pd.DataFrame,
    key: tuple[pd.Timestamp, str],
    row: pd.Series,
    values: dict[str, float],
) -> None:
    bid_mwh = _row_float(row, "afrr_energy_bid_MWh", 0.0)
    indexed.loc[key, "afrr_energy_bid_MWh"] = bid_mwh
    indexed.loc[key, "afrr_energy_bid_MW"] = _row_float(row, "afrr_energy_bid_MW", bid_mwh)
    indexed.loc[key, "afrr_energy_activated_MWh"] = values["afrr_activation"]
    indexed.loc[key, "afrr_energy_price"] = _row_float(row, "afrr_energy_price_EUR_per_MWh", pd.NA)
    indexed.loc[key, "afrr_system_activation_MWh"] = _row_float(
        row,
        "afrr_system_activation_MWh",
        0.0,
    )


def _set_operation_fields(
    indexed: pd.DataFrame,
    key: tuple[pd.Timestamp, str],
    row: pd.Series,
    values: dict[str, float],
) -> None:
    indexed.loc[key, "actual_electricity_consumption_MWh"] = values["actual_electricity"]
    indexed.loc[key, "gas_heat_MWh"] = _row_float(row, "gas_heat_MWh", 0.0)
    indexed.loc[key, "etes_charge_MWh"] = _row_float(row, "etes_charge_MWh", 0.0)
    indexed.loc[key, "etes_discharge_MWh"] = _row_float(row, "etes_discharge_MWh", 0.0)
    indexed.loc[key, "etes_soc_MWh"] = _row_float(row, "etes_soc_MWh", 0.0)


def _ensure_row(
    indexed: pd.DataFrame,
    key: tuple[pd.Timestamp, str],
    timestamp: pd.Timestamp,
    plant_name: str,
) -> None:
    if key not in indexed.index:
        indexed.loc[key, :] = _default_values_without_index(timestamp, plant_name)


def _default_record(timestamp: pd.Timestamp, plant_name: str) -> dict[str, object]:
    record: dict[str, object] = {column: pd.NA for column in MARKET_LEDGER_COLUMNS}
    record["datetime"] = pd.Timestamp(timestamp)
    record["plant_name"] = plant_name
    for column in ZERO_COLUMNS:
        record[column] = 0.0
    return record


def _default_values_without_index(timestamp: pd.Timestamp, plant_name: str) -> dict[str, object]:
    record = _default_record(timestamp, plant_name)
    record.pop("datetime")
    record.pop("plant_name")
    return record


def _row_float(row: pd.Series, column: str, default: object) -> float:
    value = row[column] if column in row.index else default
    if pd.isna(value):
        return float("nan")
    return float(value)
