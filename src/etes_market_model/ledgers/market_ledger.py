# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

MARKET_LEDGER_COLUMNS = [
    "datetime",
    "plant_name",
    "DA_position_MWh",
    "DA_price",
    "IDC_buy_MWh",
    "IDC_sell_MWh",
    "IDC_price",
    "planned_electricity_MWh",
    "afrr_capacity_reserved_MW",
    "afrr_capacity_price",
    "afrr_energy_bid_MW",
    "afrr_energy_activated_MWh",
    "afrr_energy_price",
    "actual_electricity_consumption_MWh",
    "gas_heat_MWh",
    "etes_charge_MWh",
    "etes_discharge_MWh",
    "etes_soc_MWh",
]

ZERO_COLUMNS = [
    "DA_position_MWh",
    "IDC_buy_MWh",
    "IDC_sell_MWh",
    "planned_electricity_MWh",
    "afrr_capacity_reserved_MW",
    "afrr_energy_bid_MW",
    "afrr_energy_activated_MWh",
    "actual_electricity_consumption_MWh",
    "gas_heat_MWh",
    "etes_charge_MWh",
    "etes_discharge_MWh",
    "etes_soc_MWh",
]


class MarketLedger:
    """One row per plant and dispatch time step with market positions and actions."""

    def __init__(self) -> None:
        self.rows = pd.DataFrame(columns=MARKET_LEDGER_COLUMNS)

    def initialise(self, datetimes: Iterable[pd.Timestamp], plant_names: Iterable[str]) -> None:
        records = []
        for timestamp in datetimes:
            for plant_name in plant_names:
                records.append(_default_record(timestamp, plant_name))
        self.rows = pd.DataFrame(records, columns=MARKET_LEDGER_COLUMNS)

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
            if key not in indexed.index:
                indexed.loc[key, :] = _default_values_without_index(timestamp, plant_name)
            indexed.loc[key, "DA_position_MWh"] = float(position)
            indexed.loc[key, "DA_price"] = float(price)
            indexed.loc[key, "planned_electricity_MWh"] = float(position)
        self.rows = indexed.reset_index()[MARKET_LEDGER_COLUMNS]

    def update_from_dispatch_results(self, dispatch_results: pd.DataFrame) -> None:
        if dispatch_results.empty:
            return

        datetimes = dispatch_results.index.unique()
        plant_names = dispatch_results["plant_name"].unique()
        if self.rows.empty:
            self.initialise(datetimes, plant_names)

        indexed = self.rows.set_index(["datetime", "plant_name"])
        for timestamp, row in dispatch_results.iterrows():
            plant_name = str(row["plant_name"])
            key = (pd.Timestamp(timestamp), plant_name)
            if key not in indexed.index:
                indexed.loc[key, :] = _default_values_without_index(timestamp, plant_name)
            indexed.loc[key, "DA_position_MWh"] = float(row["electricity_consumption_MWh"])
            indexed.loc[key, "DA_price"] = float(row["day_ahead_price_EUR_per_MWh"])
            indexed.loc[key, "planned_electricity_MWh"] = float(row["electricity_consumption_MWh"])
            indexed.loc[key, "actual_electricity_consumption_MWh"] = float(
                row["electricity_consumption_MWh"]
            )
            indexed.loc[key, "gas_heat_MWh"] = float(row["gas_heat_MWh"])
            indexed.loc[key, "etes_charge_MWh"] = float(row["etes_charge_MWh"])
            indexed.loc[key, "etes_discharge_MWh"] = float(row["etes_discharge_MWh"])
            indexed.loc[key, "etes_soc_MWh"] = float(row["etes_soc_MWh"])

        self.rows = indexed.reset_index()[MARKET_LEDGER_COLUMNS]
        self.rows = self.rows.sort_values(["plant_name", "datetime"]).reset_index(drop=True)

    def to_dataframe(self) -> pd.DataFrame:
        return self.rows.copy()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows.to_csv(path, index=False)
        return path


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
