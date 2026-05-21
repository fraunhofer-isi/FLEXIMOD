# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Market-facing electricity positions and realised plant operation.

The market ledger is an output table for modellers. It records electricity
procurement and activation by market with units, then adds the
operation variables needed to understand how those market positions are used.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

MARKET_LEDGER_COLUMNS = [
    "datetime",
    "plant_name",
    "day_ahead_position_MWh_el",
    "day_ahead_price_EUR_per_MWh_el",
    "additional_electricity_charge_EUR_per_MWh_el",
    "day_ahead_delivered_price_EUR_per_MWh_el",
    "intraday_buy_MWh_el",
    "intraday_sell_MWh_el",
    "intraday_price_EUR_per_MWh_el",
    "intraday_delivered_price_EUR_per_MWh_el",
    "scheduled_electricity_procurement_MWh_el",
    "afrr_energy_bid_MW_el",
    "afrr_energy_bid_MWh_el",
    "afrr_energy_activated_MWh_el",
    "afrr_energy_price_EUR_per_MWh_el",
    "afrr_energy_delivered_price_EUR_per_MWh_el",
    "afrr_energy_bid_price_EUR_per_MWh_el",
    "afrr_energy_market_spread_EUR_per_MWh_el",
    "afrr_energy_net_spread_EUR_per_MWh_el",
    "afrr_energy_pay_as_cleared_reward_EUR",
    "afrr_energy_net_value_after_charges_EUR",
    "afrr_energy_capacity_backed_bid_MWh_el",
    "afrr_energy_free_bid_MWh_el",
    "afrr_energy_capacity_backed_activated_MWh_el",
    "afrr_energy_free_activated_MWh_el",
    "afrr_system_activation_MWh_el",
    "useful_heat_cap_binding",
    "curtailed_proxy_activation_due_to_heat_cap_MWh",
    "afrr_capacity_block_id",
    "afrr_capacity_block_duration_h",
    "afrr_capacity_down_price_EUR_per_MW_h",
    "afrr_capacity_reserved_MW",
    "afrr_capacity_reserved_MWh",
    "afrr_capacity_revenue_EUR",
    "reserved_capacity_headroom_MWh",
    "available_charge_headroom_after_schedule_MWh",
    "available_storage_headroom_after_schedule_MWh",
    "actual_electricity_consumption_MWh_el",
    "gas_heat_output_MWh_th",
    "etes_charge_MWh_el",
    "etes_discharge_MWh_th",
    "etes_thermal_inventory_MWh_th",
]

ZERO_COLUMNS = [
    "day_ahead_position_MWh_el",
    "additional_electricity_charge_EUR_per_MWh_el",
    "intraday_buy_MWh_el",
    "intraday_sell_MWh_el",
    "scheduled_electricity_procurement_MWh_el",
    "afrr_energy_bid_MW_el",
    "afrr_energy_bid_MWh_el",
    "afrr_energy_activated_MWh_el",
    "afrr_energy_market_spread_EUR_per_MWh_el",
    "afrr_energy_net_spread_EUR_per_MWh_el",
    "afrr_energy_pay_as_cleared_reward_EUR",
    "afrr_energy_net_value_after_charges_EUR",
    "afrr_energy_capacity_backed_bid_MWh_el",
    "afrr_energy_free_bid_MWh_el",
    "afrr_energy_capacity_backed_activated_MWh_el",
    "afrr_energy_free_activated_MWh_el",
    "afrr_system_activation_MWh_el",
    "useful_heat_cap_binding",
    "curtailed_proxy_activation_due_to_heat_cap_MWh",
    "afrr_capacity_block_duration_h",
    "afrr_capacity_down_price_EUR_per_MW_h",
    "afrr_capacity_reserved_MW",
    "afrr_capacity_reserved_MWh",
    "afrr_capacity_revenue_EUR",
    "reserved_capacity_headroom_MWh",
    "available_charge_headroom_after_schedule_MWh",
    "available_storage_headroom_after_schedule_MWh",
    "actual_electricity_consumption_MWh_el",
    "gas_heat_output_MWh_th",
    "etes_charge_MWh_el",
    "etes_discharge_MWh_th",
    "etes_thermal_inventory_MWh_th",
]


class MarketLedger:
    """Build the market ledger from dispatch results."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def initialise(self, datetimes: Iterable[pd.Timestamp], plant_names: Iterable[str]) -> None:
        """Create empty market rows for the given plants and timestamps."""

        self._rows = [
            _empty_record(timestamp, plant_name)
            for timestamp in datetimes
            for plant_name in plant_names
        ]

    def record_day_ahead_positions(
        self,
        plant_name: str,
        datetimes: Iterable[pd.Timestamp],
        day_ahead_position_mwh_el: Iterable[float],
        day_ahead_price_eur_per_mwh_el: Iterable[float],
    ) -> None:
        """Record standalone day-ahead positions.

        The simulation normally uses ``update_from_dispatch_results``. This
        method is kept small for tests and simple DA-only experiments.
        """

        self._rows = [
            _day_ahead_record(timestamp, plant_name, position, price)
            for timestamp, position, price in zip(
                datetimes,
                day_ahead_position_mwh_el,
                day_ahead_price_eur_per_mwh_el,
                strict=False,
            )
        ]

    def update_from_dispatch_results(self, dispatch_results: pd.DataFrame) -> None:
        if dispatch_results.empty:
            return

        self._rows = [
            _record_from_dispatch_row(timestamp, row)
            for timestamp, row in dispatch_results.sort_index().iterrows()
        ]

    def to_dataframe(self) -> pd.DataFrame:
        frame = pd.DataFrame(self._rows, columns=MARKET_LEDGER_COLUMNS)
        if frame.empty:
            return frame
        for column in MARKET_LEDGER_COLUMNS:
            if column not in {"datetime", "plant_name", "afrr_capacity_block_id"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.sort_values(["plant_name", "datetime"]).reset_index(drop=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path


def _record_from_dispatch_row(timestamp: pd.Timestamp, row: pd.Series) -> dict[str, Any]:
    day_ahead_position = _value(row, "DA_position_MWh", row["electricity_consumption_MWh"])
    intraday_buy = _value(row, "IDC_buy_MWh", 0.0)
    intraday_sell = _value(row, "IDC_sell_MWh", 0.0)
    scheduled = _value(
        row,
        "final_planned_electricity_MWh",
        day_ahead_position + intraday_buy - intraday_sell,
    )
    afrr_activation = _value(row, "afrr_energy_activated_MWh", 0.0)
    actual_electricity = _value(
        row,
        "actual_electricity_consumption_MWh",
        row["electricity_consumption_MWh"],
    )
    _validate_electricity_accounting(scheduled, afrr_activation, actual_electricity)

    afrr_bid_mwh = _value(row, "afrr_energy_bid_MWh", 0.0)
    additional_charge = _value(row, "additional_electricity_charge_EUR_per_MWh_el", 0.0)
    day_ahead_price = _value(row, "day_ahead_price_EUR_per_MWh", pd.NA)
    intraday_price = _value(row, "IDC_price_EUR_per_MWh", pd.NA)
    afrr_energy_price = _value(row, "afrr_energy_price_EUR_per_MWh", pd.NA)
    return {
        "datetime": pd.Timestamp(timestamp),
        "plant_name": str(row["plant_name"]),
        "day_ahead_position_MWh_el": day_ahead_position,
        "day_ahead_price_EUR_per_MWh_el": day_ahead_price,
        "additional_electricity_charge_EUR_per_MWh_el": additional_charge,
        "day_ahead_delivered_price_EUR_per_MWh_el": _value(
            row,
            "day_ahead_delivered_price_EUR_per_MWh",
            day_ahead_price + additional_charge,
        ),
        "intraday_buy_MWh_el": intraday_buy,
        "intraday_sell_MWh_el": intraday_sell,
        "intraday_price_EUR_per_MWh_el": intraday_price,
        "intraday_delivered_price_EUR_per_MWh_el": _value(
            row,
            "IDC_delivered_price_EUR_per_MWh",
            intraday_price + additional_charge,
        ),
        "scheduled_electricity_procurement_MWh_el": scheduled,
        "afrr_energy_bid_MW_el": _value(row, "afrr_energy_bid_MW", afrr_bid_mwh),
        "afrr_energy_bid_MWh_el": afrr_bid_mwh,
        "afrr_energy_activated_MWh_el": afrr_activation,
        "afrr_energy_price_EUR_per_MWh_el": afrr_energy_price,
        "afrr_energy_delivered_price_EUR_per_MWh_el": _value(
            row,
            "afrr_energy_delivered_price_EUR_per_MWh",
            afrr_energy_price + additional_charge,
        ),
        "afrr_energy_bid_price_EUR_per_MWh_el": _value(
            row,
            "afrr_energy_bid_price_EUR_per_MWh",
            pd.NA,
        ),
        "afrr_energy_market_spread_EUR_per_MWh_el": _value(
            row,
            "afrr_energy_market_spread_EUR_per_MWh",
            0.0,
        ),
        "afrr_energy_net_spread_EUR_per_MWh_el": _value(
            row,
            "afrr_energy_net_spread_EUR_per_MWh",
            0.0,
        ),
        "afrr_energy_pay_as_cleared_reward_EUR": _value(
            row,
            "afrr_energy_pay_as_cleared_reward_EUR",
            0.0,
        ),
        "afrr_energy_net_value_after_charges_EUR": _value(
            row,
            "afrr_energy_net_value_after_charges_EUR",
            0.0,
        ),
        "afrr_energy_capacity_backed_bid_MWh_el": _value(
            row,
            "afrr_energy_capacity_backed_bid_MWh",
            0.0,
        ),
        "afrr_energy_free_bid_MWh_el": _value(row, "afrr_energy_free_bid_MWh", 0.0),
        "afrr_energy_capacity_backed_activated_MWh_el": _value(
            row,
            "afrr_energy_capacity_backed_activated_MWh",
            0.0,
        ),
        "afrr_energy_free_activated_MWh_el": _value(
            row,
            "afrr_energy_free_activated_MWh",
            0.0,
        ),
        "afrr_system_activation_MWh_el": _value(row, "afrr_system_activation_MWh", 0.0),
        "useful_heat_cap_binding": _value(row, "useful_heat_cap_binding", 0.0),
        "curtailed_proxy_activation_due_to_heat_cap_MWh": _value(
            row,
            "curtailed_proxy_activation_due_to_heat_cap_MWh",
            0.0,
        ),
        "afrr_capacity_block_id": str(
            row["afrr_capacity_block_id"] if "afrr_capacity_block_id" in row.index else ""
        ),
        "afrr_capacity_block_duration_h": _value(row, "afrr_capacity_block_duration_h", 0.0),
        "afrr_capacity_down_price_EUR_per_MW_h": _value(
            row, "afrr_capacity_down_price_EUR_per_MW_h", 0.0
        ),
        "afrr_capacity_reserved_MW": _value(row, "afrr_capacity_reserved_MW", 0.0),
        "afrr_capacity_reserved_MWh": _value(row, "afrr_capacity_reserved_MWh", 0.0),
        "afrr_capacity_revenue_EUR": _value(row, "afrr_capacity_revenue_EUR", 0.0),
        "reserved_capacity_headroom_MWh": _value(row, "reserved_capacity_headroom_MWh", 0.0),
        "available_charge_headroom_after_schedule_MWh": _value(
            row, "available_charge_headroom_after_schedule_MWh", 0.0
        ),
        "available_storage_headroom_after_schedule_MWh": _value(
            row, "available_storage_headroom_after_schedule_MWh", 0.0
        ),
        "actual_electricity_consumption_MWh_el": actual_electricity,
        "gas_heat_output_MWh_th": _value(row, "gas_heat_MWh", 0.0),
        "etes_charge_MWh_el": _value(row, "etes_charge_MWh", 0.0),
        "etes_discharge_MWh_th": _value(row, "etes_discharge_MWh", 0.0),
        "etes_thermal_inventory_MWh_th": _value(row, "etes_soc_MWh", 0.0),
    }


def _day_ahead_record(
    timestamp: pd.Timestamp,
    plant_name: str,
    position_mwh_el: float,
    price_eur_per_mwh_el: float,
) -> dict[str, Any]:
    record = _empty_record(timestamp, plant_name)
    record["day_ahead_position_MWh_el"] = float(position_mwh_el)
    record["day_ahead_price_EUR_per_MWh_el"] = float(price_eur_per_mwh_el)
    record["day_ahead_delivered_price_EUR_per_MWh_el"] = float(price_eur_per_mwh_el)
    record["scheduled_electricity_procurement_MWh_el"] = float(position_mwh_el)
    record["actual_electricity_consumption_MWh_el"] = float(position_mwh_el)
    return record


def _empty_record(timestamp: pd.Timestamp, plant_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {column: pd.NA for column in MARKET_LEDGER_COLUMNS}
    record["datetime"] = pd.Timestamp(timestamp)
    record["plant_name"] = str(plant_name)
    for column in ZERO_COLUMNS:
        record[column] = 0.0
    return record


def _validate_electricity_accounting(
    scheduled_mwh_el: float,
    afrr_activation_mwh_el: float,
    actual_mwh_el: float,
) -> None:
    expected = scheduled_mwh_el + afrr_activation_mwh_el
    if abs(actual_mwh_el - expected) > 1e-6:
        raise ValueError(
            "actual_electricity_consumption_MWh_el must equal "
            "scheduled_electricity_procurement_MWh_el plus afrr_energy_activated_MWh_el"
        )


def _value(row: pd.Series, column: str, default: object) -> float:
    value = row[column] if column in row.index else default
    if pd.isna(value):
        return float("nan")
    return float(value)
