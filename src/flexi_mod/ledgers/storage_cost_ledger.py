# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Economic ledger for the cost and origin of stored heat.

The ledger treats ETES as a thermal inventory. Every charging step adds heat to
that inventory at an electricity procurement cost. Discharging reduces the
inventory but keeps the weighted-average inventory cost unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant

EPSILON = 1e-12
NO_PROCUREMENT = "no_charge"
PROCUREMENT_MARKETS = ("day_ahead", "intraday_continuous", "afrr_energy", "other")

INVENTORY_COLUMNS = {
    "day_ahead": "thermal_inventory_day_ahead_MWh_th",
    "intraday_continuous": "thermal_inventory_intraday_continuous_MWh_th",
    "afrr_energy": "thermal_inventory_afrr_energy_MWh_th",
    "other": "thermal_inventory_other_MWh_th",
}

STORAGE_COST_LEDGER_COLUMNS = [
    "datetime",
    "plant_name",
    "procurement_market",
    "electricity_price_EUR_per_MWh_el",
    "electricity_procured_MWh_el",
    "charged_heat_MWh_th",
    "charging_cost_EUR",
    "charged_heat_cost_EUR_per_MWh_th",
    "thermal_inventory_MWh_th",
    "weighted_average_inventory_cost_EUR_per_MWh_th",
    *INVENTORY_COLUMNS.values(),
]


class StorageCostLedger:
    """Track weighted-average thermal inventory cost by procurement market."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._inventory_mwh_th: dict[str, float] = {}
        self._average_cost_eur_per_mwh_th: dict[str, float] = {}
        self._inventory_by_market: dict[str, dict[str, float]] = {}

    def record_charge(
        self,
        datetime: pd.Timestamp,
        plant_name: str,
        procurement_market: str,
        electricity_price_eur_per_mwh_el: float,
        electricity_procured_mwh_el: float,
        charged_heat_mwh_th: float,
        thermal_inventory_mwh_th: float | None = None,
    ) -> None:
        """Record one electricity procurement that charges thermal inventory."""

        previous_inventory = self._inventory_mwh_th.get(plant_name, 0.0)
        inventory_after_dispatch = (
            previous_inventory + charged_heat_mwh_th
            if thermal_inventory_mwh_th is None
            else thermal_inventory_mwh_th
        )
        self.record_storage_step(
            datetime=datetime,
            plant_name=plant_name,
            charges=[
                {
                    "procurement_market": procurement_market,
                    "electricity_price": electricity_price_eur_per_mwh_el,
                    "electricity_mwh": electricity_procured_mwh_el,
                    "charged_heat_mwh_th": charged_heat_mwh_th,
                }
            ],
            thermal_inventory_mwh_th=inventory_after_dispatch,
        )

    def record_storage_step(
        self,
        datetime: pd.Timestamp,
        plant_name: str,
        charges: list[dict[str, float | str]],
        thermal_inventory_mwh_th: float,
    ) -> None:
        """Record one dispatch step of charging and resulting inventory."""

        clean_charges = [_normalise_charge(charge) for charge in charges]
        clean_charges = [
            charge
            for charge in clean_charges
            if charge["electricity_mwh"] > EPSILON and charge["charged_heat_mwh_th"] > EPSILON
        ]

        inventory_after_dispatch = max(float(thermal_inventory_mwh_th), 0.0)
        average_cost = self._update_weighted_average_cost(
            plant_name,
            clean_charges,
            inventory_after_dispatch,
        )
        inventory_by_market = self._update_inventory_by_market(
            plant_name,
            clean_charges,
            inventory_after_dispatch,
        )

        self._inventory_mwh_th[plant_name] = inventory_after_dispatch
        self._average_cost_eur_per_mwh_th[plant_name] = average_cost
        self._inventory_by_market[plant_name] = inventory_by_market

        rows = clean_charges or [_empty_charge()]
        for charge in rows:
            self._rows.append(
                _ledger_row(
                    datetime=datetime,
                    plant_name=plant_name,
                    charge=charge,
                    thermal_inventory_mwh_th=inventory_after_dispatch,
                    average_cost_eur_per_mwh_th=average_cost,
                    inventory_by_market=inventory_by_market,
                )
            )

    def build_from_dispatch_results(
        self,
        dispatch_results: pd.DataFrame,
        plants: list[SteamGenerationPlant],
    ) -> None:
        plant_by_name = {plant.name: plant for plant in plants}
        for timestamp, row in dispatch_results.sort_index().iterrows():
            plant_name = str(row["plant_name"])
            self.record_storage_step(
                datetime=timestamp,
                plant_name=plant_name,
                charges=_charges_from_dispatch_row(row, plant_by_name[plant_name]),
                thermal_inventory_mwh_th=float(row["etes_soc_MWh"]),
            )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=STORAGE_COST_LEDGER_COLUMNS)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path

    def _update_weighted_average_cost(
        self,
        plant_name: str,
        charges: list[dict[str, float | str]],
        inventory_after_dispatch: float,
    ) -> float:
        if inventory_after_dispatch <= EPSILON:
            return 0.0

        previous_inventory = self._inventory_mwh_th.get(plant_name, 0.0)
        previous_cost = self._average_cost_eur_per_mwh_th.get(plant_name, 0.0)
        added_heat = sum(float(charge["charged_heat_mwh_th"]) for charge in charges)
        added_cost = sum(_charging_cost(charge) for charge in charges)
        inventory_after_charge = previous_inventory + added_heat

        if added_heat <= EPSILON or inventory_after_charge <= EPSILON:
            return previous_cost
        return (previous_inventory * previous_cost + added_cost) / inventory_after_charge

    def _update_inventory_by_market(
        self,
        plant_name: str,
        charges: list[dict[str, float | str]],
        inventory_after_dispatch: float,
    ) -> dict[str, float]:
        inventory = self._inventory_by_market.get(plant_name, _empty_market_inventory()).copy()
        for charge in charges:
            market = str(charge["procurement_market"])
            inventory[market] = inventory.get(market, 0.0) + float(charge["charged_heat_mwh_th"])
        return _scale_inventory_to_total(inventory, inventory_after_dispatch)


def _charges_from_dispatch_row(
    row: pd.Series,
    plant: SteamGenerationPlant,
) -> list[dict[str, float | str]]:
    actual_charge_mwh_el = max(float(row["etes_charge_MWh"]), 0.0)
    if actual_charge_mwh_el <= EPSILON:
        return []

    source_electricity = _procured_electricity_by_market(row, actual_charge_mwh_el)
    return [
        {
            "procurement_market": market,
            "electricity_price": _electricity_price(row, market),
            "electricity_mwh": electricity_mwh,
            "charged_heat_mwh_th": electricity_mwh * plant.etes.efficiency_charge,
        }
        for market, electricity_mwh in source_electricity.items()
        if electricity_mwh > EPSILON
    ]


def _procured_electricity_by_market(
    row: pd.Series,
    actual_charge_mwh_el: float,
) -> dict[str, float]:
    day_ahead = max(float(row.get("DA_position_MWh", actual_charge_mwh_el)), 0.0)
    idc_sell = max(float(row.get("IDC_sell_MWh", 0.0)), 0.0)
    volumes = {
        "day_ahead": max(day_ahead - idc_sell, 0.0),
        "intraday_continuous": max(float(row.get("IDC_buy_MWh", 0.0)), 0.0),
        "afrr_energy": max(float(row.get("afrr_energy_activated_MWh", 0.0)), 0.0),
    }

    total = sum(volumes.values())
    if total <= EPSILON:
        return {"other": actual_charge_mwh_el}
    if total <= actual_charge_mwh_el + EPSILON:
        return volumes

    scale = actual_charge_mwh_el / total
    return {market: volume * scale for market, volume in volumes.items()}


def _electricity_price(row: pd.Series, market: str) -> float:
    delivered_price_columns = {
        "day_ahead": "day_ahead_delivered_price_EUR_per_MWh",
        "intraday_continuous": "IDC_delivered_price_EUR_per_MWh",
        "afrr_energy": "afrr_energy_delivered_price_EUR_per_MWh",
        "other": "day_ahead_delivered_price_EUR_per_MWh",
    }
    raw_price_columns = {
        "day_ahead": "day_ahead_price_EUR_per_MWh",
        "intraday_continuous": "IDC_price_EUR_per_MWh",
        "afrr_energy": "afrr_energy_price_EUR_per_MWh",
        "other": "day_ahead_price_EUR_per_MWh",
    }
    delivered_price = _numeric_value(row.get(delivered_price_columns[market]))
    if delivered_price is not None:
        return delivered_price

    raw_price = _numeric_value(row.get(raw_price_columns[market]))
    additional_charge = _numeric_value(row.get("additional_electricity_charge_EUR_per_MWh_el"))
    if raw_price is None:
        return 0.0
    return raw_price + (additional_charge or 0.0)


def _normalise_charge(charge: dict[str, float | str]) -> dict[str, float | str]:
    market = _normalise_market(str(charge.get("procurement_market", "other")))
    return {
        "procurement_market": market,
        "electricity_price": float(charge.get("electricity_price", 0.0)),
        "electricity_mwh": max(float(charge.get("electricity_mwh", 0.0)), 0.0),
        "charged_heat_mwh_th": max(float(charge.get("charged_heat_mwh_th", 0.0)), 0.0),
    }


def _ledger_row(
    datetime: pd.Timestamp,
    plant_name: str,
    charge: dict[str, float | str],
    thermal_inventory_mwh_th: float,
    average_cost_eur_per_mwh_th: float,
    inventory_by_market: dict[str, float],
) -> dict[str, Any]:
    charged_heat = float(charge["charged_heat_mwh_th"])
    charging_cost = _charging_cost(charge)
    heat_cost = charging_cost / charged_heat if charged_heat > EPSILON else pd.NA
    return {
        "datetime": pd.Timestamp(datetime),
        "plant_name": plant_name,
        "procurement_market": charge["procurement_market"],
        "electricity_price_EUR_per_MWh_el": charge["electricity_price"],
        "electricity_procured_MWh_el": charge["electricity_mwh"],
        "charged_heat_MWh_th": charged_heat,
        "charging_cost_EUR": charging_cost,
        "charged_heat_cost_EUR_per_MWh_th": heat_cost,
        "thermal_inventory_MWh_th": thermal_inventory_mwh_th,
        "weighted_average_inventory_cost_EUR_per_MWh_th": average_cost_eur_per_mwh_th,
        **{
            column: float(inventory_by_market.get(market, 0.0))
            for market, column in INVENTORY_COLUMNS.items()
        },
    }


def _charging_cost(charge: dict[str, float | str]) -> float:
    return float(charge["electricity_price"]) * float(charge["electricity_mwh"])


def _numeric_value(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _scale_inventory_to_total(
    inventory_by_market: dict[str, float],
    total_inventory_mwh_th: float,
) -> dict[str, float]:
    if total_inventory_mwh_th <= EPSILON:
        return _empty_market_inventory()

    current_total = sum(max(float(value), 0.0) for value in inventory_by_market.values())
    if current_total <= EPSILON:
        inventory = _empty_market_inventory()
        inventory["other"] = total_inventory_mwh_th
        return inventory

    scale = total_inventory_mwh_th / current_total
    return {
        market: max(float(inventory_by_market.get(market, 0.0)), 0.0) * scale
        for market in PROCUREMENT_MARKETS
    }


def _normalise_market(market: str) -> str:
    value = market.strip().lower()
    if value in {"da", "day-ahead", "day_ahead"}:
        return "day_ahead"
    if value in {"idc", "intraday", "intraday_continuous"}:
        return "intraday_continuous"
    if value in {"afrr", "afrr_energy", "afrr energy"}:
        return "afrr_energy"
    if not value or value == NO_PROCUREMENT:
        return "other"
    return value if value in PROCUREMENT_MARKETS else "other"


def _empty_market_inventory() -> dict[str, float]:
    return {market: 0.0 for market in PROCUREMENT_MARKETS}


def _empty_charge() -> dict[str, float | str]:
    return {
        "procurement_market": NO_PROCUREMENT,
        "electricity_price": 0.0,
        "electricity_mwh": 0.0,
        "charged_heat_mwh_th": 0.0,
    }
