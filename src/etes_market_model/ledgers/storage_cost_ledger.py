# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pandas as pd

from etes_market_model.plants.steam_generation_plant import SteamGenerationPlant

STORAGE_COST_LEDGER_COLUMNS = [
    "datetime",
    "plant_name",
    "source_market",
    "electricity_price_EUR_per_MWh",
    "electricity_volume_MWh",
    "stored_heat_added_MWh",
    "effective_heat_cost_EUR_per_MWh_th",
    "remaining_stored_heat_MWh",
    "weighted_average_storage_cost_EUR_per_MWh_th",
    "remaining_stored_heat_day_ahead_MWh",
    "remaining_stored_heat_IDC_MWh",
    "remaining_stored_heat_afrr_energy_MWh",
    "remaining_stored_heat_other_MWh",
]

STORAGE_SOURCE_COLUMNS = {
    "day_ahead": "remaining_stored_heat_day_ahead_MWh",
    "intraday_continuous": "remaining_stored_heat_IDC_MWh",
    "afrr_energy": "remaining_stored_heat_afrr_energy_MWh",
    "other": "remaining_stored_heat_other_MWh",
}


class StorageCostLedger:
    """Weighted-average economic cost tracker for stored ETES heat."""

    def __init__(self) -> None:
        self.rows = pd.DataFrame(columns=STORAGE_COST_LEDGER_COLUMNS)
        self._state_by_plant: dict[str, tuple[float, float]] = {}
        self._source_inventory_by_plant: dict[str, dict[str, float]] = {}

    def record_storage_event(
        self,
        datetime: pd.Timestamp,
        plant_name: str,
        source_market: str,
        electricity_price_eur_per_mwh: float,
        electricity_volume_mwh: float,
        stored_heat_added_mwh: float,
        remaining_stored_heat_mwh: float | None = None,
    ) -> None:
        previous_remaining, previous_average_cost = self._state_by_plant.get(plant_name, (0.0, 0.0))
        source_inventory = self._source_inventory_by_plant.get(
            plant_name,
            {source: 0.0 for source in STORAGE_SOURCE_COLUMNS},
        )

        added_cost = electricity_price_eur_per_mwh * electricity_volume_mwh
        inventory_after_charge = previous_remaining + stored_heat_added_mwh
        if inventory_after_charge > 1e-12 and stored_heat_added_mwh > 0:
            weighted_average = (
                previous_remaining * previous_average_cost + added_cost
            ) / inventory_after_charge
        elif previous_remaining > 1e-12:
            weighted_average = previous_average_cost
        else:
            weighted_average = 0.0

        remaining = (
            inventory_after_charge
            if remaining_stored_heat_mwh is None
            else max(float(remaining_stored_heat_mwh), 0.0)
        )
        if remaining <= 1e-12:
            weighted_average = 0.0

        source = _normalise_source_market(source_market)
        if stored_heat_added_mwh > 1e-12:
            source_inventory[source] = source_inventory.get(source, 0.0) + stored_heat_added_mwh
        source_inventory = _reconcile_source_inventory(source_inventory, remaining)

        effective_heat_cost = (
            added_cost / stored_heat_added_mwh if stored_heat_added_mwh > 1e-12 else pd.NA
        )
        self._state_by_plant[plant_name] = (remaining, weighted_average)
        self._source_inventory_by_plant[plant_name] = source_inventory

        record = {
            "datetime": pd.Timestamp(datetime),
            "plant_name": plant_name,
            "source_market": source_market,
            "electricity_price_EUR_per_MWh": float(electricity_price_eur_per_mwh),
            "electricity_volume_MWh": float(electricity_volume_mwh),
            "stored_heat_added_MWh": float(stored_heat_added_mwh),
            "effective_heat_cost_EUR_per_MWh_th": effective_heat_cost,
            "remaining_stored_heat_MWh": remaining,
            "weighted_average_storage_cost_EUR_per_MWh_th": weighted_average,
            **{
                column: float(source_inventory.get(source, 0.0))
                for source, column in STORAGE_SOURCE_COLUMNS.items()
            },
        }
        self.rows.loc[len(self.rows)] = record

    def build_from_dispatch_results(
        self,
        dispatch_results: pd.DataFrame,
        plants: list[SteamGenerationPlant],
    ) -> None:
        plant_by_name = {plant.name: plant for plant in plants}
        for timestamp, row in dispatch_results.sort_index().iterrows():
            plant_name = str(row["plant_name"])
            plant = plant_by_name[plant_name]
            if plant.etes is None:
                continue
            electricity_volume = float(row["etes_charge_MWh"])
            stored_heat_added = electricity_volume * plant.etes.efficiency_charge
            self.record_storage_event(
                datetime=timestamp,
                plant_name=plant_name,
                source_market="day_ahead" if electricity_volume > 1e-12 else "",
                electricity_price_eur_per_mwh=float(row["day_ahead_price_EUR_per_MWh"]),
                electricity_volume_mwh=electricity_volume,
                stored_heat_added_mwh=stored_heat_added,
                remaining_stored_heat_mwh=float(row["etes_soc_MWh"]),
            )

    def to_dataframe(self) -> pd.DataFrame:
        return self.rows.copy()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows.to_csv(path, index=False)
        return path


def _normalise_source_market(source_market: str) -> str:
    value = str(source_market).strip().lower()
    if value in {"da", "day-ahead", "day_ahead"}:
        return "day_ahead"
    if value in {"idc", "intraday", "intraday_continuous"}:
        return "intraday_continuous"
    if value in {"afrr", "afrr_energy", "afrr energy"}:
        return "afrr_energy"
    if not value:
        return "other"
    return value if value in STORAGE_SOURCE_COLUMNS else "other"


def _reconcile_source_inventory(
    source_inventory: dict[str, float],
    remaining_total_mwh: float,
) -> dict[str, float]:
    reconciled = {
        source: max(float(source_inventory.get(source, 0.0)), 0.0)
        for source in STORAGE_SOURCE_COLUMNS
    }
    current_total = sum(reconciled.values())
    if remaining_total_mwh <= 1e-12:
        return {source: 0.0 for source in STORAGE_SOURCE_COLUMNS}
    if current_total <= 1e-12:
        reconciled["other"] = remaining_total_mwh
        return reconciled
    scale = remaining_total_mwh / current_total
    return {source: value * scale for source, value in reconciled.items()}
