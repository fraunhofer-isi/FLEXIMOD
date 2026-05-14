# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant

BASE_COLUMNS = [
    "datetime",
    "plant_name",
]

CHARGE_EVENT_COLUMNS = [
    "source_market",
    "electricity_price_EUR_per_MWh",
    "electricity_volume_MWh",
    "stored_heat_added_MWh",
    "effective_heat_cost_EUR_per_MWh_th",
]

STORAGE_STATE_COLUMNS = [
    "remaining_stored_heat_MWh",
    "weighted_average_storage_cost_EUR_per_MWh_th",
]

SOURCE_INVENTORY_COLUMNS = {
    "day_ahead": "remaining_stored_heat_day_ahead_MWh",
    "intraday_continuous": "remaining_stored_heat_IDC_MWh",
    "afrr_energy": "remaining_stored_heat_afrr_energy_MWh",
    "other": "remaining_stored_heat_other_MWh",
}

STORAGE_COST_LEDGER_COLUMNS = (
    BASE_COLUMNS
    + CHARGE_EVENT_COLUMNS
    + STORAGE_STATE_COLUMNS
    + list(SOURCE_INVENTORY_COLUMNS.values())
)

EMPTY_SOURCE = "none"


@dataclass(frozen=True)
class StorageChargeEvent:
    """One source-market contribution to ETES charging in one dispatch step."""

    source_market: str
    electricity_price_eur_per_mwh: float
    electricity_volume_mwh: float
    stored_heat_added_mwh: float

    @property
    def source(self) -> str:
        return _normalise_source_market(self.source_market)

    @property
    def electricity_cost_eur(self) -> float:
        return self.electricity_price_eur_per_mwh * self.electricity_volume_mwh

    @property
    def effective_heat_cost_eur_per_mwh_th(self) -> float | pd.NA:
        if self.stored_heat_added_mwh <= 1e-12:
            return pd.NA
        return self.electricity_cost_eur / self.stored_heat_added_mwh


class StorageCostLedger:
    """Economic cost and source attribution tracker for stored ETES heat."""

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
        previous_remaining, _ = self._plant_state(plant_name)
        remaining = (
            previous_remaining + stored_heat_added_mwh
            if remaining_stored_heat_mwh is None
            else remaining_stored_heat_mwh
        )
        self.record_storage_step(
            datetime=datetime,
            plant_name=plant_name,
            source_events=[
                StorageChargeEvent(
                    source_market=source_market,
                    electricity_price_eur_per_mwh=electricity_price_eur_per_mwh,
                    electricity_volume_mwh=electricity_volume_mwh,
                    stored_heat_added_mwh=stored_heat_added_mwh,
                )
            ],
            remaining_stored_heat_mwh=remaining,
        )

    def record_storage_step(
        self,
        datetime: pd.Timestamp,
        plant_name: str,
        source_events: list[StorageChargeEvent | tuple[str, float, float, float]],
        remaining_stored_heat_mwh: float,
    ) -> None:
        events = _normalise_events(source_events)
        remaining = max(float(remaining_stored_heat_mwh), 0.0)

        previous_remaining, previous_average_cost = self._plant_state(plant_name)
        previous_inventory = self._source_inventory(plant_name)
        average_cost = _weighted_average_cost(
            previous_remaining=previous_remaining,
            previous_average_cost=previous_average_cost,
            added_cost=sum(event.electricity_cost_eur for event in events),
            added_heat=sum(event.stored_heat_added_mwh for event in events),
            remaining_after_dispatch=remaining,
        )
        source_inventory = _updated_source_inventory(
            previous_inventory=previous_inventory,
            events=events,
            remaining_total_mwh=remaining,
        )

        self._state_by_plant[plant_name] = (remaining, average_cost)
        self._source_inventory_by_plant[plant_name] = source_inventory
        self._append_step_records(datetime, plant_name, events, remaining, average_cost)

    def build_from_dispatch_results(
        self,
        dispatch_results: pd.DataFrame,
        plants: list[SteamGenerationPlant],
    ) -> None:
        plant_by_name = {plant.name: plant for plant in plants}
        for timestamp, row in dispatch_results.sort_index().iterrows():
            plant_name = str(row["plant_name"])
            plant = plant_by_name[plant_name]
            events = _storage_events_from_dispatch_row(row, plant)
            self.record_storage_step(
                datetime=timestamp,
                plant_name=plant_name,
                source_events=events,
                remaining_stored_heat_mwh=float(row["etes_soc_MWh"]),
            )

    def to_dataframe(self) -> pd.DataFrame:
        return self.rows.copy()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows.to_csv(path, index=False)
        return path

    def _plant_state(self, plant_name: str) -> tuple[float, float]:
        return self._state_by_plant.get(plant_name, (0.0, 0.0))

    def _source_inventory(self, plant_name: str) -> dict[str, float]:
        return self._source_inventory_by_plant.get(plant_name, _empty_source_inventory())

    def _append_step_records(
        self,
        datetime: pd.Timestamp,
        plant_name: str,
        events: list[StorageChargeEvent],
        remaining_stored_heat_mwh: float,
        weighted_average_cost: float,
    ) -> None:
        row_events = events if events else [_empty_storage_event()]
        source_inventory = self._source_inventory(plant_name)
        for event in row_events:
            self.rows.loc[len(self.rows)] = {
                "datetime": pd.Timestamp(datetime),
                "plant_name": plant_name,
                "source_market": event.source_market,
                "electricity_price_EUR_per_MWh": event.electricity_price_eur_per_mwh,
                "electricity_volume_MWh": event.electricity_volume_mwh,
                "stored_heat_added_MWh": event.stored_heat_added_mwh,
                "effective_heat_cost_EUR_per_MWh_th": (event.effective_heat_cost_eur_per_mwh_th),
                "remaining_stored_heat_MWh": remaining_stored_heat_mwh,
                "weighted_average_storage_cost_EUR_per_MWh_th": weighted_average_cost,
                **{
                    column: float(source_inventory.get(source, 0.0))
                    for source, column in SOURCE_INVENTORY_COLUMNS.items()
                },
            }


def _storage_events_from_dispatch_row(
    row: pd.Series,
    plant: SteamGenerationPlant,
) -> list[StorageChargeEvent]:
    actual_charge = max(float(row["etes_charge_MWh"]), 0.0)
    if actual_charge <= 1e-12:
        return []

    source_volumes = _source_electricity_volumes(row, actual_charge)
    return [
        StorageChargeEvent(
            source_market=source,
            electricity_price_eur_per_mwh=_source_price(row, source),
            electricity_volume_mwh=volume,
            stored_heat_added_mwh=volume * plant.etes.efficiency_charge,
        )
        for source, volume in source_volumes.items()
        if volume > 1e-12
    ]


def _source_electricity_volumes(row: pd.Series, actual_charge_mwh: float) -> dict[str, float]:
    da_source = max(float(row.get("DA_position_MWh", actual_charge_mwh)), 0.0)
    idc_sell = max(float(row.get("IDC_sell_MWh", 0.0)), 0.0)
    volumes = {
        "day_ahead": max(da_source - idc_sell, 0.0),
        "intraday_continuous": max(float(row.get("IDC_buy_MWh", 0.0)), 0.0),
        "afrr_energy": max(float(row.get("afrr_energy_activated_MWh", 0.0)), 0.0),
    }
    total_source_volume = sum(volumes.values())
    if total_source_volume > actual_charge_mwh and total_source_volume > 1e-12:
        scale = actual_charge_mwh / total_source_volume
        return {source: volume * scale for source, volume in volumes.items()}
    if total_source_volume <= 1e-12:
        volumes["other"] = actual_charge_mwh
    return volumes


def _source_price(row: pd.Series, source_market: str) -> float:
    price_columns = {
        "day_ahead": "day_ahead_price_EUR_per_MWh",
        "intraday_continuous": "IDC_price_EUR_per_MWh",
        "afrr_energy": "afrr_energy_price_EUR_per_MWh",
        "other": "day_ahead_price_EUR_per_MWh",
    }
    return float(row.get(price_columns[source_market], 0.0))


def _normalise_events(
    source_events: list[StorageChargeEvent | tuple[str, float, float, float]],
) -> list[StorageChargeEvent]:
    events = [
        event
        if isinstance(event, StorageChargeEvent)
        else StorageChargeEvent(
            source_market=event[0],
            electricity_price_eur_per_mwh=float(event[1]),
            electricity_volume_mwh=float(event[2]),
            stored_heat_added_mwh=float(event[3]),
        )
        for event in source_events
    ]
    return [
        event
        for event in events
        if event.electricity_volume_mwh > 1e-12 and event.stored_heat_added_mwh > 1e-12
    ]


def _weighted_average_cost(
    previous_remaining: float,
    previous_average_cost: float,
    added_cost: float,
    added_heat: float,
    remaining_after_dispatch: float,
) -> float:
    inventory_after_charge = previous_remaining + added_heat
    if remaining_after_dispatch <= 1e-12:
        return 0.0
    if inventory_after_charge > 1e-12 and added_heat > 1e-12:
        return (previous_remaining * previous_average_cost + added_cost) / inventory_after_charge
    return previous_average_cost if previous_remaining > 1e-12 else 0.0


def _updated_source_inventory(
    previous_inventory: dict[str, float],
    events: list[StorageChargeEvent],
    remaining_total_mwh: float,
) -> dict[str, float]:
    inventory = {
        source: max(float(previous_inventory.get(source, 0.0)), 0.0)
        for source in SOURCE_INVENTORY_COLUMNS
    }
    for event in events:
        inventory[event.source] = inventory.get(event.source, 0.0) + event.stored_heat_added_mwh
    return _reconcile_source_inventory(inventory, remaining_total_mwh)


def _reconcile_source_inventory(
    source_inventory: dict[str, float],
    remaining_total_mwh: float,
) -> dict[str, float]:
    if remaining_total_mwh <= 1e-12:
        return _empty_source_inventory()

    current_total = sum(source_inventory.values())
    if current_total <= 1e-12:
        inventory = _empty_source_inventory()
        inventory["other"] = remaining_total_mwh
        return inventory

    scale = remaining_total_mwh / current_total
    return {source: value * scale for source, value in source_inventory.items()}


def _normalise_source_market(source_market: str) -> str:
    value = str(source_market).strip().lower()
    if value in {"da", "day-ahead", "day_ahead"}:
        return "day_ahead"
    if value in {"idc", "intraday", "intraday_continuous"}:
        return "intraday_continuous"
    if value in {"afrr", "afrr_energy", "afrr energy"}:
        return "afrr_energy"
    if not value or value == EMPTY_SOURCE:
        return "other"
    return value if value in SOURCE_INVENTORY_COLUMNS else "other"


def _empty_source_inventory() -> dict[str, float]:
    return {source: 0.0 for source in SOURCE_INVENTORY_COLUMNS}


def _empty_storage_event() -> StorageChargeEvent:
    return StorageChargeEvent(
        source_market=EMPTY_SOURCE,
        electricity_price_eur_per_mwh=0.0,
        electricity_volume_mwh=0.0,
        stored_heat_added_mwh=0.0,
    )
