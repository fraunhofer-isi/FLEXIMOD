# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

from etes_market_model.ledgers.storage_cost_ledger import StorageCostLedger


def test_weighted_average_storage_cost_updates() -> None:
    ledger = StorageCostLedger()
    ledger.record_storage_event(
        datetime="2025-01-01 00:00",
        plant_name="plant_1",
        source_market="day_ahead",
        electricity_price_eur_per_mwh=40.0,
        electricity_volume_mwh=1.0,
        stored_heat_added_mwh=0.8,
    )
    ledger.record_storage_event(
        datetime="2025-01-01 00:15",
        plant_name="plant_1",
        source_market="day_ahead",
        electricity_price_eur_per_mwh=80.0,
        electricity_volume_mwh=1.0,
        stored_heat_added_mwh=0.8,
    )

    frame = ledger.to_dataframe()
    assert round(frame["weighted_average_storage_cost_EUR_per_MWh_th"].iloc[-1], 6) == 75.0


def test_storage_cost_ledger_saves(tmp_path: Path) -> None:
    ledger = StorageCostLedger()
    ledger.record_storage_event(
        datetime="2025-01-01 00:00",
        plant_name="plant_1",
        source_market="day_ahead",
        electricity_price_eur_per_mwh=40.0,
        electricity_volume_mwh=1.0,
        stored_heat_added_mwh=0.8,
    )
    path = ledger.save(tmp_path / "storage_cost_ledger.csv")
    assert path.exists()
