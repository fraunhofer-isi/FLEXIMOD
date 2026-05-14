# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

from flexi_mod.ledgers.storage_cost_ledger import StorageCostLedger


def test_weighted_average_inventory_cost_updates() -> None:
    ledger = StorageCostLedger()
    ledger.record_charge(
        datetime="2025-01-01 00:00",
        plant_name="plant_1",
        procurement_market="day_ahead",
        electricity_price_eur_per_mwh_el=40.0,
        electricity_procured_mwh_el=1.0,
        charged_heat_mwh_th=0.8,
    )
    ledger.record_charge(
        datetime="2025-01-01 00:15",
        plant_name="plant_1",
        procurement_market="day_ahead",
        electricity_price_eur_per_mwh_el=80.0,
        electricity_procured_mwh_el=1.0,
        charged_heat_mwh_th=0.8,
    )

    frame = ledger.to_dataframe()
    assert round(frame["weighted_average_inventory_cost_EUR_per_MWh_th"].iloc[-1], 6) == 75.0


def test_storage_cost_ledger_saves(tmp_path: Path) -> None:
    ledger = StorageCostLedger()
    ledger.record_charge(
        datetime="2025-01-01 00:00",
        plant_name="plant_1",
        procurement_market="day_ahead",
        electricity_price_eur_per_mwh_el=40.0,
        electricity_procured_mwh_el=1.0,
        charged_heat_mwh_th=0.8,
    )
    path = ledger.save(tmp_path / "storage_cost_ledger.csv")
    assert path.exists()


def test_storage_cost_ledger_schema_uses_energy_economics_terms() -> None:
    ledger = StorageCostLedger()
    ledger.record_storage_step(
        datetime="2025-01-01 00:00",
        plant_name="plant_1",
        charges=[],
        thermal_inventory_mwh_th=0.0,
    )

    frame = ledger.to_dataframe()

    expected_columns = {
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
        "thermal_inventory_day_ahead_MWh_th",
        "thermal_inventory_intraday_continuous_MWh_th",
        "thermal_inventory_afrr_energy_MWh_th",
        "thermal_inventory_other_MWh_th",
    }
    assert set(frame.columns) == expected_columns
    assert frame["procurement_market"].iloc[0] == "no_charge"
