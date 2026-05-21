# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd

from flexi_mod.ledgers.market_ledger import MarketLedger


def test_market_ledger_initialises_and_updates_da(tmp_path: Path) -> None:
    datetimes = pd.date_range("2025-01-01", periods=2, freq="15min")
    ledger = MarketLedger()
    ledger.initialise(datetimes, ["plant_1"])
    ledger.record_day_ahead_positions(
        plant_name="plant_1",
        datetimes=datetimes,
        day_ahead_position_mwh_el=[1.0, 2.0],
        day_ahead_price_eur_per_mwh_el=[50.0, 55.0],
    )

    frame = ledger.to_dataframe()
    assert frame["day_ahead_position_MWh_el"].tolist() == [1.0, 2.0]
    assert frame["day_ahead_price_EUR_per_MWh_el"].tolist() == [50.0, 55.0]
    assert "scheduled_electricity_procurement_MWh_el" in frame.columns

    path = ledger.save(tmp_path / "market_ledger.csv")
    assert path.exists()


def test_market_ledger_schema_excludes_internal_afrr_diagnostics() -> None:
    ledger = MarketLedger()
    ledger.initialise(pd.date_range("2025-01-01", periods=1, freq="15min"), ["plant_1"])

    frame = ledger.to_dataframe()

    removed_columns = {
        "planned_" + "electricity_MWh",
        "afrr_energy_" + "price_clean",
        "afrr_raw_" + "system_activation",
        "afrr_raw_" + "system_activation_MWh",
        "afrr_system_activation_" + "MWh_clean",
        "afrr_down_" + "system_activation_MWh_clean",
        "afrr_data_" + "quality_flag",
    }
    assert removed_columns.isdisjoint(frame.columns)
    assert "afrr_system_activation_MWh_el" in frame.columns


def test_market_ledger_schema_uses_energy_economics_terms() -> None:
    ledger = MarketLedger()
    ledger.initialise(pd.date_range("2025-01-01", periods=1, freq="15min"), ["plant_1"])

    frame = ledger.to_dataframe()

    expected_columns = {
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
    }
    assert set(frame.columns) == expected_columns
