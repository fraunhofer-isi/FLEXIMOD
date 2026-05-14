# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from flexi_mod.ledgers.storage_cost_ledger import StorageCostLedger
from flexi_mod.visualisation.analytics import calculate_summary_indicators


def test_analytics_calculates_da_only_summary() -> None:
    datetimes = pd.date_range("2025-01-01", periods=2, freq="15min")
    dispatch = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_name": ["plant_1", "plant_1"],
            "heat_demand_MWh": [1.0, 1.0],
            "gas_heat_MWh": [1.0, 0.5],
            "etes_discharge_MWh": [0.0, 0.5],
            "etes_charge_MWh": [0.6, 0.0],
            "etes_soc_MWh": [0.5, 0.0],
            "electricity_consumption_MWh": [0.6, 0.0],
            "electricity_cost_EUR": [30.0, 0.0],
            "gas_cost_EUR": [50.0, 25.0],
            "co2_cost_EUR": [0.0, 0.0],
            "operating_cost_EUR": [80.0, 25.0],
            "unmet_heat_MWh": [0.0, 0.0],
        }
    )
    market = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_name": ["plant_1", "plant_1"],
            "DA_position_MWh": [0.6, 0.0],
            "actual_electricity_consumption_MWh": [0.6, 0.0],
        }
    )

    summary = calculate_summary_indicators(dispatch, market)

    assert summary["total_heat_demand_MWh"].iloc[0] == 2.0
    assert summary["total_DA_electricity_MWh"].iloc[0] == 0.6
    assert summary["total_net_operating_cost_EUR"].iloc[0] == 105.0


def test_storage_cost_ledger_tracks_source_inventory() -> None:
    ledger = StorageCostLedger()
    ledger.record_storage_event(
        datetime="2025-01-01 00:00",
        plant_name="plant_1",
        source_market="day_ahead",
        electricity_price_eur_per_mwh=40.0,
        electricity_volume_mwh=1.0,
        stored_heat_added_mwh=0.9,
        remaining_stored_heat_mwh=0.9,
    )

    frame = ledger.to_dataframe()

    assert "remaining_stored_heat_day_ahead_MWh" in frame.columns
    assert frame["remaining_stored_heat_day_ahead_MWh"].iloc[0] == 0.9
