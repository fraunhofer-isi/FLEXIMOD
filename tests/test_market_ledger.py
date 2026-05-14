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
    ledger.add_or_update_da_positions(
        plant_name="plant_1",
        datetimes=datetimes,
        da_position_mwh=[1.0, 2.0],
        da_price=[50.0, 55.0],
    )

    frame = ledger.to_dataframe()
    assert frame["DA_position_MWh"].tolist() == [1.0, 2.0]
    assert frame["DA_price"].tolist() == [50.0, 55.0]
    assert "final_planned_electricity_MWh" in frame.columns

    path = ledger.save(tmp_path / "market_ledger.csv")
    assert path.exists()


def test_market_ledger_schema_excludes_internal_afrr_diagnostics() -> None:
    ledger = MarketLedger()
    ledger.initialise(pd.date_range("2025-01-01", periods=1, freq="15min"), ["plant_1"])

    frame = ledger.to_dataframe()

    removed_columns = {
        "planned_" + "electricity_MWh",
        "afrr_capacity_" + "reserved_MW",
        "afrr_capacity_" + "price",
        "afrr_energy_" + "price_clean",
        "afrr_raw_" + "system_activation",
        "afrr_raw_" + "system_activation_MWh",
        "afrr_system_activation_" + "MWh_clean",
        "afrr_down_" + "system_activation_MWh_clean",
        "afrr_data_" + "quality_flag",
    }
    assert removed_columns.isdisjoint(frame.columns)
    assert "afrr_system_activation_MWh" in frame.columns
