# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.ledgers.storage_cost_ledger import StorageCostLedger
from flexi_mod.visualisation.analytics import calculate_summary_indicators
from flexi_mod.visualisation.plots import (
    plot_afrr_capacity_and_energy,
    plot_afrr_capacity_price_and_reserve,
    plot_idc_sell_source_and_compensation,
    plot_procurement_and_capacity_headroom,
    plot_sequential_market_position_evolution,
    plot_stagewise_gas_replacement,
)


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
        }
    )
    market = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_name": ["plant_1", "plant_1"],
            "day_ahead_position_MWh_el": [0.6, 0.0],
            "actual_electricity_consumption_MWh_el": [0.6, 0.0],
        }
    )

    summary = calculate_summary_indicators(dispatch, market)

    assert summary["total_heat_demand_MWh"].iloc[0] == 2.0
    assert summary["total_DA_electricity_MWh"].iloc[0] == 0.6
    assert summary["total_net_operating_cost_EUR"].iloc[0] == 105.0
    assert "total_unmet_heat_MWh" not in summary.columns
    assert "total_excess_heat_MWh" not in summary.columns


def test_storage_cost_ledger_tracks_procurement_market_inventory() -> None:
    ledger = StorageCostLedger()
    ledger.record_charge(
        datetime="2025-01-01 00:00",
        plant_name="plant_1",
        procurement_market="day_ahead",
        electricity_price_eur_per_mwh_el=40.0,
        electricity_procured_mwh_el=1.0,
        charged_heat_mwh_th=0.9,
        thermal_inventory_mwh_th=0.9,
    )

    frame = ledger.to_dataframe()

    assert "thermal_inventory_day_ahead_MWh_th" in frame.columns
    assert frame["thermal_inventory_day_ahead_MWh_th"].iloc[0] == 0.9


def test_sequential_market_position_plot_includes_gas_boiler(tmp_path: Path) -> None:
    datetimes = pd.date_range("2025-01-01", periods=4, freq="15min")
    market = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_name": ["plant_1"] * 4,
            "day_ahead_position_MWh_el": [1.0, 1.0, 0.5, 0.5],
            "intraday_buy_MWh_el": [0.0, 0.2, 0.0, 0.0],
            "intraday_sell_MWh_el": [0.0, 0.0, 0.1, 0.0],
            "scheduled_electricity_procurement_MWh_el": [1.0, 1.2, 0.4, 0.5],
            "afrr_energy_activated_MWh_el": [0.0, 0.0, 0.2, 0.0],
            "actual_electricity_consumption_MWh_el": [1.0, 1.2, 0.6, 0.5],
            "gas_heat_output_MWh_th": [2.0, 1.8, 2.2, 2.0],
        }
    )
    dispatch = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_name": ["plant_1"] * 4,
            "gas_heat_MWh": [2.0, 1.8, 2.2, 2.0],
        }
    )

    created = plot_sequential_market_position_evolution(market, dispatch, tmp_path)

    assert created == [tmp_path / "11_sequential_market_position_evolution.png"]
    assert created[0].exists()


def test_sequential_market_position_plot_saves_without_gas_boiler(tmp_path: Path) -> None:
    datetimes = pd.date_range("2025-01-01", periods=2, freq="15min")
    market = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_name": ["plant_1"] * 2,
            "day_ahead_position_MWh_el": [0.5, 0.4],
            "scheduled_electricity_procurement_MWh_el": [0.5, 0.4],
            "actual_electricity_consumption_MWh_el": [0.5, 0.4],
        }
    )

    with pytest.warns(UserWarning, match="gas_heat_output_MWh_th"):
        created = plot_sequential_market_position_evolution(market, pd.DataFrame(), tmp_path)

    assert created == [tmp_path / "11_sequential_market_position_evolution.png"]
    assert created[0].exists()


def test_idc_sell_source_and_compensation_plot_saves(tmp_path: Path) -> None:
    datetimes = pd.date_range("2025-01-01", periods=4, freq="15min")
    market = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_name": ["plant_1"] * 4,
            "day_ahead_position_MWh_el": [1.0, 1.0, 0.5, 0.5],
            "intraday_buy_MWh_el": [0.0, 0.0, 0.0, 0.0],
            "intraday_sell_MWh_el": [0.0, 0.4, 0.0, 0.2],
            "afrr_energy_activated_MWh_el": [0.0, 0.1, 0.0, 0.0],
            "gas_heat_output_MWh_th": [0.5, 1.0, 0.7, 1.2],
        }
    )
    dispatch = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_name": ["plant_1"] * 4,
            "heat_demand_MWh": [1.0, 1.2, 1.0, 1.2],
            "gas_heat_MWh": [0.5, 1.0, 0.7, 1.2],
            "etes_discharge_MWh": [0.5, 0.2, 0.3, 0.0],
        }
    )

    created = plot_idc_sell_source_and_compensation(market, dispatch, tmp_path)

    assert created == [tmp_path / "12_idc_sell_source_and_compensation.png"]
    assert created[0].exists()


def test_stagewise_gas_replacement_plot_saves(tmp_path: Path) -> None:
    datetimes = pd.date_range("2025-01-01", periods=4, freq="15min")
    dispatch = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_name": ["plant_1"] * 4,
            "heat_demand_MWh": [1.2, 1.2, 1.2, 1.2],
            "gas_heat_after_day_ahead_MWh": [1.0, 0.8, 1.2, 1.0],
            "gas_heat_after_intraday_MWh": [0.9, 1.0, 1.2, 0.8],
            "gas_heat_MWh": [0.7, 1.0, 0.9, 0.8],
        }
    )

    created = plot_stagewise_gas_replacement(dispatch, tmp_path)

    assert created == [tmp_path / "13_stagewise_gas_replacement.png"]
    assert created[0].exists()


def test_afrr_capacity_plots_save_when_capacity_is_present(tmp_path: Path) -> None:
    datetimes = pd.date_range("2025-01-01", periods=4, freq="15min")
    market = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_name": ["plant_1"] * 4,
            "day_ahead_position_MWh_el": [0.5, 0.5, 0.0, 0.0],
            "intraday_buy_MWh_el": [0.0, 0.0, 0.0, 0.0],
            "intraday_sell_MWh_el": [0.0, 0.0, 0.0, 0.0],
            "scheduled_electricity_procurement_MWh_el": [0.5, 0.5, 0.0, 0.0],
            "afrr_capacity_down_price_EUR_per_MW_h": [80.0] * 4,
            "afrr_capacity_reserved_MW": [2.0] * 4,
            "afrr_capacity_reserved_MWh": [0.5] * 4,
            "available_charge_headroom_after_schedule_MWh": [1.25, 1.25, 1.75, 1.75],
            "afrr_system_activation_MWh": [0.2, 0.0, 0.3, 0.1],
            "afrr_energy_activated_MWh_el": [0.2, 0.0, 0.3, 0.1],
            "afrr_energy_down_price_EUR_per_MWh": [5.0] * 4,
            "actual_electricity_consumption_MWh_el": [0.7, 0.5, 0.3, 0.1],
        }
    )

    created = []
    created.extend(plot_afrr_capacity_price_and_reserve(market, tmp_path))
    created.extend(plot_procurement_and_capacity_headroom(market, tmp_path))
    created.extend(plot_afrr_capacity_and_energy(market, tmp_path))

    assert created == [
        tmp_path / "14_afrr_capacity_price_and_reserve.png",
        tmp_path / "15_electricity_procurement_and_capacity_headroom.png",
        tmp_path / "16_afrr_capacity_and_energy.png",
    ]
    assert all(path.exists() for path in created)
