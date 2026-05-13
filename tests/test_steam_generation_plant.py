# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from etes_market_model.config.case_config import CaseConfig
from etes_market_model.data.data_loader import DataLoader
from etes_market_model.plants.steam_generation_plant import (
    AFRRDownSignals,
    DispatchSignals,
    IDCAdjustmentSignals,
    SteamGenerationPlant,
)
from etes_market_model.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DE"


def test_steam_generation_plant_builds_from_plants_csv() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plants_df = DataLoader(config, input_dir=CASE_DIR).load_plants()
    plants = SteamGenerationPlant.from_plants_dataframe(plants_df)

    assert len(plants) == 1
    assert set(plants[0].components) == {"thermal_storage", "boiler"}
    assert plants[0].etes.max_capacity_mwh > 0
    assert plants[0].gas_boiler.efficiency == 0.9


def test_steam_generation_plant_short_horizon_solves() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    loader = DataLoader(config, input_dir=CASE_DIR)
    plants_df = loader.load_plants()
    plant = SteamGenerationPlant.from_plants_dataframe(plants_df)[0]
    strategy = HybridETESGasStrategy(config)
    forecasts = pd.DataFrame(
        {
            "plant_1_heat_demand": [2.0] * 8,
            "DE_DA_price": [120.0] * 8,
            "natural_gas_price": [80.0] * 8,
            "co2_price": [0.0] * 8,
        },
        index=pd.date_range("2025-01-01 00:00", periods=8, freq="15min"),
    )
    price_col = config.market_signal("day_ahead", "price")
    benchmark = strategy.calculate_gas_based_heat_cost(plant, forecasts)
    signals = DispatchSignals(
        electricity_price_col=price_col,
        gas_price_col="natural_gas_price",
        co2_price_col="co2_price",
        gas_benchmark_eur_per_mwh_th=benchmark,
        charge_allowed=pd.Series(False, index=forecasts.index),
    )

    result = plant.solve_horizon(config, forecasts, signals)

    required_columns = {
        "etes_charge_MWh",
        "etes_discharge_MWh",
        "etes_soc_MWh",
        "gas_heat_MWh",
        "electricity_consumption_MWh",
        "unmet_heat_MWh",
    }
    assert required_columns.issubset(result.columns)
    assert result["unmet_heat_MWh"].sum() == 0
    assert (result["gas_heat_MWh"] >= result["heat_demand_MWh"] - 1e-8).all()


def test_steam_generation_plant_short_idc_adjustment_horizon_solves() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    loader = DataLoader(config, input_dir=CASE_DIR)
    plant = SteamGenerationPlant.from_plants_dataframe(loader.load_plants())[0]
    strategy = HybridETESGasStrategy(config)

    index = pd.date_range("2025-01-01 00:00", periods=4, freq="15min")
    forecasts = pd.DataFrame(
        {
            "plant_1_heat_demand": [2.0] * 4,
            "DE_DA_price": [10.0] * 4,
            "DE_ID3_price": [120.0] * 4,
            "natural_gas_price": [80.0] * 4,
        },
        index=index,
    )
    gas_benchmark = strategy.calculate_gas_based_heat_cost(plant, forecasts)
    electricity_benchmark = strategy.calculate_electricity_trading_benchmark(
        plant,
        gas_benchmark,
    )
    da_position = pd.Series([0.8] * 4, index=index)
    signals = IDCAdjustmentSignals(
        da_price_col="DE_DA_price",
        idc_price_col="DE_ID3_price",
        gas_price_col="natural_gas_price",
        da_position_mwh=da_position,
        idc_buy_upper_bound_mwh=pd.Series([0.0] * 4, index=index),
        idc_sell_upper_bound_mwh=da_position,
        gas_benchmark_eur_per_mwh_th=gas_benchmark,
        electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
    )

    result = plant.solve_intraday_adjustment_horizon(config, forecasts, signals)

    required_columns = {
        "DA_position_MWh",
        "IDC_buy_MWh",
        "IDC_sell_MWh",
        "final_planned_electricity_MWh",
        "actual_electricity_consumption_MWh",
    }
    assert required_columns.issubset(result.columns)
    assert (result["IDC_sell_MWh"] <= result["DA_position_MWh"] + 1e-8).all()
    expected = result["DA_position_MWh"] + result["IDC_buy_MWh"] - result["IDC_sell_MWh"]
    assert result["final_planned_electricity_MWh"].to_numpy() == pytest.approx(expected.to_numpy())
    assert result["unmet_heat_MWh"].sum() == 0


def test_steam_generation_plant_short_afrr_down_horizon_solves() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteamGenerationPlant.from_plants_dataframe(
        DataLoader(config, input_dir=CASE_DIR).load_plants()
    )[0]
    strategy = HybridETESGasStrategy(config)

    index = pd.date_range("2025-01-01 00:00", periods=4, freq="15min")
    forecasts = pd.DataFrame(
        {
            "plant_1_heat_demand": [2.0] * 4,
            "DE_DA_price": [120.0] * 4,
            "DE_ID3_price": [75.0] * 4,
            "aFRR_energy_down_price": [20.0] * 4,
            "natural_gas_price": [80.0] * 4,
        },
        index=index,
    )
    gas_benchmark = strategy.calculate_gas_based_heat_cost(plant, forecasts)
    electricity_benchmark = strategy.calculate_electricity_trading_benchmark(
        plant,
        gas_benchmark,
    )
    zero = pd.Series([0.0] * 4, index=index)
    activation = pd.Series([0.4] * 4, index=index)
    signals = AFRRDownSignals(
        da_price_col="DE_DA_price",
        idc_price_col="DE_ID3_price",
        gas_price_col="natural_gas_price",
        da_position_mwh=zero,
        idc_buy_mwh=zero,
        idc_sell_mwh=zero,
        final_planned_electricity_mwh=zero,
        afrr_energy_down_price_raw=pd.Series([20.0] * 4, index=index),
        afrr_energy_down_price_clean=pd.Series([20.0] * 4, index=index),
        afrr_raw_system_activation=pd.Series([1.6] * 4, index=index),
        afrr_raw_system_activation_mwh=activation,
        afrr_down_system_activation_mwh_clean=activation,
        afrr_data_quality_flag=pd.Series(["valid_activation"] * 4, index=index),
        afrr_energy_bid_mwh=activation,
        afrr_energy_activated_mwh=activation,
        gas_benchmark_eur_per_mwh_th=gas_benchmark,
        electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
    )

    result = plant.solve_afrr_down_horizon(config, forecasts, signals)

    assert result["afrr_energy_activated_MWh"].sum() == pytest.approx(1.6)
    assert result["actual_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        (result["final_planned_electricity_MWh"] + result["afrr_energy_activated_MWh"]).to_numpy()
    )
    assert result["etes_charge_MWh"].to_numpy() == pytest.approx(
        result["actual_electricity_consumption_MWh"].to_numpy()
    )
    assert result["unmet_heat_MWh"].sum() == 0
