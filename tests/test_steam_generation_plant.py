# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.markets.afrr_energy import BalancingEnergyActivation
from flexi_mod.markets.day_ahead import DayAheadMarket, DayAheadPosition
from flexi_mod.markets.electricity_settlement import (
    ElectricityMarketRequest,
    ElectricityMarketStage,
)
from flexi_mod.markets.intraday_continuous import IntradayAdjustment
from flexi_mod.plants.factory import build_plants
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant
from flexi_mod.simulation.market_stages import (
    MarketStageContext,
    MarketStageInstruction,
    MarketStageState,
)
from flexi_mod.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DA_ID_buy"


def test_steam_generation_plant_builds_from_plants_csv() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plants_df = DataLoader(config, input_dir=CASE_DIR).load_plants()
    plants = build_plants(plants_df)

    assert len(plants) == 1
    assert set(plants[0].components) == {"thermal_storage", "boiler"}
    assert plants[0].etes.max_capacity_mwh > 0
    assert plants[0].gas_boiler.efficiency == 0.85


def test_steam_generation_plant_builds_from_common_loader_input() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    loader = DataLoader(config, input_dir=CASE_DIR)
    plant_input = loader.load_plant_inputs()["steam_plant"][0]

    plant = SteamGenerationPlant.create(plant_input)

    assert plant.name == plant_input.name
    assert set(plant.components) == {"thermal_storage", "boiler"}
    assert plant.required_forecast_columns() == {"plant_1_heat_demand"}


def test_steam_generation_plant_exposes_modeler_facing_build_sequence() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = build_plants(DataLoader(config, input_dir=CASE_DIR).load_plants())[0]
    strategy = HybridETESGasStrategy(config)
    index = pd.date_range("2025-01-01 00:00", periods=4, freq="15min")
    forecasts = pd.DataFrame(
        {
            "plant_1_heat_demand": [2.0] * 4,
            "DE_DA_price": [120.0] * 4,
            "natural_gas_price": [80.0] * 4,
            "co2_price": [0.0] * 4,
        },
        index=index,
    )
    position = DayAheadPosition(
        electricity_price_col="DE_DA_price",
        gas_price_col="natural_gas_price",
        co2_price_col="co2_price",
        gas_benchmark_eur_per_mwh_th=strategy.calculate_gas_based_heat_cost(
            plant,
            forecasts,
        ),
        charge_allowed=pd.Series(False, index=index),
    )

    model = plant.build_model(ElectricityMarketStage.DAY_AHEAD, config, forecasts, position)

    assert list(model.T) == [0, 1, 2, 3]
    assert set(model.technology_blocks) == {"thermal_storage", "boiler"}
    assert hasattr(model, "market_position_matches_physical_consumption")
    assert hasattr(model, "objective")


def test_electricity_market_request_separates_policy_from_plant_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = build_plants(DataLoader(config, input_dir=CASE_DIR).load_plants())[0]
    strategy = HybridETESGasStrategy(config)
    index = pd.date_range("2025-01-01 00:00", periods=2, freq="15min")
    forecasts = pd.DataFrame(
        {
            "plant_1_heat_demand": [2.0, 2.0],
            "DE_DA_price": [120.0, 120.0],
            "natural_gas_price": [80.0, 80.0],
        },
        index=index,
    )
    context = MarketStageContext(
        market=DayAheadMarket("day_ahead", config.market("day_ahead")),
        plant=plant,
        forecasts=forecasts,
        stage_state=MarketStageState.empty(index),
    )
    executed = pd.DataFrame({"plant_name": plant.name}, index=index)

    def fake_solve(
        received_config: CaseConfig,
        received_forecasts: pd.DataFrame,
        stage: ElectricityMarketStage,
        market_input: DayAheadPosition,
        **kwargs: object,
    ) -> pd.DataFrame:
        assert received_config is config
        assert received_forecasts is forecasts
        assert stage == ElectricityMarketStage.DAY_AHEAD
        assert isinstance(market_input, DayAheadPosition)
        assert kwargs["rolling"] is False
        return executed

    monkeypatch.setattr(plant, "solve_market_stage", fake_solve)

    instruction = strategy.prepare_market_stage(context)
    assert isinstance(instruction, MarketStageInstruction)
    assert isinstance(instruction.payload, ElectricityMarketRequest)
    values = plant.solve_market_instruction(config, forecasts, instruction)
    result = strategy.settle_market_stage(context, instruction, values)

    assert values is executed
    assert result.values is executed


def test_steam_parameter_errors_identify_plant_and_technology() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    rows = DataLoader(config, input_dir=CASE_DIR).load_plants()
    rows.loc[rows["technology"] == "boiler", "efficiency"] = 1.1

    with pytest.raises(
        ValueError,
        match="Steam plant 'plant_1', technology 'boiler'.*efficiency",
    ):
        build_plants(rows)


def test_steam_forecast_errors_name_the_missing_input() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = build_plants(DataLoader(config, input_dir=CASE_DIR).load_plants())[0]
    index = pd.date_range("2025-01-01 00:00", periods=2, freq="15min")
    forecasts = pd.DataFrame(
        {
            "plant_1_heat_demand": [2.0, 2.0],
            "DE_DA_price": [120.0, 120.0],
        },
        index=index,
    )
    position = DayAheadPosition(
        electricity_price_col="DE_DA_price",
        gas_price_col="natural_gas_price",
        gas_benchmark_eur_per_mwh_th=pd.Series(0.0, index=index),
        charge_allowed=pd.Series(False, index=index),
    )

    with pytest.raises(
        ValueError,
        match="Steam plant 'plant_1'.*forecasts_df.csv.*natural_gas_price",
    ):
        plant.build_model(ElectricityMarketStage.DAY_AHEAD, config, forecasts, position)


def test_steam_generation_plant_short_horizon_solves() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    loader = DataLoader(config, input_dir=CASE_DIR)
    plants_df = loader.load_plants()
    plant = build_plants(plants_df)[0]
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
    position = DayAheadPosition(
        electricity_price_col=price_col,
        gas_price_col="natural_gas_price",
        co2_price_col="co2_price",
        gas_benchmark_eur_per_mwh_th=benchmark,
        charge_allowed=pd.Series(False, index=forecasts.index),
    )

    result = plant.solve_horizon(config, forecasts, position)

    required_columns = {
        "etes_charge_MWh",
        "etes_discharge_MWh",
        "etes_soc_MWh",
        "gas_heat_MWh",
        "electricity_consumption_MWh",
    }
    assert required_columns.issubset(result.columns)
    assert "unmet_heat_MWh" not in result.columns
    assert "excess_heat_MWh" not in result.columns
    _assert_useful_heat_matches_demand(result)


def test_steam_generation_plant_short_idc_adjustment_horizon_solves() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    loader = DataLoader(config, input_dir=CASE_DIR)
    plant = build_plants(loader.load_plants())[0]
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
    position = IntradayAdjustment(
        da_price_col="DE_DA_price",
        idc_price_col="DE_ID3_price",
        gas_price_col="natural_gas_price",
        da_position_mwh=da_position,
        idc_buy_upper_bound_mwh=pd.Series([0.0] * 4, index=index),
        idc_sell_upper_bound_mwh=da_position,
        gas_benchmark_eur_per_mwh_th=gas_benchmark,
        electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
    )

    result = plant.solve_intraday_adjustment_horizon(config, forecasts, position)

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
    _assert_useful_heat_matches_demand(result)


def test_steam_generation_plant_short_afrr_down_horizon_solves() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = build_plants(DataLoader(config, input_dir=CASE_DIR).load_plants())[0]
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
    position = BalancingEnergyActivation(
        da_price_col="DE_DA_price",
        idc_price_col="DE_ID3_price",
        gas_price_col="natural_gas_price",
        da_position_mwh=zero,
        idc_buy_mwh=zero,
        idc_sell_mwh=zero,
        final_planned_electricity_mwh=zero,
        afrr_energy_price=pd.Series([20.0] * 4, index=index),
        afrr_system_activation_mwh=activation,
        afrr_energy_bid_mwh=activation,
        afrr_energy_activated_mwh=activation,
        gas_benchmark_eur_per_mwh_th=gas_benchmark,
        electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
    )

    result = plant.solve_afrr_down_horizon(config, forecasts, position)

    assert result["afrr_energy_activated_MWh"].sum() == pytest.approx(1.6)
    assert result["actual_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        (result["final_planned_electricity_MWh"] + result["afrr_energy_activated_MWh"]).to_numpy()
    )
    assert result["etes_charge_MWh"].to_numpy() == pytest.approx(
        result["actual_electricity_consumption_MWh"].to_numpy()
    )
    _assert_useful_heat_matches_demand(result)


def test_steam_generation_plant_fails_when_real_heat_supply_is_insufficient() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = build_plants(DataLoader(config, input_dir=CASE_DIR).load_plants())[0]
    strategy = HybridETESGasStrategy(config)

    index = pd.date_range("2025-01-01 00:00", periods=4, freq="15min")
    forecasts = pd.DataFrame(
        {
            "plant_1_heat_demand": [20.0] * 4,
            "DE_DA_price": [120.0] * 4,
            "natural_gas_price": [80.0] * 4,
            "co2_price": [0.0] * 4,
        },
        index=index,
    )
    position = DayAheadPosition(
        electricity_price_col="DE_DA_price",
        gas_price_col="natural_gas_price",
        co2_price_col="co2_price",
        gas_benchmark_eur_per_mwh_th=strategy.calculate_gas_based_heat_cost(
            plant,
            forecasts,
        ),
        charge_allowed=pd.Series(False, index=index),
    )

    with pytest.raises(RuntimeError, match="infeasible"):
        plant.solve_horizon(config, forecasts, position)


def _assert_useful_heat_matches_demand(result: pd.DataFrame) -> None:
    supplied_heat = result["gas_heat_MWh"] + result["etes_discharge_MWh"]
    assert supplied_heat.to_numpy() == pytest.approx(result["heat_demand_MWh"].to_numpy())
