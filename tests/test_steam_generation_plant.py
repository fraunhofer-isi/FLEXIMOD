# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd

from etes_market_model.config.case_config import CaseConfig
from etes_market_model.data.data_loader import DataLoader
from etes_market_model.plants.steam_generation_plant import (
    DispatchSignals,
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
    required_columns = loader.required_forecast_columns(
        plants_df,
        extra_required_columns=strategy.required_forecast_columns(),
    )
    forecasts = loader.load_forecasts(required_columns).iloc[:8]
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
