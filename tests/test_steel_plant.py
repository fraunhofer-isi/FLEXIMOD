# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants.factory import build_plants
from flexi_mod.plants.steel_plant import SteelDispatchSignals, SteelPlant
from flexi_mod.plants.technologies import (
    DRIPlant,
    DRIStorage,
    ElectricArcFurnace,
    Electrolyser,
    GenericStorage,
    HydrogenBufferStorage,
    ThermalStorage,
)

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DA"


def test_steel_plant_builds_required_and_optional_technologies() -> None:
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=True))

    assert isinstance(plant.dri_plant, DRIPlant)
    assert isinstance(plant.eaf, ElectricArcFurnace)
    assert isinstance(plant.components["electrolyser"], Electrolyser)
    assert isinstance(plant.components["hydrogen_buffer_storage"], HydrogenBufferStorage)
    assert isinstance(plant.components["dri_storage"], DRIStorage)
    assert plant.steel_demand_tonnes == 4.0
    assert issubclass(ThermalStorage, GenericStorage)
    assert issubclass(HydrogenBufferStorage, GenericStorage)
    assert issubclass(DRIStorage, GenericStorage)


def test_shared_plant_factory_selects_steel_plant() -> None:
    plants = build_plants(_steel_rows(include_optional=False))

    assert len(plants) == 1
    assert isinstance(plants[0], SteelPlant)


def test_steel_plant_rejects_missing_required_technology() -> None:
    rows = _steel_rows(include_optional=False)
    rows = rows.loc[rows["technology"] != "eaf"]

    with pytest.raises(ValueError, match="missing required.*eaf"):
        SteelPlant.from_rows("steel_1", rows)


def test_steel_plant_hydrogen_route_and_material_balances_solve() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=True))
    forecasts = _steel_forecasts()

    result = plant.solve_horizon(config, forecasts, _signals())

    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert result["natural_gas_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["electrolyser_hydrogen_output_MWh"].sum() == pytest.approx(
        result["hydrogen_consumption_MWh"].sum()
        + result["hydrogen_storage_charge_MWh"].sum()
        - result["hydrogen_storage_discharge_MWh"].sum()
    )
    assert result["dri_output_t"].sum() + result["dri_storage_discharge_t"].sum() == (
        pytest.approx(result["dri_input_t"].sum() + result["dri_storage_charge_t"].sum())
    )
    assert result["total_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        (
            result["dri_electricity_consumption_MWh"]
            + result["eaf_electricity_consumption_MWh"]
            + result["electrolyser_electricity_consumption_MWh"]
        ).to_numpy()
    )


def test_steel_plant_without_electrolyser_buys_hydrogen() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=False))

    result = plant.solve_horizon(config, _steel_forecasts(), _signals())

    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert result["hydrogen_consumption_MWh"].sum() > 0
    assert "electrolyser_hydrogen_output_MWh" not in result


def test_dri_storage_can_shift_dri_production_to_later_eaf_operation() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    rows = _steel_rows(include_optional=True)
    rows = rows.loc[
        rows["technology"].isin({"dri_plant", "eaf", "dri_storage"})
    ].copy()
    rows.loc[rows["technology"] == "dri_plant", "fuel_type"] = "natural_gas"
    plant = SteelPlant.from_rows("steel_1", rows)
    forecasts = _steel_forecasts()
    forecasts["electricity_price"] = [1000.0, 1000.0, 0.0, 0.0]
    forecasts["natural_gas_price"] = [0.0, 0.0, 1000.0, 1000.0]

    result = plant.solve_horizon(config, forecasts, _signals())

    assert result["dri_storage_charge_t"].iloc[:2].sum() > 0
    assert result["dri_storage_discharge_t"].iloc[2:].sum() > 0
    assert result["dri_output_t"].iloc[:2].sum() > result["dri_input_t"].iloc[:2].sum()
    assert result["dri_input_t"].iloc[2:].sum() > result["dri_output_t"].iloc[2:].sum()


def test_quarter_hour_power_limits_are_converted_to_interval_energy() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=True))
    model = plant._build_model(config, _steel_forecasts(), _signals())

    assert float(model.technology_blocks["electrolyser"].max_power) == pytest.approx(5.0)
    assert float(model.technology_blocks["dri_plant"].max_power) == pytest.approx(2.5)
    assert float(model.technology_blocks["eaf"].max_power) == pytest.approx(2.5)


def _steel_rows(include_optional: bool) -> pd.DataFrame:
    shared = {
        "name": "steel_1",
        "unit_type": "steel_plant",
        "node": "north",
        "objective": "min_variable_cost",
        "steel_demand": 4.0,
    }
    rows = [
        {
            **shared,
            "technology": "dri_plant",
            "fuel_type": "hydrogen",
            "specific_hydrogen_consumption": 2.0,
            "specific_natural_gas_consumption": 3.0,
            "specific_electricity_consumption": 0.2,
            "specific_iron_ore_consumption": 1.5,
            "natural_gas_co2_factor": 0.2,
            "max_power": 10.0,
            "min_power": 0.0,
        },
        {
            **shared,
            "technology": "eaf",
            "specific_electricity_consumption": 0.5,
            "specific_dri_demand": 1.0,
            "specific_lime_demand": 0.05,
            "lime_co2_factor": 0.1,
            "max_power": 10.0,
            "min_power": 0.0,
        },
    ]
    if include_optional:
        rows.extend(
            [
                {
                    **shared,
                    "technology": "electrolyser",
                    "max_power": 20.0,
                    "min_power": 0.0,
                    "efficiency": 0.8,
                },
                {
                    **shared,
                    "technology": "hydrogen_buffer_storage",
                    "capacity": 20.0,
                    "min_soc": 0.0,
                    "max_soc": 1.0,
                    "initial_soc": 0.0,
                    "max_power_charge": 20.0,
                    "max_power_discharge": 20.0,
                    "efficiency_charge": 1.0,
                    "efficiency_discharge": 1.0,
                    "storage_loss_rate": 0.0,
                },
                {
                    **shared,
                    "technology": "dri_storage",
                    "capacity": 10.0,
                    "min_soc": 0.0,
                    "max_soc": 1.0,
                    "initial_soc": 0.0,
                    "max_power_charge": 10.0,
                    "max_power_discharge": 10.0,
                    "efficiency_charge": 1.0,
                    "efficiency_discharge": 1.0,
                    "storage_loss_rate": 0.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def _steel_forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "electricity_price": [30.0, 40.0, 80.0, 100.0],
            "natural_gas_price": [50.0] * 4,
            "hydrogen_price": [70.0] * 4,
            "iron_ore_price": [100.0] * 4,
            "lime_price": [20.0] * 4,
            "co2_price": [80.0] * 4,
            "steel_price": [600.0] * 4,
        },
        index=pd.date_range("2025-01-01", periods=4, freq="15min"),
    )


def _signals() -> SteelDispatchSignals:
    return SteelDispatchSignals(
        electricity_price_col="electricity_price",
        natural_gas_price_col="natural_gas_price",
        hydrogen_price_col="hydrogen_price",
        iron_ore_price_col="iron_ore_price",
        lime_price_col="lime_price",
        co2_price_col="co2_price",
        steel_price_col="steel_price",
    )
