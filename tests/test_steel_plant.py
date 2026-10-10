# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import pandas as pd
import pytest

from flexi_mod.plants.factory import build_plants


def test_hydrogen_dri_eaf_plant_solves_from_component_rows() -> None:
    plant = build_plants(_steel_rows())[0]
    forecasts = _forecasts()

    result = plant.solve_horizon(_config(), forecasts, electricity_price_column="DE_DA_price")

    assert list(result["steel_output_t"]) == pytest.approx([1.0, 1.0])
    assert list(result["dri_output_t"]) == pytest.approx([1.0, 1.0])
    assert list(result["electricity_consumption_MWh"]) == pytest.approx([0.6, 0.6])
    assert list(result["hydrogen_consumption_MWh"]) == pytest.approx([2.0, 2.0])
    assert list(result["variable_cost_EUR"]) == pytest.approx([237.25, 237.25])


def test_electrolyser_supplies_hydrogen_to_dri() -> None:
    rows = _steel_rows()
    rows.loc[0, "fuel_type"] = "hydrogen"
    rows.loc[0, "specific_hydrogen_consumption"] = 2.0
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    {
                        "name": "steel_1",
                        "unit_type": "steel_plant",
                        "technology": "electrolyser",
                        "max_power": 10.0,
                        "min_power": 0.0,
                        "efficiency": 0.5,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    forecasts = _forecasts()
    forecasts["DE_DA_price"] = 10.0

    result = build_plants(rows)[0].solve_horizon(
        _config(), forecasts, electricity_price_column="DE_DA_price"
    )

    assert list(result["hydrogen_production_MWh"]) == pytest.approx([2.0, 2.0])
    assert list(result["electricity_consumption_MWh"]) == pytest.approx([4.6, 4.6])


def test_steel_plant_requires_dri_and_eaf() -> None:
    rows = _steel_rows().iloc[[0]].copy()

    with pytest.raises(ValueError, match="exactly one 'eaf' row"):
        build_plants(rows)


def test_steel_plant_parameter_errors_identify_plant_and_technology() -> None:
    rows = _steel_rows()
    rows.loc[0, "max_power"] = float("inf")

    with pytest.raises(
        ValueError,
        match="Steel plant 'steel_1', technology 'dri_plant'.*max_power",
    ):
        build_plants(rows)


def test_steel_plant_forecast_errors_name_the_missing_input() -> None:
    plant = build_plants(_steel_rows())[0]
    forecasts = _forecasts().drop(columns="iron_ore_price")

    with pytest.raises(
        ValueError,
        match="Steel plant 'steel_1'.*forecasts_df.csv.*iron_ore_price",
    ):
        plant.build_model(_config(), forecasts, electricity_price_column="DE_DA_price")


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        timestep_minutes=60,
        solver_name="highs",
        solver_fallbacks=[],
        solver_tee=False,
    )


def _steel_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "steel_1",
                "unit_type": "steel_plant",
                "technology": "dri_plant",
                "node": "node_1",
                "objective": "min_variable_cost",
                "fuel_type": "hydrogen",
                "max_power": 10.0,
                "specific_hydrogen_consumption": 2.0,
                "specific_electricity_consumption": 0.2,
                "specific_iron_ore_consumption": 1.4,
                "natural_gas_co2_factor": 0.2,
            },
            {
                "name": "steel_1",
                "unit_type": "steel_plant",
                "technology": "eaf",
                "max_power": 20.0,
                "specific_dri_demand": 1.0,
                "specific_electricity_consumption": 0.4,
                "specific_lime_demand": 0.05,
            },
        ]
    )


def _forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "steel_1_steel_demand": [1.0, 1.0],
            "DE_DA_price": [50.0, 50.0],
            "hydrogen_price": [100.0, 100.0],
            "iron_ore_price": [5.0, 5.0],
            "lime_price": [5.0, 5.0],
            "co2_price": [0.0, 0.0],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="1h"),
    )
