# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import pandas as pd
import pytest

from flexi_mod.plants.cement_plant import CementPlant


def test_cement_kiln_line_solves_from_component_rows() -> None:
    plant = CementPlant.from_rows("cement_1", _cement_rows())

    result = plant.solve_horizon(_config(), _forecasts(), electricity_price_column="DE_DA_price")

    assert list(result["clinker_output_t"]) == pytest.approx([1.0, 1.0])
    assert list(result["raw_meal_output_t"]) == pytest.approx([1.5, 1.5])
    assert list(result["electricity_consumption_MWh"]) == pytest.approx([0.0, 0.0])


def test_cement_input_aliases_are_normalised() -> None:
    plant = CementPlant.from_rows("cement_1", _cement_rows())

    assert set(plant.components) == {"preheater", "calciner", "kiln"}


def test_cement_plant_rejects_missing_kiln() -> None:
    rows = _cement_rows().iloc[:2].copy()

    with pytest.raises(ValueError, match="exactly one 'kiln' row"):
        CementPlant.from_rows("cement_1", rows)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        timestep_minutes=60,
        solver_name="highs",
        solver_fallbacks=[],
        solver_tee=False,
    )


def _cement_rows() -> pd.DataFrame:
    common = {
        "name": "cement_1",
        "unit_type": "cement_plant",
        "node": "node_1",
        "objective": "min_variable_cost",
        "fuel_type": "fossil",
        "raw_meal_to_clinker_ratio": 1.5,
        "max_heat_out": 10.0,
        "specific_electricity_aux": 0.0,
        "eta_electric": 0.95,
        "eta_fossil": 0.9,
        "fossil_ng_share": 1.0,
        "ng_co2_factor": 0.2,
        "coal_co2_factor": 0.3,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "technology": "preheater",
                "specific_heat_demand": 1.0,
            },
            {
                **common,
                "technology": "simple_calciner",
                "specific_heat_demand": 1.0,
                "calcination_emission_factor": 0.5,
            },
            {
                **common,
                "technology": "simple_kiln",
                "specific_heat_demand": 1.0,
            },
        ]
    )


def _forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cement_1_clinker_demand": [1.0, 1.0],
            "DE_DA_price": [50.0, 50.0],
            "natural_gas_price": [10.0, 10.0],
            "coal_price": [20.0, 20.0],
            "hydrogen_price": [100.0, 100.0],
            "co2_price": [50.0, 50.0],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="1h"),
    )
