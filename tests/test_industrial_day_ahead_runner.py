# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end loader, factory, strategy, and runner coverage for industry."""

from pathlib import Path

import pandas as pd

from flexi_mod.simulation.simulation_runner import OutputOptions, SimulationRunner


def test_runner_uses_common_lifecycle_for_cement_and_steel(tmp_path: Path) -> None:
    case_dir = tmp_path / "industrial_day_ahead"
    case_dir.mkdir()
    _write_case(case_dir)

    output_paths = SimulationRunner(
        case_dir,
        output_dir=case_dir / "output",
        output_options=OutputOptions(create_plots=False),
    ).run()

    dispatch = pd.read_csv(output_paths["dispatch_results"])
    market_ledger = pd.read_csv(output_paths["market_ledger"])
    summary = pd.read_csv(output_paths["summary_indicators"])

    assert set(dispatch["plant_name"]) == {"cement_1", "steel_1"}
    assert {"DA_position_MWh", "final_planned_electricity_MWh"}.issubset(dispatch.columns)
    assert dispatch["DA_position_MWh"].tolist() == dispatch["electricity_consumption_MWh"].tolist()
    assert (
        dispatch["actual_electricity_consumption_MWh"].tolist()
        == dispatch["electricity_consumption_MWh"].tolist()
    )
    assert set(market_ledger["plant_name"]) == {"cement_1", "steel_1"}
    assert set(summary["plant_name"]) == {"cement_1", "steel_1"}


def _write_case(case_dir: Path) -> None:
    (case_dir / "config.yaml").write_text(
        """
cases:
  industrial_day_ahead:
    name: industrial_day_ahead
    country: DE
    timestep_minutes: 60
    simulation_start: "2025-01-01 00:00"
    simulation_end: "2025-01-01 01:00"
    strategy:
      name: industrial_day_ahead_cost_minimisation
      dispatch:
        dispatch_method: pyomo
        rolling_horizon_enabled: false
    solver:
      name: highs
      fallback_solvers: []
      tee: false
    market_sequence:
      - day_ahead
    markets:
      day_ahead:
        enabled: true
        signals:
          price: DE_DA_price
""".strip()
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([*_steel_rows(), *_cement_rows()]).to_csv(case_dir / "plants.csv", index=False)
    pd.DataFrame(
        {
            "datetime": ["2025-01-01 00:00", "2025-01-01 01:00"],
            "steel_1_steel_demand": [1.0, 1.0],
            "cement_1_clinker_demand": [1.0, 1.0],
            "DE_DA_price": [50.0, 50.0],
            "hydrogen_price": [100.0, 100.0],
            "iron_ore_price": [5.0, 5.0],
            "lime_price": [5.0, 5.0],
            "natural_gas_price": [10.0, 10.0],
            "coal_price": [20.0, 20.0],
            "co2_price": [50.0, 50.0],
        }
    ).to_csv(case_dir / "forecasts_df.csv", index=False)


def _steel_rows() -> list[dict[str, object]]:
    return [
        {
            "name": "steel_1",
            "unit_type": "steel_plant",
            "technology": "dri_plant",
            "node": "node_1",
            "objective": "min_variable_cost",
            "steel_demand": "steel_1_steel_demand",
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
            "node": "node_1",
            "objective": "min_variable_cost",
            "steel_demand": "steel_1_steel_demand",
            "max_power": 20.0,
            "specific_dri_demand": 1.0,
            "specific_electricity_consumption": 0.4,
            "specific_lime_demand": 0.05,
        },
    ]


def _cement_rows() -> list[dict[str, object]]:
    common = {
        "name": "cement_1",
        "unit_type": "cement_plant",
        "node": "node_1",
        "objective": "min_variable_cost",
        "clinker_demand": "cement_1_clinker_demand",
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
    return [
        {**common, "technology": "preheater", "specific_heat_demand": 1.0},
        {
            **common,
            "technology": "calciner",
            "specific_heat_demand": 1.0,
            "calcination_emission_factor": 0.5,
        },
        {**common, "technology": "kiln", "specific_heat_demand": 1.0},
    ]
