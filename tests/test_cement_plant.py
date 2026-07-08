# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.plants.cement_plant import CementDispatchSignals, CementPlant
from flexi_mod.plants.factory import build_plants
from flexi_mod.plants.technologies import CementCalciner, CementKiln, CementPreheater
from flexi_mod.simulation.simulation_runner import SimulationRunner

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DA"


def test_cement_plant_builds_preheater_calciner_kiln_route() -> None:
    plant = CementPlant.from_rows("cement_1", _cement_rows())

    assert isinstance(plant.components["preheater"], CementPreheater)
    assert isinstance(plant.components["calciner"], CementCalciner)
    assert isinstance(plant.components["kiln"], CementKiln)
    assert plant.clinker_demand_tonnes == 4.0


def test_shared_plant_factory_selects_cement_plant() -> None:
    plants = build_plants(_cement_rows())

    assert len(plants) == 1
    assert isinstance(plants[0], CementPlant)


def test_cement_plant_rejects_grinding_mill_for_now() -> None:
    rows = pd.concat(
        [
            _cement_rows(),
            pd.DataFrame(
                [
                    {
                        "name": "cement_1",
                        "unit_type": "cement_plant",
                        "technology": "cement_mill",
                        "clinker_demand": 4.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="grinding mill integration is intentionally excluded"):
        CementPlant.from_rows("cement_1", rows)


def test_cement_preheater_calciner_kiln_solve_and_balances() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = CementPlant.from_rows("cement_1", _cement_rows())

    result = plant.solve_horizon(config, _cement_forecasts(include_coal=False), _signals())

    assert result["clinker_output_t"].sum() == pytest.approx(4.0)
    assert result["preheater_raw_meal_output_t"].sum() == pytest.approx(6.0)
    assert result["calciner_clinker_output_t"].sum() == pytest.approx(4.0)
    assert result["kiln_clinker_output_t"].sum() == pytest.approx(4.0)
    assert result["natural_gas_consumption_MWh"].sum() > 0
    assert result["coal_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["hydrogen_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["co2_emissions_t"].sum() > 0


def test_cement_hydrogen_with_electrolyser_is_constrained_by_electrolyser_output() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = CementPlant.from_rows("cement_1", _hydrogen_cement_rows(include_electrolyser=True))

    result = plant.solve_horizon(config, _cement_forecasts(include_coal=False), _signals())

    assert result["clinker_output_t"].sum() == pytest.approx(4.0)
    assert result["hydrogen_consumption_MWh"].sum() > 0
    assert result["electrolyser_hydrogen_output_MWh"].sum() == pytest.approx(
        result["hydrogen_consumption_MWh"].sum()
    )


def test_cement_hydrogen_without_electrolyser_uses_external_hydrogen() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = CementPlant.from_rows("cement_1", _hydrogen_cement_rows(include_electrolyser=False))

    result = plant.solve_horizon(config, _cement_forecasts(include_coal=False), _signals())

    assert result["hydrogen_consumption_MWh"].sum() > 0
    assert "electrolyser_hydrogen_output_MWh" not in result


def test_cement_forecast_discovery_and_conditional_coal_price() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    loader = DataLoader(config, input_dir=CASE_DIR)
    total_rows = _cement_rows()

    total_required = loader.required_forecast_columns(total_rows)

    assert "cement_1_clinker_demand" not in total_required
    assert "coal_price" not in total_required

    profile_rows = total_rows.copy()
    profile_rows["clinker_demand"] = pd.NA
    profile_rows["demand"] = "custom_clinker_demand"
    profile_required = loader.required_forecast_columns(profile_rows)
    assert "custom_clinker_demand" in profile_required

    coal_rows = total_rows.copy()
    coal_rows.loc[coal_rows["technology"] == "kiln", "fossil_ng_share"] = 0.0
    coal_required = loader.required_forecast_columns(coal_rows)
    assert "coal_price" in coal_required


def test_cement_runner_uses_day_ahead_price_and_writes_outputs(tmp_path: Path) -> None:
    case_dir = tmp_path / "cement_runner"
    output_dir = tmp_path / "output"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_cement_config(), encoding="utf-8")
    _cement_rows().to_csv(case_dir / "plants.csv", index=False)
    forecasts = _cement_forecasts(include_coal=False).copy()
    forecasts = forecasts.rename(columns={"electricity_price": "connected_DA_price"})
    forecasts.reset_index(names="datetime").to_csv(case_dir / "forecasts_df.csv", index=False)

    outputs = SimulationRunner(case_dir, output_dir=output_dir).run()

    assert set(outputs) == {"dispatch_results", "summary_indicators"}
    dispatch = pd.read_csv(outputs["dispatch_results"])
    summary = pd.read_csv(outputs["summary_indicators"])
    assert dispatch["clinker_output_t"].sum() == pytest.approx(4.0)
    assert summary.loc[0, "total_clinker_production_t"] == pytest.approx(4.0)
    assert summary.loc[0, "final_clinker_demand_balance_t"] == pytest.approx(0.0)
    assert not (output_dir / "market_ledger.csv").exists()
    assert not (output_dir / "storage_cost_ledger.csv").exists()


def _cement_rows() -> pd.DataFrame:
    shared = {
        "name": "cement_1",
        "unit_type": "cement_plant",
        "node": "north",
        "objective": "min_variable_cost",
        "clinker_demand": 4.0,
        "raw_meal_to_clinker_ratio": 1.5,
        "waste_heat_per_t_clinker": 0.0,
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "technology": "preheater",
                "fuel_type": "electricity",
                "max_heat_out": 10.0,
                "specific_heat_demand": 0.1,
                "eta_electric": 1.0,
            },
            {
                **shared,
                "technology": "calciner",
                "fuel_type": "fossil",
                "fossil_ng_share": 1.0,
                "max_heat_out": 10.0,
                "specific_heat_demand": 0.5,
                "eta_fossil": 1.0,
                "calcination_emission_factor": 0.5,
                "natural_gas_co2_factor": 0.2,
            },
            {
                **shared,
                "technology": "kiln",
                "fuel_type": "fossil",
                "fossil_ng_share": 1.0,
                "max_heat_out": 10.0,
                "specific_heat_demand": 0.8,
                "eta_fossil": 1.0,
                "natural_gas_co2_factor": 0.2,
            },
        ]
    )


def _hydrogen_cement_rows(include_electrolyser: bool) -> pd.DataFrame:
    rows = _cement_rows()
    rows.loc[rows["technology"].isin(["calciner", "kiln"]), "fuel_type"] = "hydrogen"
    if include_electrolyser:
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "name": "cement_1",
                            "unit_type": "cement_plant",
                            "technology": "electrolyser",
                            "node": "north",
                            "objective": "min_variable_cost",
                            "clinker_demand": 4.0,
                            "max_power": 20.0,
                            "min_power": 0.0,
                            "efficiency": 1.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return rows


def _cement_forecasts(include_coal: bool) -> pd.DataFrame:
    data = {
        "electricity_price": [30.0, 40.0, 80.0, 100.0],
        "natural_gas_price": [50.0] * 4,
        "hydrogen_price": [70.0] * 4,
        "co2_price": [80.0] * 4,
    }
    if include_coal:
        data["coal_price"] = [15.0] * 4
    return pd.DataFrame(data, index=pd.date_range("2025-01-01", periods=4, freq="15min"))


def _signals() -> CementDispatchSignals:
    return CementDispatchSignals(electricity_price_col="electricity_price")


def _cement_config() -> str:
    return """
cases:
  cement_runner:
    name: cement_runner
    country: DE
    timestep_minutes: 15
    simulation_start: "2025-01-01 00:00"
    simulation_end: "2025-01-01 00:45"
    additional_charges: false
    strategy:
      name: cement_cost_minimization
      dispatch:
        dispatch_method: pyomo
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
          price: connected_DA_price
""".strip()
