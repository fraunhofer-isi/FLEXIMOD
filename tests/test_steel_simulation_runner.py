# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig, ConfigError
from flexi_mod.simulation.simulation_runner import SimulationRunner


def test_steel_runner_uses_connected_day_ahead_price_and_writes_outputs(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "steel_runner"
    output_dir = tmp_path / "output"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        """
cases:
  steel_runner:
    name: steel_runner
    country: DE
    timestep_minutes: 15
    simulation_start: "2025-01-01 00:00"
    simulation_end: "2025-01-01 01:45"
    additional_charges: false
    strategy:
      name: steel_cost_minimization
      dispatch:
        dispatch_method: pyomo
        rolling_horizon_enabled: true
        dispatch_horizon_hours: 1
        rolling_step_hours: 0.5
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
""".strip(),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "name": "steel_1",
                "unit_type": "steel_plant",
                "technology": "dri_plant",
                "node": "north",
                "objective": "min_variable_cost",
                "steel_demand": 4.0,
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
                "name": "steel_1",
                "unit_type": "steel_plant",
                "technology": "eaf",
                "node": "north",
                "objective": "min_variable_cost",
                "steel_demand": 4.0,
                "specific_electricity_consumption": 0.5,
                "specific_dri_demand": 1.0,
                "specific_lime_demand": 0.05,
                "lime_co2_factor": 0.1,
                "max_power": 10.0,
                "min_power": 0.0,
            },
        ]
    ).to_csv(case_dir / "plants.csv", index=False)
    periods = 8
    pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=periods, freq="15min"),
            "connected_DA_price": [20.0, 25.0, 100.0, 110.0, 30.0, 35.0, 90.0, 95.0],
            "natural_gas_price": [50.0] * periods,
            "hydrogen_price": [70.0] * periods,
            "iron_ore_price": [100.0] * periods,
            "lime_price": [20.0] * periods,
            "co2_price": [80.0] * periods,
        }
    ).to_csv(case_dir / "forecasts_df.csv", index=False)

    outputs = SimulationRunner(case_dir, output_dir=output_dir).run()

    assert set(outputs) == {"dispatch_results", "summary_indicators"}
    dispatch = pd.read_csv(outputs["dispatch_results"])
    summary = pd.read_csv(outputs["summary_indicators"])
    assert len(dispatch) == periods
    assert dispatch["steel_output_t"].sum() == pytest.approx(4.0)
    assert dispatch["remaining_steel_demand_t"].iloc[-1] == pytest.approx(0.0)
    assert summary.loc[0, "total_steel_production_t"] == pytest.approx(4.0)
    assert summary.loc[0, "final_steel_demand_balance_t"] == pytest.approx(0.0)
    assert not (output_dir / "market_ledger.csv").exists()
    assert not (output_dir / "storage_cost_ledger.csv").exists()


@pytest.mark.parametrize(
    ("day_ahead_enabled", "horizon", "step", "message"),
    [
        (False, 1.0, 0.5, "requires an enabled day_ahead"),
        (True, 0.5, 1.0, "must not exceed"),
        (True, 1.1, 0.5, "must align"),
    ],
)
def test_steel_runner_configuration_is_validated(
    tmp_path: Path,
    day_ahead_enabled: bool,
    horizon: float,
    step: float,
    message: str,
) -> None:
    case_dir = tmp_path / "invalid_steel_config"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        f"""
cases:
  invalid_steel_config:
    name: invalid_steel_config
    country: DE
    timestep_minutes: 15
    simulation_start: "2025-01-01 00:00"
    simulation_end: "2025-01-01 00:45"
    strategy:
      name: steel_cost_minimization
      dispatch:
        dispatch_method: pyomo
        rolling_horizon_enabled: true
        dispatch_horizon_hours: {horizon}
        rolling_step_hours: {step}
    solver:
      name: highs
      fallback_solvers: []
      tee: false
    market_sequence:
      - day_ahead
    markets:
      day_ahead:
        enabled: {str(day_ahead_enabled).lower()}
        signals:
          price: connected_DA_price
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
        CaseConfig.from_case_dir(case_dir)
