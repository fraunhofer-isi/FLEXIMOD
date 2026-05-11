# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from etes_market_model.simulation.simulation_runner import OutputOptions, SimulationRunner


@pytest.fixture
def day_ahead_only_case(tmp_path: Path) -> Path:
    """Create a tiny DA-only case with cheap prices first and expensive prices later."""

    case_dir = tmp_path / "day_ahead_only_case"
    case_dir.mkdir()
    _write_config(case_dir / "config.yaml")
    _write_plants(case_dir / "plants.csv")
    _write_forecasts(case_dir / "forecasts_df.csv")
    return case_dir


@pytest.fixture
def day_ahead_only_results(
    day_ahead_only_case: Path,
    tmp_path: Path,
) -> dict[str, pd.DataFrame]:
    """Run the full sequential simulation and load the output ledgers."""

    runner = SimulationRunner(
        case_dir=day_ahead_only_case,
        input_dir=day_ahead_only_case,
        output_dir=tmp_path / "output",
        output_options=OutputOptions(create_plots=False),
    )
    outputs = runner.run()

    return {
        "dispatch": pd.read_csv(outputs["dispatch_results"], parse_dates=["datetime"]),
        "market": pd.read_csv(outputs["market_ledger"], parse_dates=["datetime"]),
        "storage": pd.read_csv(outputs["storage_cost_ledger"], parse_dates=["datetime"]),
        "summary": pd.read_csv(outputs["summary_indicators"]),
    }


def test_day_ahead_only_strategy_matches_expected_plant_behaviour(
    day_ahead_only_results: dict[str, pd.DataFrame],
) -> None:
    """Cheap DA prices should charge ETES; expensive DA prices should not."""

    dispatch = day_ahead_only_results["dispatch"]
    market = day_ahead_only_results["market"]
    storage = day_ahead_only_results["storage"]
    summary = day_ahead_only_results["summary"]

    cheap_price_steps = dispatch["day_ahead_price_EUR_per_MWh"] == 10.0
    expensive_price_steps = dispatch["day_ahead_price_EUR_per_MWh"] == 120.0

    # The rule-based strategy opens the charging gate only in the cheap DA period.
    assert dispatch.loc[cheap_price_steps, "charge_allowed_by_strategy"].all()
    assert not dispatch.loc[expensive_price_steps, "charge_allowed_by_strategy"].any()

    # The Pyomo plant model then turns the gate into feasible DA electricity use.
    assert dispatch.loc[cheap_price_steps, "etes_charge_MWh"].sum() > 0
    assert dispatch.loc[expensive_price_steps, "etes_charge_MWh"].sum() == pytest.approx(0.0)

    # The plant operation remains feasible and heat demand is fully supplied.
    supplied_heat = dispatch["gas_heat_MWh"] + dispatch["etes_discharge_MWh"]
    assert (supplied_heat >= dispatch["heat_demand_MWh"] - 1e-8).all()
    assert dispatch["unmet_heat_MWh"].sum() == pytest.approx(0.0)

    # In the DA-only MVP, the market position is exactly the optimized ETES electricity use.
    assert market["DA_position_MWh"].sum() == pytest.approx(
        dispatch["electricity_consumption_MWh"].sum()
    )
    assert market["actual_electricity_consumption_MWh"].sum() == pytest.approx(
        market["DA_position_MWh"].sum()
    )

    # Stored heat is attributed to the day-ahead market and CO2 is inactive for this MVP.
    assert storage["remaining_stored_heat_day_ahead_MWh"].max() > 0
    assert summary["total_DA_electricity_MWh"].iloc[0] == pytest.approx(
        market["DA_position_MWh"].sum()
    )
    assert summary["total_unmet_heat_MWh"].iloc[0] == pytest.approx(0.0)
    assert summary["total_co2_cost_EUR"].iloc[0] == pytest.approx(0.0)


def _write_config(path: Path) -> None:
    path.write_text(
        """
case:
  name: day_ahead_strategy_test_case
  country: DE
  description: "Synthetic day-ahead strategy test case"
  timestep_minutes: 15
  simulation_start: "2025-01-01 00:00"
  simulation_end: "2025-01-01 01:45"
  timezone: "Europe/Berlin"

strategy:
  name: hybrid_etes_gas
  dispatch:
    dispatch_method: pyomo
    rolling_horizon_enabled: true
    dispatch_horizon_hours: 2
    rolling_step_hours: 1

solver:
  name: highs
  fallback_solvers:
    - gurobi
  tee: false

market_sequence:
  - day_ahead
  - intraday_continuous
  - afrr_energy
  - afrr_capacity

markets:
  day_ahead:
    enabled: true
    product_resolution: "15min"
    gate_close:
      day_relation: "D-1"
      time: "12:00"
    signals:
      price: "DE_DA_price"

  intraday_continuous:
    enabled: false
    product_resolution: "15min"
    gate_close:
      relative_to_delivery_start_minutes: -5
    signals:
      price: "DE_ID3_price"
      volume: "DE_ID3_volume"

  afrr_energy:
    enabled: false
    direction: "negative"
    product_resolution: "15min"
    gate_close:
      relative_to_delivery_start_minutes: -25
    signals:
      price: "DE_afrr_energy_neg_price"
      activation_volume: "DE_afrr_energy_neg_activation"

  afrr_capacity:
    enabled: false
    direction: "negative"
    product_length: "4h"
    min_bid_mw: 1.0
    bid_increment_mw: 1.0
    gate_close:
      day_relation: "D-1"
      time: "09:00"
    signals:
      capacity_price: "DE_afrr_capacity_neg_price"
""".strip(),
        encoding="utf-8",
    )


def _write_plants(path: Path) -> None:
    plants = pd.DataFrame(
        [
            {
                "name": "plant_1",
                "unit_type": "steam_plant",
                "technology": "thermal_storage",
                "node": "north",
                "demand": "plant_1_heat_demand",
                "storage_type": "short-term_with_generator",
                "max_power": 7.0,
                "min_power": 0.0,
                "ramp_up": 7.0,
                "ramp_down": 7.0,
                "min_operating_time": 0.0,
                "min_down_time": 0.0,
                "efficiency": 0.95,
                "max_capacity": 4.0,
                "min_capacity": 0.0,
                "max_power_charge": 7.0,
                "max_power_discharge": 7.0,
                "initial_soc": 0.0,
                "efficiency_charge": 0.92,
                "efficiency_discharge": 0.92,
                "storage_loss_rate": 0.0,
            },
            {
                "name": "plant_1",
                "unit_type": "steam_plant",
                "technology": "boiler",
                "node": "north",
                "fuel_type": "natural_gas",
                "max_power": 5.0,
                "min_power": 0.0,
                "ramp_up": 5.0,
                "ramp_down": 5.0,
                "min_operating_time": 0.0,
                "min_down_time": 0.0,
                "efficiency": 0.9,
            },
        ]
    )
    plants.to_csv(path, index=False)


def _write_forecasts(path: Path) -> None:
    datetimes = pd.date_range("2025-01-01 00:00", periods=8, freq="15min")
    forecasts = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_1_heat_demand": [2.0] * 8,
            "DE_DA_price": [10.0] * 4 + [120.0] * 4,
            "natural_gas_price": [80.0] * 8,
        }
    )
    forecasts.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
