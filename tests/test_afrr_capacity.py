# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for aFRR capacity market behavior."""

from pathlib import Path

import pandas as pd
import pytest
from conftest import case_config_text

from flexi_mod.simulation.simulation_runner import OutputOptions, SimulationRunner


@pytest.fixture
def afrr_capacity_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "afrr_capacity_case"
    case_dir.mkdir()
    _write_config(
        case_dir / "config.yaml",
        idc_enabled=True,
        afrr_enabled=True,
        afrr_capacity_enabled=True,
    )
    _write_plants(case_dir / "plants.csv", storage_capacity=4.0)
    return case_dir


def test_afrr_capacity_full_storage_no_heat_demand_should_not_bid(
    afrr_capacity_case: Path,
    tmp_path: Path,
) -> None:
    """When storage is FULL and there is NO heat demand, capacity should not be reserved.

    Rationale: If storage is full and there's no heat demand to consume the energy,
    the system cannot actually absorb aFRR activation energy. Bidding capacity would
    be impossible to fulfill.
    """
    _write_forecasts(
        afrr_capacity_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[20.0] * 8,
        afrr_quantities=[2.0] * 8,
        afrr_capacity_prices=[100.0] * 8,  # Very attractive price
        heat_demand=[0.0] * 8,  # NO heat demand!
    )
    # Override plants: set initial_soc to max_capacity (FULL)
    _write_plants(afrr_capacity_case / "plants.csv", storage_capacity=4.0, storage_initial_soc=4.0)

    results = _run_case(afrr_capacity_case, tmp_path)
    block = results["afrr_capacity_blocks"].iloc[0]

    # Storage is full, no heat demand -> should NOT bid
    # Currently this likely FAILS (bids anyway) - that's the bug we want to see
    print(f"Reserved capacity: {block['reserved_capacity_MW']}")
    print(f"Technical capacity: {block['technical_capacity_MW']}")
    print(f"Storage headroom: {block['storage_headroom_MW']}")

    # This assertion SHOULD pass after the fix:
    assert block["reserved_capacity_MW"] == pytest.approx(0.0), (
        "Should not reserve capacity when storage is full and no heat demand"
    )


def _run_case(case_dir: Path, tmp_path: Path) -> dict[str, pd.DataFrame]:
    runner = SimulationRunner(
        case_dir=case_dir,
        input_dir=case_dir,
        output_dir=tmp_path / "output",
        output_options=OutputOptions(create_plots=False),
    )
    outputs = runner.run()
    results = {
        "dispatch": pd.read_csv(outputs["dispatch_results"], parse_dates=["datetime"]),
        "market": pd.read_csv(outputs["market_ledger"], parse_dates=["datetime"]),
    }
    if "afrr_capacity_block_summary" in outputs:
        results["afrr_capacity_blocks"] = pd.read_csv(outputs["afrr_capacity_block_summary"])
    return results


def _write_config(
    path: Path,
    idc_enabled: bool = True,
    afrr_enabled: bool = True,
    afrr_capacity_enabled: bool = True,
) -> None:
    path.write_text(
        case_config_text(
            f"""
case:
  name: afrr_capacity_full_storage_test
  country: DE
  description: "Test case for full storage scenario"
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
  - afrr_capacity
  - day_ahead
  - intraday_continuous
  - afrr_energy

markets:
  day_ahead:
    enabled: true
    product_resolution: "15min"
    signals:
      price: "DE_day_ahead_price_EUR_per_MWh_el"

  intraday_continuous:
    enabled: {str(idc_enabled).lower()}
    product_resolution: "15min"
    allowed_actions:
      buy: true
      sell: true
    signals:
      price: "DE_ID3_price"

  afrr_energy:
    enabled: {str(afrr_enabled).lower()}
    direction: "down"
    product_resolution: "15min"
    gate_close:
      relative_to_delivery_start_minutes: -25
    product_rules:
      min_bid_mw: 1.0
      bid_increment_mw: 1.0
      validity_period_minutes: 15  # <-- DAS HATTE GEFEHLT!
    signals:
      price: "aFRR_energy_down_price"
      system_activation: "aFRR_energy_down_quantity"
    interpretation:
      activation_unit: "MW"

  afrr_capacity:
    enabled: {str(afrr_capacity_enabled).lower()}
    direction: "down"
    product_length: "4h"
    price_unit: "EUR_per_MW_per_h"
    gate_open:
      day_relation: "D-7"
      time: "10:00"
    gate_close:
      day_relation: "D-1"
      time: "09:00"
    product_rules:
      min_bid_mw: 1.0
      bid_increment_mw: 1.0
      divisible: true
    signals:
      price: "aFRR_capacity_down_price"
""".strip()
        ),
        encoding="utf-8",
    )


def _write_plants(
    path: Path,
    storage_initial_soc: float = 0.0,
    storage_capacity: float = 4.0,
) -> None:
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
                "max_capacity": storage_capacity,
                "min_capacity": 0.0,
                "max_power_charge": 7.0,
                "max_power_discharge": 7.0,
                "initial_soc": storage_initial_soc,
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


def _write_forecasts(
    path: Path,
    da_prices: list[float] | None = None,
    idc_prices: list[float | None] | None = None,
    afrr_prices: list[float | None] | None = None,
    afrr_quantities: list[float | None] | None = None,
    afrr_capacity_prices: list[float | None] | None = None,
    heat_demand: list[float] | None = None,
) -> None:
    datetimes = pd.date_range("2025-01-01 00:00", periods=8, freq="15min")
    da_prices = da_prices or [10.0] * 4 + [120.0] * 4
    forecasts = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_1_heat_demand": heat_demand or [2.0] * 8,
            "DE_day_ahead_price_EUR_per_MWh_el": da_prices,
            "natural_gas_price": [80.0] * 8,
        }
    )
    if idc_prices is not None:
        forecasts["DE_ID3_price"] = idc_prices
    if afrr_prices is not None:
        forecasts["aFRR_energy_down_price"] = afrr_prices
    if afrr_quantities is not None:
        forecasts["aFRR_energy_down_quantity"] = afrr_quantities
    if afrr_capacity_prices is not None:
        forecasts["aFRR_capacity_down_price"] = afrr_capacity_prices
    forecasts.to_csv(path, index=False)
