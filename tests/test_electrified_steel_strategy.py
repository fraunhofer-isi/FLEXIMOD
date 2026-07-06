# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.simulation.simulation_runner import SimulationRunner


def test_electrified_steel_cross_market_dispatch_and_ontology(tmp_path: Path) -> None:
    case_dir = tmp_path / "electrified_steel"
    output_dir = tmp_path / "output"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_config(), encoding="utf-8")
    _plants().to_csv(case_dir / "plants.csv", index=False)
    _forecasts().to_csv(case_dir / "forecasts_df.csv", index=False)

    outputs = SimulationRunner(case_dir, output_dir=output_dir).run()

    dispatch = pd.read_csv(outputs["dispatch_results"])
    ledger = pd.read_csv(outputs["market_ledger"])
    capacity = pd.read_csv(outputs["afrr_capacity_block_summary"])
    assert dispatch["steel_output_t"].sum() == pytest.approx(4.0)
    assert dispatch["natural_gas_consumption_MWh"].sum() == pytest.approx(0.0)
    assert dispatch["final_planned_electricity_MWh"].to_numpy() == pytest.approx(
        dispatch["DA_position_MWh"].to_numpy()
    )
    assert dispatch["actual_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        (
            dispatch["final_planned_electricity_MWh"]
            + dispatch["afrr_energy_activated_MWh"]
        ).to_numpy()
    )
    assert dispatch["total_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        dispatch["actual_electricity_consumption_MWh"].to_numpy()
    )
    assert (dispatch["afrr_capacity_reserved_MW"] > 0).all()
    assert dispatch["afrr_energy_free_bid_MWh"].sum() == pytest.approx(0.0)
    assert dispatch["afrr_energy_capacity_backed_bid_MWh"].to_numpy() == pytest.approx(
        dispatch["afrr_capacity_reserved_MWh"].to_numpy()
    )
    assert dispatch["afrr_energy_activated_MWh"].to_numpy() == pytest.approx(
        dispatch[
            ["afrr_energy_bid_MWh", "afrr_system_activation_MWh"]
        ].min(axis=1).to_numpy()
    )
    assert set(dispatch["afrr_capacity_pricing_rule"]) == {"pay_as_bid"}
    assert (
        dispatch["afrr_energy_bid_price_EUR_per_MWh"]
        != dispatch["afrr_energy_price_EUR_per_MWh"]
    ).all()
    assert ledger["day_ahead_position_MWh_el"].to_numpy() == pytest.approx(
        dispatch["DA_position_MWh"].to_numpy()
    )
    assert ledger["actual_electricity_consumption_MWh_el"].to_numpy() == pytest.approx(
        dispatch["total_electricity_consumption_MWh"].to_numpy()
    )
    assert capacity["capacity_pricing_rule"].eq("pay_as_bid").all()


def test_electrified_steel_rejects_gas_fallback(tmp_path: Path) -> None:
    case_dir = tmp_path / "gas_fallback"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_config(), encoding="utf-8")
    plants = _plants()
    plants.loc[plants["technology"] == "dri_plant", "fuel_type"] = "both"
    plants.to_csv(case_dir / "plants.csv", index=False)
    _forecasts().to_csv(case_dir / "forecasts_df.csv", index=False)

    with pytest.raises(ValueError, match="fuel_type.*hydrogen"):
        SimulationRunner(case_dir, output_dir=tmp_path / "output").run()


def test_free_energy_bid_is_compliant_and_activation_can_be_partial(tmp_path: Path) -> None:
    case_dir = tmp_path / "partial_activation"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_config(), encoding="utf-8")
    _plants().to_csv(case_dir / "plants.csv", index=False)
    forecasts = _forecasts()
    forecasts["aFRR_capacity_down_price"] = 0.0
    forecasts["aFRR_energy_down_quantity"] = 0.5
    forecasts.to_csv(case_dir / "forecasts_df.csv", index=False)

    outputs = SimulationRunner(case_dir, output_dir=tmp_path / "output").run()
    dispatch = pd.read_csv(outputs["dispatch_results"])
    positive = dispatch.loc[dispatch["afrr_energy_free_bid_MWh"] > 0]

    assert dispatch["afrr_capacity_reserved_MW"].sum() == pytest.approx(0.0)
    assert not positive.empty
    assert positive["afrr_energy_bid_MW"].to_numpy() == pytest.approx(
        positive["afrr_energy_bid_MW"].round().to_numpy()
    )
    assert (
        positive["afrr_energy_activated_MWh"] < positive["afrr_energy_bid_MWh"]
    ).all()
    assert positive["afrr_energy_activated_MWh"].to_numpy() == pytest.approx(
        positive["afrr_system_activation_MWh"].to_numpy()
    )


def test_expensive_afrr_energy_produces_no_free_bid(tmp_path: Path) -> None:
    case_dir = tmp_path / "expensive_afrr"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_config(), encoding="utf-8")
    _plants().to_csv(case_dir / "plants.csv", index=False)
    forecasts = _forecasts()
    forecasts["aFRR_capacity_down_price"] = 0.0
    forecasts["aFRR_energy_down_price"] = 200.0
    forecasts.to_csv(case_dir / "forecasts_df.csv", index=False)

    outputs = SimulationRunner(case_dir, output_dir=tmp_path / "output").run()
    dispatch = pd.read_csv(outputs["dispatch_results"])

    assert dispatch["afrr_energy_free_bid_MWh"].sum() == pytest.approx(0.0)
    assert dispatch["afrr_energy_activated_MWh"].sum() == pytest.approx(0.0)
    assert dispatch["DA_position_MWh"].sum() == pytest.approx(
        dispatch["total_electricity_consumption_MWh"].sum()
    )


def _config() -> str:
    return """
cases:
  electrified_steel:
    name: electrified_steel
    country: DE
    timestep_minutes: 15
    simulation_start: "2025-01-01 00:00"
    simulation_end: "2025-01-01 07:45"
    additional_charges: false
    strategy:
      name: electrified_steel
      dispatch:
        dispatch_method: pyomo
        rolling_horizon_enabled: true
        dispatch_horizon_hours: 8
        rolling_step_hours: 4
    solver:
      name: highs
      fallback_solvers: []
      tee: false
    market_sequence:
      - afrr_capacity
      - day_ahead
      - afrr_energy
    markets:
      afrr_capacity:
        enabled: true
        direction: down
        product_length: 4h
        price_unit: EUR_per_MW_per_h
        gate_open:
          day_relation: D-7
          time: "10:00"
        gate_close:
          day_relation: D-1
          time: "09:00"
        product_rules:
          min_bid_mw: 1.0
          bid_increment_mw: 1.0
          divisible: true
        signals:
          price: aFRR_capacity_down_price
      day_ahead:
        enabled: true
        product_resolution: 15min
        gate_close:
          day_relation: D-1
          time: "12:00"
        signals:
          price: DE_DA_price
      afrr_energy:
        enabled: true
        direction: down
        product_resolution: 15min
        gate_close:
          relative_to_delivery_start_minutes: -25
        product_rules:
          min_bid_mw: 1.0
          bid_increment_mw: 1.0
          validity_period_minutes: 15
        signals:
          price: aFRR_energy_down_price
          system_activation: aFRR_energy_down_quantity
        interpretation:
          activation_unit: MW
""".strip()


def _plants() -> pd.DataFrame:
    shared = {
        "name": "steel_1",
        "unit_type": "steel_plant",
        "node": "north",
        "objective": "min_variable_cost",
        "steel_demand": 4.0,
    }
    return pd.DataFrame(
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
            {
                **shared,
                "technology": "hydrogen_buffer_storage",
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


def _forecasts() -> pd.DataFrame:
    periods = 32
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=periods, freq="15min"),
            "DE_DA_price": [100.0] * periods,
            "aFRR_capacity_down_price": [10.0] * periods,
            "aFRR_energy_down_price": [0.0] * periods,
            "aFRR_energy_down_quantity": [1.0] * periods,
            "iron_ore_price": [100.0] * periods,
            "lime_price": [20.0] * periods,
            "co2_price": [80.0] * periods,
        }
    )
