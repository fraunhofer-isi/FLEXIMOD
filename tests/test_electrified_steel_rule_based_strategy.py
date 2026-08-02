# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.simulation.simulation_runner import SimulationRunner
from flexi_mod.strategies.electrified_steel_rule_based_strategy import (
    _capacity_reserved_mw_by_block,
)


def test_electrified_steel_rule_based_electricity_accounting_is_consistent(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "rule_based"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_config(), encoding="utf-8")
    _case_a_plants().to_csv(case_dir / "plants.csv", index=False)
    _case_a_forecasts().to_csv(case_dir / "forecasts_df.csv", index=False)

    outputs = SimulationRunner(case_dir, output_dir=tmp_path / "output").run()
    dispatch = pd.read_csv(outputs["dispatch_results"])
    ledger = pd.read_csv(outputs["market_ledger"])

    assert dispatch["steel_output_t"].sum() == pytest.approx(4.0)
    assert dispatch["final_planned_electricity_MWh"].to_numpy() == pytest.approx(
        dispatch["DA_position_MWh"].to_numpy()
    )
    assert dispatch["actual_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        (
            dispatch["final_planned_electricity_MWh"] + dispatch["afrr_energy_activated_MWh"]
        ).to_numpy()
    )
    assert dispatch["total_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        dispatch["actual_electricity_consumption_MWh"].to_numpy()
    )
    assert dispatch["afrr_energy_capacity_backed_bid_MWh"].to_numpy() == pytest.approx(
        dispatch["afrr_capacity_reserved_MWh"].to_numpy()
    )
    assert ledger["day_ahead_position_MWh_el"].to_numpy() == pytest.approx(
        dispatch["DA_position_MWh"].to_numpy()
    )
    assert ledger["actual_electricity_consumption_MWh_el"].to_numpy() == pytest.approx(
        dispatch["total_electricity_consumption_MWh"].to_numpy()
    )


def test_case_a_benchmark_gate_switches_fuel_route(tmp_path: Path) -> None:
    """Below the gas benchmark the electrolyser runs; above it, gas takes over."""
    case_dir = tmp_path / "benchmark_gate"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_config(), encoding="utf-8")
    _case_a_plants().to_csv(case_dir / "plants.csv", index=False)
    forecasts = _case_a_forecasts()
    # natural_gas_price=60, co2_price=80, natural_gas_co2_factor=0.2, specific_natural_gas=3.0
    # -> gas_route_cost_per_t = 3.0*(60+0.2*80) = 228; specific_hydrogen=2.0, efficiency=0.8
    # -> benchmark = 228 / (2.0/0.8) = 91.2 EUR/MWh_el
    forecasts["DE_DA_price"] = [150.0] * 16 + [10.0] * 16  # above / below the 91.2 benchmark
    forecasts["aFRR_energy_down_price"] = 200.0  # never aFRR-favorable, isolates the DA gate
    forecasts.to_csv(case_dir / "forecasts_df.csv", index=False)

    outputs = SimulationRunner(case_dir, output_dir=tmp_path / "output").run()
    dispatch = pd.read_csv(outputs["dispatch_results"])

    assert dispatch["steel_output_t"].sum() == pytest.approx(4.0)
    expensive_window = dispatch.iloc[:16]
    cheap_window = dispatch.iloc[16:]
    assert expensive_window["electrolyser_electricity_consumption_MWh"].sum() == pytest.approx(0.0)
    assert cheap_window["natural_gas_consumption_MWh"].sum() == pytest.approx(0.0)
    assert cheap_window["electrolyser_electricity_consumption_MWh"].sum() > 0.0


def test_case_b_routing_prefers_cheaper_of_da_and_afrr(tmp_path: Path) -> None:
    """No hybrid fuel choice: aFRR substitutes for day-ahead only where it's cheaper."""
    case_dir = tmp_path / "case_b_expensive"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_config(), encoding="utf-8")
    _case_b_plants().to_csv(case_dir / "plants.csv", index=False)
    forecasts = _case_b_forecasts()
    forecasts["aFRR_energy_down_price"] = 500.0  # always more expensive than DA
    forecasts.to_csv(case_dir / "forecasts_df.csv", index=False)

    outputs = SimulationRunner(case_dir, output_dir=tmp_path / "output").run()
    dispatch = pd.read_csv(outputs["dispatch_results"])

    assert dispatch["DA_position_MWh"].sum() == pytest.approx(
        dispatch["total_electricity_consumption_MWh"].sum()
    )
    assert dispatch["afrr_energy_activated_MWh"].sum() == pytest.approx(0.0)


def test_case_b_substitutes_afrr_when_cheaper_than_day_ahead(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_b_cheap"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_config(), encoding="utf-8")
    _case_b_plants().to_csv(case_dir / "plants.csv", index=False)
    forecasts = _case_b_forecasts()
    forecasts["DE_DA_price"] = 100.0
    forecasts["aFRR_energy_down_price"] = 20.0  # cheaper than DA everywhere
    forecasts["aFRR_energy_down_quantity"] = 100.0  # ample activation headroom
    forecasts.to_csv(case_dir / "forecasts_df.csv", index=False)

    outputs = SimulationRunner(case_dir, output_dir=tmp_path / "output").run()
    dispatch = pd.read_csv(outputs["dispatch_results"])

    assert dispatch["afrr_energy_activated_MWh"].sum() > 0.0
    assert dispatch["DA_position_MWh"].sum() < dispatch["total_electricity_consumption_MWh"].sum()


def _block_inputs(
    *,
    n_blocks: int = 2,
    steps_per_block: int = 16,
    total_mwh: float = 2.0,
    activation_mwh: float = 1.0,
    capacity_price: float = 10.0,
    favorable: bool = True,
) -> dict[str, pd.Series]:
    total_steps = n_blocks * steps_per_block
    block_id = pd.Series(
        [f"block_{i // steps_per_block}" for i in range(total_steps)]
    )
    return {
        "capacity_block_id": block_id,
        "total": pd.Series([total_mwh] * total_steps),
        "afrr_cheaper_than_da": pd.Series([favorable] * total_steps),
        "afrr_system_activation": pd.Series([activation_mwh] * total_steps),
        "capacity_price": pd.Series([capacity_price] * total_steps),
        "capacity_missing_price": pd.Series([False] * total_steps),
    }


def test_capacity_eligible_when_technical_and_price_and_energy_favorable():
    """A block that is technically deliverable, positively priced, and consistently
    aFRR-energy-favorable throughout should be reservable."""
    inputs = _block_inputs()

    reserved = _capacity_reserved_mw_by_block(
        capacity_product_duration_h=4.0,
        timestep_hours=0.25,
        capacity_min_bid_mw=1.0,
        capacity_bid_increment_mw=1.0,
        **inputs,
    )

    assert all(mw > 0.0 for mw in reserved.values())


def test_capacity_block_requires_every_timestep_energy_favorable():
    """One non-favorable quarter-hour disqualifies the whole block's capacity.

    This coupling was found to be necessary, not merely conservative: a version
    that dropped it let the heuristic claim more capacity revenue than the exact
    MILP's own provably-optimal choice on real data, because the MILP earns extra
    capacity revenue by jointly re-timing production around the commitment -- a
    trade-off this sequential (production-first) design cannot see. Requiring
    energy favorability throughout the block is what keeps the result a valid
    (if understated) lower bound instead of an unrealistic one.
    """
    inputs = _block_inputs(n_blocks=1)
    favorable = inputs["afrr_cheaper_than_da"].copy()
    favorable.iloc[5] = False
    inputs["afrr_cheaper_than_da"] = favorable

    reserved = _capacity_reserved_mw_by_block(
        capacity_product_duration_h=4.0,
        timestep_hours=0.25,
        capacity_min_bid_mw=1.0,
        capacity_bid_increment_mw=1.0,
        **inputs,
    )

    assert all(mw == 0.0 for mw in reserved.values())


def test_capacity_ineligible_when_price_missing_or_zero():
    inputs = _block_inputs(n_blocks=1)
    inputs["capacity_missing_price"] = pd.Series([True] * len(inputs["total"]))

    reserved = _capacity_reserved_mw_by_block(
        capacity_product_duration_h=4.0,
        timestep_hours=0.25,
        capacity_min_bid_mw=1.0,
        capacity_bid_increment_mw=1.0,
        **inputs,
    )

    assert all(mw == 0.0 for mw in reserved.values())


def test_capacity_ineligible_when_no_activation_expected():
    inputs = _block_inputs(n_blocks=1, activation_mwh=0.0)

    reserved = _capacity_reserved_mw_by_block(
        capacity_product_duration_h=4.0,
        timestep_hours=0.25,
        capacity_min_bid_mw=1.0,
        capacity_bid_increment_mw=1.0,
        **inputs,
    )

    assert all(mw == 0.0 for mw in reserved.values())


def test_capacity_bound_uses_block_minimum_and_never_exceeds_any_timestep():
    """Reserved capacity must be deliverable at *every* quarter-hour of the block, so
    it is bounded by the block's minimum draw, not its average -- a real, deliverable
    retainer can never promise more than the leanest moment can back."""
    inputs = _block_inputs(n_blocks=1, total_mwh=4.0)
    total = inputs["total"].copy()
    total.iloc[0] = 0.5  # one much leaner quarter-hour than the rest of the block
    inputs["total"] = total

    reserved = _capacity_reserved_mw_by_block(
        capacity_product_duration_h=4.0,
        timestep_hours=0.25,
        capacity_min_bid_mw=1.0,
        capacity_bid_increment_mw=1.0,
        **inputs,
    )

    reserved_mw = next(iter(reserved.values()))
    # 0.5 MWh / 0.25h = 2 MW technical ceiling; must not exceed the leanest timestep.
    assert reserved_mw == pytest.approx(2.0)
    assert reserved_mw * 0.25 <= total.min()


def test_electrified_steel_rule_based_accepted_by_config_validation(tmp_path: Path) -> None:
    case_dir = tmp_path / "config_only"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_config(), encoding="utf-8")

    config = CaseConfig.from_case_dir(case_dir, study_case="electrified_steel_rb")
    assert config.strategy_name == "electrified_steel_rule_based"


def _config() -> str:
    return """
cases:
  electrified_steel_rb:
    name: electrified_steel_rb
    country: DE
    timestep_minutes: 15
    simulation_start: "2025-01-01 00:00"
    simulation_end: "2025-01-01 07:45"
    additional_charges: false
    strategy:
      name: electrified_steel_rule_based
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


def _case_a_plants() -> pd.DataFrame:
    """Hybrid electrolyser + natural-gas DRI route -- classified Case A."""
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
                "fuel_type": "hybrid_hydrogen_natural_gas",
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
    )


def _case_a_forecasts() -> pd.DataFrame:
    periods = 32
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=periods, freq="15min"),
            "DE_DA_price": [100.0] * periods,
            "aFRR_capacity_down_price": [10.0] * periods,
            "aFRR_energy_down_price": [50.0] * periods,
            "aFRR_energy_down_quantity": [1.0] * periods,
            "iron_ore_price": [100.0] * periods,
            "lime_price": [20.0] * periods,
            "co2_price": [80.0] * periods,
            "natural_gas_price": [60.0] * periods,
            "hydrogen_price": [70.0] * periods,
        }
    )


def _case_b_plants() -> pd.DataFrame:
    """BF-BOF route: no electrolyser, no fuel substitution -- classified Case B."""
    return pd.DataFrame(
        [
            {
                "name": "steel_1",
                "unit_type": "steel_plant",
                "node": "north",
                "objective": "min_variable_cost",
                "steel_demand": 4.0,
                "technology": "bf_bof",
                "fuel_type": "natural_gas",
                "specific_natural_gas_consumption": 3.0,
                "specific_electricity_consumption": 0.2,
                "specific_iron_ore_consumption": 1.5,
                "specific_lime_demand": 0.05,
                "natural_gas_co2_factor": 0.2,
                "lime_co2_factor": 0.1,
                "max_power": 10.0,
                "min_power": 0.0,
            }
        ]
    )


def _case_b_forecasts() -> pd.DataFrame:
    periods = 32
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=periods, freq="15min"),
            "DE_DA_price": [100.0] * periods,
            "aFRR_capacity_down_price": [10.0] * periods,
            "aFRR_energy_down_price": [50.0] * periods,
            "aFRR_energy_down_quantity": [1.0] * periods,
            "iron_ore_price": [100.0] * periods,
            "lime_price": [20.0] * periods,
            "co2_price": [80.0] * periods,
            "natural_gas_price": [60.0] * periods,
            "hydrogen_price": [70.0] * periods,
        }
    )
