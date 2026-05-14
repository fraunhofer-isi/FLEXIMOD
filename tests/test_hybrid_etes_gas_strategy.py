# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.simulation.simulation_runner import OutputOptions, SimulationRunner


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
def idc_case(tmp_path: Path) -> Path:
    """Create a tiny DA + IDC case where IDC can be varied by each test."""

    case_dir = tmp_path / "idc_case"
    case_dir.mkdir()
    _write_config(case_dir / "config.yaml", idc_enabled=True)
    _write_plants(case_dir / "plants.csv")
    return case_dir


@pytest.fixture
def afrr_case(tmp_path: Path) -> Path:
    """Create a tiny DA + IDC + aFRR down case."""

    case_dir = tmp_path / "afrr_case"
    case_dir.mkdir()
    _write_config(case_dir / "config.yaml", idc_enabled=True, afrr_enabled=True)
    _write_plants(case_dir / "plants.csv")
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


def test_idc_disabled_keeps_day_ahead_position(
    day_ahead_only_results: dict[str, pd.DataFrame],
) -> None:
    market = day_ahead_only_results["market"]

    assert market["IDC_buy_MWh"].sum() == pytest.approx(0.0)
    assert market["IDC_sell_MWh"].sum() == pytest.approx(0.0)
    assert market["final_planned_electricity_MWh"].sum() == pytest.approx(
        market["DA_position_MWh"].sum()
    )


def test_cheap_idc_creates_incremental_buy(idc_case: Path, tmp_path: Path) -> None:
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[20.0] * 8,
    )

    results = _run_case(idc_case, tmp_path)
    market = results["market"]

    assert market["IDC_buy_MWh"].sum() > 0.0
    assert market["IDC_sell_MWh"].sum() == pytest.approx(0.0)
    _assert_final_planned_balance(market)
    _assert_heat_is_feasible(results["dispatch"])


def test_expensive_idc_creates_sell_without_exceeding_da(
    idc_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[10.0] * 8,
        idc_prices=[120.0] * 8,
    )

    results = _run_case(idc_case, tmp_path)
    market = results["market"]

    assert market["IDC_sell_MWh"].sum() > 0.0
    assert market["IDC_buy_MWh"].sum() == pytest.approx(0.0)
    assert (market["IDC_sell_MWh"] <= market["DA_position_MWh"] + 1e-8).all()
    _assert_final_planned_balance(market)
    _assert_heat_is_feasible(results["dispatch"])


def test_neutral_idc_creates_no_adjustment(idc_case: Path, tmp_path: Path) -> None:
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[10.0] * 8,
        idc_prices=[75.0] * 8,
    )

    results = _run_case(idc_case, tmp_path)
    market = results["market"]

    assert market["IDC_buy_MWh"].sum() == pytest.approx(0.0)
    assert market["IDC_sell_MWh"].sum() == pytest.approx(0.0)
    _assert_final_planned_balance(market)


def test_missing_idc_values_create_no_action_timestep(
    idc_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[20.0, None, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    )

    with pytest.warns(UserWarning, match="IDC price contains missing values"):
        results = _run_case(idc_case, tmp_path)
    market = results["market"].sort_values("datetime")
    missing_price_row = market[market["IDC_price"].isna()].iloc[0]

    assert missing_price_row["IDC_buy_MWh"] == pytest.approx(0.0)
    assert missing_price_row["IDC_sell_MWh"] == pytest.approx(0.0)


def test_missing_idc_price_column_raises_clear_error(
    idc_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[20.0] * 8,
    )
    forecasts = pd.read_csv(idc_case / "forecasts_df.csv")
    forecasts = forecasts.drop(columns=["DE_ID3_price"])
    forecasts.to_csv(idc_case / "forecasts_df.csv", index=False)

    with pytest.raises(ValueError, match="DE_ID3_price"):
        _run_case(idc_case, tmp_path)


def test_afrr_disabled_keeps_final_planned_electricity(
    idc_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
    )

    results = _run_case(idc_case, tmp_path)
    market = results["market"]

    assert market["afrr_energy_bid_MWh"].sum() == pytest.approx(0.0)
    assert market["afrr_energy_activated_MWh"].sum() == pytest.approx(0.0)
    assert market["actual_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        market["final_planned_electricity_MWh"].to_numpy()
    )


def test_cheap_afrr_down_creates_proxy_activation(
    afrr_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[20.0] * 8,
        afrr_quantities=[2.0] * 8,
        heat_demand=[0.0] * 8,
    )

    results = _run_case(afrr_case, tmp_path)
    market = results["market"]
    dispatch = results["dispatch"]

    assert market["afrr_energy_bid_MWh"].sum() > 0.0
    assert market["afrr_energy_activated_MWh"].sum() > 0.0
    assert (market["afrr_energy_activated_MWh"] <= market["afrr_energy_bid_MWh"] + 1e-8).all()
    assert (
        market["afrr_energy_activated_MWh"]
        <= market["afrr_down_system_activation_MWh_clean"] + 1e-8
    ).all()
    _assert_actual_electricity_with_afrr(market)
    assert dispatch["etes_charge_MWh"].to_numpy() == pytest.approx(
        dispatch["actual_electricity_consumption_MWh"].to_numpy()
    )
    _assert_heat_is_feasible(dispatch)


def test_expensive_afrr_down_creates_no_bid_or_activation(
    afrr_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[120.0] * 8,
        afrr_quantities=[2.0] * 8,
    )

    results = _run_case(afrr_case, tmp_path)
    market = results["market"]

    assert market["afrr_energy_bid_MWh"].sum() == pytest.approx(0.0)
    assert market["afrr_energy_activated_MWh"].sum() == pytest.approx(0.0)
    _assert_actual_electricity_with_afrr(market)


def test_afrr_missing_price_blocks_bid_even_with_activation(
    afrr_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[None] + [20.0] * 7,
        afrr_quantities=[2.0] * 8,
    )

    with pytest.warns(UserWarning, match="aFRR down price contains missing values"):
        results = _run_case(afrr_case, tmp_path)
    first = results["market"].sort_values("datetime").iloc[0]

    assert first["afrr_energy_bid_MWh"] == pytest.approx(0.0)
    assert first["afrr_energy_activated_MWh"] == pytest.approx(0.0)
    assert first["afrr_energy_price_clean"] == pytest.approx(0.0)
    assert first["afrr_data_quality_flag"] == "activation_without_price"


def test_afrr_minimum_bid_rule_uses_mw_headroom(
    afrr_case: Path,
    tmp_path: Path,
) -> None:
    _write_plants(afrr_case / "plants.csv", storage_initial_soc=3.8)
    _write_forecasts(
        afrr_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[20.0] * 8,
        afrr_quantities=[2.0] * 8,
        heat_demand=[0.0] * 8,
    )

    results = _run_case(afrr_case, tmp_path)
    first = results["market"].sort_values("datetime").iloc[0]

    assert first["afrr_energy_bid_MWh"] == pytest.approx(0.0)
    assert first["afrr_energy_activated_MWh"] == pytest.approx(0.0)


def test_afrr_bid_uses_storage_capacity_headroom(
    afrr_case: Path,
    tmp_path: Path,
) -> None:
    _write_plants(afrr_case / "plants.csv", storage_initial_soc=3.6)
    _write_forecasts(
        afrr_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[20.0] * 8,
        afrr_quantities=[10.0] * 8,
        heat_demand=[0.0] * 8,
    )

    results = _run_case(afrr_case, tmp_path)
    market = results["market"].sort_values("datetime")
    first = market.iloc[0]

    assert first["afrr_energy_bid_MWh"] <= (4.0 - 3.6) / 0.92 + 1e-8
    _assert_actual_electricity_with_afrr(market)


def test_missing_afrr_down_price_column_raises_clear_error(
    afrr_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[20.0] * 8,
        afrr_quantities=[2.0] * 8,
    )
    forecasts = pd.read_csv(afrr_case / "forecasts_df.csv")
    forecasts = forecasts.drop(columns=["aFRR_energy_down_price"])
    forecasts.to_csv(afrr_case / "forecasts_df.csv", index=False)

    with pytest.raises(ValueError, match="aFRR_energy_down_price"):
        _run_case(afrr_case, tmp_path)


def test_missing_afrr_down_activation_column_raises_clear_error(
    afrr_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[20.0] * 8,
        afrr_quantities=[2.0] * 8,
    )
    forecasts = pd.read_csv(afrr_case / "forecasts_df.csv")
    forecasts = forecasts.drop(columns=["aFRR_energy_down_quantity"])
    forecasts.to_csv(afrr_case / "forecasts_df.csv", index=False)

    with pytest.raises(ValueError, match="aFRR_energy_down_quantity"):
        _run_case(afrr_case, tmp_path)


def _run_case(case_dir: Path, tmp_path: Path) -> dict[str, pd.DataFrame]:
    runner = SimulationRunner(
        case_dir=case_dir,
        input_dir=case_dir,
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


def _assert_final_planned_balance(market: pd.DataFrame) -> None:
    expected = market["DA_position_MWh"] + market["IDC_buy_MWh"] - market["IDC_sell_MWh"]
    assert market["final_planned_electricity_MWh"].to_numpy() == pytest.approx(expected.to_numpy())
    assert market["actual_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        market["final_planned_electricity_MWh"].to_numpy()
    )


def _assert_heat_is_feasible(dispatch: pd.DataFrame) -> None:
    supplied_heat = dispatch["gas_heat_MWh"] + dispatch["etes_discharge_MWh"]
    assert (supplied_heat >= dispatch["heat_demand_MWh"] - 1e-8).all()
    assert dispatch["unmet_heat_MWh"].sum() == pytest.approx(0.0)


def _assert_actual_electricity_with_afrr(market: pd.DataFrame) -> None:
    expected = market["final_planned_electricity_MWh"] + market["afrr_energy_activated_MWh"]
    assert market["actual_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        expected.to_numpy()
    )


def _write_config(path: Path, idc_enabled: bool = False, afrr_enabled: bool = False) -> None:
    path.write_text(
        f"""
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
    enabled: {str(idc_enabled).lower()}
    product_resolution: "15min"
    gate_close:
      relative_to_delivery_start_minutes: -5
    signals:
      price: "DE_ID3_price"
      volume: "DE_ID3_volume"

  afrr_energy:
    enabled: {str(afrr_enabled).lower()}
    direction: "down"
    product_resolution: "15min"
    gate_close:
      relative_to_delivery_start_minutes: -25
    product_rules:
      min_bid_mw: 1.0
      bid_increment_mw: 1.0
      validity_period_minutes: 15
    signals:
      price: "aFRR_energy_down_price"
      system_activation: "aFRR_energy_down_quantity"
    interpretation:
      activation_unit: "MW"

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


def _write_plants(path: Path, storage_initial_soc: float = 0.0) -> None:
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
    heat_demand: list[float] | None = None,
) -> None:
    datetimes = pd.date_range("2025-01-01 00:00", periods=8, freq="15min")
    da_prices = da_prices or [10.0] * 4 + [120.0] * 4
    forecasts = pd.DataFrame(
        {
            "datetime": datetimes,
            "plant_1_heat_demand": heat_demand or [2.0] * 8,
            "DE_DA_price": da_prices,
            "natural_gas_price": [80.0] * 8,
        }
    )
    if idc_prices is not None:
        forecasts["DE_ID3_price"] = idc_prices
    if afrr_prices is not None:
        forecasts["aFRR_energy_down_price"] = afrr_prices
    if afrr_quantities is not None:
        forecasts["aFRR_energy_down_quantity"] = afrr_quantities
    forecasts.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
