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
def afrr_capacity_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "afrr_capacity_case"
    case_dir.mkdir()
    _write_config(
        case_dir / "config.yaml",
        idc_enabled=True,
        afrr_enabled=True,
        afrr_capacity_enabled=True,
    )
    _write_plants(case_dir / "plants.csv", storage_capacity=24.0)
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
    assert supplied_heat.to_numpy() == pytest.approx(dispatch["heat_demand_MWh"].to_numpy())
    assert "unmet_heat_MWh" not in dispatch.columns
    assert "excess_heat_MWh" not in dispatch.columns

    # In the DA-only MVP, the market position is exactly the optimized ETES electricity use.
    assert market["day_ahead_position_MWh_el"].sum() == pytest.approx(
        dispatch["electricity_consumption_MWh"].sum()
    )
    assert market["actual_electricity_consumption_MWh_el"].sum() == pytest.approx(
        market["day_ahead_position_MWh_el"].sum()
    )

    # Stored heat is attributed to the day-ahead market and CO2 is inactive for this MVP.
    assert storage["thermal_inventory_day_ahead_MWh_th"].max() > 0
    assert summary["total_DA_electricity_MWh"].iloc[0] == pytest.approx(
        market["day_ahead_position_MWh_el"].sum()
    )
    assert "total_unmet_heat_MWh" not in summary.columns
    assert "total_excess_heat_MWh" not in summary.columns
    assert summary["total_co2_cost_EUR"].iloc[0] == pytest.approx(0.0)


def test_idc_disabled_keeps_day_ahead_position(
    day_ahead_only_results: dict[str, pd.DataFrame],
) -> None:
    market = day_ahead_only_results["market"]

    assert market["intraday_buy_MWh_el"].sum() == pytest.approx(0.0)
    assert market["intraday_sell_MWh_el"].sum() == pytest.approx(0.0)
    assert market["scheduled_electricity_procurement_MWh_el"].sum() == pytest.approx(
        market["day_ahead_position_MWh_el"].sum()
    )


def test_cheap_idc_creates_incremental_buy(idc_case: Path, tmp_path: Path) -> None:
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[20.0] * 8,
    )

    results = _run_case(idc_case, tmp_path)
    market = results["market"]

    assert market["intraday_buy_MWh_el"].sum() > 0.0
    assert market["intraday_sell_MWh_el"].sum() == pytest.approx(0.0)
    _assert_final_planned_balance(market)
    _assert_heat_is_feasible(results["dispatch"])


def test_buy_only_idc_allows_cheap_buy(idc_case: Path, tmp_path: Path) -> None:
    _write_config(
        idc_case / "config.yaml",
        idc_enabled=True,
        idc_buy_enabled=True,
        idc_sell_enabled=False,
    )
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[20.0] * 8,
    )

    results = _run_case(idc_case, tmp_path)
    market = results["market"]

    assert market["intraday_buy_MWh_el"].sum() > 0.0
    assert market["intraday_sell_MWh_el"].sum() == pytest.approx(0.0)
    _assert_final_planned_balance(market)


def test_buy_only_idc_blocks_expensive_sell(idc_case: Path, tmp_path: Path) -> None:
    _write_config(
        idc_case / "config.yaml",
        idc_enabled=True,
        idc_buy_enabled=True,
        idc_sell_enabled=False,
    )
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[10.0] * 8,
        idc_prices=[120.0] * 8,
    )

    results = _run_case(idc_case, tmp_path)
    market = results["market"]

    assert market["intraday_buy_MWh_el"].sum() == pytest.approx(0.0)
    assert market["intraday_sell_MWh_el"].sum() == pytest.approx(0.0)
    _assert_final_planned_balance(market)


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

    assert market["intraday_sell_MWh_el"].sum() > 0.0
    assert market["intraday_buy_MWh_el"].sum() == pytest.approx(0.0)
    assert (market["intraday_sell_MWh_el"] <= market["day_ahead_position_MWh_el"] + 1e-8).all()
    _assert_final_planned_balance(market)
    _assert_heat_is_feasible(results["dispatch"])


def test_sell_only_idc_allows_sell_and_blocks_buy(idc_case: Path, tmp_path: Path) -> None:
    _write_config(
        idc_case / "config.yaml",
        idc_enabled=True,
        idc_buy_enabled=False,
        idc_sell_enabled=True,
    )
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[10.0] * 4 + [120.0] * 4,
        idc_prices=[120.0] * 4 + [20.0] * 4,
    )

    results = _run_case(idc_case, tmp_path)
    market = results["market"]

    assert market["intraday_sell_MWh_el"].sum() > 0.0
    assert market["intraday_buy_MWh_el"].sum() == pytest.approx(0.0)
    assert (market["intraday_sell_MWh_el"] <= market["day_ahead_position_MWh_el"] + 1e-8).all()
    _assert_final_planned_balance(market)


def test_observe_only_idc_records_price_without_trading(
    idc_case: Path,
    tmp_path: Path,
) -> None:
    _write_config(
        idc_case / "config.yaml",
        idc_enabled=True,
        idc_buy_enabled=False,
        idc_sell_enabled=False,
    )
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[10.0] * 4 + [120.0] * 4,
        idc_prices=[120.0] * 4 + [20.0] * 4,
    )

    results = _run_case(idc_case, tmp_path)
    market = results["market"]

    assert market["intraday_price_EUR_per_MWh_el"].notna().all()
    assert market["intraday_buy_MWh_el"].sum() == pytest.approx(0.0)
    assert market["intraday_sell_MWh_el"].sum() == pytest.approx(0.0)
    _assert_final_planned_balance(market)


def test_neutral_idc_creates_no_adjustment(idc_case: Path, tmp_path: Path) -> None:
    _write_forecasts(
        idc_case / "forecasts_df.csv",
        da_prices=[10.0] * 8,
        idc_prices=[75.0] * 8,
    )

    results = _run_case(idc_case, tmp_path)
    market = results["market"]

    assert market["intraday_buy_MWh_el"].sum() == pytest.approx(0.0)
    assert market["intraday_sell_MWh_el"].sum() == pytest.approx(0.0)
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
    missing_price_row = market[market["intraday_price_EUR_per_MWh_el"].isna()].iloc[0]

    assert missing_price_row["intraday_buy_MWh_el"] == pytest.approx(0.0)
    assert missing_price_row["intraday_sell_MWh_el"] == pytest.approx(0.0)


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

    assert market["afrr_energy_bid_MWh_el"].sum() == pytest.approx(0.0)
    assert market["afrr_energy_activated_MWh_el"].sum() == pytest.approx(0.0)
    assert market["actual_electricity_consumption_MWh_el"].to_numpy() == pytest.approx(
        market["scheduled_electricity_procurement_MWh_el"].to_numpy()
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
        heat_demand=[2.0] * 8,
    )

    results = _run_case(afrr_case, tmp_path)
    market = results["market"]
    dispatch = results["dispatch"]

    assert market["afrr_energy_bid_MWh_el"].sum() > 0.0
    assert market["afrr_energy_activated_MWh_el"].sum() > 0.0
    assert (market["afrr_energy_activated_MWh_el"] <= market["afrr_energy_bid_MWh_el"] + 1e-8).all()
    assert (
        market["afrr_energy_activated_MWh_el"] <= market["afrr_system_activation_MWh_el"] + 1e-8
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

    assert market["afrr_energy_bid_MWh_el"].sum() == pytest.approx(0.0)
    assert market["afrr_energy_activated_MWh_el"].sum() == pytest.approx(0.0)
    _assert_actual_electricity_with_afrr(market)


def test_afrr_down_margin_blocks_price_too_close_to_benchmark(
    afrr_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[72.0] * 8,
        afrr_quantities=[2.0] * 8,
        heat_demand=[2.0] * 8,
    )

    results = _run_case(afrr_case, tmp_path)
    market = results["market"]

    assert market["afrr_energy_bid_MWh_el"].sum() == pytest.approx(0.0)
    assert market["afrr_energy_activated_MWh_el"].sum() == pytest.approx(0.0)
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

    assert first["afrr_energy_bid_MWh_el"] == pytest.approx(0.0)
    assert first["afrr_energy_activated_MWh_el"] == pytest.approx(0.0)
    assert first["afrr_energy_price_EUR_per_MWh_el"] == pytest.approx(0.0)
    assert results["afrr_quality"]["aFRR_down_activation_without_price_rows"].iloc[0] == 1


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

    assert first["afrr_energy_bid_MWh_el"] == pytest.approx(0.0)
    assert first["afrr_energy_activated_MWh_el"] == pytest.approx(0.0)


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
        afrr_quantities=[0.1] * 8,
        heat_demand=[2.0] * 8,
    )

    results = _run_case(afrr_case, tmp_path)
    market = results["market"].sort_values("datetime")
    first = market.iloc[0]

    first_step_heat_demand_mwh = 2.0 * 0.25
    baseline_soc_after_heat_dispatch = 3.6 - first_step_heat_demand_mwh / 0.92
    expected_capacity_limited_bid = (4.0 - baseline_soc_after_heat_dispatch) / 0.92
    assert first["afrr_energy_bid_MWh_el"] <= expected_capacity_limited_bid + 1e-8
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


def test_afrr_capacity_reserves_headroom_and_caps_activation(
    afrr_capacity_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_capacity_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[20.0] * 8,
        afrr_quantities=[2.0] * 8,
        afrr_capacity_prices=[100.0] * 8,
        heat_demand=[2.0] * 8,
    )

    results = _run_case(afrr_capacity_case, tmp_path)
    market = results["market"]

    assert results["afrr_capacity_blocks"]["reserved_capacity_MW"].sum() > 0.0
    assert market["afrr_capacity_reserved_MW"].nunique() == 1
    assert (
        market["scheduled_electricity_procurement_MWh_el"] + market["afrr_capacity_reserved_MWh"]
        <= 7.0 * 0.25 + 1e-8
    ).all()
    assert (
        market["afrr_energy_activated_MWh_el"] <= market["afrr_capacity_reserved_MWh"] + 1e-8
    ).all()
    assert market["afrr_capacity_revenue_EUR"].sum() > 0.0


def test_afrr_capacity_low_capacity_price_blocks_reservation(
    afrr_capacity_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_capacity_case / "forecasts_df.csv",
        da_prices=[10.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[20.0] * 8,
        afrr_quantities=[2.0] * 8,
        afrr_capacity_prices=[0.0] * 8,
        heat_demand=[2.0] * 8,
    )

    results = _run_case(afrr_capacity_case, tmp_path)

    assert results["afrr_capacity_blocks"]["reserved_capacity_MW"].sum() == pytest.approx(0.0)
    assert not results["afrr_capacity_blocks"]["capacity_profitability_check_passed"].iloc[0]


def test_afrr_capacity_high_activation_energy_price_blocks_reservation(
    afrr_capacity_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_capacity_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[200.0] * 8,
        afrr_quantities=[2.0] * 8,
        afrr_capacity_prices=[100.0] * 8,
        heat_demand=[2.0] * 8,
    )

    results = _run_case(afrr_capacity_case, tmp_path)
    block = results["afrr_capacity_blocks"].iloc[0]

    assert block["reserved_capacity_MW"] == pytest.approx(0.0)
    assert not bool(block["activation_safety_check_passed"])
    assert block["activation_price_check_failed_timesteps"] == 8


def test_afrr_capacity_activation_without_price_blocks_reservation(
    afrr_capacity_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_capacity_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[None] + [20.0] * 7,
        afrr_quantities=[2.0] * 8,
        afrr_capacity_prices=[100.0] * 8,
        heat_demand=[2.0] * 8,
    )

    with pytest.warns(UserWarning, match="aFRR down price contains missing values"):
        results = _run_case(afrr_capacity_case, tmp_path)
    block = results["afrr_capacity_blocks"].iloc[0]

    assert block["reserved_capacity_MW"] == pytest.approx(0.0)
    assert block["activation_without_price_timesteps"] == 1
    assert block["activation_price_check_failed_timesteps"] == 1
    assert results["afrr_quality"]["aFRR_down_activation_without_price_rows"].iloc[0] == 1


def test_afrr_capacity_no_activation_block_can_reserve(
    afrr_capacity_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_capacity_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[200.0] * 8,
        afrr_quantities=[0.0] * 8,
        afrr_capacity_prices=[100.0] * 8,
        heat_demand=[2.0] * 8,
    )

    results = _run_case(afrr_capacity_case, tmp_path)
    block = results["afrr_capacity_blocks"].iloc[0]

    assert block["reserved_capacity_MW"] > 0.0
    assert bool(block["activation_safety_check_passed"])
    assert block["activation_relevant_timesteps"] == 0


def test_afrr_capacity_reports_useful_heat_cap_binding(
    afrr_capacity_case: Path,
    tmp_path: Path,
) -> None:
    _write_forecasts(
        afrr_capacity_case / "forecasts_df.csv",
        da_prices=[120.0] * 8,
        idc_prices=[75.0] * 8,
        afrr_prices=[20.0] * 8,
        afrr_quantities=[100.0] * 8,
        afrr_capacity_prices=[100.0] * 8,
        heat_demand=[0.5] * 8,
    )

    results = _run_case(afrr_capacity_case, tmp_path)
    market = results["market"]

    assert market["useful_heat_cap_binding"].sum() > 0.0
    assert market["curtailed_proxy_activation_due_to_heat_cap_MWh"].sum() > 0.0
    assert (
        market["afrr_energy_activated_MWh_el"] <= market["afrr_capacity_reserved_MWh"] + 1e-8
    ).all()


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
        "storage": pd.read_csv(outputs["storage_cost_ledger"], parse_dates=["datetime"]),
        "summary": pd.read_csv(outputs["summary_indicators"]),
    }
    if "afrr_energy_data_quality_summary" in outputs:
        results["afrr_quality"] = pd.read_csv(outputs["afrr_energy_data_quality_summary"])
    if "afrr_capacity_block_summary" in outputs:
        results["afrr_capacity_blocks"] = pd.read_csv(outputs["afrr_capacity_block_summary"])
    return results


def _assert_final_planned_balance(market: pd.DataFrame) -> None:
    expected = (
        market["day_ahead_position_MWh_el"]
        + market["intraday_buy_MWh_el"]
        - market["intraday_sell_MWh_el"]
    )
    assert market["scheduled_electricity_procurement_MWh_el"].to_numpy() == pytest.approx(
        expected.to_numpy()
    )
    assert market["actual_electricity_consumption_MWh_el"].to_numpy() == pytest.approx(
        market["scheduled_electricity_procurement_MWh_el"].to_numpy()
    )


def _assert_heat_is_feasible(dispatch: pd.DataFrame) -> None:
    supplied_heat = dispatch["gas_heat_MWh"] + dispatch["etes_discharge_MWh"]
    assert supplied_heat.to_numpy() == pytest.approx(dispatch["heat_demand_MWh"].to_numpy())
    assert "unmet_heat_MWh" not in dispatch.columns
    assert "excess_heat_MWh" not in dispatch.columns


def _assert_actual_electricity_with_afrr(market: pd.DataFrame) -> None:
    expected = (
        market["scheduled_electricity_procurement_MWh_el"] + market["afrr_energy_activated_MWh_el"]
    )
    assert market["actual_electricity_consumption_MWh_el"].to_numpy() == pytest.approx(
        expected.to_numpy()
    )


def _write_config(
    path: Path,
    idc_enabled: bool = False,
    afrr_enabled: bool = False,
    idc_buy_enabled: bool = True,
    idc_sell_enabled: bool = True,
    afrr_capacity_enabled: bool = False,
) -> None:
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
  - afrr_capacity
  - day_ahead
  - intraday_continuous
  - afrr_energy

markets:
  day_ahead:
    enabled: true
    product_resolution: "15min"
    gate_close:
      day_relation: "D-1"
      time: "12:00"
    signals:
      price: "DE_day_ahead_price_EUR_per_MWh_el"

  intraday_continuous:
    enabled: {str(idc_enabled).lower()}
    product_resolution: "15min"
    gate_close:
      relative_to_delivery_start_minutes: -5
    allowed_actions:
      buy: {str(idc_buy_enabled).lower()}
      sell: {str(idc_sell_enabled).lower()}
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
""".strip(),
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
