# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from flexi_mod.regulations import (
    GermanGridFeeRegulation,
    GridFeeConfigError,
    NullGridFeeRegulation,
    build_grid_fee_regulation,
)

LEVIES = 2.77 + 8.16 + 1.1 + 0.5  # CHP + offshore + concession + tax = 12.53


def full_charges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component": [
                "Grid energy charge >=2500 h/a",
                "Grid energy charge <2500 h/a",
                "Grid capacity charge >=2500 h/a",
                "Grid capacity charge <2500 h/a",
                "CHP surcharge",
                "Offshore grid levy",
                "Surcharge for special network use (group A)",
                "Surcharge for special network use (group B)",
                "Concession fee",
                "Electricity tax",
            ],
            "unit": [
                "EUR/MWh",
                "EUR/MWh",
                "EUR/MW.a",
                "EUR/MW.a",
                "EUR/MWh",
                "EUR/MWh",
                "EUR/MWh",
                "EUR/MWh",
                "EUR/MWh",
                "EUR/MWh",
            ],
            "value": [36.9, 45.6, 66570, 44850, 2.77, 8.16, 15.58, 0.5, 1.1, 0.5],
        }
    )


def make_reg(**kwargs) -> GermanGridFeeRegulation:
    return GermanGridFeeRegulation(full_charges(), **kwargs)


def dispatch(index: pd.DatetimeIndex, consumption) -> pd.DataFrame:
    return pd.DataFrame(
        {"actual_electricity_consumption_MWh": np.asarray(consumption, dtype=float)},
        index=index,
    )


def year_index() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01 00:00", "2025-12-31 23:45", freq="15min")


# --------------------------------------------------------------- marginal charge
def test_marginal_charge_high_tier():
    assert make_reg(assumed_tier="high").marginal_charge_eur_per_mwh() == pytest.approx(
        LEVIES + 36.9 + 0.5
    )


def test_marginal_charge_low_tier():
    assert make_reg(assumed_tier="low").marginal_charge_eur_per_mwh() == pytest.approx(
        LEVIES + 45.6 + 0.5
    )


# -------------------------------------------------------- high-load-window rules
@pytest.mark.parametrize(
    "timestamp, expected",
    [
        ("2025-01-06 09:30", 0),  # winter Mon, before window
        ("2025-01-06 09:45", 1),  # winter Mon, window start (inclusive)
        ("2025-01-06 19:15", 1),  # winter Mon, last in-window step
        ("2025-01-06 19:30", 0),  # winter Mon, window end (exclusive)
        ("2025-10-06 11:15", 1),  # autumn Mon, morning window start
        ("2025-10-06 13:30", 1),  # autumn Mon, last morning step
        ("2025-10-06 13:45", 0),  # autumn Mon, gap between windows
        ("2025-10-06 15:00", 1),  # autumn Mon, afternoon window start
        ("2025-10-06 19:00", 1),  # autumn Mon, last afternoon step
        ("2025-04-15 12:00", 0),  # spring -> no window
        ("2025-07-15 12:00", 0),  # summer -> no window
        ("2025-01-11 12:00", 0),  # Saturday -> never in window
        ("2025-01-12 12:00", 0),  # Sunday -> never in window
    ],
)
def test_compute_high_load_window(timestamp, expected):
    flag = GermanGridFeeRegulation.compute_high_load_window(pd.DatetimeIndex([timestamp]))
    assert int(flag.iloc[0]) == expected


# ------------------------------------------------------------------ settlement
def test_settle_high_tier_held_and_window_avoided():
    idx = year_index()
    win = GermanGridFeeRegulation.compute_high_load_window(idx).to_numpy()
    consumption = np.where(win == 1, 0.0, 1.0)  # 4 MW outside windows, 0 inside
    reg = make_reg(assumed_tier="high")
    res = reg.settle(dispatch(idx, consumption), timestep_minutes=15)

    assert res.realized_tier == "high"
    assert res.tier_assumption_held is True
    assert res.warnings == []
    assert res.window_peak_MW == pytest.approx(0.0)
    assert res.capacity_charge_EUR == pytest.approx(0.0)  # atypical grid use -> ~0
    assert res.energy_charge_EUR == pytest.approx(res.grid_energy_MWh * 36.9)
    # reconciliation: ex-post adds only capacity + group-A premium + tier true-up
    in_dispatch = (LEVIES + 36.9 + 0.5) * res.grid_energy_MWh
    assert res.ex_post_addition_EUR == pytest.approx(res.grid_fee_total_EUR - in_dispatch)


def test_settle_low_tier_warns_and_trues_up():
    idx = year_index()
    consumption = np.zeros(len(idx))
    consumption[:100] = 10.0  # peaky -> low full-load hours
    reg = make_reg(assumed_tier="high", capacity_peak_basis="annual")
    res = reg.settle(dispatch(idx, consumption), timestep_minutes=15)

    assert res.realized_tier == "low"
    assert res.tier_assumption_held is False
    assert res.warnings  # a tier-mismatch warning is recorded
    assert res.full_load_hours < GermanGridFeeRegulation.FULL_LOAD_HOURS_THRESHOLD_H
    # energy billed at the realized (low) tier
    assert res.energy_charge_EUR == pytest.approx(res.grid_energy_MWh * 45.6)
    # true-up component is part of ex-post addition
    in_dispatch = (LEVIES + 36.9 + 0.5) * res.grid_energy_MWh
    assert res.ex_post_addition_EUR == pytest.approx(res.grid_fee_total_EUR - in_dispatch)


@pytest.mark.parametrize(
    "energy_mwh, expected_special",
    [
        (1000.0, 1000.0 * 15.58),  # exactly the threshold -> all group A
        (2000.0, 1000.0 * 15.58 + 1000.0 * 0.5),  # split A then B
        (500.0, 500.0 * 15.58),  # below threshold -> all group A
    ],
)
def test_special_network_use_ab_split(energy_mwh, expected_special):
    idx = pd.date_range("2025-04-01 00:00", periods=4, freq="15min")  # spring, no window
    per_step = energy_mwh / len(idx)
    res = make_reg().settle(dispatch(idx, [per_step] * len(idx)), timestep_minutes=15)
    assert res.grid_energy_MWh == pytest.approx(energy_mwh)
    assert res.special_network_use_EUR == pytest.approx(expected_special)


def test_capacity_basis_window_vs_annual():
    # Winter Monday: 08:00 outside window (16 MW), 10:00 inside window (4 MW).
    idx = pd.DatetimeIndex(["2025-01-06 08:00", "2025-01-06 10:00"])
    consumption = [4.0, 1.0]  # MWh/step -> 16 MW and 4 MW at 15-min resolution

    window = make_reg(capacity_peak_basis="high_load_window").settle(
        dispatch(idx, consumption), timestep_minutes=15
    )
    annual = make_reg(capacity_peak_basis="annual").settle(
        dispatch(idx, consumption), timestep_minutes=15
    )
    assert window.annual_peak_MW == pytest.approx(16.0)
    assert window.window_peak_MW == pytest.approx(4.0)
    assert window.billed_peak_MW == pytest.approx(4.0)
    assert annual.billed_peak_MW == pytest.approx(16.0)
    # both use the low tier here (short horizon), capacity rate 44850 EUR/MW.a
    assert window.capacity_charge_EUR == pytest.approx(44850 * 4.0)
    assert annual.capacity_charge_EUR == pytest.approx(44850 * 16.0)


def test_settle_zero_consumption_is_quiet():
    idx = pd.date_range("2025-01-06 00:00", periods=8, freq="15min")
    res = make_reg(assumed_tier="high").settle(dispatch(idx, [0.0] * 8), timestep_minutes=15)
    assert res.grid_fee_total_EUR == pytest.approx(0.0)
    assert res.tier_assumption_held is True
    assert res.warnings == []


# ----------------------------------------------------------- parsing / factory
def test_null_regulation_when_charges_none():
    reg = build_grid_fee_regulation("DE", None)
    assert isinstance(reg, NullGridFeeRegulation)
    assert reg.marginal_charge_eur_per_mwh() == 0.0
    idx = pd.date_range("2025-01-06 00:00", periods=4, freq="15min")
    assert not reg.charging_block_mask(pd.DataFrame(index=idx)).any()
    assert reg.settle(dispatch(idx, [1.0] * 4), 15).grid_fee_total_EUR == 0.0


def test_factory_unknown_country_raises():
    with pytest.raises(GridFeeConfigError, match="No grid-fee regulation"):
        build_grid_fee_regulation("FR", full_charges())


def test_incomplete_tier_raises():
    charges = pd.DataFrame(
        {
            "component": ["Grid energy charge >=2500 h/a", "CHP surcharge"],
            "unit": ["EUR/MWh", "EUR/MWh"],
            "value": [36.9, 2.77],
        }
    )
    with pytest.raises(GridFeeConfigError, match="incomplete tiered component"):
        GermanGridFeeRegulation(charges)


def test_absent_categories_default_to_zero():
    # A pure-levy tariff (no grid energy/capacity/special) loads with zero tiers.
    charges = pd.DataFrame(
        {
            "component": ["Network consumption price"],
            "unit": ["EUR/MWh"],
            "value": [10.0],
        }
    )
    reg = GermanGridFeeRegulation(charges)
    assert reg.marginal_charge_eur_per_mwh() == pytest.approx(10.0)
    idx = pd.date_range("2025-04-01 00:00", periods=4, freq="15min")
    res = reg.settle(dispatch(idx, [1.0] * 4), timestep_minutes=15)
    assert res.energy_charge_EUR == pytest.approx(0.0)
    assert res.capacity_charge_EUR == pytest.approx(0.0)
    assert res.special_network_use_EUR == pytest.approx(0.0)
    assert res.levies_EUR == pytest.approx(res.grid_energy_MWh * 10.0)


def test_charging_block_mask_uses_forecast_column():
    idx = pd.date_range("2025-04-01 00:00", periods=4, freq="15min")  # spring, rules give 0
    forecasts = pd.DataFrame({"high_load_window": [0, 1, 1, 0]}, index=idx)
    mask = make_reg().charging_block_mask(forecasts).to_numpy()
    assert list(mask.astype(int)) == [0, 1, 1, 0]


def test_charging_block_mask_absent_column_blocks_nothing():
    idx = pd.date_range("2025-01-06 10:00", periods=4, freq="15min")  # would be in-window
    with pytest.warns(UserWarning, match="high_load_window"):
        mask = make_reg().charging_block_mask(pd.DataFrame(index=idx))
    assert not mask.any()
