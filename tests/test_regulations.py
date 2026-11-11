# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from flexi_mod.regulations import (
    EnflexAgnesGridFeeRegulation,
    FrenchGridFeeRegulation,
    GermanGridFeeRegulation,
    GridFeeConfigError,
    NullGridFeeRegulation,
    SpanishGridFeeRegulation,
    build_grid_fee_regulation,
)

LEVIES = 2.77 + 8.16 + 1.1 + 0.5  # CHP + offshore + concession + tax = 12.53


def full_charges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component": [
                "grid_energy_charge_high",
                "grid_energy_charge_low",
                "grid_capacity_charge_high",
                "grid_capacity_charge_low",
                "chp_surcharge",
                "offshore_grid_levy",
                "special_network_use_a",
                "special_network_use_b",
                "concession_fee",
                "electricity_tax",
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
    return GermanGridFeeRegulation.from_charges_frame(full_charges(), **kwargs)


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
        build_grid_fee_regulation("NL", full_charges())


def test_factory_selects_regulation_by_name():
    reg = build_grid_fee_regulation("de_atypical_grid_use", full_charges())
    assert isinstance(reg, GermanGridFeeRegulation)
    spain = build_grid_fee_regulation(
        "es_peajes_6xtd",
        pd.DataFrame(
            {
                "component": ["grid_capacity_charge"],
                "unit": ["EUR/MW.a"],
                "value": [19629.0],
            }
        ),
    )
    assert isinstance(spain, SpanishGridFeeRegulation)
    france = build_grid_fee_regulation(
        "fr_turpe",
        pd.DataFrame(
            {
                "component": ["turpe_fix"],
                "unit": ["EUR/MW.a"],
                "value": [12948.94],
            }
        ),
    )
    assert isinstance(france, FrenchGridFeeRegulation)


def test_factory_country_codes_resolve_to_the_same_regulation_as_the_name():
    by_name = build_grid_fee_regulation("de_atypical_grid_use", full_charges())
    for code in ("DE", "de", " De "):
        by_code = build_grid_fee_regulation(code, full_charges())
        assert type(by_code) is type(by_name)
        assert by_code.marginal_charge_eur_per_mwh() == by_name.marginal_charge_eur_per_mwh()


def test_factory_unknown_name_lists_available_regulations():
    with pytest.raises(GridFeeConfigError, match="de_atypical_grid_use.*fr_turpe"):
        build_grid_fee_regulation("de_not_a_regulation", full_charges())


def test_incomplete_tier_raises():
    charges = pd.DataFrame(
        {
            "component": ["grid_energy_charge_high", "chp_surcharge"],
            "unit": ["EUR/MWh", "EUR/MWh"],
            "value": [36.9, 2.77],
        }
    )
    with pytest.raises(GridFeeConfigError, match="incomplete tiered component"):
        GermanGridFeeRegulation.from_charges_frame(charges)


def test_unknown_charge_name_raises():
    # A misspelled/unknown component no longer disappears into a levy bucket.
    charges = pd.DataFrame(
        {"component": ["grid_capcity_charge_high"], "unit": ["EUR/MW.a"], "value": [66570.0]}
    )
    with pytest.raises(GridFeeConfigError, match="Unknown grid-fee charge"):
        GermanGridFeeRegulation.from_charges_frame(charges)


def test_absent_categories_default_to_zero():
    # A pure-levy tariff (no grid energy/capacity/special) loads with zero tiers.
    charges = pd.DataFrame(
        {
            "component": ["electricity_tax"],
            "unit": ["EUR/MWh"],
            "value": [10.0],
        }
    )
    reg = GermanGridFeeRegulation.from_charges_frame(charges)
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


# --------------------------------------------------------------------- Spain
def test_spanish_settle_separates_iee_from_levies():
    """IEE is reported in electricity_tax_EUR, not folded into levies_EUR."""
    charges = pd.DataFrame(
        {"component": ["grid_capacity_charge"], "unit": ["EUR/MW.a"], "value": [19629.0]}
    )
    reg = SpanishGridFeeRegulation.from_charges_frame(charges)

    idx = pd.date_range("2025-01-01 00:00", periods=4, freq="15min")
    dr = pd.DataFrame(
        {
            "actual_electricity_consumption_MWh": [1.0, 1.0, 1.0, 1.0],  # 4 MWh, 4 MW peak
            "additional_electricity_charges_cost_EUR": [10.0, 10.0, 10.0, 10.0],  # peajes = 40
            "electricity_market_cost_EUR": [100.0, 100.0, 100.0, 100.0],  # market = 400
        },
        index=idx,
    )
    res = reg.settle(dr, timestep_minutes=15)

    energy_charge = 40.0
    taxable_base = 400.0 + energy_charge  # market + peajes (Spain has no static EUR/MWh levy)
    iee = taxable_base * SpanishGridFeeRegulation.ELECTRICITY_TAX_RATE

    assert res.levies_EUR == pytest.approx(0.0)  # pure; IEE not bundled in here
    assert res.electricity_tax_EUR == pytest.approx(iee)  # IEE lives in its own field
    assert res.capacity_charge_EUR == pytest.approx(19629.0 * 4.0)  # annual, no proration
    assert res.grid_fee_total_EUR == pytest.approx(
        res.energy_charge_EUR
        + res.capacity_charge_EUR
        + res.special_network_use_EUR
        + res.levies_EUR
        + res.electricity_tax_EUR
    )


def test_spanish_rejects_unknown_charge():
    """Spain's per-MWh charges belong in the dynamic column; a stray levy row raises."""
    charges = pd.DataFrame({"component": ["some_static_levy"], "unit": ["EUR/MWh"], "value": [2.0]})
    with pytest.raises(GridFeeConfigError, match="Unknown grid-fee charge"):
        SpanishGridFeeRegulation.from_charges_frame(charges)


# --------------------------------------------------------------------- France
def test_french_settle_sums_all_fixed_annual_charges():
    """All EUR/MW.a charges are summed (not just Capacity Obligation) and not prorated."""
    charges = pd.DataFrame(
        {
            "component": [
                "capacity_obligation",
                "turpe_management",
                "turpe_metering",
                "turpe_fix",
            ],
            "unit": ["EUR/MW.a", "EUR/MW.a", "EUR/MW.a", "EUR/MW.a"],
            "value": [14650.0, 11545.32, 3800.04, 12948.94],
        }
    )
    reg = FrenchGridFeeRegulation.from_charges_frame(charges)

    idx = pd.date_range("2025-01-01 00:00", periods=4, freq="15min")
    dr = pd.DataFrame({"actual_electricity_consumption_MWh": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    res = reg.settle(dr, timestep_minutes=15)

    total_rate = 14650.0 + 11545.32 + 3800.04 + 12948.94  # 42944.30
    assert res.capacity_charge_EUR == pytest.approx(total_rate * 4.0)  # 4 MW peak, no proration
    assert res.electricity_tax_EUR == pytest.approx(0.0)  # France has no multiplicative tax


# ------------------------------------------------------- Germany (AgNes design)
AGNES_CHARGE_NAMES = [
    "grid_ordered_capacity",
    "grid_ordered_capacity_charge",
    "grid_energy_charge_within_order",
    "grid_energy_charge_above_order",
    "chp_surcharge",
    "offshore_grid_levy",
    "concession_fee",
    "electricity_tax",
]
AGNES_CHARGE_UNITS = ["MW", "EUR/MW.a"] + ["EUR/MWh"] * 6


def agnes_charges(**values: float) -> pd.DataFrame:
    """The example case values (5 MW order, KP 100 EUR/kW.a, AP1 20, AP2 60 EUR/MWh)."""
    defaults = dict(
        zip(AGNES_CHARGE_NAMES, [5.0, 100000.0, 20.0, 60.0, 2.77, 8.16, 1.1, 0.5], strict=True)
    )
    defaults.update(values)
    return pd.DataFrame(
        {
            "component": AGNES_CHARGE_NAMES,
            "unit": AGNES_CHARGE_UNITS,
            "value": [defaults[name] for name in AGNES_CHARGE_NAMES],
        }
    )


def test_agnes_is_selected_by_name_and_marginal_charge_is_within_order_rate_plus_levies():
    reg = build_grid_fee_regulation("de_agnes_enflex", agnes_charges())
    assert isinstance(reg, EnflexAgnesGridFeeRegulation)
    assert reg.marginal_charge_eur_per_mwh() == pytest.approx(20.0 + LEVIES)
    assert reg.ordered_capacity_mw == 5.0
    assert reg.excess_energy_surcharge_eur_per_mwh == pytest.approx(40.0)
    assert reg.electricity_tax_rate == 0.0
    assert reg.dynamic_charge_column is None


def test_agnes_never_blocks_charging():
    reg = build_grid_fee_regulation("de_agnes_enflex", agnes_charges())
    forecasts = pd.DataFrame({"high_load_window": [1, 1, 0]}, index=year_index()[:3])
    assert not reg.charging_block_mask(forecasts).any()


def test_agnes_settle_splits_energy_at_the_ordered_capacity():
    reg = build_grid_fee_regulation("de_agnes_enflex", agnes_charges())
    index = pd.date_range("2025-01-01 00:00", periods=4, freq="15min")
    # 5 MW order = 1.25 MWh per 15 min. Steps of 1.0, 2.0, 1.25 and 0.0 MWh.
    result = reg.settle(dispatch(index, [1.0, 2.0, 1.25, 0.0]), timestep_minutes=15)

    assert result.grid_energy_MWh == pytest.approx(4.25)
    assert result.excess_over_order_MWh == pytest.approx(0.75)  # only the 2.0 MWh step
    assert result.annual_peak_MW == pytest.approx(8.0)
    assert result.billed_peak_MW == pytest.approx(5.0)
    assert result.ordered_capacity_MW == pytest.approx(5.0)
    assert result.energy_charge_EUR == pytest.approx(3.5 * 20.0 + 0.75 * 60.0)
    assert result.capacity_charge_EUR == pytest.approx(100000.0 * 5.0)  # not prorated
    assert result.levies_EUR == pytest.approx(LEVIES * 4.25)
    assert result.grid_fee_total_EUR == pytest.approx(
        result.energy_charge_EUR + result.capacity_charge_EUR + result.levies_EUR
    )
    # The dispatch books the within-order rate, the levies and the above-order surcharge.
    # Only the capacity charge is added after the run.
    assert result.ex_post_addition_EUR == pytest.approx(500000.0)


def test_agnes_load_below_the_order_has_no_excess():
    reg = build_grid_fee_regulation("de_agnes_enflex", agnes_charges())
    index = pd.date_range("2025-01-01 00:00", periods=3, freq="15min")
    result = reg.settle(dispatch(index, [1.25, 0.5, 0.0]), timestep_minutes=15)
    assert result.excess_over_order_MWh == pytest.approx(0.0)
    assert result.energy_charge_EUR == pytest.approx(1.75 * 20.0)


def test_agnes_without_consumption_still_pays_the_ordered_capacity():
    reg = build_grid_fee_regulation("de_agnes_enflex", agnes_charges())
    index = pd.date_range("2025-01-01 00:00", periods=4, freq="15min")
    result = reg.settle(dispatch(index, [0.0, 0.0, 0.0, 0.0]), timestep_minutes=15)
    assert result.grid_fee_total_EUR == pytest.approx(500000.0)
    assert result.full_load_hours == 0.0


def test_agnes_summary_has_order_columns_but_other_regulations_do_not():
    index = pd.date_range("2025-01-01 00:00", periods=2, freq="15min")
    agnes = build_grid_fee_regulation("de_agnes_enflex", agnes_charges())
    summary = agnes.settle(dispatch(index, [1.0, 2.0]), 15).as_summary_dict()
    assert summary["grid_ordered_capacity_MW"] == pytest.approx(5.0)
    assert summary["grid_excess_over_order_MWh"] == pytest.approx(0.75)

    german = build_grid_fee_regulation("de_atypical_grid_use", full_charges())
    german_summary = german.settle(dispatch(index, [1.0, 2.0]), 15).as_summary_dict()
    assert "grid_ordered_capacity_MW" not in german_summary
    assert "grid_excess_over_order_MWh" not in german_summary


# Reference values of "Analyse AgNes Kapazitätsbestellung.xlsx", sheet "50 % Auslastungsgrad":
# cost per kW of peak capacity for an ordered capacity of 0, 10, ..., 100 % of the peak
# (KP 100 EUR/kW.a, AP1 2 ct/kWh, AP2 6 ct/kWh, 8760 h), excluding the AP1 base energy
# cost, and the resulting average price in ct/kWh. Each case is a load-duration curve in
# ten deciles of the year, as a share of the peak.
AGNES_EXCEL_50_PERCENT = {
    "worst": {
        "profile": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "cost_eur_per_kw": [175.2, 167.68, 160.16, 152.64, 145.12, 137.6]
        + [130.08, 122.56, 115.04, 107.52, 100.0],
        "price_ct_per_kwh": [6.0, 5.828310502283105, 5.65662100456621, 5.484931506849314]
        + [5.31324200913242, 5.141552511415525, 4.96986301369863, 4.798173515981735]
        + [4.62648401826484, 4.454794520547945, 4.28310502283105],
    },
    "real": {
        "profile": [1, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0],
        "cost_eur_per_kw": [175.2, 153.664, 135.632, 121.104, 110.08, 102.56]
        + [95.04, 91.024, 90.512, 93.504, 100.0],
        "price_ct_per_kwh": [6.0, 5.508310502283105, 5.0966210045662095, 4.764931506849315]
        + [4.51324200913242, 4.341552511415525, 4.16986301369863, 4.078173515981735]
        + [4.06648401826484, 4.134794520547945, 4.28310502283105],
    },
    "best": {
        "profile": [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0],
        "cost_eur_per_kw": [175.2, 153.664, 132.128, 110.592, 89.056, 67.52]
        + [74.016, 80.512, 87.008, 93.504, 100.0],
        "price_ct_per_kwh": [6.0, 5.508310502283105, 5.016621004566209, 4.524931506849315]
        + [4.03324200913242, 3.5415525114155253, 3.6898630136986297, 3.8381735159817345]
        + [3.9864840182648402, 4.134794520547945, 4.28310502283105],
    },
}


@pytest.mark.parametrize("case", sorted(AGNES_EXCEL_50_PERCENT))
def test_agnes_reproduces_the_reference_excel(case):
    """A plant with a 1 kW peak follows the Excel's load-duration curve for a whole year."""
    reference = AGNES_EXCEL_50_PERCENT[case]
    hourly_load_mwh = np.repeat(np.asarray(reference["profile"], dtype=float), 876) * 0.001
    index = pd.date_range("2025-01-01 00:00", periods=8760, freq="h")
    load = dispatch(index, hourly_load_mwh)

    for step, (cost, price) in enumerate(
        zip(reference["cost_eur_per_kw"], reference["price_ct_per_kwh"], strict=True)
    ):
        order_mw = 0.001 * step / 10.0
        charges = agnes_charges(
            grid_ordered_capacity=order_mw,
            grid_energy_charge_within_order=20.0,
            grid_energy_charge_above_order=60.0,
            chp_surcharge=0.0,
            offshore_grid_levy=0.0,
            concession_fee=0.0,
            electricity_tax=0.0,
        )
        result = build_grid_fee_regulation("de_agnes_enflex", charges).settle(load, 60)
        # The Excel cost = capacity charge + surcharge on the energy above the order.
        surcharge = 40.0  # AP2 - AP1 in EUR/MWh
        excel_cost = result.capacity_charge_EUR + surcharge * result.excess_over_order_MWh
        assert excel_cost == pytest.approx(cost, abs=1e-9), (case, step)
        average_ct_per_kwh = result.grid_fee_total_EUR / result.grid_energy_MWh / 10.0
        assert average_ct_per_kwh == pytest.approx(price, abs=1e-9), (case, step)


def test_agnes_missing_required_charge_raises():
    frame = agnes_charges()
    frame = frame[frame["component"] != "grid_ordered_capacity"]
    with pytest.raises(GridFeeConfigError, match="missing required charge.*grid_ordered_capacity"):
        build_grid_fee_regulation("de_agnes_enflex", frame)


def test_agnes_unknown_charge_raises():
    frame = pd.concat(
        [
            agnes_charges(),
            pd.DataFrame({"component": ["grid_ordered_power"], "unit": ["MW"], "value": [1.0]}),
        ]
    )
    with pytest.raises(GridFeeConfigError, match="Unknown grid-fee charge 'grid_ordered_power'"):
        build_grid_fee_regulation("de_agnes_enflex", frame)


def test_agnes_rejects_a_charge_in_the_wrong_unit():
    frame = agnes_charges()
    frame.loc[frame["component"] == "grid_ordered_capacity_charge", "unit"] = "EUR/MWh"
    with pytest.raises(GridFeeConfigError, match="grid_ordered_capacity_charge.*EUR/MW.a"):
        build_grid_fee_regulation("de_agnes_enflex", frame)


def test_agnes_rejects_a_cheaper_rate_above_the_order():
    with pytest.raises(GridFeeConfigError, match="above_order must not be lower"):
        build_grid_fee_regulation(
            "de_agnes_enflex", agnes_charges(grid_energy_charge_above_order=10.0)
        )


def test_agnes_rejects_negative_charges():
    with pytest.raises(GridFeeConfigError, match="must not be negative.*grid_ordered_capacity"):
        build_grid_fee_regulation("de_agnes_enflex", agnes_charges(grid_ordered_capacity=-1.0))
