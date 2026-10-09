# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reusable Pyomo formulation for electricity-market positions and settlement.

The configured market modules describe the product rules and prepare input
data.  Strategies construct one of the typed stage inputs defined alongside
those products.  This module translates that input into generic electricity
positions and settlement expressions on a plant's Pyomo model.

It deliberately does not know about a particular plant or technology.  A
plant must provide ``model.T`` and an ``additional_electricity_charge``
parameter indexed by ``model.T`` before adding a stage.  It can then connect
``required_electricity_consumption_mwh`` to its own physical equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from flexi_mod.markets.afrr_capacity import BalancingCapacityAward
from flexi_mod.markets.afrr_energy import BalancingEnergyActivation
from flexi_mod.markets.day_ahead import DayAheadPosition
from flexi_mod.markets.intraday_continuous import IntradayAdjustment
from flexi_mod.modeling.pyomo_utils import series_or_zero

__all__ = [
    "BalancingEnergyActivation",
    "DayAheadPosition",
    "ElectricityMarketStage",
    "ElectricityMarketStageInput",
    "ElectricityMarketRequest",
    "IntradayAdjustment",
    "add_balancing_energy_activation_to_model",
    "add_day_ahead_position_to_model",
    "add_electricity_settlement_cost_expressions",
    "add_intraday_adjustment_to_model",
    "add_stage_to_model",
    "market_name_for_stage",
    "stage_label",
    "stage_warning_label",
    "validate_fixed_afrr_instruction",
    "validate_stage_input",
]


class ElectricityMarketStage(StrEnum):
    """Electricity-market stage whose position a plant must fulfil."""

    DAY_AHEAD = "day_ahead"
    INTRADAY = "intraday"
    AFRR_ENERGY = "afrr_energy"


type ElectricityMarketStageInput = DayAheadPosition | IntradayAdjustment | BalancingEnergyActivation


@dataclass(frozen=True)
class ElectricityMarketRequest:
    """One electricity-product position that a plant must physically fulfil.

    The request is deliberately independent of a particular plant family.  A
    strategy selects the product position; the receiving plant connects that
    position to its own physical feasibility model.
    """

    stage: ElectricityMarketStage
    position: ElectricityMarketStageInput
    capacity_award: BalancingCapacityAward | None = None


def add_stage_to_model(
    model: pyo.ConcreteModel,
    stage: ElectricityMarketStage,
    forecasts: pd.DataFrame,
    market_input: ElectricityMarketStageInput,
) -> None:
    """Add one electricity product's position and settlement to ``model``.

    The caller owns plant-specific physical constraints.  This function only
    creates the generic commercial position, its resulting electricity demand,
    and market settlement expressions.
    """

    validate_stage_input(stage, market_input)
    if stage == ElectricityMarketStage.DAY_AHEAD:
        add_day_ahead_position_to_model(
            model,
            forecasts,
            market_input,
        )
    elif stage == ElectricityMarketStage.INTRADAY:
        add_intraday_adjustment_to_model(
            model,
            forecasts,
            market_input,
        )
    elif stage == ElectricityMarketStage.AFRR_ENERGY:
        add_balancing_energy_activation_to_model(
            model,
            forecasts,
            market_input,
        )
    else:  # Defensive guard for future enum extensions.
        raise ValueError(f"Unsupported electricity market stage '{stage}'")


def add_day_ahead_position_to_model(
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    position: DayAheadPosition,
) -> None:
    """Add a day-ahead electricity procurement position to ``model``."""

    steps = list(model.T)
    price = forecasts[position.electricity_price_col].astype(float).to_numpy()
    charge_allowed = position.charge_allowed.astype(bool).reindex(forecasts.index).fillna(False)
    model.da_price = pyo.Param(
        model.T,
        initialize={t: float(price[t]) for t in steps},
    )
    model.idc_price = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.afrr_energy_price = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.electricity_price = pyo.Param(
        model.T,
        initialize={
            t: float(price[t]) + float(pyo.value(model.additional_electricity_charge[t]))
            for t in steps
        },
    )
    model.charge_allowed = pyo.Param(
        model.T,
        within=pyo.Binary,
        initialize={t: int(bool(charge_allowed.iloc[t])) for t in steps},
    )
    model.da_position_mwh = pyo.Var(
        model.T,
        within=pyo.NonNegativeReals,
    )
    model.idc_buy_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.idc_sell_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.afrr_energy_bid_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.afrr_energy_activated_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.final_planned_electricity_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.da_position_mwh[t],
    )
    model.actual_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.final_planned_electricity_mwh[t],
    )
    model.required_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.actual_electricity_consumption_mwh[t],
    )
    add_electricity_settlement_cost_expressions(model)


def add_intraday_adjustment_to_model(
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    adjustment: IntradayAdjustment,
) -> None:
    """Add an intraday buy/sell adjustment to a day-ahead position."""

    steps = list(model.T)
    da_price = forecasts[adjustment.da_price_col].astype(float).to_numpy()
    idc_price = forecasts[adjustment.idc_price_col].astype(float).fillna(0.0).to_numpy()
    da_position = series_or_zero(
        adjustment.da_position_mwh,
        forecasts.index,
    ).to_numpy()
    buy_upper = (
        series_or_zero(
            adjustment.idc_buy_upper_bound_mwh,
            forecasts.index,
        )
        .clip(lower=0.0)
        .to_numpy()
    )
    sell_upper = (
        series_or_zero(
            adjustment.idc_sell_upper_bound_mwh,
            forecasts.index,
        )
        .clip(lower=0.0)
        .to_numpy()
    )
    model.da_price = pyo.Param(
        model.T,
        initialize={t: float(da_price[t]) for t in steps},
    )
    model.idc_price = pyo.Param(
        model.T,
        initialize={t: float(idc_price[t]) for t in steps},
    )
    model.afrr_energy_price = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.electricity_price = pyo.Param(
        model.T,
        initialize={
            t: float(idc_price[t]) + float(pyo.value(model.additional_electricity_charge[t]))
            for t in steps
        },
    )
    model.da_position_mwh = pyo.Param(
        model.T,
        initialize={t: float(da_position[t]) for t in steps},
    )
    model.idc_buy_upper_bound_mwh = pyo.Param(
        model.T,
        initialize={t: float(buy_upper[t]) for t in steps},
    )
    model.idc_sell_upper_bound_mwh = pyo.Param(
        model.T,
        initialize={t: float(sell_upper[t]) for t in steps},
    )
    model.idc_buy_mwh = pyo.Var(
        model.T,
        within=pyo.NonNegativeReals,
    )
    model.idc_sell_mwh = pyo.Var(
        model.T,
        within=pyo.NonNegativeReals,
    )
    model.final_planned_electricity_mwh = pyo.Var(
        model.T,
        within=pyo.NonNegativeReals,
    )
    model.afrr_energy_bid_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.afrr_energy_activated_mwh = pyo.Param(
        model.T,
        initialize={t: 0.0 for t in steps},
    )
    model.idc_buy_limit = pyo.Constraint(
        model.T,
        rule=lambda mm, t: mm.idc_buy_mwh[t] <= mm.idc_buy_upper_bound_mwh[t],
    )
    model.idc_sell_limit = pyo.Constraint(
        model.T,
        rule=lambda mm, t: mm.idc_sell_mwh[t] <= mm.idc_sell_upper_bound_mwh[t],
    )
    model.idc_sell_da_limit = pyo.Constraint(
        model.T,
        rule=lambda mm, t: mm.idc_sell_mwh[t] <= mm.da_position_mwh[t],
    )
    model.final_planned_position = pyo.Constraint(
        model.T,
        rule=lambda mm, t: (
            mm.final_planned_electricity_mwh[t]
            == mm.da_position_mwh[t] + mm.idc_buy_mwh[t] - mm.idc_sell_mwh[t]
        ),
    )
    model.actual_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.final_planned_electricity_mwh[t],
    )
    model.required_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.actual_electricity_consumption_mwh[t],
    )
    add_electricity_settlement_cost_expressions(model)


def add_balancing_energy_activation_to_model(
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    activation: BalancingEnergyActivation,
) -> None:
    """Add a fixed balancing-energy bid and realised activation to ``model``."""

    steps = list(model.T)
    da_price = forecasts[activation.da_price_col].astype(float).to_numpy()
    idc_price = forecasts[activation.idc_price_col].astype(float).fillna(0.0).to_numpy()
    afrr_price = series_or_zero(
        activation.afrr_energy_price,
        forecasts.index,
    ).to_numpy()
    da_position = series_or_zero(
        activation.da_position_mwh,
        forecasts.index,
    ).to_numpy()
    idc_buy = series_or_zero(
        activation.idc_buy_mwh,
        forecasts.index,
    ).to_numpy()
    idc_sell = series_or_zero(
        activation.idc_sell_mwh,
        forecasts.index,
    ).to_numpy()
    final_planned = series_or_zero(
        activation.final_planned_electricity_mwh,
        forecasts.index,
    ).to_numpy()
    afrr_bid = series_or_zero(
        activation.afrr_energy_bid_mwh,
        forecasts.index,
    ).to_numpy()
    afrr_activation = series_or_zero(
        activation.afrr_energy_activated_mwh,
        forecasts.index,
    ).to_numpy()
    system_activation = series_or_zero(
        activation.afrr_system_activation_mwh,
        forecasts.index,
    ).to_numpy()
    validate_fixed_afrr_instruction(
        bid_mwh=afrr_bid,
        activation_mwh=afrr_activation,
        system_activation_mwh=system_activation,
    )

    model.da_price = pyo.Param(
        model.T,
        initialize={t: float(da_price[t]) for t in steps},
    )
    model.idc_price = pyo.Param(
        model.T,
        initialize={t: float(idc_price[t]) for t in steps},
    )
    model.afrr_energy_price = pyo.Param(
        model.T,
        initialize={t: float(afrr_price[t]) for t in steps},
    )
    model.electricity_price = pyo.Param(
        model.T,
        initialize={
            t: float(afrr_price[t]) + float(pyo.value(model.additional_electricity_charge[t]))
            for t in steps
        },
    )
    model.da_position_mwh = pyo.Param(
        model.T,
        initialize={t: float(da_position[t]) for t in steps},
    )
    model.idc_buy_mwh = pyo.Param(
        model.T,
        initialize={t: float(idc_buy[t]) for t in steps},
    )
    model.idc_sell_mwh = pyo.Param(
        model.T,
        initialize={t: float(idc_sell[t]) for t in steps},
    )
    model.final_planned_electricity_mwh = pyo.Param(
        model.T,
        initialize={t: float(final_planned[t]) for t in steps},
    )
    model.afrr_energy_bid_mwh = pyo.Param(
        model.T,
        initialize={t: float(afrr_bid[t]) for t in steps},
    )
    model.afrr_energy_activated_mwh = pyo.Param(
        model.T,
        initialize={t: float(afrr_activation[t]) for t in steps},
    )
    model.actual_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.final_planned_electricity_mwh[t] + mm.afrr_energy_activated_mwh[t],
    )
    model.required_electricity_consumption_mwh = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.actual_electricity_consumption_mwh[t],
    )
    add_electricity_settlement_cost_expressions(model)


def add_electricity_settlement_cost_expressions(model: pyo.ConcreteModel) -> None:
    """Add common DA, intraday, and balancing-energy settlement expressions."""

    model.da_electricity_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.da_position_mwh[t] * mm.da_price[t],
    )
    model.idc_buy_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.idc_buy_mwh[t] * mm.idc_price[t],
    )
    model.idc_sell_revenue = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.idc_sell_mwh[t] * mm.idc_price[t],
    )
    model.afrr_energy_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.afrr_energy_activated_mwh[t] * mm.afrr_energy_price[t],
    )
    model.electricity_market_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: (
            mm.da_electricity_cost[t]
            + mm.idc_buy_cost[t]
            - mm.idc_sell_revenue[t]
            + mm.afrr_energy_cost[t]
        ),
    )


def validate_stage_input(
    stage: ElectricityMarketStage,
    market_input: ElectricityMarketStageInput,
) -> None:
    """Ensure a stage receives the matching typed electricity-market input."""

    expected_type: type[ElectricityMarketStageInput]
    if stage == ElectricityMarketStage.DAY_AHEAD:
        expected_type = DayAheadPosition
    elif stage == ElectricityMarketStage.INTRADAY:
        expected_type = IntradayAdjustment
    elif stage == ElectricityMarketStage.AFRR_ENERGY:
        expected_type = BalancingEnergyActivation
    else:
        raise ValueError(f"Unsupported electricity market stage '{stage}'")
    if not isinstance(market_input, expected_type):
        raise TypeError(
            f"Stage '{stage.value}' requires {expected_type.__name__}, "
            f"received {type(market_input).__name__}"
        )


def market_name_for_stage(stage: ElectricityMarketStage) -> str:
    """Return the configured market name that owns a physical stage."""

    return {
        ElectricityMarketStage.DAY_AHEAD: "day_ahead",
        ElectricityMarketStage.INTRADAY: "intraday_continuous",
        ElectricityMarketStage.AFRR_ENERGY: "afrr_energy",
    }[stage]


def stage_label(stage: ElectricityMarketStage) -> str:
    """Return the solve label used in errors for an electricity stage."""

    return {
        ElectricityMarketStage.DAY_AHEAD: "Dispatch",
        ElectricityMarketStage.INTRADAY: "IDC adjustment",
        ElectricityMarketStage.AFRR_ENERGY: "aFRR down adjustment",
    }[stage]


def stage_warning_label(stage: ElectricityMarketStage) -> str:
    """Return the user-facing warning label for an electricity stage."""

    return {
        ElectricityMarketStage.DAY_AHEAD: "day-ahead dispatch",
        ElectricityMarketStage.INTRADAY: "intraday adjustment",
        ElectricityMarketStage.AFRR_ENERGY: "aFRR down adjustment",
    }[stage]


def validate_fixed_afrr_instruction(
    *,
    bid_mwh: np.ndarray,
    activation_mwh: np.ndarray,
    system_activation_mwh: np.ndarray,
) -> None:
    """Validate a fixed aFRR-down bid and its realised activation."""

    tolerance = 1e-9
    if (bid_mwh < -tolerance).any() or (activation_mwh < -tolerance).any():
        raise ValueError("Fixed aFRR bid and activation values must be non-negative")
    if (activation_mwh > bid_mwh + tolerance).any():
        raise ValueError("Fixed aFRR activation cannot exceed the submitted aFRR bid")
    if (activation_mwh > system_activation_mwh + tolerance).any():
        raise ValueError("Fixed aFRR activation cannot exceed the system activation signal")
