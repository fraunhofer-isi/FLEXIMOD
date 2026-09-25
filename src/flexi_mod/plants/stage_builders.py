# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Market-stage builders that attach day-ahead, intraday, and aFRR variables."""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo

from flexi_mod.plants._helpers import _validate_fixed_afrr_instruction
from flexi_mod.plants.model_utils import series_or_zero
from flexi_mod.plants.objective import _add_market_cost_expressions
from flexi_mod.plants.signals.afrr_down_signals import AFRRDownSignals
from flexi_mod.plants.signals.dispatch_signals import DispatchSignals
from flexi_mod.plants.signals.idc_adjustment_signals import IDCAdjustmentSignals


def _add_day_ahead_stage(
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    signals: DispatchSignals,
) -> None:
    steps = list(model.T)
    price = forecasts[signals.electricity_price_col].astype(float).to_numpy()
    charge_allowed = signals.charge_allowed.astype(bool).reindex(forecasts.index).fillna(False)
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
    _add_market_cost_expressions(model)


def _add_intraday_stage(
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    signals: IDCAdjustmentSignals,
) -> None:
    steps = list(model.T)
    da_price = forecasts[signals.da_price_col].astype(float).to_numpy()
    idc_price = forecasts[signals.idc_price_col].astype(float).fillna(0.0).to_numpy()
    da_position = series_or_zero(
        signals.da_position_mwh,
        forecasts.index,
    ).to_numpy()
    buy_upper = (
        series_or_zero(
            signals.idc_buy_upper_bound_mwh,
            forecasts.index,
        )
        .clip(lower=0.0)
        .to_numpy()
    )
    sell_upper = (
        series_or_zero(
            signals.idc_sell_upper_bound_mwh,
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
    _add_market_cost_expressions(model)


def _add_afrr_energy_stage(
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    signals: AFRRDownSignals,
) -> None:
    steps = list(model.T)
    da_price = forecasts[signals.da_price_col].astype(float).to_numpy()
    idc_price = forecasts[signals.idc_price_col].astype(float).fillna(0.0).to_numpy()
    afrr_price = series_or_zero(
        signals.afrr_energy_price,
        forecasts.index,
    ).to_numpy()
    da_position = series_or_zero(
        signals.da_position_mwh,
        forecasts.index,
    ).to_numpy()
    idc_buy = series_or_zero(
        signals.idc_buy_mwh,
        forecasts.index,
    ).to_numpy()
    idc_sell = series_or_zero(
        signals.idc_sell_mwh,
        forecasts.index,
    ).to_numpy()
    final_planned = series_or_zero(
        signals.final_planned_electricity_mwh,
        forecasts.index,
    ).to_numpy()
    afrr_bid = series_or_zero(
        signals.afrr_energy_bid_mwh,
        forecasts.index,
    ).to_numpy()
    afrr_activation = series_or_zero(
        signals.afrr_energy_activated_mwh,
        forecasts.index,
    ).to_numpy()
    system_activation = series_or_zero(
        signals.afrr_system_activation_mwh,
        forecasts.index,
    ).to_numpy()
    _validate_fixed_afrr_instruction(
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
    _add_market_cost_expressions(model)
