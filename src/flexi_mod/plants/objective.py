# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Market-cost expressions and the shared minimisation objective."""

from __future__ import annotations

import pyomo.environ as pyo

from flexi_mod.plants.routes.base import SteamRouteProcess


def _add_market_cost_expressions(model: pyo.ConcreteModel) -> None:
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


def _add_common_costs_and_objective(
    model: pyo.ConcreteModel,
    route: SteamRouteProcess,
) -> None:
    boiler = model.technology_blocks["boiler"]
    model.additional_electricity_charges_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: (
            mm.actual_electricity_consumption_mwh[t] * mm.additional_electricity_charge[t]
        ),
    )
    model.electricity_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: (
            mm.electricity_market_cost[t] + mm.additional_electricity_charges_cost[t]
        ),
    )
    model.tax_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: mm.electricity_cost[t] * mm.tax_rate,
    )
    model.gas_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: boiler.operating_cost[t],
    )
    model.co2_cost = pyo.Expression(
        model.T,
        rule=lambda mm, t: boiler.co2_cost[t],
    )
    model.objective = pyo.Objective(
        sense=pyo.minimize,
        expr=pyo.quicksum(
            model.electricity_cost[t]
            + model.gas_cost[t]
            + model.co2_cost[t]
            + model.tax_cost[t]
            + route.objective_penalty(model, t)
            for t in model.T
        ),
    )
