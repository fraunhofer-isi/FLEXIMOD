# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Day-ahead cost-minimisation strategy for cement and steel plants."""

from __future__ import annotations

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.day_ahead import DayAheadMarket
from flexi_mod.plants.cement_plant import CementPlant
from flexi_mod.plants.steel_plant import SteelPlant
from flexi_mod.simulation.market_stages import MarketStageContext, MarketStageResult
from flexi_mod.strategies.base_strategy import BaseStrategy

IndustrialPlant = CementPlant | SteelPlant


class IndustrialDayAheadCostMinimisationStrategy(BaseStrategy):
    """Buy the feasible cement/steel electricity demand in the day-ahead market.

    The physical plant model owns the Pyomo objective and all process
    constraints. This strategy supplies the configured day-ahead price and
    records the resulting electricity use as the plant's day-ahead position.
    Future industrial strategies can add flexibility products without changing
    either physical model or the runner's stage contract.
    """

    def __init__(self, config: CaseConfig):
        self.config = config
        self.electricity_price_column = config.market_signal("day_ahead", "price")

    def required_forecast_columns(self) -> set[str]:
        return {self.electricity_price_column}

    def decide_market_stage(self, context: MarketStageContext) -> MarketStageResult:
        """Solve the supported plant against the configured day-ahead product."""

        if not isinstance(context.plant, (CementPlant, SteelPlant)):
            raise TypeError(
                "Strategy 'industrial_day_ahead_cost_minimisation' requires CementPlant "
                f"or SteelPlant, received {type(context.plant).__name__}"
            )
        if not isinstance(context.market, DayAheadMarket):
            raise NotImplementedError(
                "Strategy 'industrial_day_ahead_cost_minimisation' supports only day_ahead"
            )
        return context.result(
            self.decide_day_ahead(
                context.plant,
                context.forecasts,
                market=context.market,
            )
        )

    def decide_day_ahead(
        self,
        plant: IndustrialPlant,
        forecasts: pd.DataFrame,
        *,
        market: DayAheadMarket | None = None,
    ) -> pd.DataFrame:
        """Return a feasible physical dispatch and its DA procurement position."""

        market = market or DayAheadMarket("day_ahead", self.config.market("day_ahead"))
        market_data = market.prepare_market_data(forecasts)
        dispatch = plant.solve_horizon(
            self.config,
            forecasts,
            electricity_price_column=market.signal_column("price"),
        )
        return _with_day_ahead_commitment(dispatch, market_data)


def _with_day_ahead_commitment(
    dispatch: pd.DataFrame,
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Give industrial dispatch the common market-ledger output contract."""

    required = {"electricity_consumption_MWh", "variable_cost_EUR"}
    missing = sorted(required - set(dispatch.columns))
    if missing:
        raise ValueError(
            "Industrial day-ahead dispatch is missing required result column(s): "
            + ", ".join(missing)
        )

    result = dispatch.copy()
    electricity = pd.to_numeric(result["electricity_consumption_MWh"], errors="raise")
    price = pd.to_numeric(
        market_data.reindex(result.index)["day_ahead_price_EUR_per_MWh"],
        errors="raise",
    )
    variable_cost = pd.to_numeric(result["variable_cost_EUR"], errors="raise")

    result["DA_position_MWh"] = electricity
    result["final_planned_electricity_MWh"] = electricity
    result["actual_electricity_consumption_MWh"] = electricity
    result["day_ahead_price_EUR_per_MWh"] = price
    result["DA_electricity_cost_EUR"] = electricity * price
    result["electricity_market_cost_EUR"] = result["DA_electricity_cost_EUR"]
    result["electricity_cost_EUR"] = result["DA_electricity_cost_EUR"]
    result["operating_cost_EUR"] = variable_cost
    result["gross_operating_cost_EUR"] = variable_cost
    result["net_operating_cost_EUR"] = variable_cost
    return result
