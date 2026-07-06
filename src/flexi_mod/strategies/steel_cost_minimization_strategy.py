# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants.steel_plant import SteelDispatchSignals, SteelPlant

NATURAL_GAS_PRICE_SIGNAL = "natural_gas_price"
HYDROGEN_PRICE_SIGNAL = "hydrogen_price"
IRON_ORE_PRICE_SIGNAL = "iron_ore_price"
LIME_PRICE_SIGNAL = "lime_price"
CO2_PRICE_SIGNAL = "co2_price"


class SteelCostMinimizationStrategy:
    """Temporary price-taking steel dispatch before market bidding is added."""

    def __init__(self, config: CaseConfig):
        self.config = config
        self.signals = SteelDispatchSignals(
            electricity_price_col=config.market_signal("day_ahead", "price"),
            natural_gas_price_col=NATURAL_GAS_PRICE_SIGNAL,
            hydrogen_price_col=HYDROGEN_PRICE_SIGNAL,
            iron_ore_price_col=IRON_ORE_PRICE_SIGNAL,
            lime_price_col=LIME_PRICE_SIGNAL,
            co2_price_col=CO2_PRICE_SIGNAL,
        )

    def required_forecast_columns(self) -> set[str]:
        return {
            self.signals.electricity_price_col,
            self.signals.natural_gas_price_col,
            self.signals.hydrogen_price_col,
            self.signals.iron_ore_price_col,
            self.signals.lime_price_col,
            self.signals.co2_price_col,
        }

    def dispatch(self, plant: SteelPlant, forecasts: pd.DataFrame) -> pd.DataFrame:
        if bool(self.config.dispatch_setting("rolling_horizon_enabled", True)):
            return plant.solve_rolling(self.config, forecasts, self.signals)
        return plant.solve_horizon(self.config, forecasts, self.signals)
