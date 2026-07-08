# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants.cement_plant import CementDispatchSignals, CementPlant

NATURAL_GAS_PRICE_SIGNAL = "natural_gas_price"
HYDROGEN_PRICE_SIGNAL = "hydrogen_price"
COAL_PRICE_SIGNAL = "coal_price"
CO2_PRICE_SIGNAL = "co2_price"


class CementCostMinimizationStrategy:
    """Temporary price-taking cement clinker dispatch before market bidding is added."""

    def __init__(self, config: CaseConfig):
        self.config = config
        self.signals = CementDispatchSignals(
            electricity_price_col=config.market_signal("day_ahead", "price"),
            natural_gas_price_col=NATURAL_GAS_PRICE_SIGNAL,
            hydrogen_price_col=HYDROGEN_PRICE_SIGNAL,
            coal_price_col=COAL_PRICE_SIGNAL,
            co2_price_col=CO2_PRICE_SIGNAL,
        )

    def required_forecast_columns(self) -> set[str]:
        return {
            self.signals.electricity_price_col,
            self.signals.natural_gas_price_col,
            self.signals.hydrogen_price_col,
            self.signals.co2_price_col,
        }

    def dispatch(self, plant: CementPlant, forecasts: pd.DataFrame) -> pd.DataFrame:
        return plant.solve_horizon(self.config, forecasts, self.signals)
