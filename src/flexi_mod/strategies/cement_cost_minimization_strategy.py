# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants.cement_plant import CementDispatchSignals, CementPlant

NATURAL_GAS_PRICE_SIGNAL = "natural_gas_price"
HYDROGEN_PRICE_SIGNAL = "hydrogen_price"
COAL_PRICE_SIGNAL = "coal_price"
BIOMASS_PRICE_SIGNAL = "biomass_price"
RDF_PRICE_SIGNAL = "rdf_price"
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
            biomass_price_col=BIOMASS_PRICE_SIGNAL,
            rdf_price_col=RDF_PRICE_SIGNAL,
            co2_price_col=CO2_PRICE_SIGNAL,
        )

    def required_forecast_columns(self) -> set[str]:
        return {
            self.signals.electricity_price_col,
            self.signals.natural_gas_price_col,
            self.signals.hydrogen_price_col,
            self.signals.co2_price_col,
        }

    def dispatch(
        self,
        plant: CementPlant,
        forecasts: pd.DataFrame,
        progress_callback: Callable[[pd.Timestamp, pd.Timestamp], None] | None = None,
    ) -> pd.DataFrame:
        if self.config.dispatch_setting("rolling_horizon_enabled", True):
            return plant.solve_rolling(
                self.config,
                forecasts,
                self.signals,
                progress_callback=progress_callback,
            )
        return plant.solve_horizon(self.config, forecasts, self.signals)
