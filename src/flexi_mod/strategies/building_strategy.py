# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Electricity-price strategy for the Building model."""

from __future__ import annotations

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants.building import Building
from flexi_mod.strategies.base_strategy import BaseStrategy


class BuildingStrategy(BaseStrategy):
    """Optimize building imports, charging, and V2G exports against prices."""

    def __init__(self, config: CaseConfig):
        self.config = config
        day_ahead = config.market("day_ahead")
        signals = day_ahead["signals"]
        self.import_price_column = str(signals["price"])
        self.export_price_column = str(signals.get("export_price", signals["price"]))

    def required_forecast_columns(self) -> set[str]:
        return {self.import_price_column, self.export_price_column}

    def decide_day_ahead(
        self,
        building: Building,
        forecasts: pd.DataFrame,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        """Return the least-cost feasible building schedule for one horizon."""

        return building.solve_horizon(
            config=self.config,
            forecasts=forecasts,
            electricity_price_column=self.import_price_column,
            export_price_column=self.export_price_column,
            initial_soc_mwh=initial_soc_mwh,
        )
