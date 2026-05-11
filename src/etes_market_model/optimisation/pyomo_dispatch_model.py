# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from etes_market_model.config.case_config import CaseConfig
from etes_market_model.plants.steam_generation_plant import (
    DispatchSignals,
    SteamGenerationPlant,
)


class PyomoDispatchModel:
    """Backward-compatible wrapper around ``SteamGenerationPlant`` dispatch.

    The physical Pyomo model now lives in
    ``etes_market_model.plants.steam_generation_plant`` and component blocks live
    in ``etes_market_model.plants.technologies``.
    """

    def __init__(self, config: CaseConfig):
        self.config = config

    def solve_rolling(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        signals: DispatchSignals,
    ) -> pd.DataFrame:
        return plant.solve_rolling(self.config, forecasts, signals)
