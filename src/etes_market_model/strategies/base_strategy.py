# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from etes_market_model.plants.steam_generation_plant import SteamGenerationPlant


class BaseStrategy:
    """Interface for sequential market strategies."""

    def required_forecast_columns(self) -> set[str]:
        return set()

    def decide_day_ahead(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.DataFrame(index=forecasts.index)

    def decide_intraday_continuous(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
    ) -> pd.DataFrame:
        # TODO: Implement IDC adjustments without overwriting fixed DA positions.
        return pd.DataFrame(index=forecasts.index)

    def decide_afrr_energy(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
    ) -> pd.DataFrame:
        # TODO: Implement negative aFRR energy using remaining ETES charging headroom.
        return pd.DataFrame(index=forecasts.index)

    def decide_afrr_capacity(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
    ) -> pd.DataFrame:
        # TODO: Reserve headroom before DA/IDC once aFRR capacity is activated.
        return pd.DataFrame(index=forecasts.index)
