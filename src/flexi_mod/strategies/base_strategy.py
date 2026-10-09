# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from flexi_mod.simulation.market_stages import MarketStageContext, MarketStageResult


class BaseStrategy:
    """Interface for sequential market strategies."""

    def decide_market_stage(self, context: MarketStageContext) -> MarketStageResult:
        """Apply this strategy to one prepared market stage.

        The returned values become market results available to the remaining
        stages of the decision window. Concrete strategies keep their
        market-specific rules in their existing ``decide_*`` methods.
        """

        raise NotImplementedError

    def required_forecast_columns(self) -> set[str]:
        return set()
