# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed hand-off objects for one synchronous market-decision stage.

FLEXIMOD is a price-taking dispatch model: markets prepare product data,
strategies apply commercial rules, and plants check the resulting instruction
with Pyomo. These classes make that simulation hand-off explicit without
introducing an agent, order-book, or market-clearing layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from flexi_mod.markets.base_market import MarketCommitmentKind

if TYPE_CHECKING:
    from flexi_mod.markets.base_market import BaseMarket
    from flexi_mod.plants.base_plant import BasePlant


@dataclass(frozen=True)
class MarketCommitments:
    """Positions fixed by earlier stages of the current decision window."""

    positions: pd.DataFrame
    capacity_reservation: pd.DataFrame

    @classmethod
    def empty(cls, index: pd.Index) -> MarketCommitments:
        """Create the no-commitment state for a new decision window."""

        return cls(
            positions=pd.DataFrame(index=index),
            capacity_reservation=pd.DataFrame(index=index),
        )


@dataclass(frozen=True)
class MarketStageContext:
    """Everything a strategy needs to decide one configured market stage."""

    market: BaseMarket
    plant: BasePlant
    forecasts: pd.DataFrame
    commitments: MarketCommitments
    initial_soc_mwh: float | None = None
    rolling: bool = False

    def result(self, values: pd.DataFrame) -> MarketStageResult:
        """Attach this context's market identity and commitment semantics."""

        return MarketStageResult(
            market_name=self.market.name,
            commitment_kind=self.market.commitment_kind,
            values=values,
        )


@dataclass(frozen=True)
class MarketStageResult:
    """The values a strategy stage contributes to the sequential schedule."""

    market_name: str
    commitment_kind: MarketCommitmentKind
    values: pd.DataFrame
