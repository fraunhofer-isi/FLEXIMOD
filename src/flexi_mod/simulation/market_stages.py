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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from flexi_mod.markets.base_market import MarketResultKind

if TYPE_CHECKING:
    from flexi_mod.markets.base_market import BaseMarket
    from flexi_mod.plants.base_plant import BasePlant


@dataclass(frozen=True)
class MarketStageState:
    """Market results already available in the current decision window."""

    operating_schedule: pd.DataFrame
    capacity_award: pd.DataFrame

    @classmethod
    def empty(cls, index: pd.Index) -> MarketStageState:
        """Create the empty market-result state for a new decision window."""

        return cls(
            operating_schedule=pd.DataFrame(index=index),
            capacity_award=pd.DataFrame(index=index),
        )


@dataclass(frozen=True)
class MarketStageContext:
    """Everything a strategy needs to decide one configured market stage."""

    market: BaseMarket
    plant: BasePlant
    forecasts: pd.DataFrame
    stage_state: MarketStageState
    initial_soc_mwh: float | None = None
    rolling: bool = False

    def result(self, values: pd.DataFrame) -> MarketStageResult:
        """Attach this context's market identity and result semantics."""

        return MarketStageResult(
            market_name=self.market.name,
            result_kind=self.market.result_kind,
            values=values,
        )

    def instruction(
        self,
        payload: object,
        *,
        execution_forecasts: pd.DataFrame | None = None,
        initial_state: Mapping[str, float] | None = None,
        rolling: bool | None = None,
    ) -> MarketStageInstruction:
        """Attach this context's market identity to a plant instruction.

        A strategy owns the commercial policy and returns an instruction.  The
        plant subsequently validates and executes its typed payload through its
        physical model.  Keeping the market identity here lets the runner
        reject an instruction intended for a different market before solving.
        """

        return MarketStageInstruction(
            market_name=self.market.name,
            result_kind=self.market.result_kind,
            payload=payload,
            execution_forecasts=execution_forecasts,
            initial_state=dict(initial_state or {}),
            rolling=self.rolling if rolling is None else rolling,
        )


@dataclass(frozen=True)
class MarketStageResult:
    """The values a strategy stage contributes to the sequential schedule."""

    market_name: str
    result_kind: MarketResultKind
    values: pd.DataFrame


@dataclass(frozen=True)
class MarketStageInstruction:
    """A strategy's typed commercial instruction for one plant-market stage.

    ``payload`` is a typed product request. For example, electricity-market
    strategies use an
    :class:`~flexi_mod.markets.electricity_settlement.ElectricityMarketRequest`;
    a plant then checks that product against its own physical model. Future
    plant families can introduce their own typed requests without changing the
    market runner.

    ``initial_state`` deliberately uses a small named mapping instead of a
    plant-specific field. A storage plant can pass its state of charge while a
    stateless plant leaves it empty. This keeps the hand-off reusable without
    making the simulation layer depend on a technology class.
    """

    market_name: str
    result_kind: MarketResultKind
    payload: object
    execution_forecasts: pd.DataFrame | None = None
    initial_state: Mapping[str, float] = field(default_factory=dict)
    rolling: bool = False

    def result(self, values: pd.DataFrame) -> MarketStageResult:
        """Attach executed values while retaining this instruction's identity."""

        return MarketStageResult(
            market_name=self.market_name,
            result_kind=self.result_kind,
            values=values,
        )
