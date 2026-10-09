# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the explicit market-strategy-plant stage hand-off."""

import pandas as pd

from flexi_mod.markets.afrr_capacity import AFRRCapacityMarket
from flexi_mod.markets.base_market import MarketCommitmentKind
from flexi_mod.markets.day_ahead import DayAheadMarket
from flexi_mod.plants.base_plant import BasePlant
from flexi_mod.simulation.market_stages import MarketCommitments, MarketStageContext


def test_stage_context_marks_day_ahead_as_a_dispatch_commitment() -> None:
    index = pd.date_range("2025-01-01", periods=1, freq="h")
    context = MarketStageContext(
        market=DayAheadMarket("day_ahead", {}),
        plant=BasePlant(name="plant", unit_type="test", node="node"),
        forecasts=pd.DataFrame(index=index),
        commitments=MarketCommitments.empty(index),
    )

    result = context.result(pd.DataFrame(index=index))

    assert result.market_name == "day_ahead"
    assert result.commitment_kind == MarketCommitmentKind.DISPATCH


def test_capacity_market_marks_its_result_as_a_capacity_reservation() -> None:
    market = AFRRCapacityMarket("afrr_capacity", {})

    assert market.commitment_kind == MarketCommitmentKind.CAPACITY_RESERVATION
