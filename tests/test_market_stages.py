# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the explicit market-strategy-plant stage hand-off."""

import pandas as pd

from flexi_mod.markets.afrr_capacity import AFRRCapacityMarket
from flexi_mod.markets.base_market import MarketResultKind
from flexi_mod.markets.day_ahead import DayAheadMarket
from flexi_mod.plants.base_plant import BasePlant
from flexi_mod.simulation.market_stages import MarketStageContext, MarketStageState


def test_empty_stage_state_keeps_schedule_and_capacity_award_separate() -> None:
    index = pd.date_range("2025-01-01", periods=1, freq="h")

    state = MarketStageState.empty(index)

    assert state.operating_schedule.empty
    assert state.capacity_award.empty
    assert state.operating_schedule.index.equals(index)
    assert state.capacity_award.index.equals(index)


def test_stage_context_marks_day_ahead_as_an_operating_schedule() -> None:
    index = pd.date_range("2025-01-01", periods=1, freq="h")
    context = MarketStageContext(
        market=DayAheadMarket("day_ahead", {}),
        plant=BasePlant(name="plant", unit_type="test", node="node"),
        forecasts=pd.DataFrame(index=index),
        stage_state=MarketStageState.empty(index),
    )

    result = context.result(pd.DataFrame(index=index))

    assert result.market_name == "day_ahead"
    assert result.result_kind == MarketResultKind.OPERATING_SCHEDULE


def test_stage_instruction_keeps_market_identity_until_execution() -> None:
    index = pd.date_range("2025-01-01", periods=1, freq="h")
    context = MarketStageContext(
        market=DayAheadMarket("day_ahead", {}),
        plant=BasePlant(name="plant", unit_type="test", node="node"),
        forecasts=pd.DataFrame(index=index),
        stage_state=MarketStageState.empty(index),
    )

    instruction = context.instruction(payload={"commercial_rule": "buy_when_cheap"})
    result = instruction.result(pd.DataFrame(index=index))

    assert instruction.market_name == "day_ahead"
    assert instruction.result_kind == MarketResultKind.OPERATING_SCHEDULE
    assert result.market_name == "day_ahead"
    assert result.result_kind == MarketResultKind.OPERATING_SCHEDULE


def test_capacity_market_marks_its_result_as_a_capacity_award() -> None:
    market = AFRRCapacityMarket("afrr_capacity", {})

    assert market.result_kind == MarketResultKind.CAPACITY_AWARD
