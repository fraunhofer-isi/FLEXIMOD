# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from enum import StrEnum


class MarketStage(StrEnum):
    DAY_AHEAD = "day_ahead"
    INTRADAY_CONTINUOUS = "intraday_continuous"
    AFRR_ENERGY = "afrr_energy"
    AFRR_CAPACITY = "afrr_capacity"
