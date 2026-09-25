# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Market stage whose electricity position must be physically delivered."""

from enum import StrEnum


class SteamMarketStage(StrEnum):
    """Market stage whose electricity position must be physically delivered."""

    DAY_AHEAD = "day_ahead"
    INTRADAY = "intraday"
    AFRR_ENERGY = "afrr_energy"
