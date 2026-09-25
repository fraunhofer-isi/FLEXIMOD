# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Union type alias for all steam-generation plant market signals."""

from __future__ import annotations

from flexi_mod.plants.signals.afrr_down_signals import AFRRDownSignals
from flexi_mod.plants.signals.dispatch_signals import DispatchSignals
from flexi_mod.plants.signals.idc_adjustment_signals import IDCAdjustmentSignals

SteamSignals = DispatchSignals | IDCAdjustmentSignals | AFRRDownSignals
