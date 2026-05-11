# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BasePlant:
    """Common plant metadata used by strategies and ledgers."""

    name: str
    unit_type: str
    node: str
    objective: str = "min_variable_cost"
    heat_demand_column: str = ""

    @property
    def plant_type(self) -> str:
        return self.unit_type
