# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flexi_mod.regulations import GridFeeRegulation


@dataclass
class BasePlant:
    """Common plant metadata used by strategies and ledgers."""

    name: str
    unit_type: str
    node: str
    objective: str = "min_variable_cost"
    heat_demand_column: str = ""
    additional_electricity_charge_eur_per_mwh: float = 0.0
    grid_fee_regulation: GridFeeRegulation | None = None

    @property
    def plant_type(self) -> str:
        return self.unit_type
