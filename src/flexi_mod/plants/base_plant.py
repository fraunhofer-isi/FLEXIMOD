# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flexi_mod.regulations import GridFeeRegulation


DEFAULT_GAS_EMISSIONS_FACTOR_KG_PER_MWH = 201.0


@dataclass
class BasePlant:
    """Common plant metadata used by strategies and ledgers."""

    name: str
    unit_type: str
    node: str
    objective: str = "min_variable_cost"
    heat_demand_column: str = ""
    additional_electricity_charge_eur_per_mwh: float = 0.0
    gas_emissions_factor_kg_per_mwh: float = DEFAULT_GAS_EMISSIONS_FACTOR_KG_PER_MWH
    grid_fee_regulation: GridFeeRegulation | None = None

    @property
    def plant_type(self) -> str:
        return self.unit_type
