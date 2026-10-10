# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from flexi_mod.config.case_config import CaseConfig
    from flexi_mod.regulations import GridFeeRegulation
    from flexi_mod.simulation.market_stages import MarketStageInstruction


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

    def required_forecast_columns(self) -> set[str]:
        """Return plant-owned forecast profiles needed before market decisions.

        Strategies add their configured market signals separately. Subclasses
        override this for their physical demand and availability profiles.
        """

        return set()

    def solve_market_instruction(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        instruction: MarketStageInstruction,
    ) -> pd.DataFrame:
        """Execute one strategy instruction through this plant's physical model.

        Subclasses define the instruction payload that they accept.  The
        runner uses this single plant-facing entry point, so market and
        strategy code never need to call a technology-specific Pyomo solver.
        """

        del config, forecasts, instruction
        raise NotImplementedError(
            f"Plant '{type(self).__name__}' does not implement market-instruction execution"
        )
