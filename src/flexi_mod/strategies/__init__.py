# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from flexi_mod.strategies.building_strategy import BuildingStrategy
from flexi_mod.strategies.hybrid_electric_gas_boiler_strategy import (
    HybridElectricGasBoilerStrategy,
)
from flexi_mod.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy
from flexi_mod.strategies.industrial_day_ahead_strategy import (
    IndustrialDayAheadCostMinimisationStrategy,
)

STRATEGY_REGISTRY = {
    "building_v2g": BuildingStrategy,
    "industrial_day_ahead_cost_minimisation": IndustrialDayAheadCostMinimisationStrategy,
    "hybrid_electric_gas_boiler": HybridElectricGasBoilerStrategy,
    "hybrid_etes_gas": HybridETESGasStrategy,
    # Pay-as-cleared is a config-selected capacity-pricing rule inside
    # HybridETESGasStrategy, not a separate class. This name is kept as a
    # backward-compatible alias that selects the pay-as-cleared rule.
    "hybrid_etes_gas_pay_as_cleared_capacity": HybridETESGasStrategy,
}


def build_strategy(name: str, config):
    """Build the configured strategy or raise a clear error for an unknown name."""

    try:
        strategy_class = STRATEGY_REGISTRY[name]
    except KeyError as exc:
        options = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {options}") from exc
    return strategy_class(config)


__all__ = [
    "BuildingStrategy",
    "HybridElectricGasBoilerStrategy",
    "HybridETESGasStrategy",
    "IndustrialDayAheadCostMinimisationStrategy",
    "STRATEGY_REGISTRY",
    "build_strategy",
]
