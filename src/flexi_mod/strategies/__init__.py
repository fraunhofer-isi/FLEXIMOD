# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from flexi_mod.strategies.electrified_steel_strategy import ElectrifiedSteelStrategy
from flexi_mod.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy
from flexi_mod.strategies.pay_as_cleared_capacity_strategy import (
    PayAsClearedCapacityHybridETESGasStrategy,
)
from flexi_mod.strategies.steel_cost_minimization_strategy import (
    SteelCostMinimizationStrategy,
)

STRATEGY_REGISTRY = {
    "hybrid_etes_gas": HybridETESGasStrategy,
    # Pay-as-cleared is a config-selected capacity-pricing rule inside
    # HybridETESGasStrategy, not a separate class. This name is kept as a
    # backward-compatible alias that selects the pay-as-cleared rule.
    "hybrid_etes_gas_pay_as_cleared_capacity": HybridETESGasStrategy,
    "steel_cost_minimization": SteelCostMinimizationStrategy,
    "electrified_steel": ElectrifiedSteelStrategy,
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
    "HybridETESGasStrategy",
    "SteelCostMinimizationStrategy",
    "ElectrifiedSteelStrategy",
    "STRATEGY_REGISTRY",
    "build_strategy",
]
