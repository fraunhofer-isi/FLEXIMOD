# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from flexi_mod.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy
from flexi_mod.strategies.pay_as_cleared_capacity_strategy import (
    PayAsClearedCapacityHybridETESGasStrategy,
)

STRATEGY_REGISTRY = {
    "hybrid_etes_gas": HybridETESGasStrategy,
    "hybrid_etes_gas_pay_as_cleared_capacity": PayAsClearedCapacityHybridETESGasStrategy,
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
    "PayAsClearedCapacityHybridETESGasStrategy",
    "STRATEGY_REGISTRY",
    "build_strategy",
]
