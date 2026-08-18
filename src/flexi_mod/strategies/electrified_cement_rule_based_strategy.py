# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rule-based market-timing variant of :class:`ElectrifiedCementStrategy`.

Physical dispatch is still solved exactly by :meth:`CementPlant.solve_rolling`; only the
market-timing decision is rule-based. The whole rule set is inherited from the steel
variant, which no longer contains anything steel-specific - the fuel-substitution
economics come from the plant's own ``afrr_fuel_substitution`` hook, and the bid bound
from ``afrr_aggregate_max_power_mw``.

The same caveat as the steel variant applies, and applies more sharply to cement:
production timing is decided first, with no awareness that aFRR capacity exists, and
capacity is then sized from whatever that fixed schedule happens to leave spare. Since
cement's only real flexibility *is* the room left in its thermal store, a schedule chosen
without regard to capacity value will leave less of it than the exact MILP would. Prefer
:class:`ElectrifiedCementStrategy` when capacity economics need to be accurate.
"""

from __future__ import annotations

from typing import Any

from flexi_mod.plants.cement_plant import CementDispatchSignals
from flexi_mod.strategies.electrified_cement_strategy import (
    BIOMASS_PRICE_SIGNAL,
    CO2_PRICE_SIGNAL,
    HYDROGEN_PRICE_SIGNAL,
    NATURAL_GAS_PRICE_SIGNAL,
    RDF_PRICE_SIGNAL,
    ElectrifiedCementStrategy,
)
from flexi_mod.strategies.electrified_steel_rule_based_strategy import (
    ElectrifiedSteelRuleBasedStrategy,
)

COAL_PRICE_SIGNAL = "coal_price"


class HybridStrategyCement(
    ElectrifiedCementStrategy, ElectrifiedSteelRuleBasedStrategy
):
    """DA/aFRR strategy for hybrid electricity/fuel switching in a clinker line.

    Takes its market configuration and commodity columns from
    :class:`ElectrifiedCementStrategy`, and its ``dispatch`` from the rule-based steel
    variant. The method resolution order matters: cement must come first so that
    ``commodity_forecast_columns`` and ``_build_signals`` are cement's, while ``dispatch``
    falls through to the rule-based implementation.
    """

    def _build_dispatch_signals(
        self,
        plant: Any,
        *,
        electricity_price_col: str,
        gate_column: str | None,
    ) -> CementDispatchSignals:
        # Cement has no discrete gated load - a hybrid stage blends continuously - so
        # gate_column is always None here and there is nothing to pass on.
        return CementDispatchSignals(
            electricity_price_col=electricity_price_col,
            natural_gas_price_col=NATURAL_GAS_PRICE_SIGNAL,
            hydrogen_price_col=HYDROGEN_PRICE_SIGNAL,
            coal_price_col=COAL_PRICE_SIGNAL,
            biomass_price_col=BIOMASS_PRICE_SIGNAL,
            rdf_price_col=RDF_PRICE_SIGNAL,
            co2_price_col=CO2_PRICE_SIGNAL,
        )


# Backward-compatible import alias for existing user code. New case configurations
# should use the clearer ``hybrid_strategy_cement`` strategy identifier.
ElectrifiedCementRuleBasedStrategy = HybridStrategyCement
