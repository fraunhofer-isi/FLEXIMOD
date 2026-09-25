# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Direct electric-boiler process backed by a natural-gas boiler."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyomo.environ as pyo

from flexi_mod.plants._helpers import (
    _electric_boiler,
    _empty_capacity_result_fields,
)
from flexi_mod.plants.constants import DIRECT_ELECTRIC_GAS_BOILER_ROUTE
from flexi_mod.plants.model_utils import pyomo_value
from flexi_mod.plants.routes.base import SteamRouteProcess
from flexi_mod.plants.stage import SteamMarketStage


class DirectElectricGasBoilerProcess(SteamRouteProcess):
    """Direct electric-boiler process backed by a natural-gas boiler."""

    name = DIRECT_ELECTRIC_GAS_BOILER_ROUTE
    required_components = frozenset({"electric_boiler", "boiler"})

    def add_process_constraints(
        self,
        model: pyo.ConcreteModel,
        components: dict[str, object],
        stage: SteamMarketStage,
        timestep_hours: float,
        heat_demand_mwh: list[float],
    ) -> None:
        electric_boiler = model.technology_blocks["electric_boiler"]
        gas_boiler = model.technology_blocks["boiler"]
        model.electricity_balance = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                mm.electricity_consumption[t] == electric_boiler.electricity_consumption[t]
            ),
        )
        model.heat_balance = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                electric_boiler.heat_out[t] + gas_boiler.heat_out[t] == mm.heat_demand[t]
            ),
        )

    def validate_capacity_reservation(
        self,
        reserved_capacity_mwh: pd.Series,
    ) -> None:
        if (reserved_capacity_mwh.abs() > 1e-9).any():
            raise ValueError(
                "The electric-boiler + gas-boiler route does not support aFRR-capacity reservations"
            )

    def physical_result_fields(
        self,
        model: pyo.ConcreteModel,
        position: int,
    ) -> dict[str, object]:
        electric_boiler = model.technology_blocks["electric_boiler"]
        gas_boiler = model.technology_blocks["boiler"]
        return {
            "technology_route": self.name,
            "electric_boiler_electricity_consumption_MWh": pyomo_value(
                electric_boiler.electricity_consumption[position]
            ),
            "electric_boiler_heat_MWh": pyomo_value(electric_boiler.heat_out[position]),
            "gas_heat_MWh": pyomo_value(gas_boiler.heat_out[position]),
            "gas_input_MWh": pyomo_value(gas_boiler.fuel_input[position]),
        }

    def capacity_result_fields(
        self,
        signals: Any,
        timestamp: pd.Timestamp,
        timestep_hours: float,
        final_planned_mwh: float,
        physical_fields: dict[str, object],
        components: dict[str, object],
    ) -> dict[str, object]:
        electric_boiler = _electric_boiler(components)
        max_load_mwh = electric_boiler.max_electricity_input_mw * timestep_hours
        return {
            **_empty_capacity_result_fields(),
            "reserved_capacity_headroom_MWh": 0.0,
            "available_load_headroom_after_schedule_MWh": max(
                0.0,
                max_load_mwh - float(final_planned_mwh),
            ),
        }

    def infeasibility_message(self, stage_label: str, solver_name: str) -> str:
        return (
            f"{stage_label} solve is infeasible with solver '{solver_name}' for "
            "the direct electric-boiler + gas-boiler route. Gas heat plus "
            "electric-boiler heat must equal heat demand in every timestep, with "
            "no storage, unmet-heat, or heat-dump slack. Check fixed electricity "
            "positions, aFRR activation, electric-boiler input capacity, "
            "gas-boiler heat capacity, and heat demand."
        )
