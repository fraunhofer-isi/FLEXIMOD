# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ETES charging/storage process backed by a natural-gas boiler."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
import pyomo.environ as pyo

from flexi_mod.plants._helpers import _capacity_signal_values, _thermal_storage
from flexi_mod.plants.constants import (
    DELIVERY_GUARANTEE_SLACK_PENALTY_EUR_PER_MWH,
    THERMAL_STORAGE_GAS_BOILER_ROUTE,
)
from flexi_mod.plants.model_utils import pyomo_value
from flexi_mod.plants.routes.base import SteamRouteProcess
from flexi_mod.plants.stage import SteamMarketStage


class ThermalStorageGasBoilerProcess(SteamRouteProcess):
    """ETES charging/storage process backed by a natural-gas boiler."""

    name = THERMAL_STORAGE_GAS_BOILER_ROUTE
    required_components = frozenset({"thermal_storage", "boiler"})

    def initial_soc(
        self,
        components: dict[str, object],
        requested_soc_mwh: float | None,
    ) -> float:
        storage = _thermal_storage(components)
        if requested_soc_mwh is None:
            return float(storage.initial_soc_mwh)
        return float(requested_soc_mwh)

    def add_process_constraints(
        self,
        model: pyo.ConcreteModel,
        components: dict[str, object],
        stage: SteamMarketStage,
        timestep_hours: float,
        heat_demand_mwh: list[float],
    ) -> None:
        storage = model.technology_blocks["thermal_storage"]
        boiler = model.technology_blocks["boiler"]

        model.electricity_balance = pyo.Constraint(
            model.T,
            rule=lambda mm, t: mm.electricity_consumption[t] == storage.electricity_consumption[t],
        )
        model.heat_balance = pyo.Constraint(
            model.T,
            rule=lambda mm, t: storage.discharge_heat[t] + boiler.heat_out[t] == mm.heat_demand[t],
        )

        if stage not in {SteamMarketStage.DAY_AHEAD, SteamMarketStage.INTRADAY}:
            return

        max_charge_mwh = _thermal_storage(components).max_power_charge_mw * timestep_hours
        model.reserve_charge_power_headroom = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                storage.electric_charge_to_storage[t] + mm.reserved_capacity_mwh[t]
                <= max_charge_mwh
            ),
        )
        model.reserve_storage_headroom = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                storage.soc[t] + mm.reserved_capacity_mwh[t] * storage.efficiency_charge
                <= storage.max_capacity_mwh
            ),
        )
        self._add_reserved_capacity_delivery_guarantee(
            model,
            components,
            heat_demand_mwh,
            timestep_hours,
        )

    def validate_capacity_reservation(
        self,
        reserved_capacity_mwh: pd.Series,
    ) -> None:
        if (reserved_capacity_mwh < -1e-9).any():
            raise ValueError("Reserved aFRR capacity must be non-negative")

    def physical_result_fields(
        self,
        model: pyo.ConcreteModel,
        position: int,
    ) -> dict[str, object]:
        storage = model.technology_blocks["thermal_storage"]
        boiler = model.technology_blocks["boiler"]
        return {
            "etes_charge_MWh": pyomo_value(storage.electric_charge_to_storage[position]),
            "etes_discharge_MWh": pyomo_value(storage.discharge_heat[position]),
            "etes_soc_MWh": pyomo_value(storage.soc[position]),
            "gas_heat_MWh": pyomo_value(boiler.heat_out[position]),
            "gas_input_MWh": pyomo_value(boiler.fuel_input[position]),
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
        storage = _thermal_storage(components)
        values = _capacity_signal_values(signals, timestamp, timestep_hours)
        max_charge_mwh = storage.max_power_charge_mw * timestep_hours
        values.update(
            {
                "reserved_capacity_headroom_MWh": values["afrr_capacity_reserved_MWh"],
                "available_charge_headroom_after_schedule_MWh": max(
                    0.0,
                    max_charge_mwh - float(final_planned_mwh),
                ),
                "available_storage_headroom_after_schedule_MWh": (
                    max(
                        0.0,
                        storage.max_capacity_mwh - float(physical_fields["etes_soc_MWh"]),
                    )
                    / storage.efficiency_charge
                ),
            }
        )
        return values

    def next_soc(self, committed_row: pd.Series) -> float:
        return float(committed_row["etes_soc_MWh"])

    def objective_penalty(
        self,
        model: pyo.ConcreteModel,
        position: int,
    ) -> pyo.NumericValue | float:
        if not hasattr(model, "delivery_guarantee_slack_mwh"):
            return 0.0
        return (
            DELIVERY_GUARANTEE_SLACK_PENALTY_EUR_PER_MWH
            * model.delivery_guarantee_slack_mwh[position]
        )

    def warn_after_solve(self, model: pyo.ConcreteModel, stage_label: str) -> None:
        if not hasattr(model, "delivery_guarantee_slack_mwh"):
            return
        worst = max(
            (float(pyo.value(model.delivery_guarantee_slack_mwh[t])) for t in model.T),
            default=0.0,
        )
        if worst > 1e-6:
            warnings.warn(
                f"{stage_label}: reserved aFRR-down capacity exceeds what the plant "
                f"can deliver under continuous activation by up to {worst:.3f} MWh "
                "per step even after rescheduling. The capacity bid sizing and the "
                "plant parameters are inconsistent.",
                stacklevel=2,
            )

    def infeasibility_message(self, stage_label: str, solver_name: str) -> str:
        return (
            f"{stage_label} solve is infeasible with solver '{solver_name}'. "
            "FLEXIMOD now enforces strict useful heat dispatch: gas heat plus "
            "storage discharge must equal heat demand in every timestep, with no "
            "unmet-heat or heat-dump slack. Check fixed market electricity "
            "positions, aFRR activation, ETES storage headroom, and heat demand."
        )

    @staticmethod
    def _add_reserved_capacity_delivery_guarantee(
        model: pyo.ConcreteModel,
        components: dict[str, object],
        heat_demand_mwh: list[float],
        timestep_hours: float,
    ) -> None:
        storage_component = _thermal_storage(components)
        storage = model.technology_blocks["thermal_storage"]
        max_discharge_mwh = storage_component.max_power_discharge_mw * timestep_hours
        model.max_heat_outlet_mwh = pyo.Param(
            model.T,
            initialize={
                t: min(float(heat_demand_mwh[t]), max_discharge_mwh)
                / storage_component.efficiency_discharge
                for t in model.T
            },
        )
        model.soc_under_full_activation_mwh = pyo.Var(
            model.T,
            within=pyo.NonNegativeReals,
        )
        model.delivery_guarantee_slack_mwh = pyo.Var(
            model.T,
            within=pyo.NonNegativeReals,
        )
        model.full_activation_soc_tracks_schedule = pyo.Constraint(
            model.T,
            rule=lambda mm, t: mm.soc_under_full_activation_mwh[t] >= storage.soc[t],
        )

        def full_activation_soc_balance(
            mm: pyo.ConcreteModel,
            t: int,
        ) -> pyo.Constraint:
            previous = (
                storage.initial_soc_mwh if t == 0 else mm.soc_under_full_activation_mwh[t - 1]
            )
            return mm.soc_under_full_activation_mwh[t] >= (
                previous * (1.0 - storage.storage_loss_rate)
                + (storage.electric_charge_to_storage[t] + mm.reserved_capacity_mwh[t])
                * storage.efficiency_charge
                - mm.max_heat_outlet_mwh[t]
            )

        model.full_activation_soc_balance = pyo.Constraint(
            model.T,
            rule=full_activation_soc_balance,
        )
        model.reserve_delivery_headroom = pyo.Constraint(
            model.T,
            rule=lambda mm, t: (
                mm.soc_under_full_activation_mwh[t]
                <= storage.max_capacity_mwh + mm.delivery_guarantee_slack_mwh[t]
            ),
        )
