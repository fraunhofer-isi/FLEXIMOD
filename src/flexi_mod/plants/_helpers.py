# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Low-level component accessors and capacity-signal helpers shared across routes."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from flexi_mod.plants.model_utils import series_value
from flexi_mod.plants.signals.afrr_down_signals import AFRRDownSignals
from flexi_mod.plants.signals.dispatch_signals import DispatchSignals
from flexi_mod.plants.signals.idc_adjustment_signals import IDCAdjustmentSignals
from flexi_mod.plants.signals.steam_signals import SteamSignals
from flexi_mod.plants.stage import SteamMarketStage
from flexi_mod.plants.technologies import ElectricBoiler, ThermalStorage


def _capacity_signal_values(
    signals: SteamSignals,
    timestamp: pd.Timestamp,
    timestep_hours: float,
) -> dict[str, object]:
    reserved_mwh = float(series_value(signals.reserved_capacity_mwh, timestamp, 0.0))
    if signals.afrr_capacity_reserved_mw is not None:
        reserved_mw = float(series_value(signals.afrr_capacity_reserved_mw, timestamp, 0.0))
    else:
        reserved_mw = reserved_mwh / timestep_hours if timestep_hours > 0 else 0.0
    clearing_price = float(
        series_value(
            signals.afrr_capacity_price_eur_per_mw_h,
            timestamp,
            0.0,
        )
    )
    return {
        "afrr_capacity_block_id": str(series_value(signals.afrr_capacity_block_id, timestamp, "")),
        "afrr_capacity_block_duration_h": float(
            series_value(
                signals.afrr_capacity_block_duration_h,
                timestamp,
                0.0,
            )
        ),
        "afrr_capacity_pricing_rule": str(
            series_value(signals.afrr_capacity_pricing_rule, timestamp, "")
        ),
        "afrr_capacity_bid_price_EUR_per_MW_h": float(
            series_value(
                signals.afrr_capacity_bid_price_eur_per_mw_h,
                timestamp,
                0.0,
            )
        ),
        "afrr_capacity_clearing_price_EUR_per_MW_h": clearing_price,
        "afrr_capacity_settlement_price_EUR_per_MW_h": float(
            series_value(
                signals.afrr_capacity_settlement_price_eur_per_mw_h,
                timestamp,
                0.0,
            )
        ),
        "afrr_capacity_down_price_EUR_per_MW_h": clearing_price,
        "afrr_capacity_reserved_MW": reserved_mw,
        "afrr_capacity_reserved_MWh": reserved_mwh,
        "afrr_capacity_revenue_EUR": float(
            series_value(signals.afrr_capacity_revenue_eur, timestamp, 0.0)
        ),
        "afrr_capacity_opportunity_cost_EUR": float(
            series_value(
                signals.afrr_capacity_opportunity_cost_eur,
                timestamp,
                0.0,
            )
        ),
        "afrr_capacity_market_surplus_EUR": float(
            series_value(
                signals.afrr_capacity_market_surplus_eur,
                timestamp,
                0.0,
            )
        ),
        "afrr_capacity_net_value_EUR": float(
            series_value(signals.afrr_capacity_net_value_eur, timestamp, 0.0)
        ),
    }


def _empty_capacity_result_fields() -> dict[str, object]:
    return {
        "afrr_capacity_block_id": "",
        "afrr_capacity_block_duration_h": 0.0,
        "afrr_capacity_pricing_rule": "",
        "afrr_capacity_bid_price_EUR_per_MW_h": 0.0,
        "afrr_capacity_clearing_price_EUR_per_MW_h": 0.0,
        "afrr_capacity_settlement_price_EUR_per_MW_h": 0.0,
        "afrr_capacity_down_price_EUR_per_MW_h": 0.0,
        "afrr_capacity_reserved_MW": 0.0,
        "afrr_capacity_reserved_MWh": 0.0,
        "afrr_capacity_revenue_EUR": 0.0,
        "afrr_capacity_opportunity_cost_EUR": 0.0,
        "afrr_capacity_market_surplus_EUR": 0.0,
        "afrr_capacity_net_value_EUR": 0.0,
    }


def _thermal_storage(components: dict[str, object]) -> ThermalStorage:
    component = components.get("thermal_storage")
    if not isinstance(component, ThermalStorage):
        raise ValueError("Thermal-storage route has no thermal_storage component")
    return component


def _electric_boiler(components: dict[str, object]) -> ElectricBoiler:
    component = components.get("electric_boiler")
    if not isinstance(component, ElectricBoiler):
        raise ValueError("Direct-boiler route has no electric_boiler component")
    return component


def _validate_fixed_afrr_instruction(
    *,
    bid_mwh: np.ndarray,
    activation_mwh: np.ndarray,
    system_activation_mwh: np.ndarray,
) -> None:
    tolerance = 1e-9
    if (bid_mwh < -tolerance).any() or (activation_mwh < -tolerance).any():
        raise ValueError("Fixed aFRR bid and activation values must be non-negative")
    if (activation_mwh > bid_mwh + tolerance).any():
        raise ValueError("Fixed aFRR activation cannot exceed the submitted aFRR bid")
    if (activation_mwh > system_activation_mwh + tolerance).any():
        raise ValueError("Fixed aFRR activation cannot exceed the system activation signal")


def _validate_stage_signal_type(
    stage: SteamMarketStage,
    signals: SteamSignals,
) -> None:
    expected_type: type[SteamSignals]
    if stage == SteamMarketStage.DAY_AHEAD:
        expected_type = DispatchSignals
    elif stage == SteamMarketStage.INTRADAY:
        expected_type = IDCAdjustmentSignals
    else:
        expected_type = AFRRDownSignals
    if not isinstance(signals, expected_type):
        raise TypeError(
            f"Stage '{stage.value}' requires {expected_type.__name__}, "
            f"received {type(signals).__name__}"
        )


def _stage_label(stage: SteamMarketStage) -> str:
    return {
        SteamMarketStage.DAY_AHEAD: "Dispatch",
        SteamMarketStage.INTRADAY: "IDC adjustment",
        SteamMarketStage.AFRR_ENERGY: "aFRR down adjustment",
    }[stage]


def _stage_warning_label(stage: SteamMarketStage) -> str:
    return {
        SteamMarketStage.DAY_AHEAD: "day-ahead dispatch",
        SteamMarketStage.INTRADAY: "intraday adjustment",
        SteamMarketStage.AFRR_ENERGY: "aFRR down adjustment",
    }[stage]


def _charge_allowed_flag(
    stage: SteamMarketStage,
    signals: SteamSignals,
    position: int,
    direct_route: bool,
) -> bool:
    if stage == SteamMarketStage.DAY_AHEAD:
        return bool(cast(DispatchSignals, signals).charge_allowed.iloc[position])
    if stage == SteamMarketStage.INTRADAY and not direct_route:
        return bool(
            cast(
                IDCAdjustmentSignals,
                signals,
            ).idc_buy_upper_bound_mwh.iloc[position]
            > 1e-12
        )
    return False


def _preserve_legacy_stage_schema(
    row: dict[str, object],
    stage: SteamMarketStage,
    direct_route: bool,
) -> None:
    if stage != SteamMarketStage.AFRR_ENERGY:
        row.pop("afrr_energy_delivered_bid_price_EUR_per_MWh", None)
    if direct_route:
        return
    if stage == SteamMarketStage.DAY_AHEAD:
        for column in (
            "IDC_price_signal",
            "IDC_delivered_price_EUR_per_MWh",
            "afrr_energy_delivered_price_EUR_per_MWh",
            "electricity_trading_benchmark_EUR_per_MWh_el",
            "idc_buy_allowed_by_strategy",
            "idc_sell_allowed_by_strategy",
            "afrr_energy_bid_allowed_by_strategy",
            "afrr_headroom_binding",
            "afrr_curtailment_MWh",
        ):
            row.pop(column, None)
    elif stage == SteamMarketStage.INTRADAY:
        for column in (
            "afrr_energy_delivered_price_EUR_per_MWh",
            "afrr_energy_bid_allowed_by_strategy",
            "afrr_headroom_binding",
            "afrr_curtailment_MWh",
        ):
            row.pop(column, None)
