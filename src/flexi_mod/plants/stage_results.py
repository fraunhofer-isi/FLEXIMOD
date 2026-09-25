# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assembly of the per-stage result DataFrame from a solved steam-plant model."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd
import pyomo.environ as pyo

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.plants._helpers import (
    _charge_allowed_flag,
    _preserve_legacy_stage_schema,
)
from flexi_mod.plants.constants import DIRECT_ELECTRIC_GAS_BOILER_ROUTE
from flexi_mod.plants.model_utils import (
    pyomo_value,
    series_float_or_nan,
    series_value,
)
from flexi_mod.plants.signals.afrr_down_signals import AFRRDownSignals
from flexi_mod.plants.signals.dispatch_signals import DispatchSignals
from flexi_mod.plants.signals.idc_adjustment_signals import IDCAdjustmentSignals
from flexi_mod.plants.signals.steam_signals import SteamSignals
from flexi_mod.plants.stage import SteamMarketStage

if TYPE_CHECKING:
    from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant


def _extract_stage_results(
    plant: SteamGenerationPlant,
    stage: SteamMarketStage,
    model: pyo.ConcreteModel,
    config: CaseConfig,
    forecasts: pd.DataFrame,
    signals: SteamSignals,
    solver_name: str,
) -> pd.DataFrame:
    route = plant.route_process
    direct_route = route.name == DIRECT_ELECTRIC_GAS_BOILER_ROUTE
    dt_hours = config.timestep_minutes / 60.0
    da_price_col = (
        signals.electricity_price_col
        if isinstance(signals, DispatchSignals)
        else signals.da_price_col
    )
    idc_price_col = (
        signals.idc_price_col
        if isinstance(signals, (IDCAdjustmentSignals, AFRRDownSignals))
        else None
    )
    electricity_benchmark = getattr(
        signals,
        "electricity_trading_benchmark_eur_per_mwh_el",
        None,
    )
    rows: list[dict[str, object]] = []

    for position, timestamp in enumerate(forecasts.index):
        physical = route.physical_result_fields(model, position)
        electricity = pyomo_value(model.electricity_consumption[position])
        da_position = pyomo_value(model.da_position_mwh[position])
        idc_buy = pyomo_value(model.idc_buy_mwh[position])
        idc_sell = pyomo_value(model.idc_sell_mwh[position])
        final_planned = pyomo_value(model.final_planned_electricity_mwh[position])
        actual_electricity = pyomo_value(model.actual_electricity_consumption_mwh[position])
        afrr_bid = pyomo_value(model.afrr_energy_bid_mwh[position])
        afrr_activation = pyomo_value(model.afrr_energy_activated_mwh[position])
        day_ahead_price = float(forecasts[da_price_col].iloc[position])
        idc_price = (
            series_float_or_nan(
                forecasts[idc_price_col],
                position,
            )
            if idc_price_col is not None
            else float("nan")
        )
        additional_charge = pyomo_value(model.additional_electricity_charge[position])
        afrr_price = (
            pyomo_value(model.afrr_energy_price[position])
            if stage == SteamMarketStage.AFRR_ENERGY
            else float("nan")
        )
        tax_rate = pyomo_value(model.tax_rate)
        benchmark = (
            float(electricity_benchmark.iloc[position])
            if electricity_benchmark is not None
            else float("nan")
        )
        default_raw_bid_price = (
            benchmark / (1.0 + tax_rate) - additional_charge
            if stage == SteamMarketStage.AFRR_ENERGY
            else float("nan")
        )
        afrr_bid_price = (
            float(
                series_value(
                    cast(AFRRDownSignals, signals).afrr_energy_bid_price,
                    timestamp,
                    default_raw_bid_price,
                )
            )
            if stage == SteamMarketStage.AFRR_ENERGY
            else float("nan")
        )
        afrr_delivered_price = (afrr_price + additional_charge) * (1.0 + tax_rate)
        afrr_delivered_bid_price = (
            float(
                series_value(
                    cast(AFRRDownSignals, signals).afrr_energy_delivered_bid_price,
                    timestamp,
                    (afrr_bid_price + additional_charge) * (1.0 + tax_rate),
                )
            )
            if stage == SteamMarketStage.AFRR_ENERGY
            else float("nan")
        )
        afrr_market_spread = (
            afrr_delivered_bid_price - afrr_price if stage == SteamMarketStage.AFRR_ENERGY else 0.0
        )
        afrr_net_spread = (
            benchmark - afrr_delivered_price if stage == SteamMarketStage.AFRR_ENERGY else 0.0
        )
        afrr_reward = afrr_activation * afrr_market_spread
        afrr_net_value = afrr_activation * afrr_net_spread
        co2_price = (
            float(forecasts[signals.co2_price_col].iloc[position])
            if signals.co2_price_col and signals.co2_price_col in forecasts.columns
            else 0.0
        )
        afrr_signals = (
            cast(AFRRDownSignals, signals) if stage == SteamMarketStage.AFRR_ENERGY else None
        )
        default_free_volume = afrr_bid if direct_route else 0.0
        free_bid = (
            float(
                series_value(
                    afrr_signals.afrr_energy_free_bid_mwh,
                    timestamp,
                    default_free_volume,
                )
            )
            if afrr_signals is not None
            else 0.0
        )
        free_activation = (
            float(
                series_value(
                    afrr_signals.afrr_energy_free_activated_mwh,
                    timestamp,
                    afrr_activation if direct_route else 0.0,
                )
            )
            if afrr_signals is not None
            else 0.0
        )
        capacity_backed_bid = (
            0.0
            if direct_route or afrr_signals is None
            else float(
                series_value(
                    afrr_signals.afrr_energy_capacity_backed_bid_mwh,
                    timestamp,
                    0.0,
                )
            )
        )
        capacity_backed_activation = (
            0.0
            if direct_route or afrr_signals is None
            else float(
                series_value(
                    afrr_signals.afrr_energy_capacity_backed_activated_mwh,
                    timestamp,
                    0.0,
                )
            )
        )
        electricity_market_cost = pyomo_value(model.electricity_market_cost[position])
        additional_cost = pyomo_value(model.additional_electricity_charges_cost[position])
        electricity_cost = pyomo_value(model.electricity_cost[position])
        tax_cost = pyomo_value(model.tax_cost[position])
        gas_cost = pyomo_value(model.gas_cost[position])
        co2_cost = pyomo_value(model.co2_cost[position])

        row: dict[str, object] = {
            "datetime": timestamp,
            "plant_name": plant.name,
            "heat_demand_MWh": (
                float(forecasts[plant.heat_demand_column].iloc[position]) * dt_hours
            ),
            "day_ahead_price_EUR_per_MWh": day_ahead_price,
            "IDC_price_EUR_per_MWh": idc_price,
            "additional_electricity_charge_EUR_per_MWh_el": (additional_charge),
            "day_ahead_delivered_price_EUR_per_MWh": (
                (day_ahead_price + additional_charge) * (1.0 + tax_rate)
            ),
            "IDC_delivered_price_EUR_per_MWh": ((idc_price + additional_charge) * (1.0 + tax_rate)),
            "afrr_energy_delivered_price_EUR_per_MWh": (afrr_delivered_price),
            "gas_price_EUR_per_MWh": float(forecasts[signals.gas_price_col].iloc[position]),
            "co2_price_EUR_per_t": co2_price,
            "day_ahead_price_signal": da_price_col,
            "IDC_price_signal": idc_price_col or "",
            "gas_price_signal": signals.gas_price_col,
            "co2_price_signal": signals.co2_price_col or "",
            "gas_based_heat_benchmark_EUR_per_MWh_th": float(
                signals.gas_benchmark_eur_per_mwh_th.iloc[position]
            ),
            "electricity_trading_benchmark_EUR_per_MWh_el": benchmark,
            **physical,
            "electricity_consumption_MWh": electricity,
            "DA_position_MWh": da_position,
            "IDC_buy_MWh": idc_buy,
            "IDC_sell_MWh": idc_sell,
            "final_planned_electricity_MWh": final_planned,
            "actual_electricity_consumption_MWh": actual_electricity,
            "DA_electricity_cost_EUR": pyomo_value(model.da_electricity_cost[position]),
            "IDC_buy_cost_EUR": pyomo_value(model.idc_buy_cost[position]),
            "IDC_sell_revenue_EUR": pyomo_value(model.idc_sell_revenue[position]),
            "afrr_energy_bid_MWh": afrr_bid,
            "afrr_energy_bid_MW": (afrr_bid / dt_hours if dt_hours > 0 else 0.0),
            "afrr_energy_activated_MWh": afrr_activation,
            "afrr_energy_price_EUR_per_MWh": afrr_price,
            "afrr_system_activation_MWh": (
                float(afrr_signals.afrr_system_activation_mwh.iloc[position])
                if afrr_signals is not None
                else 0.0
            ),
            "afrr_energy_bid_price_EUR_per_MWh": afrr_bid_price,
            "afrr_energy_delivered_bid_price_EUR_per_MWh": (afrr_delivered_bid_price),
            "afrr_energy_market_spread_EUR_per_MWh": afrr_market_spread,
            "afrr_energy_net_spread_EUR_per_MWh": afrr_net_spread,
            "afrr_energy_cost_EUR": pyomo_value(model.afrr_energy_cost[position]),
            "afrr_energy_savings_vs_benchmark_EUR": afrr_net_value,
            "afrr_energy_pay_as_cleared_reward_EUR": afrr_reward,
            "afrr_energy_net_value_after_charges_EUR": afrr_net_value,
            "afrr_energy_capacity_backed_bid_MWh": capacity_backed_bid,
            "afrr_energy_free_bid_MWh": free_bid,
            "afrr_energy_capacity_backed_activated_MWh": (capacity_backed_activation),
            "afrr_energy_free_activated_MWh": free_activation,
            "afrr_headroom_binding": (
                bool(
                    series_value(
                        afrr_signals.afrr_headroom_binding,
                        timestamp,
                        False,
                    )
                )
                if afrr_signals is not None
                else False
            ),
            "afrr_curtailment_MWh": (
                float(
                    series_value(
                        afrr_signals.afrr_curtailment_mwh,
                        timestamp,
                        0.0,
                    )
                )
                if afrr_signals is not None
                else 0.0
            ),
            "electricity_market_cost_EUR": electricity_market_cost,
            "additional_electricity_charges_cost_EUR": additional_cost,
            "electricity_cost_EUR": electricity_cost,
            "tax_cost_EUR": tax_cost,
            "gas_cost_EUR": gas_cost,
            "co2_cost_EUR": co2_cost,
            "operating_cost_EUR": (electricity_cost + gas_cost + co2_cost + tax_cost),
            "charge_allowed_by_strategy": _charge_allowed_flag(
                stage,
                signals,
                position,
                direct_route,
            ),
            "idc_buy_allowed_by_strategy": (
                bool(
                    cast(
                        IDCAdjustmentSignals,
                        signals,
                    ).idc_buy_upper_bound_mwh.iloc[position]
                    > 1e-12
                )
                if stage == SteamMarketStage.INTRADAY
                else False
            ),
            "idc_sell_allowed_by_strategy": (
                bool(
                    cast(
                        IDCAdjustmentSignals,
                        signals,
                    ).idc_sell_upper_bound_mwh.iloc[position]
                    > 1e-12
                )
                if stage == SteamMarketStage.INTRADAY
                else False
            ),
            "afrr_energy_bid_allowed_by_strategy": bool(afrr_bid > 1e-12),
            "solver": solver_name,
        }
        row.update(
            route.capacity_result_fields(
                signals,
                timestamp,
                dt_hours,
                final_planned,
                physical,
                plant.components,
            )
        )
        row["gross_operating_cost_EUR"] = row["operating_cost_EUR"]
        row["net_operating_cost_EUR"] = float(row["gross_operating_cost_EUR"]) - float(
            row["afrr_capacity_revenue_EUR"]
        )
        _preserve_legacy_stage_schema(row, stage, direct_route)
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("datetime")
    numeric_columns = frame.select_dtypes(include=["number"]).columns
    frame[numeric_columns] = frame[numeric_columns].mask(
        frame[numeric_columns].abs() < 1e-9,
        0.0,
    )
    return frame
