# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Map solved plant models onto stable, file-ready dispatch-result tables.

This is FLEXIMOD's synchronous equivalent of ASSUME's output role.  Plants
retain their physical equations; mappers read solved Pyomo values and assemble
the public result schema consumed by ledgers, analytics, plots, and CSV output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import pyomo.environ as pyo

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_capacity import (
    BalancingCapacityAward,
    capacity_award_result_fields,
)
from flexi_mod.markets.afrr_energy import BalancingEnergyActivation
from flexi_mod.markets.day_ahead import DayAheadPosition
from flexi_mod.markets.electricity_settlement import (
    ElectricityMarketStage,
    ElectricityMarketStageInput,
)
from flexi_mod.markets.intraday_continuous import IntradayAdjustment
from flexi_mod.modeling.pyomo_utils import pyomo_value, series_float_or_nan, series_value
from flexi_mod.plants.base_plant import BasePlant
from flexi_mod.plants.building import Building
from flexi_mod.plants.cement_plant import CementPlant
from flexi_mod.plants.steam_generation_plant import (
    DIRECT_ELECTRIC_GAS_BOILER_ROUTE,
    SteamGenerationPlant,
)
from flexi_mod.plants.steel_plant import SteelPlant


@dataclass(frozen=True)
class ResultContext:
    """Data needed to map a solved plant model to dispatch-result rows."""

    forecasts: pd.DataFrame
    solver_name: str
    config: CaseConfig | None = None
    stage: ElectricityMarketStage | None = None
    market_input: ElectricityMarketStageInput | None = None
    capacity_award: BalancingCapacityAward | None = None


def extract_dispatch_results(
    plant: BasePlant,
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    solver_name: str,
    *,
    config: CaseConfig | None = None,
    stage: ElectricityMarketStage | None = None,
    market_input: ElectricityMarketStageInput | None = None,
    capacity_award: BalancingCapacityAward | None = None,
) -> pd.DataFrame:
    """Use the registered mapper for ``plant`` to create dispatch-result rows."""

    try:
        mapper = RESULT_MAPPERS[type(plant)]
    except KeyError as exc:
        raise ValueError(f"No result mapper is registered for {type(plant).__name__}") from exc
    return mapper(
        plant,
        model,
        ResultContext(
            forecasts=forecasts,
            solver_name=solver_name,
            config=config,
            stage=stage,
            market_input=market_input,
            capacity_award=capacity_award,
        ),
    )


def _map_building(
    plant: BasePlant,
    model: pyo.ConcreteModel,
    context: ResultContext,
) -> pd.DataFrame:
    building = _expect_plant_type(plant, Building)
    dt_hours = pyo.value(model.dt_hours)
    rows: list[dict[str, object]] = []
    for t, timestamp in enumerate(context.forecasts.index):
        grid_import = pyomo_value(model.grid_import_mwh[t])
        grid_export = pyomo_value(model.grid_export_mwh[t])
        charge = pyomo_value(model.electric_vehicle.charge_mwh[t])
        discharge = pyomo_value(model.electric_vehicle.discharge_mwh[t])
        soc = pyomo_value(model.electric_vehicle.soc_mwh[t])
        rows.append(
            {
                "datetime": timestamp,
                "plant_name": building.name,
                "building_demand_MWh": pyomo_value(model.building_demand_mwh[t]),
                "bus_availability_fraction": pyomo_value(model.electric_vehicle.availability[t]),
                "bus_trip_energy_MWh": pyomo_value(model.electric_vehicle.trip_energy_mwh[t]),
                "bus_charge_MWh": charge,
                "bus_discharge_MWh": discharge,
                "bus_soc_MWh": soc,
                "bus_soc_fraction": soc / building.electric_vehicle.total_capacity_mwh,
                "grid_import_MWh": grid_import,
                "grid_export_MWh": grid_export,
                "net_grid_import_MWh": grid_import - grid_export,
                "grid_import_MW": grid_import / dt_hours,
                "grid_export_MW": grid_export / dt_hours,
                "electricity_import_price_EUR_per_MWh": pyomo_value(model.import_price[t]),
                "electricity_export_price_EUR_per_MWh": pyomo_value(model.export_price[t]),
                "variable_cost_EUR": pyomo_value(model.variable_cost[t]),
                "v2g_enabled": building.v2g_enabled,
                "solver": context.solver_name,
            }
        )

    result = pd.DataFrame(rows).set_index("datetime")
    numeric_columns = result.select_dtypes(include=["number"]).columns
    result[numeric_columns] = result[numeric_columns].mask(
        result[numeric_columns].abs() < 1e-9,
        0.0,
    )
    return result


def _map_steel(
    plant: BasePlant,
    model: pyo.ConcreteModel,
    context: ResultContext,
) -> pd.DataFrame:
    steel_plant = _expect_plant_type(plant, SteelPlant)
    records = []
    for t, _timestamp in enumerate(context.forecasts.index):
        row = {
            "plant_name": steel_plant.name,
            "unit_type": steel_plant.unit_type,
            "steel_demand_t": pyomo_value(model.steel_demand[t]),
            "steel_output_t": pyomo_value(model.eaf.steel_output[t]),
            "dri_output_t": pyomo_value(model.dri_plant.dri_output[t]),
            "electricity_consumption_MWh": pyomo_value(model.total_power_input[t]),
            "natural_gas_consumption_MWh": pyomo_value(model.dri_plant.natural_gas_in[t]),
            "hydrogen_consumption_MWh": pyomo_value(model.dri_plant.hydrogen_in[t]),
            "variable_cost_EUR": pyomo_value(model.variable_cost[t]),
            "solver": context.solver_name,
        }
        if hasattr(model, "electrolyser"):
            row["hydrogen_production_MWh"] = pyomo_value(model.electrolyser.hydrogen_out[t])
        records.append(row)
    return pd.DataFrame(records, index=context.forecasts.index)


def _map_cement(
    plant: BasePlant,
    model: pyo.ConcreteModel,
    context: ResultContext,
) -> pd.DataFrame:
    cement_plant = _expect_plant_type(plant, CementPlant)
    records = []
    for t, _timestamp in enumerate(context.forecasts.index):
        records.append(
            {
                "plant_name": cement_plant.name,
                "unit_type": cement_plant.unit_type,
                "clinker_demand_t": pyomo_value(model.clinker_demand[t]),
                "clinker_output_t": pyomo_value(model.kiln.clinker_out[t]),
                "raw_meal_output_t": pyomo_value(model.preheater.raw_meal_out[t]),
                "electricity_consumption_MWh": pyomo_value(model.total_power_input[t]),
                "variable_cost_EUR": pyomo_value(model.variable_cost[t]),
                "solver": context.solver_name,
            }
        )
    return pd.DataFrame(records, index=context.forecasts.index)


def _map_steam(
    plant: BasePlant,
    model: pyo.ConcreteModel,
    context: ResultContext,
) -> pd.DataFrame:
    steam_plant = _expect_plant_type(plant, SteamGenerationPlant)
    if context.config is None or context.stage is None or context.market_input is None:
        raise ValueError("Steam result mapping requires config, stage, and market input")

    config = context.config
    stage = context.stage
    market_input = context.market_input
    capacity_award = context.capacity_award
    route = steam_plant.route_process
    direct_route = route.name == DIRECT_ELECTRIC_GAS_BOILER_ROUTE
    dt_hours = config.timestep_minutes / 60.0
    da_price_col = (
        market_input.electricity_price_col
        if isinstance(market_input, DayAheadPosition)
        else market_input.da_price_col
    )
    idc_price_col = (
        market_input.idc_price_col
        if isinstance(market_input, (IntradayAdjustment, BalancingEnergyActivation))
        else None
    )
    electricity_benchmark = getattr(
        market_input,
        "electricity_trading_benchmark_eur_per_mwh_el",
        None,
    )
    rows: list[dict[str, object]] = []

    for position, timestamp in enumerate(context.forecasts.index):
        physical = route.physical_result_fields(model, position)
        electricity = pyomo_value(model.electricity_consumption[position])
        da_position = pyomo_value(model.da_position_mwh[position])
        idc_buy = pyomo_value(model.idc_buy_mwh[position])
        idc_sell = pyomo_value(model.idc_sell_mwh[position])
        final_planned = pyomo_value(model.final_planned_electricity_mwh[position])
        actual_electricity = pyomo_value(model.actual_electricity_consumption_mwh[position])
        afrr_bid = pyomo_value(model.afrr_energy_bid_mwh[position])
        afrr_activation = pyomo_value(model.afrr_energy_activated_mwh[position])
        day_ahead_price = float(context.forecasts[da_price_col].iloc[position])
        idc_price = (
            series_float_or_nan(
                context.forecasts[idc_price_col],
                position,
            )
            if idc_price_col is not None
            else float("nan")
        )
        additional_charge = pyomo_value(model.additional_electricity_charge[position])
        afrr_price = (
            pyomo_value(model.afrr_energy_price[position])
            if stage == ElectricityMarketStage.AFRR_ENERGY
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
            if stage == ElectricityMarketStage.AFRR_ENERGY
            else float("nan")
        )
        afrr_bid_price = (
            float(
                series_value(
                    cast(BalancingEnergyActivation, market_input).afrr_energy_bid_price,
                    timestamp,
                    default_raw_bid_price,
                )
            )
            if stage == ElectricityMarketStage.AFRR_ENERGY
            else float("nan")
        )
        afrr_delivered_price = (afrr_price + additional_charge) * (1.0 + tax_rate)
        afrr_delivered_bid_price = (
            float(
                series_value(
                    cast(BalancingEnergyActivation, market_input).afrr_energy_delivered_bid_price,
                    timestamp,
                    (afrr_bid_price + additional_charge) * (1.0 + tax_rate),
                )
            )
            if stage == ElectricityMarketStage.AFRR_ENERGY
            else float("nan")
        )
        afrr_market_spread = (
            afrr_delivered_bid_price - afrr_price
            if stage == ElectricityMarketStage.AFRR_ENERGY
            else 0.0
        )
        afrr_net_spread = (
            benchmark - afrr_delivered_price if stage == ElectricityMarketStage.AFRR_ENERGY else 0.0
        )
        afrr_reward = afrr_activation * afrr_market_spread
        afrr_net_value = afrr_activation * afrr_net_spread
        co2_price = (
            float(context.forecasts[market_input.co2_price_col].iloc[position])
            if (
                market_input.co2_price_col
                and market_input.co2_price_col in context.forecasts.columns
            )
            else 0.0
        )
        activation = (
            cast(BalancingEnergyActivation, market_input)
            if stage == ElectricityMarketStage.AFRR_ENERGY
            else None
        )
        default_free_volume = afrr_bid if direct_route else 0.0
        free_bid = (
            float(
                series_value(
                    activation.afrr_energy_free_bid_mwh,
                    timestamp,
                    default_free_volume,
                )
            )
            if activation is not None
            else 0.0
        )
        free_activation = (
            float(
                series_value(
                    activation.afrr_energy_free_activated_mwh,
                    timestamp,
                    afrr_activation if direct_route else 0.0,
                )
            )
            if activation is not None
            else 0.0
        )
        capacity_backed_bid = (
            0.0
            if direct_route or activation is None
            else float(
                series_value(
                    activation.afrr_energy_capacity_backed_bid_mwh,
                    timestamp,
                    0.0,
                )
            )
        )
        capacity_backed_activation = (
            0.0
            if direct_route or activation is None
            else float(
                series_value(
                    activation.afrr_energy_capacity_backed_activated_mwh,
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
        capacity_market_fields = capacity_award_result_fields(
            capacity_award,
            timestamp,
            dt_hours,
        )

        row: dict[str, object] = {
            "datetime": timestamp,
            "plant_name": steam_plant.name,
            "heat_demand_MWh": (
                float(context.forecasts[steam_plant.heat_demand_column].iloc[position]) * dt_hours
            ),
            "day_ahead_price_EUR_per_MWh": day_ahead_price,
            "IDC_price_EUR_per_MWh": idc_price,
            "additional_electricity_charge_EUR_per_MWh_el": (additional_charge),
            "day_ahead_delivered_price_EUR_per_MWh": (
                (day_ahead_price + additional_charge) * (1.0 + tax_rate)
            ),
            "IDC_delivered_price_EUR_per_MWh": ((idc_price + additional_charge) * (1.0 + tax_rate)),
            "afrr_energy_delivered_price_EUR_per_MWh": (afrr_delivered_price),
            "gas_price_EUR_per_MWh": float(
                context.forecasts[market_input.gas_price_col].iloc[position]
            ),
            "co2_price_EUR_per_t": co2_price,
            "day_ahead_price_signal": da_price_col,
            "IDC_price_signal": idc_price_col or "",
            "gas_price_signal": market_input.gas_price_col,
            "co2_price_signal": market_input.co2_price_col or "",
            "gas_based_heat_benchmark_EUR_per_MWh_th": float(
                market_input.gas_benchmark_eur_per_mwh_th.iloc[position]
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
                float(activation.afrr_system_activation_mwh.iloc[position])
                if activation is not None
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
                        activation.afrr_headroom_binding,
                        timestamp,
                        False,
                    )
                )
                if activation is not None
                else False
            ),
            "afrr_curtailment_MWh": (
                float(
                    series_value(
                        activation.afrr_curtailment_mwh,
                        timestamp,
                        0.0,
                    )
                )
                if activation is not None
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
                market_input,
                position,
                direct_route,
            ),
            "idc_buy_allowed_by_strategy": (
                bool(
                    cast(
                        IntradayAdjustment,
                        market_input,
                    ).idc_buy_upper_bound_mwh.iloc[position]
                    > 1e-12
                )
                if stage == ElectricityMarketStage.INTRADAY
                else False
            ),
            "idc_sell_allowed_by_strategy": (
                bool(
                    cast(
                        IntradayAdjustment,
                        market_input,
                    ).idc_sell_upper_bound_mwh.iloc[position]
                    > 1e-12
                )
                if stage == ElectricityMarketStage.INTRADAY
                else False
            ),
            "afrr_energy_bid_allowed_by_strategy": bool(afrr_bid > 1e-12),
            "solver": context.solver_name,
        }
        row.update(capacity_market_fields)
        row.update(
            route.capacity_physical_result_fields(
                float(capacity_market_fields["afrr_capacity_reserved_MWh"]),
                dt_hours,
                final_planned,
                physical,
                steam_plant.components,
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


def _charge_allowed_flag(
    stage: ElectricityMarketStage,
    market_input: ElectricityMarketStageInput,
    position: int,
    direct_route: bool,
) -> bool:
    if stage == ElectricityMarketStage.DAY_AHEAD:
        return bool(cast(DayAheadPosition, market_input).charge_allowed.iloc[position])
    if stage == ElectricityMarketStage.INTRADAY and not direct_route:
        return bool(
            cast(
                IntradayAdjustment,
                market_input,
            ).idc_buy_upper_bound_mwh.iloc[position]
            > 1e-12
        )
    return False


def _preserve_legacy_stage_schema(
    row: dict[str, object],
    stage: ElectricityMarketStage,
    direct_route: bool,
) -> None:
    if stage != ElectricityMarketStage.AFRR_ENERGY:
        row.pop("afrr_energy_delivered_bid_price_EUR_per_MWh", None)
    if direct_route:
        return
    if stage == ElectricityMarketStage.DAY_AHEAD:
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
    elif stage == ElectricityMarketStage.INTRADAY:
        for column in (
            "afrr_energy_delivered_price_EUR_per_MWh",
            "afrr_energy_bid_allowed_by_strategy",
            "afrr_headroom_binding",
            "afrr_curtailment_MWh",
        ):
            row.pop(column, None)


def _expect_plant_type[PlantType: BasePlant](
    plant: BasePlant,
    expected_type: type[PlantType],
) -> PlantType:
    if not isinstance(plant, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__} for its registered result mapper, "
            f"received {type(plant).__name__}"
        )
    return plant


RESULT_MAPPERS: dict[type[BasePlant], Any] = {
    Building: _map_building,
    SteelPlant: _map_steel,
    CementPlant: _map_cement,
    SteamGenerationPlant: _map_steam,
}
