# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import math
import warnings

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_capacity import AFRRCapacityMarket
from flexi_mod.markets.afrr_energy import AFRRDownEnergyMarket
from flexi_mod.markets.day_ahead import DayAheadMarket
from flexi_mod.markets.intraday_continuous import IntradayContinuousMarket
from flexi_mod.plants.steam_generation_plant import (
    AFRRDownSignals,
    DispatchSignals,
    IDCAdjustmentSignals,
    SteamGenerationPlant,
)
from flexi_mod.strategies.base_strategy import BaseStrategy

GAS_PRICE_SIGNAL = "natural_gas_price"
ELECTRICITY_PRICE_SAFETY_MARGIN_EUR_PER_MWH = 0.0
# TODO: Move IDC_MARGIN_EUR_PER_MWH to config.yaml once multi-country cases
# or sensitivity analyses are implemented.
IDC_MARGIN_EUR_PER_MWH = 0.0
AFRR_ENERGY_BID_MARGIN_EUR_PER_MWH = 0.0
AFRR_CAPACITY_MARGIN_EUR_PER_MW_H = 0.0


class HybridETESGasStrategy(BaseStrategy):
    """Operator strategy for electricity procurement and plant operation.

    The configured market classes describe market products and prepare market
    inputs. This strategy decides how the industrial operator acts on those
    inputs using the gas benchmark, margins and plant flexibility.
    """

    def __init__(self, config: CaseConfig):
        self.config = config
        self.afrr_energy_data_quality_summary = pd.DataFrame()
        self.afrr_capacity_block_summary = pd.DataFrame()
        self._afrr_down_energy_data_cache = {}

    def required_forecast_columns(self) -> set[str]:
        required = {GAS_PRICE_SIGNAL}
        if "intraday_continuous" in self.config.enabled_markets:
            required.add(self.config.market_signal("intraday_continuous", "price"))
        if "afrr_energy" in self.config.enabled_markets:
            required.add(self.config.market_signal("afrr_energy", "price"))
            required.add(self.config.market_signal("afrr_energy", "system_activation"))
        return required

    def decide_day_ahead(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        market = DayAheadMarket("day_ahead", self.config.market("day_ahead"))
        market_data = market.prepare_market_data(forecasts)
        price_col = market.signal_column("price")

        benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        delivered_da_price = self._delivered_electricity_price(
            plant,
            market_data["day_ahead_price_EUR_per_MWh"],
        )
        charge_allowed = self._calculate_charge_gate(
            plant=plant,
            electricity_price=delivered_da_price,
            benchmark=benchmark,
        )
        charge_allowed = charge_allowed & ~self._grid_charging_block(plant, forecasts)

        signals = DispatchSignals(
            electricity_price_col=price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            gas_benchmark_eur_per_mwh_th=benchmark,
            charge_allowed=charge_allowed,
            additional_electricity_charge_eur_per_mwh=(
                plant.additional_electricity_charge_eur_per_mwh
            ),
            **_capacity_signal_kwargs(capacity_reservation, forecasts.index),
        )
        if rolling:
            return plant.solve_rolling(
                self.config,
                forecasts,
                signals,
                initial_soc_mwh=initial_soc_mwh,
            )
        return plant.solve_horizon(
            self.config,
            forecasts,
            signals,
            initial_soc_mwh=initial_soc_mwh,
        )

    def decide_intraday_continuous(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        idc_market = IntradayContinuousMarket(
            "intraday_continuous",
            self.config.market("intraday_continuous"),
        )
        idc_data = idc_market.prepare_market_data(forecasts)
        idc_price_col = idc_market.signal_column("price")
        da_price_col = self.config.market_signal("day_ahead", "price")
        if da_price_col not in forecasts.columns:
            forecasts = forecasts.copy()
            forecasts[da_price_col] = 0.0

        da_position = self._fixed_da_position(fixed_positions, forecasts.index)
        idc_price = idc_data["IDC_price_EUR_per_MWh"]
        delivered_idc_price = self._delivered_electricity_price(plant, idc_price)
        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_heat_benchmark,
        )

        missing_price = ~idc_data["IDC_price_available"]
        if missing_price.any():
            warnings.warn(
                "IDC price contains missing values. IDC action is set to zero for "
                f"{int(missing_price.sum())} timestep(s).",
                stacklevel=2,
            )

        grid_block = self._grid_charging_block(plant, forecasts)
        buy_allowed = (
            (delivered_idc_price < (electricity_benchmark - IDC_MARGIN_EUR_PER_MWH))
            & ~missing_price
            & ~grid_block
        )
        sell_allowed = (
            delivered_idc_price > (electricity_benchmark + IDC_MARGIN_EUR_PER_MWH)
        ) & ~missing_price
        if not idc_market.buy_enabled:
            buy_allowed = pd.Series(False, index=forecasts.index)
        if not idc_market.sell_enabled:
            sell_allowed = pd.Series(False, index=forecasts.index)

        timestep_hours = self.config.timestep_minutes / 60.0
        max_charge_mwh = plant.etes.max_power_charge_mw * timestep_hours
        idc_buy_upper_bound = pd.Series(0.0, index=forecasts.index)
        idc_sell_upper_bound = pd.Series(0.0, index=forecasts.index)
        idc_buy_upper_bound.loc[buy_allowed] = (max_charge_mwh - da_position.loc[buy_allowed]).clip(
            lower=0.0
        )
        idc_sell_upper_bound.loc[sell_allowed] = da_position.loc[sell_allowed].clip(lower=0.0)

        signals = IDCAdjustmentSignals(
            da_price_col=da_price_col,
            idc_price_col=idc_price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            da_position_mwh=da_position,
            idc_buy_upper_bound_mwh=idc_buy_upper_bound,
            idc_sell_upper_bound_mwh=idc_sell_upper_bound,
            gas_benchmark_eur_per_mwh_th=gas_heat_benchmark,
            electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
            additional_electricity_charge_eur_per_mwh=(
                plant.additional_electricity_charge_eur_per_mwh
            ),
            **_capacity_signal_kwargs(capacity_reservation, forecasts.index),
        )
        if rolling:
            return plant.solve_intraday_adjustment_rolling(
                self.config,
                forecasts,
                signals,
                initial_soc_mwh=initial_soc_mwh,
            )
        return plant.solve_intraday_adjustment_horizon(
            self.config,
            forecasts,
            signals,
            initial_soc_mwh=initial_soc_mwh,
        )

    def decide_afrr_energy(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        da_price_col = self.config.market_signal("day_ahead", "price")
        if da_price_col not in forecasts.columns:
            forecasts = forecasts.copy()
            forecasts[da_price_col] = 0.0
        idc_price_col = self.config.market_signal("intraday_continuous", "price")
        if idc_price_col not in forecasts.columns:
            forecasts = forecasts.copy()
            forecasts[idc_price_col] = 0.0
        timestep_hours = self.config.timestep_minutes / 60.0
        afrr_market = AFRRDownEnergyMarket("afrr_energy", self.config.market("afrr_energy"))
        product_rules = afrr_market.product_rules
        min_bid_mw = float(product_rules.get("min_bid_mw", 0.0))

        cleaned = self._prepare_afrr_down_energy_data(forecasts, timestep_hours)
        clean_afrr = cleaned.frame

        da_position = self._series_from_fixed_positions(
            fixed_positions,
            "DA_position_MWh",
            forecasts.index,
        )
        idc_buy = self._series_from_fixed_positions(
            fixed_positions,
            "IDC_buy_MWh",
            forecasts.index,
            default=0.0,
        )
        idc_sell = self._series_from_fixed_positions(
            fixed_positions,
            "IDC_sell_MWh",
            forecasts.index,
            default=0.0,
        )
        final_planned = self._series_from_fixed_positions(
            fixed_positions,
            "final_planned_electricity_MWh",
            forecasts.index,
            default=None,
        )
        if final_planned is None:
            final_planned = da_position + idc_buy - idc_sell

        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_heat_benchmark,
        )

        reserved_capacity = _capacity_column(
            capacity_reservation,
            forecasts.index,
            "afrr_capacity_reserved_MWh",
        )
        baseline_storage_soc = self._series_from_fixed_positions(
            fixed_positions,
            "etes_soc_MWh",
            forecasts.index,
            default=plant.etes.initial_soc_mwh,
        )
        baseline_gas_heat = self._series_from_fixed_positions(
            fixed_positions,
            "gas_heat_MWh",
            forecasts.index,
            default=0.0,
        )
        storage_capacity_headroom = (plant.etes.max_capacity_mwh - baseline_storage_soc).clip(
            lower=0.0
        ) / plant.etes.efficiency_charge

        valid_price = clean_afrr["afrr_price_available"]
        afrr_energy_bid_price = electricity_benchmark + AFRR_ENERGY_BID_MARGIN_EUR_PER_MWH
        delivered_afrr_price = self._delivered_electricity_price(
            plant,
            clean_afrr["afrr_energy_down_price_EUR_per_MWh"],
        )
        # aFRR energy is offered only when the deal is profitable for the plant:
        # the market clearing price plus industrial electricity charges must stay
        # below the benchmark bid price derived from gas-based heat value.
        price_allowed = (
            (delivered_afrr_price <= afrr_energy_bid_price)
            & valid_price
            & ~self._grid_charging_block(plant, forecasts)
        )
        max_charge_mwh = plant.etes.max_power_charge_mw * timestep_hours
        charge_power_headroom_after_reserve = (
            max_charge_mwh - final_planned - reserved_capacity
        ).clip(lower=0.0)
        storage_headroom_after_reserve = (storage_capacity_headroom - reserved_capacity).clip(
            lower=0.0
        )
        free_bid_potential = pd.concat(
            [charge_power_headroom_after_reserve, storage_headroom_after_reserve],
            axis=1,
        ).min(axis=1)
        free_bid_potential = free_bid_potential.where(price_allowed, 0.0).clip(lower=0.0)
        remaining_headroom_mw = free_bid_potential / timestep_hours
        free_bid_upper_bound = free_bid_potential.where(remaining_headroom_mw >= min_bid_mw, 0.0)
        free_bid_upper_bound = free_bid_upper_bound.clip(lower=0.0)
        # TODO: Enforce strict bid increments using integer or discretised variables
        # if exact market-compliant bid granularity is required later.

        if capacity_reservation is not None and not capacity_reservation.empty:
            capacity_backed_bid = reserved_capacity.clip(lower=0.0)
            bid_upper_bound, activated, split = self._strict_afrr_down_offer_and_activation_split(
                plant=plant,
                forecasts=forecasts,
                final_planned=final_planned,
                capacity_backed_bid=capacity_backed_bid,
                free_bid_upper_bound=free_bid_upper_bound,
                system_activation_mwh=clean_afrr["afrr_system_activation_MWh"],
                baseline_storage_soc=baseline_storage_soc,
                baseline_gas_heat=baseline_gas_heat,
                min_bid_mw=min_bid_mw,
                timestep_hours=timestep_hours,
            )
        else:
            zero_capacity = pd.Series(0.0, index=forecasts.index)
            bid_upper_bound, activated, split = self._strict_afrr_down_offer_and_activation_split(
                plant=plant,
                forecasts=forecasts,
                final_planned=final_planned,
                capacity_backed_bid=zero_capacity,
                free_bid_upper_bound=free_bid_upper_bound,
                system_activation_mwh=clean_afrr["afrr_system_activation_MWh"],
                baseline_storage_soc=baseline_storage_soc,
                baseline_gas_heat=baseline_gas_heat,
                min_bid_mw=min_bid_mw,
                timestep_hours=timestep_hours,
            )
        useful_heat_cap_binding = split["useful_heat_cap_binding"]
        curtailed_activation = split["curtailed_proxy_activation_due_to_heat_cap_MWh"]

        signals = AFRRDownSignals(
            da_price_col=da_price_col,
            idc_price_col=idc_price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            da_position_mwh=da_position,
            idc_buy_mwh=idc_buy,
            idc_sell_mwh=idc_sell,
            final_planned_electricity_mwh=final_planned,
            afrr_energy_price=clean_afrr["afrr_energy_down_price_EUR_per_MWh"],
            afrr_system_activation_mwh=clean_afrr["afrr_system_activation_MWh"],
            afrr_energy_bid_mwh=bid_upper_bound,
            afrr_energy_activated_mwh=activated,
            afrr_energy_bid_price=afrr_energy_bid_price,
            afrr_energy_capacity_backed_bid_mwh=split["afrr_energy_capacity_backed_bid_MWh"],
            afrr_energy_free_bid_mwh=split["afrr_energy_free_bid_MWh"],
            afrr_energy_capacity_backed_activated_mwh=split[
                "afrr_energy_capacity_backed_activated_MWh"
            ],
            afrr_energy_free_activated_mwh=split["afrr_energy_free_activated_MWh"],
            useful_heat_cap_binding=useful_heat_cap_binding,
            curtailed_proxy_activation_due_to_heat_cap_mwh=curtailed_activation,
            gas_benchmark_eur_per_mwh_th=gas_heat_benchmark,
            electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
            additional_electricity_charge_eur_per_mwh=(
                plant.additional_electricity_charge_eur_per_mwh
            ),
            **_capacity_signal_kwargs(capacity_reservation, forecasts.index),
        )
        if rolling:
            return plant.solve_afrr_down_rolling(
                self.config,
                forecasts,
                signals,
                initial_soc_mwh=initial_soc_mwh,
            )
        return plant.solve_afrr_down_horizon(
            self.config,
            forecasts,
            signals,
            initial_soc_mwh=initial_soc_mwh,
        )

    def _strict_afrr_down_offer_and_activation_split(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        final_planned: pd.Series,
        capacity_backed_bid: pd.Series,
        free_bid_upper_bound: pd.Series,
        system_activation_mwh: pd.Series,
        baseline_storage_soc: pd.Series,
        baseline_gas_heat: pd.Series,
        min_bid_mw: float,
        timestep_hours: float,
    ) -> tuple[pd.Series, pd.Series, dict[str, pd.Series]]:
        """Limit aFRR down bids and activation to useful heat trajectories.

        Capacity-backed bid volume is the mandatory energy bid behind awarded
        capacity. Free bid volume is optional and enters only after the strategy
        has found the aFRR energy deal profitable. Activation is allocated to
        capacity-backed volume first, then to optional free volume.
        """

        capacity_bid_values: list[float] = []
        free_bid_values: list[float] = []
        total_bid_values: list[float] = []
        capacity_activated_values: list[float] = []
        free_activated_values: list[float] = []
        total_activated_values: list[float] = []
        binding_values: list[bool] = []
        curtailed_values: list[float] = []
        max_charge_mwh = plant.etes.max_power_charge_mw * timestep_hours

        for timestamp in forecasts.index:
            planned_charge = max(0.0, float(final_planned.loc[timestamp]))
            useful_extra_heat_outlet = max(0.0, float(baseline_gas_heat.loc[timestamp]))
            baseline_soc = max(0.0, float(baseline_storage_soc.loc[timestamp]))
            storage_capacity_offer = max(0.0, plant.etes.max_capacity_mwh - baseline_soc) / (
                plant.etes.efficiency_charge
            )
            power_offer = max_charge_mwh - planned_charge
            immediate_use_offer = useful_extra_heat_outlet / (
                plant.etes.efficiency_charge * plant.etes.efficiency_discharge
            )
            physical_activation_cap = max(
                0.0,
                min(power_offer, storage_capacity_offer, immediate_use_offer),
            )

            capacity_bid = max(0.0, float(capacity_backed_bid.loc[timestamp]))
            free_bid = max(0.0, float(free_bid_upper_bound.loc[timestamp]))
            free_room_after_capacity = max(0.0, physical_activation_cap - capacity_bid)
            free_bid = min(free_bid, free_room_after_capacity)
            if timestep_hours <= 0 or free_bid / timestep_hours < min_bid_mw:
                free_bid = 0.0

            total_bid = capacity_bid + free_bid
            system_activation = max(0.0, float(system_activation_mwh.loc[timestamp]))
            proxy_activation = min(total_bid, system_activation)
            feasible_activation = min(proxy_activation, physical_activation_cap)
            capacity_activated = min(capacity_bid, feasible_activation)
            free_activated = min(free_bid, max(0.0, feasible_activation - capacity_activated))
            total_activated = capacity_activated + free_activated
            curtailment = max(0.0, proxy_activation - total_activated)

            capacity_bid_values.append(capacity_bid)
            free_bid_values.append(free_bid)
            total_bid_values.append(total_bid)
            capacity_activated_values.append(capacity_activated)
            free_activated_values.append(free_activated)
            total_activated_values.append(total_activated)
            binding_values.append(
                proxy_activation > 1e-12 and physical_activation_cap < proxy_activation - 1e-12
            )
            curtailed_values.append(curtailment)

        split = {
            "afrr_energy_capacity_backed_bid_MWh": pd.Series(
                capacity_bid_values, index=forecasts.index
            ),
            "afrr_energy_free_bid_MWh": pd.Series(free_bid_values, index=forecasts.index),
            "afrr_energy_capacity_backed_activated_MWh": pd.Series(
                capacity_activated_values, index=forecasts.index
            ),
            "afrr_energy_free_activated_MWh": pd.Series(
                free_activated_values, index=forecasts.index
            ),
            "useful_heat_cap_binding": pd.Series(binding_values, index=forecasts.index),
            "curtailed_proxy_activation_due_to_heat_cap_MWh": pd.Series(
                curtailed_values, index=forecasts.index
            ),
        }
        return (
            pd.Series(total_bid_values, index=forecasts.index, name="afrr_energy_bid_MWh"),
            pd.Series(
                total_activated_values,
                index=forecasts.index,
                name="afrr_energy_activated_MWh",
            ),
            split,
        )

    def decide_afrr_capacity(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        capacity_market = AFRRCapacityMarket(
            "afrr_capacity",
            self.config.market("afrr_capacity"),
        )
        timestep_hours = self.config.timestep_minutes / 60.0
        capacity_data = capacity_market.prepare_market_data(
            forecasts,
            timestep_hours=timestep_hours,
        )
        capacity_frame = capacity_data.frame.copy()
        block_summary = capacity_data.block_summary.copy()

        da_price_col = self.config.market_signal("day_ahead", "price")
        afrr_energy_enabled = "afrr_energy" in self.config.enabled_markets
        if afrr_energy_enabled:
            cleaned_afrr_energy = self._prepare_afrr_down_energy_data(forecasts, timestep_hours)
            afrr_energy = cleaned_afrr_energy.frame
        else:
            afrr_energy = pd.DataFrame(
                {
                    "afrr_energy_down_price_EUR_per_MWh": 0.0,
                    "afrr_price_available": False,
                    "afrr_system_activation_MWh": 0.0,
                    "afrr_activation_without_price": False,
                },
                index=forecasts.index,
            )
        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_heat_benchmark,
        )
        reference_price = self._delivered_electricity_price(
            plant,
            forecasts[da_price_col].astype(float),
        )
        opportunity_cost = (electricity_benchmark - reference_price).clip(lower=0.0)
        afrr_energy_bid_price = electricity_benchmark + AFRR_ENERGY_BID_MARGIN_EUR_PER_MWH
        delivered_afrr_energy_price = self._delivered_electricity_price(
            plant,
            afrr_energy["afrr_energy_down_price_EUR_per_MWh"],
        )
        activation_relevant = (afrr_energy["afrr_system_activation_MWh"] > 1e-12) | afrr_energy[
            "afrr_activation_without_price"
        ].astype(bool)
        activation_price_allowed = afrr_energy["afrr_price_available"].astype(bool) & (
            delivered_afrr_energy_price <= afrr_energy_bid_price
        )

        max_charge_power_mw = plant.etes.max_power_charge_mw
        min_bid_mw = float(capacity_market.product_rules.get("min_bid_mw", 0.0))
        bid_increment_mw = float(capacity_market.product_rules.get("bid_increment_mw", 1.0))
        if bid_increment_mw <= 0:
            raise ValueError("afrr_capacity.product_rules.bid_increment_mw must be positive")
        expected_soc = plant.etes.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
        # Under atypical grid use, do not commit aFRR-down capacity in blocks that overlap a
        # high-load window: a mandatory capacity-backed activation there would raise the billed
        # window peak and forfeit the §19(2) capacity-charge saving.
        grid_block = self._grid_charging_block(plant, forecasts)
        records = []
        for _, block in block_summary.iterrows():
            block_id = str(block["block_id"])
            mask = capacity_frame["afrr_capacity_block_id"] == block_id
            block_duration_h = float(block["block_duration_h"])
            opportunity_cost_block = float(
                (opportunity_cost.loc[mask] * timestep_hours).sum() / block_duration_h
            )
            block_relevant = activation_relevant.loc[mask]
            block_price_allowed = activation_price_allowed.loc[mask]
            block_without_price = afrr_energy["afrr_activation_without_price"].loc[mask]
            relevant_timesteps = int(block_relevant.sum())
            activation_without_price_timesteps = int(block_without_price.sum())
            price_failed_timesteps = int((block_relevant & ~block_price_allowed).sum())
            activation_safety_allowed = price_failed_timesteps == 0
            relevant_with_price = block_relevant & afrr_energy["afrr_price_available"].loc[mask]
            if relevant_with_price.any():
                activation_price_margin = (
                    afrr_energy_bid_price.loc[mask] - delivered_afrr_energy_price.loc[mask]
                )
                min_activation_price_margin = float(
                    activation_price_margin.loc[relevant_with_price].min()
                )
            else:
                min_activation_price_margin = float("nan")
            storage_capacity_mw = max(
                0.0,
                (plant.etes.max_capacity_mwh - expected_soc)
                / (plant.etes.efficiency_charge * block_duration_h),
            )
            max_activation_need_mwh = float(
                afrr_energy["afrr_system_activation_MWh"].loc[mask].max()
            )
            max_activation_need_mw = (
                max_activation_need_mwh / timestep_hours if timestep_hours > 0 else 0.0
            )
            technical_capacity = min(max_charge_power_mw, storage_capacity_mw)
            raw_feasible_capacity = min(technical_capacity, max_activation_need_mw)
            market_compliant_capacity = _round_capacity_down_to_increment(
                raw_feasible_capacity,
                min_bid_mw=min_bid_mw,
                bid_increment_mw=bid_increment_mw,
            )
            activation_need_cap_binding = max_activation_need_mw + 1e-12 < technical_capacity
            capacity_limited_by_activation_need = max(
                0.0,
                technical_capacity - raw_feasible_capacity,
            )
            capacity_profitability_allowed = (
                not bool(block["missing_capacity_price_flag"])
                and float(block["capacity_price_EUR_per_MW_h"])
                >= opportunity_cost_block + AFRR_CAPACITY_MARGIN_EUR_PER_MW_H
            )
            physical_feasibility_allowed = market_compliant_capacity >= min_bid_mw
            block_overlaps_high_load_window = bool(grid_block.loc[mask].any())
            economic_allowed = (
                capacity_profitability_allowed
                and activation_safety_allowed
                and physical_feasibility_allowed
                and not block_overlaps_high_load_window
            )
            if not economic_allowed:
                reserved_mw = 0.0
            else:
                reserved_mw = market_compliant_capacity
            revenue = reserved_mw * float(block["capacity_price_EUR_per_MW_h"]) * block_duration_h
            records.append(
                {
                    **block.to_dict(),
                    "opportunity_cost_EUR_per_MW_h": opportunity_cost_block,
                    "capacity_profitability_check_passed": bool(capacity_profitability_allowed),
                    "activation_safety_check_passed": bool(activation_safety_allowed),
                    "activation_relevant_timesteps": relevant_timesteps,
                    "activation_price_check_failed_timesteps": price_failed_timesteps,
                    "activation_without_price_timesteps": activation_without_price_timesteps,
                    "min_activation_price_margin_EUR_per_MWh": min_activation_price_margin,
                    "physical_feasibility_check_passed": bool(physical_feasibility_allowed),
                    "economic_bid_allowed": bool(economic_allowed),
                    "max_afrr_down_activation_need_MWh_in_block": max_activation_need_mwh,
                    "max_afrr_down_activation_need_MW_in_block": max_activation_need_mw,
                    "raw_feasible_capacity_potential_MW": raw_feasible_capacity,
                    "market_compliant_capacity_potential_MW": market_compliant_capacity,
                    "activation_need_cap_binding": bool(activation_need_cap_binding),
                    "capacity_limited_by_activation_need_MW": capacity_limited_by_activation_need,
                    "bid_increment_mw": bid_increment_mw,
                    "feasible_capacity_potential_MW": market_compliant_capacity,
                    "reserved_capacity_MW": reserved_mw,
                    "capacity_revenue_EUR": revenue,
                    "min_final_planned_headroom_MW": reserved_mw,
                    "min_storage_headroom_MW": storage_capacity_mw,
                    "total_afrr_energy_activated_MWh_in_block": 0.0,
                    "total_afrr_energy_cost_EUR_in_block": 0.0,
                }
            )

        self.afrr_capacity_block_summary = pd.DataFrame(records)
        enriched = capacity_frame.join(
            self.afrr_capacity_block_summary.set_index("block_id")[
                [
                    "opportunity_cost_EUR_per_MW_h",
                    "economic_bid_allowed",
                    "feasible_capacity_potential_MW",
                    "reserved_capacity_MW",
                    "capacity_revenue_EUR",
                ]
            ],
            on="afrr_capacity_block_id",
        )
        enriched["afrr_capacity_reserved_MW"] = enriched["reserved_capacity_MW"].fillna(0.0)
        enriched["afrr_capacity_reserved_MWh"] = (
            enriched["afrr_capacity_reserved_MW"] * timestep_hours
        )
        enriched["afrr_capacity_revenue_EUR"] = (
            enriched["afrr_capacity_reserved_MW"]
            * enriched["capacity_price_EUR_per_MW_h"]
            * timestep_hours
        )
        return enriched

    def _prepare_afrr_down_energy_data(
        self,
        forecasts: pd.DataFrame,
        timestep_hours: float,
    ):
        cache_key = (id(forecasts), timestep_hours)
        if cache_key not in self._afrr_down_energy_data_cache:
            afrr_energy_market = AFRRDownEnergyMarket(
                "afrr_energy",
                self.config.market("afrr_energy"),
            )
            self._afrr_down_energy_data_cache[cache_key] = afrr_energy_market.prepare_market_data(
                forecasts,
                timestep_hours=timestep_hours,
            )
        cleaned = self._afrr_down_energy_data_cache[cache_key]
        self.afrr_energy_data_quality_summary = cleaned.quality_summary
        return cleaned

    def calculate_gas_based_heat_cost(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
    ) -> pd.Series:
        if plant.gas_boiler is None:
            raise ValueError(f"Plant '{plant.name}' has no gas boiler")

        gas_input_per_mwh_heat = 1.0 / plant.gas_boiler.efficiency
        benchmark = forecasts[GAS_PRICE_SIGNAL].astype(float) * gas_input_per_mwh_heat
        benchmark.name = "gas_based_heat_benchmark_EUR_per_MWh_th"
        # TODO: Add CO2 cost to this benchmark when CO2 is enabled in gas cost accounting.
        return benchmark

    def calculate_electricity_trading_benchmark(
        self,
        plant: SteamGenerationPlant,
        gas_heat_benchmark: pd.Series,
    ) -> pd.Series:
        if plant.etes is None:
            raise ValueError(f"Plant '{plant.name}' has no ETES component")

        delivered_heat_per_mwh_electric = (
            plant.etes.efficiency_charge * plant.etes.efficiency_discharge
        )
        if delivered_heat_per_mwh_electric <= 0:
            raise ValueError("ETES charge/discharge efficiencies must be positive")

        benchmark = gas_heat_benchmark.astype(float) * delivered_heat_per_mwh_electric
        benchmark.name = "electricity_trading_benchmark_EUR_per_MWh_el"
        return benchmark

    @staticmethod
    def _delivered_electricity_price(
        plant: SteamGenerationPlant,
        market_price: pd.Series,
    ) -> pd.Series:
        """Return market electricity price plus plant consumption charges."""

        return market_price.astype(float) + float(plant.additional_electricity_charge_eur_per_mwh)

    @staticmethod
    def _grid_charging_block(
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
    ) -> pd.Series:
        """Per-timestep mask, True where the grid-fee regulation blocks grid-charging.

        Under atypical grid use (§19(2) StromNEV) the plant avoids drawing grid
        power during DSO high-load windows to keep its billed capacity peak low.
        """

        regulation = getattr(plant, "grid_fee_regulation", None)
        if regulation is None:
            return pd.Series(False, index=forecasts.index)
        mask = regulation.charging_block_mask(forecasts)
        return mask.reindex(forecasts.index).fillna(False).astype(bool)

    def _calculate_charge_gate(
        self,
        plant: SteamGenerationPlant,
        electricity_price: pd.Series,
        benchmark: pd.Series,
    ) -> pd.Series:
        if plant.etes is None:
            raise ValueError(f"Plant '{plant.name}' has no ETES component")

        safety_margin = ELECTRICITY_PRICE_SAFETY_MARGIN_EUR_PER_MWH
        delivered_heat_per_mwh_electric = (
            plant.etes.efficiency_charge * plant.etes.efficiency_discharge
        )
        if delivered_heat_per_mwh_electric <= 0:
            raise ValueError("ETES charge/discharge efficiencies must be positive")

        effective_electric_heat_cost = (
            electricity_price.astype(float) / delivered_heat_per_mwh_electric
        )
        return effective_electric_heat_cost <= (benchmark.astype(float) - safety_margin)

    @staticmethod
    def _fixed_da_position(
        fixed_positions: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        if "DA_position_MWh" in fixed_positions.columns:
            da_position = fixed_positions["DA_position_MWh"]
        elif "electricity_consumption_MWh" in fixed_positions.columns:
            da_position = fixed_positions["electricity_consumption_MWh"]
        else:
            raise ValueError(
                "IDC stage requires fixed day-ahead positions, but neither "
                "'DA_position_MWh' nor 'electricity_consumption_MWh' was found."
            )

        da_position = da_position.astype(float).reindex(index)
        if da_position.isna().any():
            raise ValueError("Fixed day-ahead positions are not aligned with forecast timestamps")
        return da_position.clip(lower=0.0)

    @staticmethod
    def _series_from_fixed_positions(
        fixed_positions: pd.DataFrame,
        column: str,
        index: pd.DatetimeIndex,
        default: float | None = 0.0,
    ) -> pd.Series | None:
        if column in fixed_positions.columns:
            series = fixed_positions[column].astype(float).reindex(index)
        elif default is None:
            return None
        else:
            series = pd.Series(float(default), index=index)
        if series.isna().any():
            raise ValueError(f"Fixed position column '{column}' is not aligned with forecasts")
        return series


def _capacity_signal_kwargs(
    capacity_reservation: pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    if capacity_reservation is None or capacity_reservation.empty:
        return {}

    frame = capacity_reservation.reindex(index)
    return {
        "reserved_capacity_mwh": _capacity_column(frame, index, "afrr_capacity_reserved_MWh"),
        "afrr_capacity_block_id": _capacity_object_column(
            frame,
            index,
            "afrr_capacity_block_id",
            "",
        ),
        "afrr_capacity_block_duration_h": _capacity_column(frame, index, "block_duration_h"),
        "afrr_capacity_price_eur_per_mw_h": _capacity_column(
            frame,
            index,
            "capacity_price_EUR_per_MW_h",
        ),
        "afrr_capacity_reserved_mw": _capacity_column(frame, index, "afrr_capacity_reserved_MW"),
        "afrr_capacity_revenue_eur": _capacity_column(frame, index, "afrr_capacity_revenue_EUR"),
    }


def _round_capacity_down_to_increment(
    capacity_mw: float,
    min_bid_mw: float,
    bid_increment_mw: float,
) -> float:
    """Return the largest market-compliant bid not exceeding available capacity."""

    if capacity_mw < min_bid_mw:
        return 0.0
    rounded = math.floor((capacity_mw + 1e-12) / bid_increment_mw) * bid_increment_mw
    if rounded < min_bid_mw:
        return 0.0
    return float(rounded)


def _capacity_column(
    capacity_reservation: pd.DataFrame | None,
    index: pd.DatetimeIndex,
    column: str,
) -> pd.Series:
    if (
        capacity_reservation is None
        or capacity_reservation.empty
        or column not in capacity_reservation
    ):
        return pd.Series(0.0, index=index)
    return capacity_reservation[column].astype(float).reindex(index).fillna(0.0)


def _capacity_object_column(
    capacity_reservation: pd.DataFrame,
    index: pd.DatetimeIndex,
    column: str,
    default: object,
) -> pd.Series:
    if column not in capacity_reservation:
        return pd.Series(default, index=index)
    return capacity_reservation[column].reindex(index).fillna(default)
