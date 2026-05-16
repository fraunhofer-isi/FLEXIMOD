# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

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
IDC_MARGIN_EUR_PER_MWH = 5.0
AFRR_ENERGY_MARGIN_EUR_PER_MWH = 5.0
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
    ) -> pd.DataFrame:
        market = DayAheadMarket("day_ahead", self.config.market("day_ahead"))
        market_data = market.prepare_market_data(forecasts)
        price_col = market.signal_column("price")

        benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        charge_allowed = self._calculate_charge_gate(
            plant=plant,
            electricity_price=market_data["day_ahead_price_EUR_per_MWh"],
            benchmark=benchmark,
        )

        signals = DispatchSignals(
            electricity_price_col=price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            gas_benchmark_eur_per_mwh_th=benchmark,
            charge_allowed=charge_allowed,
            **_capacity_signal_kwargs(capacity_reservation, forecasts.index),
        )
        return plant.solve_rolling(self.config, forecasts, signals)

    def decide_intraday_continuous(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        idc_market = IntradayContinuousMarket(
            "intraday_continuous",
            self.config.market("intraday_continuous"),
        )
        idc_data = idc_market.prepare_market_data(forecasts)
        idc_price_col = idc_market.signal_column("price")
        da_price_col = self.config.market_signal("day_ahead", "price")

        da_position = self._fixed_da_position(fixed_positions, forecasts.index)
        idc_price = idc_data["IDC_price_EUR_per_MWh"]
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

        buy_allowed = (
            idc_price < (electricity_benchmark - IDC_MARGIN_EUR_PER_MWH)
        ) & ~missing_price
        sell_allowed = (
            idc_price > (electricity_benchmark + IDC_MARGIN_EUR_PER_MWH)
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
            **_capacity_signal_kwargs(capacity_reservation, forecasts.index),
        )
        return plant.solve_intraday_adjustment_rolling(self.config, forecasts, signals)

    def decide_afrr_energy(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        da_price_col = self.config.market_signal("day_ahead", "price")
        idc_price_col = self.config.market_signal("intraday_continuous", "price")
        if idc_price_col not in forecasts.columns:
            forecasts = forecasts.copy()
            forecasts[idc_price_col] = 0.0
        timestep_hours = self.config.timestep_minutes / 60.0
        afrr_market = AFRRDownEnergyMarket("afrr_energy", self.config.market("afrr_energy"))
        product_rules = afrr_market.product_rules
        min_bid_mw = float(product_rules.get("min_bid_mw", 0.0))

        cleaned = afrr_market.prepare_market_data(forecasts, timestep_hours=timestep_hours)
        self.afrr_energy_data_quality_summary = cleaned.quality_summary
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

        max_charge_mwh = plant.etes.max_power_charge_mw * timestep_hours
        reserved_capacity = _capacity_column(
            capacity_reservation,
            forecasts.index,
            "afrr_capacity_reserved_MWh",
        )
        charge_power_headroom = (max_charge_mwh - final_planned - reserved_capacity).clip(lower=0.0)
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
        feasible_bid_potential = pd.concat(
            [charge_power_headroom, storage_capacity_headroom],
            axis=1,
        ).min(axis=1)

        valid_price = clean_afrr["afrr_price_available"]
        price_allowed = (
            clean_afrr["afrr_energy_down_price_EUR_per_MWh"]
            <= electricity_benchmark - AFRR_ENERGY_MARGIN_EUR_PER_MWH
        ) & valid_price
        feasible_bid_potential = feasible_bid_potential.where(price_allowed, 0.0)
        remaining_headroom_mw = feasible_bid_potential / timestep_hours
        bid_upper_bound = feasible_bid_potential.where(remaining_headroom_mw >= min_bid_mw, 0.0)
        bid_upper_bound = bid_upper_bound.clip(lower=0.0)
        # TODO: Enforce strict bid increments using integer or discretised variables
        # if exact market-compliant bid granularity is required later.

        if capacity_reservation is not None and not capacity_reservation.empty:
            capacity_backed_bid = reserved_capacity.clip(lower=0.0)
            # TODO: If aFRR capacity is awarded, the plant must submit energy bids
            # at least equal to the awarded capacity volume. Later, allow optional
            # extra aFRR energy bids above this mandatory capacity-backed volume
            # when additional ETES headroom and attractive energy prices are available.
            bid_upper_bound = capacity_backed_bid
            activated = self._strict_capacity_backed_afrr_activation(
                plant=plant,
                forecasts=forecasts,
                final_planned=final_planned,
                capacity_backed_bid=capacity_backed_bid,
                system_activation_mwh=clean_afrr["afrr_system_activation_MWh"],
                baseline_storage_soc=baseline_storage_soc,
                baseline_gas_heat=baseline_gas_heat,
            )
        else:
            bid_upper_bound, activated = self._strict_afrr_down_offer_and_activation(
                plant=plant,
                forecasts=forecasts,
                final_planned=final_planned,
                price_allowed=price_allowed,
                system_activation_mwh=clean_afrr["afrr_system_activation_MWh"],
                baseline_bid_upper_bound=bid_upper_bound,
                baseline_storage_soc=baseline_storage_soc,
                baseline_gas_heat=baseline_gas_heat,
                min_bid_mw=min_bid_mw,
                timestep_hours=timestep_hours,
            )

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
            gas_benchmark_eur_per_mwh_th=gas_heat_benchmark,
            electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
            **_capacity_signal_kwargs(capacity_reservation, forecasts.index),
        )
        return plant.solve_afrr_down_rolling(self.config, forecasts, signals)

    def _strict_afrr_down_offer_and_activation(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        final_planned: pd.Series,
        price_allowed: pd.Series,
        system_activation_mwh: pd.Series,
        baseline_bid_upper_bound: pd.Series,
        baseline_storage_soc: pd.Series,
        baseline_gas_heat: pd.Series,
        min_bid_mw: float,
        timestep_hours: float,
    ) -> tuple[pd.Series, pd.Series]:
        """Limit aFRR down activation to trajectories that need no heat dumping.

        With strict useful heat dispatch, additional aFRR electricity can only be accepted
        when it can replace useful gas heat. The first strict implementation is conservative:
        it accepts only activation that can be converted to useful heat in the same timestep,
        avoiding hidden carry-over that would later require heat dumping.
        """

        bid_values: list[float] = []
        activated_values: list[float] = []
        max_charge_mwh = plant.etes.max_power_charge_mw * timestep_hours

        for timestamp in forecasts.index:
            planned_charge = max(0.0, float(final_planned.loc[timestamp]))
            useful_extra_heat_outlet = max(0.0, float(baseline_gas_heat.loc[timestamp]))
            baseline_soc = max(0.0, float(baseline_storage_soc.loc[timestamp]))
            storage_capacity_offer = max(0.0, plant.etes.max_capacity_mwh - baseline_soc)
            power_offer = max_charge_mwh - planned_charge
            immediate_use_offer = useful_extra_heat_outlet / (
                plant.etes.efficiency_charge * plant.etes.efficiency_discharge
            )
            strict_offer = max(0.0, min(power_offer, storage_capacity_offer, immediate_use_offer))

            if not bool(price_allowed.loc[timestamp]):
                strict_offer = 0.0
            strict_offer = min(
                strict_offer,
                max(0.0, float(baseline_bid_upper_bound.loc[timestamp])),
            )
            if timestep_hours <= 0 or strict_offer / timestep_hours < min_bid_mw:
                strict_offer = 0.0

            activation = min(strict_offer, max(0.0, float(system_activation_mwh.loc[timestamp])))

            bid_values.append(strict_offer)
            activated_values.append(activation)

        return (
            pd.Series(bid_values, index=forecasts.index, name="afrr_energy_bid_MWh"),
            pd.Series(activated_values, index=forecasts.index, name="afrr_energy_activated_MWh"),
        )

    def _strict_capacity_backed_afrr_activation(
        self,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        final_planned: pd.Series,
        capacity_backed_bid: pd.Series,
        system_activation_mwh: pd.Series,
        baseline_storage_soc: pd.Series,
        baseline_gas_heat: pd.Series,
    ) -> pd.Series:
        """Assign capacity-backed proxy activation only when useful heat remains feasible.

        Capacity reservation reports the plant's availability. The system activation signal is
        still a proxy, so the modelled plant receives only the activation volume that can be
        absorbed by ETES and later represented as useful heat without introducing heat dumping.
        """

        timestep_hours = self.config.timestep_minutes / 60.0
        max_charge_mwh = plant.etes.max_power_charge_mw * timestep_hours
        activated_values: list[float] = []

        for timestamp in forecasts.index:
            planned_charge = max(0.0, float(final_planned.loc[timestamp]))
            baseline_soc = max(0.0, float(baseline_storage_soc.loc[timestamp]))
            useful_extra_heat_outlet = max(0.0, float(baseline_gas_heat.loc[timestamp]))
            power_headroom = max(0.0, max_charge_mwh - planned_charge)
            storage_headroom_el = max(0.0, plant.etes.max_capacity_mwh - baseline_soc) / (
                plant.etes.efficiency_charge
            )
            useful_heat_replacement_el = useful_extra_heat_outlet / (
                plant.etes.efficiency_charge * plant.etes.efficiency_discharge
            )
            feasible_activation = min(
                max(0.0, float(capacity_backed_bid.loc[timestamp])),
                max(0.0, float(system_activation_mwh.loc[timestamp])),
                power_headroom,
                storage_headroom_el,
                useful_heat_replacement_el,
            )
            activated_values.append(max(0.0, feasible_activation))

        return pd.Series(
            activated_values,
            index=forecasts.index,
            name="afrr_energy_activated_MWh",
        )

    def decide_afrr_capacity(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
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
        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_heat_benchmark,
        )
        reference_price = forecasts[da_price_col].astype(float)
        opportunity_cost = (electricity_benchmark - reference_price).clip(lower=0.0)

        max_charge_power_mw = plant.etes.max_power_charge_mw
        min_bid_mw = float(capacity_market.product_rules.get("min_bid_mw", 0.0))
        expected_soc = plant.etes.initial_soc_mwh
        records = []
        for _, block in block_summary.iterrows():
            block_id = str(block["block_id"])
            mask = capacity_frame["afrr_capacity_block_id"] == block_id
            block_duration_h = float(block["block_duration_h"])
            opportunity_cost_block = float(
                (opportunity_cost.loc[mask] * timestep_hours).sum() / block_duration_h
            )
            storage_capacity_mw = max(
                0.0,
                (plant.etes.max_capacity_mwh - expected_soc)
                / (plant.etes.efficiency_charge * block_duration_h),
            )
            feasible_capacity = min(max_charge_power_mw, storage_capacity_mw)
            economic_allowed = (
                not bool(block["missing_capacity_price_flag"])
                and float(block["capacity_price_EUR_per_MW_h"])
                >= opportunity_cost_block + AFRR_CAPACITY_MARGIN_EUR_PER_MW_H
            )
            if feasible_capacity < min_bid_mw or not economic_allowed:
                reserved_mw = 0.0
            else:
                reserved_mw = feasible_capacity
            revenue = reserved_mw * float(block["capacity_price_EUR_per_MW_h"]) * block_duration_h
            records.append(
                {
                    **block.to_dict(),
                    "opportunity_cost_EUR_per_MW_h": opportunity_cost_block,
                    "economic_bid_allowed": bool(economic_allowed),
                    "feasible_capacity_potential_MW": feasible_capacity,
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
