# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_capacity import AFRRCapacityMarket
from flexi_mod.plants.capabilities import PlantCapabilities
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant
from flexi_mod.strategies._benchmarks import (
    GAS_PRICE_SIGNAL,
    delivered_electricity_price,
    electricity_trading_benchmark,
    gas_based_heat_cost,
)
from flexi_mod.strategies._bids import (
    _round_bid_down_to_increment,
    _validate_bid_rules,
    configured_afrr_energy_bid_margin,
)
from flexi_mod.strategies._charge_gate import (
    charge_gate,
    fixed_da_position,
    grid_charging_block,
    series_from_fixed_positions,
)
from flexi_mod.strategies._charges import (
    calculate_additional_charges_t,
    get_dynamic_charge_column,
    get_tax_rate,
)
from flexi_mod.strategies._clearing import (
    capacity_bid_price,
    capacity_pricing_rule,
    capacity_settlement_price,
    resolve_capacity_clearing_mechanism,
    resolve_energy_clearing_mechanism,
)
from flexi_mod.strategies.base_strategy import BaseStrategy
from flexi_mod.strategies.deciders.afrr_energy import (
    AFRREnergyDecider,
    prepare_afrr_down_energy_data,
)
from flexi_mod.strategies.deciders.day_ahead import DayAheadDecider
from flexi_mod.strategies.deciders.intraday_continuous import IntradayContinuousDecider

# TODO: Move IDC_MARGIN_EUR_PER_MWH to config.yaml once multi-country cases
# or sensitivity analyses are implemented.
IDC_MARGIN_EUR_PER_MWH = 0.0
AFRR_CAPACITY_MARGIN_EUR_PER_MW_H = 0.0


class HybridETESGasStrategy(BaseStrategy):
    """Operator strategy for electricity procurement and plant operation.

    The configured market classes describe market products and prepare market
    inputs. This strategy decides how the industrial operator acts on those
    inputs using the gas benchmark, margins and plant flexibility.
    """

    def __init__(self, config: CaseConfig):
        self.config = config
        self._capacity_clearing_mechanism = resolve_capacity_clearing_mechanism(config)
        self._energy_clearing_mechanism = resolve_energy_clearing_mechanism(config)
        self.afrr_energy_bid_margin_eur_per_mwh = configured_afrr_energy_bid_margin(config)
        self.afrr_energy_data_quality_summary = pd.DataFrame()
        self.afrr_capacity_block_summary = pd.DataFrame()
        self._afrr_down_energy_data_cache = {}

    @property
    def capacity_pricing_rule(self) -> str:
        """Return the aFRR-capacity clearing mechanism in force for this case."""

        return capacity_pricing_rule(self._capacity_clearing_mechanism)

    def capacity_settlement_price(
        self,
        capacity_bid_price_eur_per_mw_h: float,
        clearing_price_eur_per_mw_h: float,
    ) -> float:
        """Return the awarded-capacity settlement price in EUR/MW/h."""

        return capacity_settlement_price(
            self._capacity_clearing_mechanism,
            capacity_bid_price_eur_per_mw_h,
            clearing_price_eur_per_mw_h,
        )

    def capacity_bid_price(
        self,
        minimum_acceptable_price_eur_per_mw_h: float,
        market_reference_price_eur_per_mw_h: float,
    ) -> float:
        """Return the submitted bid price for the capacity product in EUR/MW/h."""

        return capacity_bid_price(
            self._capacity_clearing_mechanism,
            minimum_acceptable_price_eur_per_mw_h,
            market_reference_price_eur_per_mw_h,
        )

    # ─── Regulation-aware helpers ──────────────────────────────────

    @staticmethod
    def _get_tax_rate(plant: SteamGenerationPlant) -> float:
        """Read the multiplicative electricity tax rate from the plant's regulation."""

        return get_tax_rate(plant)

    @staticmethod
    def _get_dynamic_charge_column(plant: SteamGenerationPlant) -> str | None:
        """Read the dynamic charge column name from the plant's regulation."""

        return get_dynamic_charge_column(plant)

    def calculate_additional_charges_t(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
    ) -> pd.Series:
        """Return per-timestep additional electricity charges.

        - If the regulation declares a ``dynamic_charge_column``, that column must
          be present in the forecasts and its time series is used. A declared but
          missing column is a configuration error and raises — it is never
          silently replaced by the scalar, which would zero the per-MWh grid fee
          for countries whose charge lives entirely in that column (ES, FR).
        - Otherwise (no dynamic column declared, e.g. Germany): use the scalar
          marginal charge from the regulation.
        """

        return calculate_additional_charges_t(plant, forecasts)

    # ─── Core interface ────────────────────────────────────────────

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
        plant: PlantCapabilities,
        forecasts: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        gas_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        return DayAheadDecider(
            self.config, plant, forecasts, gas_benchmark
        ).decide(capacity_reservation, initial_soc_mwh, rolling)

    def decide_intraday_continuous(
        self,
        plant: PlantCapabilities,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant, gas_heat_benchmark
        )
        return IntradayContinuousDecider(
            self.config,
            plant,
            forecasts,
            gas_heat_benchmark,
            electricity_benchmark,
            IDC_MARGIN_EUR_PER_MWH,
        ).decide(fixed_positions, capacity_reservation, initial_soc_mwh, rolling)

    def decide_afrr_energy(
        self,
        plant: PlantCapabilities,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant, gas_heat_benchmark
        )
        decider = AFRREnergyDecider(
            self.config,
            plant,
            forecasts,
            gas_heat_benchmark,
            electricity_benchmark,
            self.afrr_energy_bid_margin_eur_per_mwh,
        )
        result = decider.decide(fixed_positions, capacity_reservation, initial_soc_mwh, rolling)
        self.afrr_energy_data_quality_summary = decider.quality_summary
        return result

    def decide_afrr_capacity(
        self,
        plant: PlantCapabilities,
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

        tax_rate = self._get_tax_rate(plant)
        additional_charges_t = self.calculate_additional_charges_t(plant, forecasts)

        gas_heat_benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        electricity_benchmark = self.calculate_electricity_trading_benchmark(
            plant,
            gas_heat_benchmark,
        )
        reference_price = self._delivered_electricity_price(
            forecasts[da_price_col].astype(float),
            tax_rate,
            additional_charges_t,
        )
        opportunity_cost = (electricity_benchmark - reference_price).clip(lower=0.0)
        afrr_energy_delivered_bid_price = (
            electricity_benchmark - self.afrr_energy_bid_margin_eur_per_mwh
        )
        delivered_afrr_energy_price = self._delivered_electricity_price(
            afrr_energy["afrr_energy_down_price_EUR_per_MWh"],
            tax_rate,
            additional_charges_t,
        )
        activation_relevant = (afrr_energy["afrr_system_activation_MWh"] > 1e-12) | afrr_energy[
            "afrr_activation_without_price"
        ].astype(bool)
        activation_price_allowed = afrr_energy["afrr_price_available"].astype(bool) & (
            delivered_afrr_energy_price <= afrr_energy_delivered_bid_price
        )

        max_charge_power_mw = plant.etes.max_power_charge_mw
        min_bid_mw = float(capacity_market.product_rules.get("min_bid_mw", 0.0))
        bid_increment_mw = float(capacity_market.product_rules.get("bid_increment_mw", 1.0))
        _validate_bid_rules("afrr_capacity", min_bid_mw, bid_increment_mw)
        heat_demand_mwh = forecasts[plant.heat_demand_column].astype(float) * timestep_hours
        expected_soc = plant.etes.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
        # Chain the projected storage level across blocks. A block's charge headroom is what
        # remains after the earlier reserved blocks are (worst-case) fully activated and the
        # process has drained the store. Sizing every block against the same day-start
        # snapshot instead lets consecutive reserved blocks each claim the one shared buffer
        # in full, over-committing capacity the plant cannot sustain under continuous
        # activation (the store saturates and the surplus is curtailed).
        projected_soc = expected_soc
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
            activation_expected = relevant_timesteps > 0
            activation_profitable = price_failed_timesteps == 0
            relevant_with_price = block_relevant & afrr_energy["afrr_price_available"].loc[mask]
            if relevant_with_price.any():
                activation_price_margin = (
                    afrr_energy_delivered_bid_price.loc[mask]
                    - delivered_afrr_energy_price.loc[mask]
                )
                min_activation_price_margin = float(
                    activation_price_margin.loc[relevant_with_price].min()
                )
            else:
                min_activation_price_margin = float("nan")
            storage_capacity_mw = max(
                0.0,
                (plant.etes.max_capacity_mwh - projected_soc)
                / (plant.etes.efficiency_charge * block_duration_h),
            )
            # Direct use: charging straight into the heat demand is deliverable even
            # when the ETES is full. The bid must survive continuous activation, so
            # the block's weakest step (min of heat demand and discharge power) sets
            # this ceiling, not the block average.
            round_trip = plant.etes.efficiency_charge * plant.etes.efficiency_discharge
            max_discharge_mwh_step = plant.etes.max_power_discharge_mw * timestep_hours
            block_min_heat_outlet_mwh = float(
                heat_demand_mwh.loc[mask].clip(upper=max_discharge_mwh_step).min()
            )
            direct_use_mw = (
                block_min_heat_outlet_mwh / (round_trip * timestep_hours)
                if round_trip > 0 and timestep_hours > 0
                else 0.0
            )
            deliverable_capacity_mw = min(max_charge_power_mw, storage_capacity_mw + direct_use_mw)
            max_activation_need_mwh = float(
                afrr_energy["afrr_system_activation_MWh"].loc[mask].max()
            )
            max_activation_need_mw = (
                max_activation_need_mwh / timestep_hours if timestep_hours > 0 else 0.0
            )
            # Limit the reservation by three independent ceilings: physical delivery,
            # expected peak activation, and capacity-market demand. The result is then
            # rounded down to the market increment. This prevents the representative
            # plant from claiming either unused physical headroom or more capacity than
            # the exogenous market quantity.
            technical_capacity = deliverable_capacity_mw
            market_capacity_mw = float(block["capacity_quantity_MW"])
            target_capacity_mw = min(
                deliverable_capacity_mw,
                max(max_activation_need_mw, min_bid_mw),
                market_capacity_mw,
            )
            compliant_capacity = _round_bid_down_to_increment(
                target_capacity_mw,
                min_bid_mw=min_bid_mw,
                bid_increment_mw=bid_increment_mw,
            )
            minimum_acceptable_price = opportunity_cost_block + AFRR_CAPACITY_MARGIN_EUR_PER_MW_H
            clearing_price = float(block["capacity_price_EUR_per_MW_h"])
            capacity_bid_price = self.capacity_bid_price(
                minimum_acceptable_price,
                clearing_price,
            )
            capacity_profitable = (
                not bool(block["missing_capacity_price_flag"])
                and clearing_price >= minimum_acceptable_price
                and clearing_price >= capacity_bid_price
            )
            technically_feasible = compliant_capacity > 1e-12 and compliant_capacity >= min_bid_mw
            capacity_quantity_available = (
                not bool(block["missing_capacity_quantity_flag"])
                and market_capacity_mw >= min_bid_mw
            )
            block_overlaps_high_load_window = bool(grid_block.loc[mask].any())
            bid_eligible = (
                activation_expected
                and capacity_profitable
                and activation_profitable
                and technically_feasible
                and capacity_quantity_available
                and not block_overlaps_high_load_window
            )
            if not bid_eligible:
                reserved_mw = 0.0
            else:
                reserved_mw = compliant_capacity
            # Advance the projected storage level for the next block, assuming this block's
            # reserved capacity is fully activated (the worst case a delivery guarantee must
            # survive) while the process keeps draining the store. Mirrors the dispatch-side
            # projection in project_reserved_capacity_claim so sizing and delivery agree.
            block_heat_thermal_mwh = float(heat_demand_mwh.loc[mask].sum())
            soc_in_mwh = reserved_mw * block_duration_h * plant.etes.efficiency_charge
            soc_out_mwh = min(
                block_heat_thermal_mwh / plant.etes.efficiency_discharge,
                plant.etes.max_power_discharge_mw * block_duration_h,
            )
            projected_soc = min(
                plant.etes.max_capacity_mwh,
                max(0.0, projected_soc + soc_in_mwh - soc_out_mwh),
            )
            settlement_price = self.capacity_settlement_price(
                capacity_bid_price,
                clearing_price,
            )
            revenue = reserved_mw * settlement_price * block_duration_h
            opportunity_cost_total = reserved_mw * opportunity_cost_block * block_duration_h
            market_surplus = (
                reserved_mw * (settlement_price - capacity_bid_price) * block_duration_h
            )
            net_value = revenue - opportunity_cost_total
            records.append(
                {
                    **block.to_dict(),
                    "capacity_pricing_rule": self.capacity_pricing_rule,
                    "opportunity_cost_EUR_per_MW_h": opportunity_cost_block,
                    "capacity_bid_price_EUR_per_MW_h": capacity_bid_price,
                    "capacity_clearing_price_EUR_per_MW_h": clearing_price,
                    "capacity_settlement_price_EUR_per_MW_h": settlement_price,
                    "activation_expected": bool(activation_expected),
                    "capacity_profitable": bool(capacity_profitable),
                    "activation_profitable": bool(activation_profitable),
                    "technically_feasible": bool(technically_feasible),
                    "bid_eligible": bool(bid_eligible),
                    "activation_steps": relevant_timesteps,
                    "price_fail_steps": price_failed_timesteps,
                    "missing_price_steps": activation_without_price_timesteps,
                    "min_price_margin_EUR_per_MWh": min_activation_price_margin,
                    "peak_activation_MW": max_activation_need_mw,
                    "technical_capacity_MW": technical_capacity,
                    "market_capacity_MW": market_capacity_mw,
                    "compliant_capacity_MW": compliant_capacity,
                    "capacity_quantity_binding": bool(
                        market_capacity_mw
                        <= min(
                            deliverable_capacity_mw,
                            max(max_activation_need_mw, min_bid_mw),
                        )
                    ),
                    "bid_increment_MW": bid_increment_mw,
                    "reserved_capacity_MW": reserved_mw,
                    "capacity_revenue_EUR": revenue,
                    "capacity_opportunity_cost_EUR": opportunity_cost_total,
                    "capacity_market_surplus_EUR": market_surplus,
                    "capacity_net_value_EUR": net_value,
                    "charge_headroom_MW": max_charge_power_mw,
                    "storage_headroom_MW": storage_capacity_mw,
                    "activated_energy_MWh": 0.0,
                    "activation_cost_EUR": 0.0,
                }
            )

        self.afrr_capacity_block_summary = pd.DataFrame(records)
        enriched = capacity_frame.join(
            self.afrr_capacity_block_summary.set_index("block_id")[
                [
                    "opportunity_cost_EUR_per_MW_h",
                    "capacity_pricing_rule",
                    "capacity_bid_price_EUR_per_MW_h",
                    "capacity_clearing_price_EUR_per_MW_h",
                    "capacity_settlement_price_EUR_per_MW_h",
                    "bid_eligible",
                    "compliant_capacity_MW",
                    "reserved_capacity_MW",
                    "capacity_revenue_EUR",
                    "capacity_opportunity_cost_EUR",
                    "capacity_market_surplus_EUR",
                    "capacity_net_value_EUR",
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
            * enriched["capacity_settlement_price_EUR_per_MW_h"]
            * timestep_hours
        )
        enriched["afrr_capacity_opportunity_cost_EUR"] = (
            enriched["afrr_capacity_reserved_MW"]
            * enriched["opportunity_cost_EUR_per_MW_h"]
            * timestep_hours
        )
        enriched["afrr_capacity_market_surplus_EUR"] = (
            enriched["afrr_capacity_reserved_MW"]
            * (
                enriched["capacity_settlement_price_EUR_per_MW_h"]
                - enriched["capacity_bid_price_EUR_per_MW_h"]
            )
            * timestep_hours
        )
        enriched["afrr_capacity_net_value_EUR"] = (
            enriched["afrr_capacity_revenue_EUR"] - enriched["afrr_capacity_opportunity_cost_EUR"]
        )
        return enriched

    def _prepare_afrr_down_energy_data(
        self,
        forecasts: pd.DataFrame,
        timestep_hours: float,
    ):
        cleaned = prepare_afrr_down_energy_data(self.config, forecasts, timestep_hours)
        self.afrr_energy_data_quality_summary = cleaned.quality_summary
        return cleaned

    def calculate_gas_based_heat_cost(
        self, plant: SteamGenerationPlant, forecasts: pd.DataFrame
    ) -> pd.Series:
        return gas_based_heat_cost(plant, forecasts)

    def calculate_electricity_trading_benchmark(
        self,
        plant: SteamGenerationPlant,
        gas_heat_benchmark: pd.Series,
    ) -> pd.Series:
        return electricity_trading_benchmark(plant, gas_heat_benchmark)

    @staticmethod
    def _delivered_electricity_price(
        market_price: pd.Series,
        tax_rate: float,
        additional_charges: pd.Series,
    ) -> pd.Series:
        return delivered_electricity_price(market_price, tax_rate, additional_charges)

    @staticmethod
    def _grid_charging_block(
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
    ) -> pd.Series:
        return grid_charging_block(plant, forecasts)

    @staticmethod
    def _calculate_charge_gate(
        plant: SteamGenerationPlant,
        electricity_price: pd.Series,
        benchmark: pd.Series,
    ) -> pd.Series:
        return charge_gate(plant, electricity_price, benchmark)

    @staticmethod
    def _fixed_da_position(
        fixed_positions: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        return fixed_da_position(fixed_positions, index)

    @staticmethod
    def _series_from_fixed_positions(
        fixed_positions: pd.DataFrame,
        column: str,
        index: pd.DatetimeIndex,
        default: float | None = 0.0,
    ) -> pd.Series | None:
        return series_from_fixed_positions(fixed_positions, column, index, default)
