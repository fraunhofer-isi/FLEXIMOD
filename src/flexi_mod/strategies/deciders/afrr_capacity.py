# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""aFRR capacity market decider for the hybrid ETES + gas strategy.

Builds the per-block capacity reservation bid, sizing each block against the
projected storage headroom and the exogenous market quantity. The gas and
electricity benchmarks are computed by the strategy (preserving the subclass
override hooks) and passed in; the clearing mechanism, the aFRR-energy bid
margin and the capacity margin constant are also passed in.
"""

from __future__ import annotations

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_capacity import AFRRCapacityMarket
from flexi_mod.plants.capabilities import PlantCapabilities
from flexi_mod.strategies._benchmarks import delivered_electricity_price
from flexi_mod.strategies._bids import _round_bid_down_to_increment, _validate_bid_rules
from flexi_mod.strategies._charge_gate import grid_charging_block
from flexi_mod.strategies._charges import calculate_additional_charges_t, get_tax_rate
from flexi_mod.strategies._clearing import (
    capacity_bid_price,
    capacity_pricing_rule,
    capacity_settlement_price,
)
from flexi_mod.strategies.deciders.afrr_energy import prepare_afrr_down_energy_data


class AFRRCapacityDecider:
    """Build aFRR capacity reservation bids per delivery block."""

    def __init__(
        self,
        config: CaseConfig,
        plant: PlantCapabilities,
        gas_heat_benchmark: pd.Series,
        electricity_benchmark: pd.Series,
        capacity_clearing_mechanism: str,
        afrr_energy_bid_margin_eur_per_mwh: float,
        capacity_margin_eur_per_mw_h: float,
    ):
        self.config = config
        self.plant = plant
        self.gas_heat_benchmark = gas_heat_benchmark
        self.electricity_benchmark = electricity_benchmark
        self.capacity_clearing_mechanism = capacity_clearing_mechanism
        self.afrr_energy_bid_margin_eur_per_mwh = afrr_energy_bid_margin_eur_per_mwh
        self.capacity_margin_eur_per_mw_h = capacity_margin_eur_per_mw_h
        self.quality_summary: pd.DataFrame | None = None
        self.block_summary = pd.DataFrame()

    def decide(
        self,
        forecasts: pd.DataFrame,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        self._prepare_market_data(forecasts)
        self._prepare_afrr_energy(forecasts)
        self._prepare_price_gates()
        self._prepare_block_params(forecasts, initial_soc_mwh)
        records = self._build_block_records()
        self.block_summary = pd.DataFrame(records)
        return self._enrich()

    def _prepare_market_data(self, forecasts: pd.DataFrame) -> None:
        self.forecasts = forecasts
        self.capacity_market = AFRRCapacityMarket(
            "afrr_capacity", self.config.market("afrr_capacity")
        )
        self.timestep_hours = self.config.timestep_minutes / 60.0
        capacity_data = self.capacity_market.prepare_market_data(
            forecasts, timestep_hours=self.timestep_hours
        )
        self.capacity_frame = capacity_data.frame.copy()
        self.block_summary_raw = capacity_data.block_summary.copy()
        self.da_price_col = self.config.market_signal("day_ahead", "price")

    def _prepare_afrr_energy(self, forecasts: pd.DataFrame) -> None:
        self.afrr_energy_enabled = "afrr_energy" in self.config.enabled_markets
        if self.afrr_energy_enabled:
            cleaned = prepare_afrr_down_energy_data(self.config, forecasts, self.timestep_hours)
            self.quality_summary = cleaned.quality_summary
            self.afrr_energy = cleaned.frame
        else:
            self.afrr_energy = pd.DataFrame(
                {
                    "afrr_energy_down_price_EUR_per_MWh": 0.0,
                    "afrr_price_available": False,
                    "afrr_system_activation_MWh": 0.0,
                    "afrr_activation_without_price": False,
                },
                index=forecasts.index,
            )

    def _prepare_price_gates(self) -> None:
        self.tax_rate = get_tax_rate(self.plant)
        self.additional_charges_t = calculate_additional_charges_t(self.plant, self.forecasts)
        reference_price = delivered_electricity_price(
            self.forecasts[self.da_price_col].astype(float),
            self.tax_rate,
            self.additional_charges_t,
        )
        self.opportunity_cost = (self.electricity_benchmark - reference_price).clip(lower=0.0)
        self.afrr_energy_delivered_bid_price = (
            self.electricity_benchmark - self.afrr_energy_bid_margin_eur_per_mwh
        )
        self.delivered_afrr_energy_price = delivered_electricity_price(
            self.afrr_energy["afrr_energy_down_price_EUR_per_MWh"],
            self.tax_rate,
            self.additional_charges_t,
        )
        self.activation_relevant = (
            self.afrr_energy["afrr_system_activation_MWh"] > 1e-12
        ) | self.afrr_energy["afrr_activation_without_price"].astype(bool)
        self.activation_price_allowed = self.afrr_energy["afrr_price_available"].astype(bool) & (
            self.delivered_afrr_energy_price <= self.afrr_energy_delivered_bid_price
        )

    def _prepare_block_params(self, forecasts: pd.DataFrame, initial_soc_mwh: float | None) -> None:
        self.max_charge_power_mw = self.plant.etes.max_power_charge_mw
        self.min_bid_mw = float(self.capacity_market.product_rules.get("min_bid_mw", 0.0))
        self.bid_increment_mw = float(
            self.capacity_market.product_rules.get("bid_increment_mw", 1.0)
        )
        _validate_bid_rules("afrr_capacity", self.min_bid_mw, self.bid_increment_mw)
        self.heat_demand_mwh = (
            forecasts[self.plant.heat_demand_column].astype(float) * self.timestep_hours
        )
        self.projected_soc = (
            self.plant.etes.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
        )
        self.grid_block = grid_charging_block(self.plant, forecasts)

    def _build_block_records(self) -> list[dict]:
        records = []
        for _, block in self.block_summary_raw.iterrows():
            record = self._evaluate_block(block)
            records.append(record)
        return records

    def _evaluate_block(self, block: pd.Series) -> dict:
        block_id = str(block["block_id"])
        mask = self.capacity_frame["afrr_capacity_block_id"] == block_id
        block_duration_h = float(block["block_duration_h"])
        flags = self._block_activation_flags(mask, block_duration_h)
        sizing = self._block_capacity_sizing(mask, block_duration_h, flags)
        market_capacity_mw = float(block["capacity_quantity_MW"])
        compliant = self._compliant_capacity(sizing, market_capacity_mw)
        eligibility = self._block_eligibility(block, mask, compliant, market_capacity_mw, flags)
        reserved_mw = compliant if eligibility["bid_eligible"] else 0.0
        self.projected_soc = self._advance_projected_soc(mask, block_duration_h, reserved_mw)
        return self._build_block_record(
            block,
            block_duration_h,
            mask,
            compliant,
            reserved_mw,
            market_capacity_mw,
            sizing,
            flags,
            eligibility,
        )

    def _block_activation_flags(self, mask: pd.Series, block_duration_h: float) -> dict:
        opportunity_cost_block = float(
            (self.opportunity_cost.loc[mask] * self.timestep_hours).sum() / block_duration_h
        )
        block_relevant = self.activation_relevant.loc[mask]
        block_price_allowed = self.activation_price_allowed.loc[mask]
        block_without_price = self.afrr_energy["afrr_activation_without_price"].loc[mask]
        relevant_timesteps = int(block_relevant.sum())
        activation_without_price_timesteps = int(block_without_price.sum())
        price_failed_timesteps = int((block_relevant & ~block_price_allowed).sum())
        relevant_with_price = block_relevant & self.afrr_energy["afrr_price_available"].loc[mask]
        min_activation_price_margin = self._min_activation_price_margin(mask, relevant_with_price)
        max_activation_need_mw = self._max_activation_need_mw(mask)
        return {
            "opportunity_cost_block": opportunity_cost_block,
            "relevant_timesteps": relevant_timesteps,
            "activation_without_price_timesteps": activation_without_price_timesteps,
            "price_failed_timesteps": price_failed_timesteps,
            "activation_expected": relevant_timesteps > 0,
            "activation_profitable": price_failed_timesteps == 0,
            "min_activation_price_margin": min_activation_price_margin,
            "max_activation_need_mw": max_activation_need_mw,
        }

    def _min_activation_price_margin(
        self, mask: pd.Series, relevant_with_price: pd.Series
    ) -> float:
        if relevant_with_price.any():
            activation_price_margin = (
                self.afrr_energy_delivered_bid_price.loc[mask]
                - self.delivered_afrr_energy_price.loc[mask]
            )
            return float(activation_price_margin.loc[relevant_with_price].min())
        return float("nan")

    def _max_activation_need_mw(self, mask: pd.Series) -> float:
        max_activation_need_mwh = float(
            self.afrr_energy["afrr_system_activation_MWh"].loc[mask].max()
        )
        return max_activation_need_mwh / self.timestep_hours if self.timestep_hours > 0 else 0.0

    def _block_capacity_sizing(self, mask: pd.Series, block_duration_h: float, flags: dict) -> dict:
        storage_capacity_mw = max(
            0.0,
            (self.plant.etes.max_capacity_mwh - self.projected_soc)
            / (self.plant.etes.efficiency_charge * block_duration_h),
        )
        round_trip = self.plant.etes.efficiency_charge * self.plant.etes.efficiency_discharge
        max_discharge_mwh_step = self.plant.etes.max_power_discharge_mw * self.timestep_hours
        block_min_heat_outlet_mwh = float(
            self.heat_demand_mwh.loc[mask].clip(upper=max_discharge_mwh_step).min()
        )
        direct_use_mw = (
            block_min_heat_outlet_mwh / (round_trip * self.timestep_hours)
            if round_trip > 0 and self.timestep_hours > 0
            else 0.0
        )
        deliverable_capacity_mw = min(self.max_charge_power_mw, storage_capacity_mw + direct_use_mw)
        return {
            "storage_capacity_mw": storage_capacity_mw,
            "deliverable_capacity_mw": deliverable_capacity_mw,
            "max_activation_need_mw": flags["max_activation_need_mw"],
        }

    def _compliant_capacity(self, sizing: dict, market_capacity_mw: float) -> float:
        target_capacity_mw = min(
            sizing["deliverable_capacity_mw"],
            max(sizing["max_activation_need_mw"], self.min_bid_mw),
            market_capacity_mw,
        )
        return _round_bid_down_to_increment(
            target_capacity_mw,
            min_bid_mw=self.min_bid_mw,
            bid_increment_mw=self.bid_increment_mw,
        )

    def _block_eligibility(
        self,
        block: pd.Series,
        mask: pd.Series,
        compliant_capacity: float,
        market_capacity_mw: float,
        flags: dict,
    ) -> dict:
        minimum_acceptable_price = (
            flags["opportunity_cost_block"] + self.capacity_margin_eur_per_mw_h
        )
        clearing_price = float(block["capacity_price_EUR_per_MW_h"])
        bid_price = capacity_bid_price(
            self.capacity_clearing_mechanism,
            minimum_acceptable_price,
            clearing_price,
        )
        capacity_profitable = (
            not bool(block["missing_capacity_price_flag"])
            and clearing_price >= minimum_acceptable_price
            and clearing_price >= bid_price
        )
        technically_feasible = compliant_capacity > 1e-12 and compliant_capacity >= self.min_bid_mw
        capacity_quantity_available = (
            not bool(block["missing_capacity_quantity_flag"])
            and market_capacity_mw >= self.min_bid_mw
        )
        block_overlaps_high_load_window = bool(self.grid_block.loc[mask].any())
        bid_eligible = (
            flags["activation_expected"]
            and capacity_profitable
            and flags["activation_profitable"]
            and technically_feasible
            and capacity_quantity_available
            and not block_overlaps_high_load_window
        )
        return {
            "minimum_acceptable_price": minimum_acceptable_price,
            "clearing_price": clearing_price,
            "bid_price": bid_price,
            "capacity_profitable": capacity_profitable,
            "technically_feasible": technically_feasible,
            "capacity_quantity_available": capacity_quantity_available,
            "block_overlaps_high_load_window": block_overlaps_high_load_window,
            "bid_eligible": bid_eligible,
        }

    def _advance_projected_soc(
        self, mask: pd.Series, block_duration_h: float, reserved_mw: float
    ) -> float:
        block_heat_thermal_mwh = float(self.heat_demand_mwh.loc[mask].sum())
        soc_in_mwh = reserved_mw * block_duration_h * self.plant.etes.efficiency_charge
        soc_out_mwh = min(
            block_heat_thermal_mwh / self.plant.etes.efficiency_discharge,
            self.plant.etes.max_power_discharge_mw * block_duration_h,
        )
        return min(
            self.plant.etes.max_capacity_mwh,
            max(0.0, self.projected_soc + soc_in_mwh - soc_out_mwh),
        )

    def _build_block_record(
        self,
        block: pd.Series,
        block_duration_h: float,
        mask: pd.Series,
        compliant_capacity: float,
        reserved_mw: float,
        market_capacity_mw: float,
        sizing: dict,
        flags: dict,
        eligibility: dict,
    ) -> dict:
        settlement_price = capacity_settlement_price(
            self.capacity_clearing_mechanism,
            eligibility["bid_price"],
            eligibility["clearing_price"],
        )
        revenue = reserved_mw * settlement_price * block_duration_h
        opportunity_cost_total = reserved_mw * flags["opportunity_cost_block"] * block_duration_h
        market_surplus = (
            reserved_mw * (settlement_price - eligibility["bid_price"]) * block_duration_h
        )
        net_value = revenue - opportunity_cost_total
        deliverable = sizing["deliverable_capacity_mw"]
        return {
            **block.to_dict(),
            "capacity_pricing_rule": capacity_pricing_rule(self.capacity_clearing_mechanism),
            "opportunity_cost_EUR_per_MW_h": flags["opportunity_cost_block"],
            "capacity_bid_price_EUR_per_MW_h": eligibility["bid_price"],
            "capacity_clearing_price_EUR_per_MW_h": eligibility["clearing_price"],
            "capacity_settlement_price_EUR_per_MW_h": settlement_price,
            "activation_expected": bool(flags["activation_expected"]),
            "capacity_profitable": bool(eligibility["capacity_profitable"]),
            "activation_profitable": bool(flags["activation_profitable"]),
            "technically_feasible": bool(eligibility["technically_feasible"]),
            "bid_eligible": bool(eligibility["bid_eligible"]),
            "activation_steps": flags["relevant_timesteps"],
            "price_fail_steps": flags["price_failed_timesteps"],
            "missing_price_steps": flags["activation_without_price_timesteps"],
            "min_price_margin_EUR_per_MWh": flags["min_activation_price_margin"],
            "peak_activation_MW": flags["max_activation_need_mw"],
            "technical_capacity_MW": deliverable,
            "market_capacity_MW": market_capacity_mw,
            "compliant_capacity_MW": compliant_capacity,
            "capacity_quantity_binding": bool(
                market_capacity_mw
                <= min(
                    deliverable,
                    max(flags["max_activation_need_mw"], self.min_bid_mw),
                )
            ),
            "bid_increment_MW": self.bid_increment_mw,
            "reserved_capacity_MW": reserved_mw,
            "capacity_revenue_EUR": revenue,
            "capacity_opportunity_cost_EUR": opportunity_cost_total,
            "capacity_market_surplus_EUR": market_surplus,
            "capacity_net_value_EUR": net_value,
            "charge_headroom_MW": self.max_charge_power_mw,
            "storage_headroom_MW": sizing["storage_capacity_mw"],
            "activated_energy_MWh": 0.0,
            "activation_cost_EUR": 0.0,
        }

    def _enrich(self) -> pd.DataFrame:
        enriched = self.capacity_frame.join(
            self.block_summary.set_index("block_id")[
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
            enriched["afrr_capacity_reserved_MW"] * self.timestep_hours
        )
        enriched["afrr_capacity_revenue_EUR"] = (
            enriched["afrr_capacity_reserved_MW"]
            * enriched["capacity_settlement_price_EUR_per_MW_h"]
            * self.timestep_hours
        )
        enriched["afrr_capacity_opportunity_cost_EUR"] = (
            enriched["afrr_capacity_reserved_MW"]
            * enriched["opportunity_cost_EUR_per_MW_h"]
            * self.timestep_hours
        )
        enriched["afrr_capacity_market_surplus_EUR"] = (
            enriched["afrr_capacity_reserved_MW"]
            * (
                enriched["capacity_settlement_price_EUR_per_MW_h"]
                - enriched["capacity_bid_price_EUR_per_MW_h"]
            )
            * self.timestep_hours
        )
        enriched["afrr_capacity_net_value_EUR"] = (
            enriched["afrr_capacity_revenue_EUR"] - enriched["afrr_capacity_opportunity_cost_EUR"]
        )
        return enriched
