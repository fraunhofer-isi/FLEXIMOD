# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""aFRR down-energy market decider for the hybrid ETES + gas strategy.

Builds the aFRR down-energy bid and activation split, then asks the plant
to solve the rolling or horizon aFRR-down schedule. The gas and electricity
benchmarks are computed by the strategy (preserving the subclass override
hooks) and passed in; the ``afrr_energy_bid_margin`` is also passed in.
"""

from __future__ import annotations

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_energy import AFRRDownEnergyMarket
from flexi_mod.plants.capabilities import PlantCapabilities
from flexi_mod.plants.signals.afrr_down_signals import AFRRDownSignals
from flexi_mod.strategies._benchmarks import GAS_PRICE_SIGNAL, delivered_electricity_price
from flexi_mod.strategies._bids import (
    _round_bid_down_to_increment,
    _validate_bid_rules,
    raw_electricity_bid_price,
)
from flexi_mod.strategies._capacity import (
    _capacity_column,
    capacity_signal_kwargs,
    future_storage_input_headroom_mwh,
    project_reserved_capacity_claim,
)
from flexi_mod.strategies._charge_gate import grid_charging_block, series_from_fixed_positions
from flexi_mod.strategies._charges import calculate_additional_charges_t, get_tax_rate


def prepare_afrr_down_energy_data(
    config: CaseConfig,
    forecasts: pd.DataFrame,
    timestep_hours: float,
):
    """Return cleaned aFRR down-energy market data (no side effects).

    NOTE: do not memoize by id(forecasts). CPython reuses object ids after
    garbage collection, so a transient per-window forecasts copy (the direct
    boiler strategy always copies) can collide with a stale cache entry from
    an earlier window -- returning aFRR data for the wrong index/length (e.g.
    across a DST-shortened window) and silently corrupting results or raising
    an index-mismatch. prepare_market_data is a cheap, pure function of
    forecasts.
    """
    afrr_energy_market = AFRRDownEnergyMarket("afrr_energy", config.market("afrr_energy"))
    return afrr_energy_market.prepare_market_data(forecasts, timestep_hours=timestep_hours)


class AFRREnergyDecider:
    """Build aFRR down-energy signals and solve the plant schedule."""

    def __init__(
        self,
        config: CaseConfig,
        plant: PlantCapabilities,
        forecasts: pd.DataFrame,
        gas_benchmark: pd.Series,
        electricity_benchmark: pd.Series,
        afrr_energy_bid_margin: float,
    ):
        self.config = config
        self.plant = plant
        self.forecasts = forecasts
        self.gas_benchmark = gas_benchmark
        self.electricity_benchmark = electricity_benchmark
        self.afrr_energy_bid_margin = afrr_energy_bid_margin
        self.quality_summary = None

    def decide(
        self,
        fixed_positions: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None,
        initial_soc_mwh: float | None,
        rolling: bool,
    ) -> pd.DataFrame:
        self._prepare_forecasts()
        self._prepare_market_data()
        self._prepare_charges()
        self._prepare_positions(fixed_positions)
        self._prepare_baselines(fixed_positions, capacity_reservation)
        self._compute_price_gate()
        self._compute_free_bid()
        self._split_offer_and_activation(capacity_reservation)
        signals = self._build_signals(capacity_reservation)
        return self._solve(signals, initial_soc_mwh, rolling)

    def _prepare_forecasts(self) -> None:
        self.da_price_col = self.config.market_signal("day_ahead", "price")
        if self.da_price_col not in self.forecasts.columns:
            self.forecasts = self.forecasts.copy()
            self.forecasts[self.da_price_col] = 0.0
        self.idc_price_col = self.config.market_signal("intraday_continuous", "price")
        if self.idc_price_col not in self.forecasts.columns:
            self.forecasts = self.forecasts.copy()
            self.forecasts[self.idc_price_col] = 0.0
        self.timestep_hours = self.config.timestep_minutes / 60.0

    def _prepare_market_data(self) -> None:
        afrr_market = AFRRDownEnergyMarket("afrr_energy", self.config.market("afrr_energy"))
        product_rules = afrr_market.product_rules
        self.min_bid_mw = float(product_rules.get("min_bid_mw", 0.0))
        self.bid_increment_mw = float(product_rules.get("bid_increment_mw", 1.0))
        _validate_bid_rules("afrr_energy", self.min_bid_mw, self.bid_increment_mw)
        cleaned = prepare_afrr_down_energy_data(self.config, self.forecasts, self.timestep_hours)
        self.quality_summary = cleaned.quality_summary
        self.clean_afrr = cleaned.frame

    def _prepare_charges(self) -> None:
        self.tax_rate = get_tax_rate(self.plant)
        self.additional_charges = calculate_additional_charges_t(self.plant, self.forecasts)

    def _prepare_positions(self, fixed_positions: pd.DataFrame) -> None:
        index = self.forecasts.index
        self.da_position = series_from_fixed_positions(fixed_positions, "DA_position_MWh", index)
        self.idc_buy = series_from_fixed_positions(
            fixed_positions, "IDC_buy_MWh", index, default=0.0
        )
        self.idc_sell = series_from_fixed_positions(
            fixed_positions, "IDC_sell_MWh", index, default=0.0
        )
        final_planned = series_from_fixed_positions(
            fixed_positions, "final_planned_electricity_MWh", index, default=None
        )
        if final_planned is None:
            final_planned = self.da_position + self.idc_buy - self.idc_sell
        self.final_planned = final_planned

    def _prepare_baselines(
        self, fixed_positions: pd.DataFrame, capacity_reservation: pd.DataFrame | None
    ) -> None:
        index = self.forecasts.index
        self.reserved_capacity = _capacity_column(
            capacity_reservation, index, "afrr_capacity_reserved_MWh"
        )
        self.baseline_storage_soc = series_from_fixed_positions(
            fixed_positions, "etes_soc_MWh", index, default=self.plant.etes.initial_soc_mwh
        )
        self.baseline_gas_heat = series_from_fixed_positions(
            fixed_positions, "gas_heat_MWh", index, default=0.0
        )
        self.baseline_storage_discharge = series_from_fixed_positions(
            fixed_positions, "etes_discharge_MWh", index, default=0.0
        )
        self.storage_capacity_headroom = (
            self.plant.etes.max_capacity_mwh - self.baseline_storage_soc
        ).clip(lower=0.0) / self.plant.etes.efficiency_charge

    def _compute_price_gate(self) -> None:
        valid_price = self.clean_afrr["afrr_price_available"]
        self.afrr_energy_delivered_bid_price = (
            self.electricity_benchmark - self.afrr_energy_bid_margin
        )
        self.afrr_energy_bid_price = raw_electricity_bid_price(
            self.afrr_energy_delivered_bid_price, self.tax_rate, self.additional_charges
        )
        delivered_afrr_price = delivered_electricity_price(
            self.clean_afrr["afrr_energy_down_price_EUR_per_MWh"],
            self.tax_rate,
            self.additional_charges,
        )
        self.price_allowed = (
            (delivered_afrr_price <= self.afrr_energy_delivered_bid_price)
            & valid_price
            & ~grid_charging_block(self.plant, self.forecasts)
        )

    def _compute_free_bid(self) -> None:
        max_charge_mwh = self.plant.etes.max_power_charge_mw * self.timestep_hours
        charge_power_headroom = (max_charge_mwh - self.final_planned - self.reserved_capacity).clip(
            lower=0.0
        )
        storage_headroom = (self.storage_capacity_headroom - self.reserved_capacity).clip(lower=0.0)
        free_bid_potential = pd.concat([charge_power_headroom, storage_headroom], axis=1).min(
            axis=1
        )
        free_bid_potential = free_bid_potential.where(self.price_allowed, 0.0).clip(lower=0.0)
        self.free_bid_upper_bound = (
            free_bid_potential.div(self.timestep_hours)
            .map(
                lambda feasible_mw: _round_bid_down_to_increment(
                    feasible_mw,
                    min_bid_mw=self.min_bid_mw,
                    bid_increment_mw=self.bid_increment_mw,
                )
            )
            .mul(self.timestep_hours)
        )
        self.system_activation_for_bid = self.clean_afrr["afrr_system_activation_MWh"].where(
            self.price_allowed, 0.0
        )

    def _split_offer_and_activation(self, capacity_reservation: pd.DataFrame | None) -> None:
        if capacity_reservation is not None and not capacity_reservation.empty:
            capacity_backed_bid = self.reserved_capacity.clip(lower=0.0)
        else:
            capacity_backed_bid = pd.Series(0.0, index=self.forecasts.index)
        self.bid_upper_bound, self.activated, self.split = (
            self._strict_afrr_down_offer_and_activation_split(capacity_backed_bid)
        )

    def _build_signals(self, capacity_reservation: pd.DataFrame | None) -> AFRRDownSignals:
        split = self.split
        return AFRRDownSignals(
            da_price_col=self.da_price_col,
            idc_price_col=self.idc_price_col,
            gas_price_col=GAS_PRICE_SIGNAL,
            da_position_mwh=self.da_position,
            idc_buy_mwh=self.idc_buy,
            idc_sell_mwh=self.idc_sell,
            final_planned_electricity_mwh=self.final_planned,
            afrr_energy_price=self.clean_afrr["afrr_energy_down_price_EUR_per_MWh"],
            afrr_system_activation_mwh=self.clean_afrr["afrr_system_activation_MWh"],
            afrr_energy_bid_mwh=self.bid_upper_bound,
            afrr_energy_activated_mwh=self.activated,
            afrr_energy_bid_price=self.afrr_energy_bid_price,
            afrr_energy_delivered_bid_price=self.afrr_energy_delivered_bid_price,
            afrr_energy_capacity_backed_bid_mwh=split["afrr_energy_capacity_backed_bid_MWh"],
            afrr_energy_free_bid_mwh=split["afrr_energy_free_bid_MWh"],
            afrr_energy_capacity_backed_activated_mwh=split[
                "afrr_energy_capacity_backed_activated_MWh"
            ],
            afrr_energy_free_activated_mwh=split["afrr_energy_free_activated_MWh"],
            afrr_headroom_binding=split["afrr_headroom_binding"],
            afrr_curtailment_mwh=split["afrr_curtailment_MWh"],
            gas_benchmark_eur_per_mwh_th=self.gas_benchmark,
            electricity_trading_benchmark_eur_per_mwh_el=self.electricity_benchmark,
            additional_electricity_charge_eur_per_mwh=self.additional_charges,
            tax_rate=self.tax_rate,
            **capacity_signal_kwargs(capacity_reservation, self.forecasts.index),
        )

    def _solve(
        self,
        signals: AFRRDownSignals,
        initial_soc_mwh: float | None,
        rolling: bool,
    ) -> pd.DataFrame:
        if rolling:
            return self.plant.solve_afrr_down_rolling(
                self.config, self.forecasts, signals, initial_soc_mwh=initial_soc_mwh
            )
        return self.plant.solve_afrr_down_horizon(
            self.config, self.forecasts, signals, initial_soc_mwh=initial_soc_mwh
        )

    def _strict_afrr_down_offer_and_activation_split(
        self, capacity_backed_bid: pd.Series
    ) -> tuple[pd.Series, pd.Series, dict[str, pd.Series]]:
        """Limit aFRR down bids and activation to market and ETES headroom.

        Capacity-backed bid volume is the mandatory energy bid behind awarded
        capacity. Free bid volume is optional and enters only after the strategy
        has found the aFRR energy deal profitable. Activation is allocated to
        capacity-backed volume first, then to optional free volume.
        """
        return _compute_offer_and_activation_split(
            plant=self.plant,
            forecasts=self.forecasts,
            final_planned=self.final_planned,
            capacity_backed_bid=capacity_backed_bid,
            free_bid_upper_bound=self.free_bid_upper_bound,
            system_activation_mwh=self.system_activation_for_bid,
            baseline_storage_soc=self.baseline_storage_soc,
            baseline_gas_heat=self.baseline_gas_heat,
            baseline_storage_discharge=self.baseline_storage_discharge,
            min_bid_mw=self.min_bid_mw,
            bid_increment_mw=self.bid_increment_mw,
            timestep_hours=self.timestep_hours,
        )


def _compute_offer_and_activation_split(
    plant: PlantCapabilities,
    forecasts: pd.DataFrame,
    final_planned: pd.Series,
    capacity_backed_bid: pd.Series,
    free_bid_upper_bound: pd.Series,
    system_activation_mwh: pd.Series,
    baseline_storage_soc: pd.Series,
    baseline_gas_heat: pd.Series,
    baseline_storage_discharge: pd.Series,
    min_bid_mw: float,
    bid_increment_mw: float,
    timestep_hours: float,
) -> tuple[pd.Series, pd.Series, dict[str, pd.Series]]:
    max_charge_mwh = plant.etes.max_power_charge_mw * timestep_hours
    max_discharge_mwh = plant.etes.max_power_discharge_mw * timestep_hours
    baseline_soc_values = baseline_storage_soc.reindex(forecasts.index).fillna(0.0).clip(lower=0.0)
    replaceable_gas_heat = pd.concat(
        [
            baseline_gas_heat.reindex(forecasts.index).fillna(0.0).clip(lower=0.0),
            (
                max_discharge_mwh - baseline_storage_discharge.reindex(forecasts.index).fillna(0.0)
            ).clip(lower=0.0),
        ],
        axis=1,
    ).min(axis=1)
    future_storage_input_headroom = future_storage_input_headroom_mwh(
        plant=plant,
        baseline_soc=baseline_soc_values,
        replaceable_gas_heat=replaceable_gas_heat,
    )
    claimed_soc, remaining_gas_heat = project_reserved_capacity_claim(
        plant=plant,
        capacity_backed_bid=capacity_backed_bid,
        baseline_soc=baseline_soc_values,
        replaceable_gas_heat=replaceable_gas_heat,
    )
    unclaimed_future_headroom = future_storage_input_headroom_mwh(
        plant=plant,
        baseline_soc=claimed_soc,
        replaceable_gas_heat=remaining_gas_heat,
    )
    return _collect_per_timestep_offers(
        plant=plant,
        forecasts=forecasts,
        final_planned=final_planned,
        capacity_backed_bid=capacity_backed_bid,
        free_bid_upper_bound=free_bid_upper_bound,
        system_activation_mwh=system_activation_mwh,
        baseline_soc_values=baseline_soc_values,
        replaceable_gas_heat=replaceable_gas_heat,
        future_storage_input_headroom=future_storage_input_headroom,
        unclaimed_future_headroom=unclaimed_future_headroom,
        min_bid_mw=min_bid_mw,
        bid_increment_mw=bid_increment_mw,
        timestep_hours=timestep_hours,
        max_charge_mwh=max_charge_mwh,
    )


def _collect_per_timestep_offers(
    plant: PlantCapabilities,
    forecasts: pd.DataFrame,
    final_planned: pd.Series,
    capacity_backed_bid: pd.Series,
    free_bid_upper_bound: pd.Series,
    system_activation_mwh: pd.Series,
    baseline_soc_values: pd.Series,
    replaceable_gas_heat: pd.Series,
    future_storage_input_headroom: pd.Series,
    unclaimed_future_headroom: pd.Series,
    min_bid_mw: float,
    bid_increment_mw: float,
    timestep_hours: float,
    max_charge_mwh: float,
) -> tuple[pd.Series, pd.Series, dict[str, pd.Series]]:
    capacity_bid_values: list[float] = []
    free_bid_values: list[float] = []
    total_bid_values: list[float] = []
    capacity_activated_values: list[float] = []
    free_activated_values: list[float] = []
    total_activated_values: list[float] = []
    binding_values: list[bool] = []
    curtailed_values: list[float] = []
    additional_soc_mwh = 0.0

    for position, timestamp in enumerate(forecasts.index):
        planned_charge = max(0.0, float(final_planned.loc[timestamp]))
        baseline_soc = float(baseline_soc_values.loc[timestamp])
        additional_soc_mwh *= 1.0 - plant.etes.storage_loss_rate
        replaceable_heat = float(replaceable_gas_heat.loc[timestamp])
        storage_capacity_offer = (
            max(
                0.0,
                plant.etes.max_capacity_mwh
                - baseline_soc
                - additional_soc_mwh
                + replaceable_heat / plant.etes.efficiency_discharge,
            )
            / plant.etes.efficiency_charge
        )
        power_offer = max_charge_mwh - planned_charge
        physical_activation_cap = max(0.0, min(power_offer, storage_capacity_offer))
        future_storage_cap = max(
            0.0,
            (float(future_storage_input_headroom.iloc[position]) - additional_soc_mwh)
            / plant.etes.efficiency_charge,
        )
        horizon_activation_cap = min(physical_activation_cap, future_storage_cap)

        capacity_bid = max(0.0, float(capacity_backed_bid.loc[timestamp]))
        free_bid = max(0.0, float(free_bid_upper_bound.loc[timestamp]))
        free_room_after_capacity = max(0.0, horizon_activation_cap - capacity_bid)
        unclaimed_future_cap = max(
            0.0,
            (float(unclaimed_future_headroom.iloc[position]) - additional_soc_mwh)
            / plant.etes.efficiency_charge,
        )
        feasible_free_bid_mwh = min(free_bid, free_room_after_capacity, unclaimed_future_cap)
        free_bid = (
            _round_bid_down_to_increment(
                feasible_free_bid_mwh / timestep_hours,
                min_bid_mw=min_bid_mw,
                bid_increment_mw=bid_increment_mw,
            )
            * timestep_hours
        )

        total_bid = capacity_bid + free_bid
        system_activation = max(0.0, float(system_activation_mwh.loc[timestamp]))
        proxy_activation = min(total_bid, system_activation)
        feasible_activation = min(proxy_activation, horizon_activation_cap)
        capacity_activated = min(capacity_bid, feasible_activation)
        free_activated = min(free_bid, max(0.0, feasible_activation - capacity_activated))
        total_activated = capacity_activated + free_activated
        curtailment = max(0.0, proxy_activation - total_activated)

        additional_soc_mwh += total_activated * plant.etes.efficiency_charge
        additional_discharge_heat = min(
            replaceable_heat,
            additional_soc_mwh * plant.etes.efficiency_discharge,
        )
        additional_soc_mwh -= additional_discharge_heat / plant.etes.efficiency_discharge

        capacity_bid_values.append(capacity_bid)
        free_bid_values.append(free_bid)
        total_bid_values.append(total_bid)
        capacity_activated_values.append(capacity_activated)
        free_activated_values.append(free_activated)
        total_activated_values.append(total_activated)
        binding_values.append(
            proxy_activation > 1e-12 and horizon_activation_cap < proxy_activation - 1e-12
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
        "afrr_energy_free_activated_MWh": pd.Series(free_activated_values, index=forecasts.index),
        "afrr_headroom_binding": pd.Series(binding_values, index=forecasts.index),
        "afrr_curtailment_MWh": pd.Series(curtailed_values, index=forecasts.index),
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
