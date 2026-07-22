# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import math
import warnings

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataValidationError
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

# aFRR clearing mechanisms. ``pay_as_bid`` pays each awarded bid its own
# submitted price; ``pay_as_cleared`` pays every awarded bid the marginal
# clearing price. The mechanism is a property of each market and is configured
# under ``markets.<market>.clearing_mechanism`` — see
# HybridETESGasStrategy._resolve_capacity_clearing_mechanism.
_VALID_CLEARING_MECHANISMS = ("pay_as_bid", "pay_as_cleared")
_AFRR_CAPACITY_MARKET = "afrr_capacity"
_AFRR_ENERGY_MARKET = "afrr_energy"
# Legacy strategy name that selected pay-as-cleared capacity before the
# clearing mechanism moved onto the market blocks. Kept for backward compatibility.
_PAY_AS_CLEARED_STRATEGY_NAME = "hybrid_etes_gas_pay_as_cleared_capacity"


class HybridETESGasStrategy(BaseStrategy):
    """Operator strategy for electricity procurement and plant operation.

    The configured market classes describe market products and prepare market
    inputs. This strategy decides how the industrial operator acts on those
    inputs using the gas benchmark, margins and plant flexibility.
    """

    def __init__(self, config: CaseConfig):
        self.config = config
        self._capacity_clearing_mechanism = self._resolve_capacity_clearing_mechanism(config)
        self._energy_clearing_mechanism = self._resolve_energy_clearing_mechanism(config)
        self.afrr_energy_data_quality_summary = pd.DataFrame()
        self.afrr_capacity_block_summary = pd.DataFrame()
        self._afrr_down_energy_data_cache = {}

    def _market_clearing_mechanism(self, config: CaseConfig, market_name: str) -> str | None:
        """Return ``markets.<market_name>.clearing_mechanism`` (lowercased) or None."""

        markets = config.case.get("markets") or {}
        market_cfg = markets.get(market_name) or {}
        value = market_cfg.get("clearing_mechanism")
        return None if value is None else str(value).strip().lower()

    def _resolve_capacity_clearing_mechanism(self, config: CaseConfig) -> str:
        """Determine the aFRR-capacity clearing mechanism from configuration.

        Resolution order (first match wins):
        1. ``markets.afrr_capacity.clearing_mechanism`` — the preferred location,
           since the clearing rule is a property of the market.
        2. Legacy ``strategy.clearing_mechanism`` field.
        3. Legacy strategy name ``hybrid_etes_gas_pay_as_cleared_capacity``.
        4. Default ``"pay_as_bid"``.
        """

        mechanism = self._market_clearing_mechanism(config, _AFRR_CAPACITY_MARKET)
        if mechanism is None:
            strategy_cfg = config.case.get("strategy") or {}
            legacy_field = strategy_cfg.get("clearing_mechanism")
            if legacy_field is not None:
                mechanism = str(legacy_field).strip().lower()
            elif config.strategy_name == _PAY_AS_CLEARED_STRATEGY_NAME:
                mechanism = "pay_as_cleared"
            else:
                mechanism = "pay_as_bid"
        if mechanism not in _VALID_CLEARING_MECHANISMS:
            options = ", ".join(_VALID_CLEARING_MECHANISMS)
            raise ValueError(
                f"Unknown afrr_capacity clearing_mechanism '{mechanism}'. Valid options: {options}."
            )
        return mechanism

    def _resolve_energy_clearing_mechanism(self, config: CaseConfig) -> str:
        """Determine the aFRR-energy clearing mechanism from configuration.

        aFRR energy is settled at the market clearing price, so ``pay_as_cleared``
        is the only mechanism implemented today. The field is accepted under
        ``markets.afrr_energy.clearing_mechanism`` for symmetry with capacity;
        a ``pay_as_bid`` value raises a clear error rather than silently being
        ignored. Defaults to ``"pay_as_cleared"``.
        """

        mechanism = self._market_clearing_mechanism(config, _AFRR_ENERGY_MARKET)
        if mechanism is None:
            return "pay_as_cleared"
        if mechanism not in _VALID_CLEARING_MECHANISMS:
            options = ", ".join(_VALID_CLEARING_MECHANISMS)
            raise ValueError(
                f"Unknown afrr_energy clearing_mechanism '{mechanism}'. Valid options: {options}."
            )
        if mechanism != "pay_as_cleared":
            raise ValueError(
                "afrr_energy clearing_mechanism 'pay_as_bid' is not implemented; "
                "aFRR energy is currently settled pay_as_cleared. "
                "Set clearing_mechanism: pay_as_cleared or omit the field."
            )
        return mechanism

    @property
    def capacity_pricing_rule(self) -> str:
        """Return the aFRR-capacity clearing mechanism in force for this case.

        Selected from configuration (``markets.afrr_capacity.clearing_mechanism``,
        the legacy ``strategy.clearing_mechanism`` field, or the legacy
        ``hybrid_etes_gas_pay_as_cleared_capacity`` strategy name) and defaulting
        to ``"pay_as_bid"``. Reported in outputs as ``afrr_capacity_pricing_rule``.
        """

        return self._capacity_clearing_mechanism

    def capacity_settlement_price(
        self,
        capacity_bid_price_eur_per_mw_h: float,
        clearing_price_eur_per_mw_h: float,
    ) -> float:
        """Return the awarded-capacity settlement price in EUR/MW/h.

        - ``pay_as_bid``: the operator is paid its own submitted bid price.
        - ``pay_as_cleared``: every awarded bid is paid the marginal clearing
          price, independent of the submitted bid.
        """

        if self._capacity_clearing_mechanism == "pay_as_cleared":
            return clearing_price_eur_per_mw_h
        return capacity_bid_price_eur_per_mw_h

    def capacity_bid_price(
        self,
        minimum_acceptable_price_eur_per_mw_h: float,
        market_reference_price_eur_per_mw_h: float,
    ) -> float:
        """Return the submitted bid price for the capacity product in EUR/MW/h.

        - ``pay_as_bid``: the configured capacity-price signal is interpreted as
          the submitted and awarded bid price, preserving prior behaviour.
        - ``pay_as_cleared``: the operator bids its true reservation price (the
          minimum acceptable price, i.e. opportunity cost plus capacity margin)
          and is awarded whenever the clearing price covers it.
        """

        if self._capacity_clearing_mechanism == "pay_as_cleared":
            return minimum_acceptable_price_eur_per_mw_h
        return market_reference_price_eur_per_mw_h

    # ─── Regulation-aware helpers ──────────────────────────────────

    @staticmethod
    def _get_tax_rate(plant: SteamGenerationPlant) -> float:
        """Read the multiplicative electricity tax rate from the plant's regulation."""
        regulation = getattr(plant, "grid_fee_regulation", None)
        if regulation is None:
            return 0.0
        return float(getattr(regulation, "electricity_tax_rate", 0.0))

    @staticmethod
    def _get_dynamic_charge_column(plant: SteamGenerationPlant) -> str | None:
        """Read the dynamic charge column name from the plant's regulation."""
        regulation = getattr(plant, "grid_fee_regulation", None)
        if regulation is None:
            return None
        return getattr(regulation, "dynamic_charge_column", None)

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
        ac_column = self._get_dynamic_charge_column(plant)
        scalar = float(getattr(plant, "additional_electricity_charge_eur_per_mwh", 0.0))
        if ac_column is not None:
            if ac_column not in forecasts.columns:
                raise DataValidationError(
                    f"Grid-fee regulation for plant '{plant.name}' declares dynamic "
                    f"charge column '{ac_column}', but it is missing from "
                    "forecasts_df.csv. Add the column (or fix its name); falling back "
                    "to the scalar charge would silently drop the per-MWh grid fee."
                )
            # The dynamic column carries the time-varying per-MWh grid charge. Any
            # static EUR/MWh levy is a *separate* component and is added on top, so
            # dispatch matches the ex-post settlement (which also sums both). For
            # ES/FR the static levy is 0 today, so this is a no-op there.
            if scalar != 0.0:
                warnings.warn(
                    f"Plant '{plant.name}' has a static per-MWh levy ({scalar} EUR/MWh) "
                    f"alongside dynamic charge column '{ac_column}'; both are summed for "
                    "dispatch and settlement. Verify the levy is not already included in "
                    "the dynamic column to avoid double counting.",
                    stacklevel=2,
                )
            charges = forecasts[ac_column].astype(float) + scalar
            charges.name = "additional_charges_EUR_per_MWh"
            return charges

        return pd.Series(scalar, index=forecasts.index, name="additional_charges_EUR_per_MWh")

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
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        market = DayAheadMarket("day_ahead", self.config.market("day_ahead"))
        market_data = market.prepare_market_data(forecasts)
        price_col = market.signal_column("price")

        tax_rate = self._get_tax_rate(plant)
        additional_charges_t = self.calculate_additional_charges_t(plant, forecasts)

        benchmark = self.calculate_gas_based_heat_cost(plant, forecasts)
        delivered_da_price = self._delivered_electricity_price(
            market_data["day_ahead_price_EUR_per_MWh"],
            tax_rate,
            additional_charges_t,
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
            additional_electricity_charge_eur_per_mwh=additional_charges_t,
            tax_rate=tax_rate,
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

        tax_rate = self._get_tax_rate(plant)
        additional_charges_t = self.calculate_additional_charges_t(plant, forecasts)

        da_position = self._fixed_da_position(fixed_positions, forecasts.index)
        idc_price = idc_data["IDC_price_EUR_per_MWh"]
        delivered_idc_price = self._delivered_electricity_price(
            idc_price,
            tax_rate,
            additional_charges_t,
        )
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
            additional_electricity_charge_eur_per_mwh=additional_charges_t,
            tax_rate=tax_rate,
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
        bid_increment_mw = float(product_rules.get("bid_increment_mw", 1.0))
        _validate_bid_rules("afrr_energy", min_bid_mw, bid_increment_mw)

        tax_rate = self._get_tax_rate(plant)
        additional_charges_t = self.calculate_additional_charges_t(plant, forecasts)

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
        baseline_storage_discharge = self._series_from_fixed_positions(
            fixed_positions,
            "etes_discharge_MWh",
            forecasts.index,
            default=0.0,
        )
        storage_capacity_headroom = (plant.etes.max_capacity_mwh - baseline_storage_soc).clip(
            lower=0.0
        ) / plant.etes.efficiency_charge

        valid_price = clean_afrr["afrr_price_available"]
        afrr_energy_bid_price = electricity_benchmark + AFRR_ENERGY_BID_MARGIN_EUR_PER_MWH
        delivered_afrr_price = self._delivered_electricity_price(
            clean_afrr["afrr_energy_down_price_EUR_per_MWh"],
            tax_rate,
            additional_charges_t,
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
        free_bid_upper_bound = (
            free_bid_potential.div(timestep_hours)
            .map(
                lambda feasible_mw: _round_bid_down_to_increment(
                    feasible_mw,
                    min_bid_mw=min_bid_mw,
                    bid_increment_mw=bid_increment_mw,
                )
            )
            .mul(timestep_hours)
        )

        system_activation_for_bid = clean_afrr["afrr_system_activation_MWh"].where(
            price_allowed, 0.0
        )
        if capacity_reservation is not None and not capacity_reservation.empty:
            capacity_backed_bid = reserved_capacity.clip(lower=0.0)
            bid_upper_bound, activated, split = self._strict_afrr_down_offer_and_activation_split(
                plant=plant,
                forecasts=forecasts,
                final_planned=final_planned,
                capacity_backed_bid=capacity_backed_bid,
                free_bid_upper_bound=free_bid_upper_bound,
                system_activation_mwh=system_activation_for_bid,
                baseline_storage_soc=baseline_storage_soc,
                baseline_gas_heat=baseline_gas_heat,
                baseline_storage_discharge=baseline_storage_discharge,
                min_bid_mw=min_bid_mw,
                bid_increment_mw=bid_increment_mw,
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
                system_activation_mwh=system_activation_for_bid,
                baseline_storage_soc=baseline_storage_soc,
                baseline_gas_heat=baseline_gas_heat,
                baseline_storage_discharge=baseline_storage_discharge,
                min_bid_mw=min_bid_mw,
                bid_increment_mw=bid_increment_mw,
                timestep_hours=timestep_hours,
            )
        headroom_binding = split["afrr_headroom_binding"]
        curtailed_activation = split["afrr_curtailment_MWh"]

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
            afrr_headroom_binding=headroom_binding,
            afrr_curtailment_mwh=curtailed_activation,
            gas_benchmark_eur_per_mwh_th=gas_heat_benchmark,
            electricity_trading_benchmark_eur_per_mwh_el=electricity_benchmark,
            additional_electricity_charge_eur_per_mwh=additional_charges_t,
            tax_rate=tax_rate,
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
        baseline_storage_discharge: pd.Series,
        min_bid_mw: float,
        bid_increment_mw: float,
        timestep_hours: float,
    ) -> tuple[pd.Series, pd.Series, dict[str, pd.Series]]:
        """Limit aFRR down bids and activation to market and ETES headroom.

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
        max_discharge_mwh = plant.etes.max_power_discharge_mw * timestep_hours
        baseline_soc_values = (
            baseline_storage_soc.reindex(forecasts.index).fillna(0.0).clip(lower=0.0)
        )
        replaceable_gas_heat = pd.concat(
            [
                baseline_gas_heat.reindex(forecasts.index).fillna(0.0).clip(lower=0.0),
                (
                    max_discharge_mwh
                    - baseline_storage_discharge.reindex(forecasts.index).fillna(0.0)
                ).clip(lower=0.0),
            ],
            axis=1,
        ).min(axis=1)
        future_storage_input_headroom = _future_storage_input_headroom_mwh(
            plant=plant,
            baseline_soc=baseline_soc_values,
            replaceable_gas_heat=replaceable_gas_heat,
        )
        # Free bids only get room the standing capacity promise does not claim.
        claimed_soc, remaining_gas_heat = _project_reserved_capacity_claim(
            plant=plant,
            capacity_backed_bid=capacity_backed_bid,
            baseline_soc=baseline_soc_values,
            replaceable_gas_heat=replaceable_gas_heat,
        )
        unclaimed_future_headroom = _future_storage_input_headroom_mwh(
            plant=plant,
            baseline_soc=claimed_soc,
            replaceable_gas_heat=remaining_gas_heat,
        )
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
            physical_activation_cap = max(
                0.0,
                min(power_offer, storage_capacity_offer),
            )
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
            "afrr_energy_free_activated_MWh": pd.Series(
                free_activated_values, index=forecasts.index
            ),
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
        afrr_energy_bid_price = electricity_benchmark + AFRR_ENERGY_BID_MARGIN_EUR_PER_MWH
        delivered_afrr_energy_price = self._delivered_electricity_price(
            afrr_energy["afrr_energy_down_price_EUR_per_MWh"],
            tax_rate,
            additional_charges_t,
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
        _validate_bid_rules("afrr_capacity", min_bid_mw, bid_increment_mw)
        heat_demand_mwh = forecasts[plant.heat_demand_column].astype(float) * timestep_hours
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
            activation_expected = relevant_timesteps > 0
            activation_profitable = price_failed_timesteps == 0
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
            # Reserve the volume the market will actually activate (the block's peak
            # 15-min activation), floored at the minimum bid for market compliance, and
            # never beyond what the plant can physically deliver. Reserving the full
            # deliverable capacity (ignoring the activation volume) over-reserves — and
            # is paid for — capacity that will never be called.
            technical_capacity = deliverable_capacity_mw
            target_capacity_mw = min(
                deliverable_capacity_mw, max(max_activation_need_mw, min_bid_mw)
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
            block_overlaps_high_load_window = bool(grid_block.loc[mask].any())
            bid_eligible = (
                activation_expected
                and capacity_profitable
                and activation_profitable
                and technically_feasible
                and not block_overlaps_high_load_window
            )
            if not bid_eligible:
                reserved_mw = 0.0
            else:
                reserved_mw = compliant_capacity
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
                    "compliant_capacity_MW": compliant_capacity,
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
        market_price: pd.Series,
        tax_rate: float,
        additional_charges: pd.Series,
    ) -> pd.Series:
        """Return total delivered electricity price including charges and tax.

        Formula: (market_price + additional_charges) × (1 + tax_rate)
        - DE: tax_rate=0.0, charges=scalar → price + scalar
        - ES: tax_rate>0, charges=time-series → (price + charges) × (1 + tax)
        """
        base = market_price.astype(float) + additional_charges.astype(float)
        return base * (1.0 + tax_rate)

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
            "capacity_clearing_price_EUR_per_MW_h",
        ),
        "afrr_capacity_pricing_rule": _capacity_object_column(
            frame,
            index,
            "capacity_pricing_rule",
            "",
        ),
        "afrr_capacity_bid_price_eur_per_mw_h": _capacity_column(
            frame,
            index,
            "capacity_bid_price_EUR_per_MW_h",
        ),
        "afrr_capacity_settlement_price_eur_per_mw_h": _capacity_column(
            frame,
            index,
            "capacity_settlement_price_EUR_per_MW_h",
        ),
        "afrr_capacity_reserved_mw": _capacity_column(frame, index, "afrr_capacity_reserved_MW"),
        "afrr_capacity_revenue_eur": _capacity_column(frame, index, "afrr_capacity_revenue_EUR"),
        "afrr_capacity_opportunity_cost_eur": _capacity_column(
            frame,
            index,
            "afrr_capacity_opportunity_cost_EUR",
        ),
        "afrr_capacity_market_surplus_eur": _capacity_column(
            frame,
            index,
            "afrr_capacity_market_surplus_EUR",
        ),
        "afrr_capacity_net_value_eur": _capacity_column(
            frame,
            index,
            "afrr_capacity_net_value_EUR",
        ),
    }


def _validate_bid_rules(market_name: str, min_bid_mw: float, bid_increment_mw: float) -> None:
    if min_bid_mw < 0:
        raise ValueError(f"{market_name}.product_rules.min_bid_mw cannot be negative")
    if bid_increment_mw <= 0:
        raise ValueError(f"{market_name}.product_rules.bid_increment_mw must be positive")


def _round_bid_down_to_increment(
    feasible_bid_mw: float,
    min_bid_mw: float,
    bid_increment_mw: float,
) -> float:
    """Return the largest market-compliant bid not exceeding physical capability."""

    if feasible_bid_mw < min_bid_mw:
        return 0.0
    rounded = math.floor((feasible_bid_mw + 1e-12) / bid_increment_mw) * bid_increment_mw
    if rounded < min_bid_mw:
        return 0.0
    return float(rounded)


def _future_storage_input_headroom_mwh(
    plant: SteamGenerationPlant,
    baseline_soc: pd.Series,
    replaceable_gas_heat: pd.Series,
) -> pd.Series:
    """Return feasible extra thermal inventory before each timestep's heat outlet.

    A volume can fit at delivery time but still overfill ETES later when fixed
    day-ahead or intraday electricity raises the baseline storage trajectory.
    Working backward makes each value reserve enough room for every later fixed
    charge while crediting only replaceable gas heat as a valid storage outlet.
    """

    baseline_soc_values = baseline_soc.to_numpy(dtype=float)
    replaceable_heat_values = replaceable_gas_heat.to_numpy(dtype=float)
    retention = 1.0 - plant.etes.storage_loss_rate
    discharge_efficiency = plant.etes.efficiency_discharge
    max_capacity = plant.etes.max_capacity_mwh
    allowed_before = [0.0] * len(baseline_soc_values)
    next_allowed_before = float("inf")

    for position in range(len(baseline_soc_values) - 1, -1, -1):
        capacity_headroom = max(0.0, max_capacity - baseline_soc_values[position])
        if retention > 0.0:
            allowed_after_outlet = min(capacity_headroom, next_allowed_before / retention)
        else:
            allowed_after_outlet = capacity_headroom
        allowed_before[position] = (
            allowed_after_outlet + replaceable_heat_values[position] / discharge_efficiency
        )
        next_allowed_before = allowed_before[position]

    return pd.Series(allowed_before, index=baseline_soc.index)


def _project_reserved_capacity_claim(
    plant: SteamGenerationPlant,
    capacity_backed_bid: pd.Series,
    baseline_soc: pd.Series,
    replaceable_gas_heat: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Project the baseline SOC as if every reserved interval were fully activated.

    Returns the claimed SOC trajectory and the replaceable gas heat left over
    for free bids. Sizing free bids against these keeps them out of the storage
    room a later capacity-backed activation is entitled to.
    """

    retention = 1.0 - plant.etes.storage_loss_rate
    charge_efficiency = plant.etes.efficiency_charge
    discharge_efficiency = plant.etes.efficiency_discharge
    max_capacity = plant.etes.max_capacity_mwh
    claim = 0.0
    claimed_soc = []
    remaining_gas_heat = []

    for position in range(len(baseline_soc)):
        claim *= retention
        claim += max(0.0, float(capacity_backed_bid.iloc[position])) * charge_efficiency
        outlet = min(float(replaceable_gas_heat.iloc[position]), claim * discharge_efficiency)
        claim -= outlet / discharge_efficiency
        claimed_soc.append(min(max_capacity, float(baseline_soc.iloc[position]) + claim))
        remaining_gas_heat.append(float(replaceable_gas_heat.iloc[position]) - outlet)

    index = baseline_soc.index
    return pd.Series(claimed_soc, index=index), pd.Series(remaining_gas_heat, index=index)


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
