# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rule-based (non-MILP) market-timing variant of :class:`ElectrifiedSteelStrategy`.

Physical dispatch (ramp, technology balances, storage) is still solved exactly by
:meth:`SteelPlant.solve_rolling` -- the same Pyomo LP used by
``steel_cost_minimization``. Only the *market-timing* decision (which market
supplies each MWh, when to reserve aFRR capacity) is rule-based, mirroring the
day-ahead/gas-benchmark gate and all-or-nothing capacity-block eligibility rule
already used for steam plants by ``HybridETESGasStrategy``. This trades the exact
MILP's joint optimality for roughly an order of magnitude in solve time, since no
integer bid-sizing variables are solved at all.

Two cases, matching the plant's own fuel/technology mix:

* Case A -- a hybrid electrolyser+gas route (``dri_plant`` or ``bf_bof`` fuel_type
  ``hybrid_hydrogen_natural_gas``, with an on-site ``electrolyser``): a
  natural-gas-based benchmark price gates whether the electrolyser runs at all
  (day-ahead-favorable, or aFRR-energy-rescued when day-ahead alone is not).
* Case B -- every other route: no fuel substitution exists, so the day-ahead
  schedule is the baseline procurement plan and aFRR energy simply substitutes
  in wherever it is cheaper than the day-ahead price already being paid.

Known limitation -- aFRR capacity revenue is a conservative lower bound, not an
estimate: this strategy decides production timing first (via the plain LP above,
with no awareness that aFRR capacity exists), then sizes capacity from whatever
that already-fixed schedule happens to be consuming. The exact MILP instead
decides both *jointly*, and can earn meaningfully more capacity revenue by
deliberately running below full power at some hours specifically to free up room
for a larger commitment elsewhere -- a trade-off this sequential design cannot
see. Real-data validation (a 4-day slice, 9 plants) measured this understatement
at roughly 42% higher net cost on hybrid electrolyser+gas routes (where the
electrolyser is the dominant, most schedule-flexible load) versus ~0.4-1.7% on
routes with no fuel-substitution choice, where the gap is small. Use the exact
MILP (``ElectrifiedSteelStrategy``) when aFRR-capacity economics on
hybrid-electrolyser routes need to be accurate, not just directionally right.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from flexi_mod.markets.afrr_capacity import AFRRCapacityMarket
from flexi_mod.markets.afrr_energy import AFRRDownEnergyMarket
from flexi_mod.markets.day_ahead import DayAheadMarket
from flexi_mod.plants.steel_plant import STEEL_AFRR_RESULT_COLUMNS, SteelDispatchSignals, SteelPlant
from flexi_mod.strategies.electrified_steel_strategy import (
    CO2_PRICE_SIGNAL,
    HYDROGEN_PRICE_SIGNAL,
    IRON_ORE_PRICE_SIGNAL,
    LIME_PRICE_SIGNAL,
    NATURAL_GAS_PRICE_SIGNAL,
    ElectrifiedSteelStrategy,
    _capacity_block_summary,
    _duration_hours,
)

_ELECTROLYSER_ALLOWED = "__electrolyser_allowed"
_EFFECTIVE_ELECTRICITY_PRICE = "__effective_electricity_price_EUR_per_MWh"
HYBRID_FUEL_TYPE = "hybrid_hydrogen_natural_gas"


class ElectrifiedSteelRuleBasedStrategy(ElectrifiedSteelStrategy):
    """DA/aFRR market-timing decided by rules instead of a joint MILP.

    Subclasses :class:`ElectrifiedSteelStrategy` to reuse its ``__init__``,
    ``_validate_configuration`` (market_sequence, German gate calendars), and
    ``required_forecast_columns`` unchanged -- only :meth:`dispatch` differs.
    """

    def dispatch(
        self,
        plant: SteelPlant,
        forecasts: pd.DataFrame,
        progress_callback: Callable[[pd.Timestamp, pd.Timestamp], None] | None = None,
    ) -> pd.DataFrame:
        timestep_hours = self.config.timestep_minutes / 60.0
        day_ahead = DayAheadMarket("day_ahead", self.config.market("day_ahead"))
        capacity = AFRRCapacityMarket("afrr_capacity", self.config.market("afrr_capacity"))
        energy = AFRRDownEnergyMarket("afrr_energy", self.config.market("afrr_energy"))

        day_ahead.prepare_market_data(forecasts)
        capacity_data = capacity.prepare_market_data(forecasts, timestep_hours=timestep_hours)
        energy_data = energy.prepare_market_data(forecasts, timestep_hours=timestep_hours)
        self.afrr_energy_data_quality_summary = energy_data.quality_summary.copy()

        energy_rules = energy.product_rules
        capacity_rules = capacity.product_rules
        energy_min_bid_mw = float(energy_rules.get("min_bid_mw", 0.0))
        energy_bid_increment_mw = float(energy_rules.get("bid_increment_mw", 1.0))
        capacity_min_bid_mw = float(capacity_rules.get("min_bid_mw", 0.0))
        capacity_bid_increment_mw = float(capacity_rules.get("bid_increment_mw", 1.0))
        _validate_bid_rules("afrr_energy", energy_min_bid_mw, energy_bid_increment_mw)
        _validate_bid_rules("afrr_capacity", capacity_min_bid_mw, capacity_bid_increment_mw)
        capacity_product_duration_h = _duration_hours(capacity.product_length)

        da_price_col = day_ahead.signal_column("price")
        da_price = forecasts[da_price_col].astype(float)
        afrr_price = energy_data.frame["afrr_energy_down_price_EUR_per_MWh"].astype(float)
        afrr_available = energy_data.frame["afrr_price_available"].astype(bool)
        afrr_system_activation = energy_data.frame["afrr_system_activation_MWh"].astype(float)
        capacity_block_id = capacity_data.frame["afrr_capacity_block_id"]
        capacity_block_duration = capacity_data.frame["block_duration_h"].astype(float)
        capacity_price = capacity_data.frame["capacity_price_EUR_per_MW_h"].astype(float)
        capacity_missing_price = capacity_data.frame["missing_capacity_price_flag"].astype(bool)

        additional_charge = float(plant.additional_electricity_charge_eur_per_mwh)
        delivered_da_price = da_price + additional_charge
        delivered_afrr_price = afrr_price + additional_charge

        case_a, hybrid_component = _classify_case_a(plant)

        prepared = forecasts.copy()
        effective_price = da_price.copy()
        electrolyser_da_favored = pd.Series(False, index=forecasts.index)
        electrolyser_afrr_rescued = pd.Series(False, index=forecasts.index)
        benchmark = pd.Series(np.nan, index=forecasts.index)
        if case_a:
            benchmark = _gas_based_electricity_benchmark(
                plant, hybrid_component, forecasts
            )
            electrolyser_da_favored = delivered_da_price <= benchmark
            electrolyser_afrr_rescued = (
                ~electrolyser_da_favored & afrr_available & (delivered_afrr_price <= benchmark)
            )
            gate_open = electrolyser_da_favored | electrolyser_afrr_rescued
            prepared[_ELECTROLYSER_ALLOWED] = gate_open
            effective_price = effective_price.where(
                ~electrolyser_afrr_rescued, np.minimum(da_price, afrr_price)
            )
        prepared[_EFFECTIVE_ELECTRICITY_PRICE] = effective_price

        signals = SteelDispatchSignals(
            electricity_price_col=_EFFECTIVE_ELECTRICITY_PRICE,
            natural_gas_price_col=NATURAL_GAS_PRICE_SIGNAL,
            hydrogen_price_col=HYDROGEN_PRICE_SIGNAL,
            iron_ore_price_col=IRON_ORE_PRICE_SIGNAL,
            lime_price_col=LIME_PRICE_SIGNAL,
            co2_price_col=CO2_PRICE_SIGNAL,
            electrolyser_allowed_col=_ELECTROLYSER_ALLOWED if case_a else None,
        )
        physical = plant.solve_rolling(
            self.config,
            prepared,
            signals,
            progress_callback=progress_callback,
        )

        dispatch = _apply_market_timing_rules(
            plant=plant,
            physical=physical,
            effective_price=effective_price.reindex(physical.index),
            da_price=da_price.reindex(physical.index),
            afrr_price=afrr_price.reindex(physical.index),
            afrr_available=afrr_available.reindex(physical.index),
            afrr_system_activation=afrr_system_activation.reindex(physical.index),
            capacity_block_id=capacity_block_id.reindex(physical.index),
            capacity_block_duration=capacity_block_duration.reindex(physical.index),
            capacity_price=capacity_price.reindex(physical.index),
            capacity_missing_price=capacity_missing_price.reindex(physical.index),
            capacity_product_duration_h=capacity_product_duration_h,
            additional_charge=additional_charge,
            timestep_hours=timestep_hours,
            energy_min_bid_mw=energy_min_bid_mw,
            energy_bid_increment_mw=energy_bid_increment_mw,
            capacity_min_bid_mw=capacity_min_bid_mw,
            capacity_bid_increment_mw=capacity_bid_increment_mw,
            case_a=case_a,
            benchmark=benchmark.reindex(physical.index),
        )
        self.afrr_capacity_block_summary = _capacity_block_summary(dispatch)
        return dispatch


def _classify_case_a(plant: SteelPlant) -> tuple[bool, object | None]:
    """Return (is_case_a, hybrid_fuel_component) for the plant's technology mix.

    Case A requires an on-site electrolyser AND a hybrid-fuel terminal/DRI block
    that can genuinely substitute hydrogen for gas. Checks whichever of
    ``dri_plant``/``bf_bof`` is present -- both share the same ``fuel_type`` /
    ``specific_hydrogen_consumption_mwh_per_t`` / ``specific_natural_gas_consumption_mwh_per_t``
    attributes, and both support the hybrid fuel split in their own Pyomo balance.
    """

    if "electrolyser" not in plant.components:
        return False, None
    hybrid_component = plant.components.get("dri_plant") or plant.components.get("bf_bof")
    if hybrid_component is None:
        return False, None
    if getattr(hybrid_component, "fuel_type", None) != HYBRID_FUEL_TYPE:
        return False, None
    return True, hybrid_component


def _gas_based_electricity_benchmark(
    plant: SteelPlant,
    hybrid_component: object,
    forecasts: pd.DataFrame,
) -> pd.Series:
    """Electricity price at which hydrogen (via electrolyser) matches gas cost per t.

    ``gas_route_cost_per_t = natural_gas_price * specific_natural_gas_consumption + CO2``
    ``hydrogen_mwh_el_per_t = specific_hydrogen_consumption / electrolyser.efficiency``
    ``benchmark = gas_route_cost_per_t / hydrogen_mwh_el_per_t``

    CO2 is included (the exact MILP prices it on the gas path too), unlike the
    ETES benchmark's TODO-flagged omission -- a deliberate fidelity choice, not
    an oversight. Iron-ore and the block's own baseline electricity consumption
    are fuel-route-independent and correctly excluded.
    """

    natural_gas_price = forecasts[NATURAL_GAS_PRICE_SIGNAL].astype(float)
    co2_price = forecasts[CO2_PRICE_SIGNAL].astype(float)
    specific_natural_gas = float(hybrid_component.specific_natural_gas_consumption_mwh_per_t)
    natural_gas_co2_factor = float(hybrid_component.natural_gas_co2_factor_t_per_mwh)
    specific_hydrogen = float(hybrid_component.specific_hydrogen_consumption_mwh_per_t)
    electrolyser = plant.components["electrolyser"]
    efficiency = float(electrolyser.efficiency)

    gas_route_cost_per_t = specific_natural_gas * (
        natural_gas_price + natural_gas_co2_factor * co2_price
    )
    hydrogen_mwh_el_per_t = specific_hydrogen / efficiency
    benchmark = gas_route_cost_per_t / hydrogen_mwh_el_per_t
    benchmark.name = "gas_based_electricity_benchmark_EUR_per_MWh_el"
    return benchmark


def _capacity_reserved_mw_by_block(
    *,
    capacity_block_id: pd.Series,
    total: pd.Series,
    afrr_cheaper_than_da: pd.Series,
    afrr_system_activation: pd.Series,
    capacity_price: pd.Series,
    capacity_missing_price: pd.Series,
    capacity_product_duration_h: float,
    timestep_hours: float,
    capacity_min_bid_mw: float,
    capacity_bid_increment_mw: float,
) -> dict[object, float]:
    """Return reserved capacity MW per aFRR capacity block.

    In principle capacity is a *retainer*, independent of whether the energy trade is
    separately worthwhile -- real aFRR capacity and energy are separately auctioned
    products, and the exact MILP's own eligibility rule never checks energy-market
    favorability, only that the capacity-price data is valid/complete. A decoupled
    version of this rule was tried and rejected empirically: the MILP earns its
    capacity revenue partly by *jointly* choosing to run below full power at some
    hours specifically to free up room for a larger capacity commitment elsewhere --
    a genuine trade-off between production timing and capacity headroom that only
    exists when both are solved together. This heuristic decides production timing
    first, with no awareness that capacity exists, then bounds capacity by whatever
    that already-fixed schedule happens to be consuming. Without also requiring
    energy-market favorability, that bound came out *larger* than the exact MILP's
    own (provably optimal) choice in real-data testing -- an impossible result if
    the comparison is fair, confirming this heuristic cannot safely size capacity
    independent of the energy decision the way the MILP can. Requiring the whole
    block to also be energy-favorable is therefore kept as a real constraint of this
    design, not merely conservatism: it is the only way found to guarantee the
    result stays a valid (if understated) lower bound on the true opportunity,
    particularly on hybrid electrolyser+gas routes where this understatement is
    largest -- see ``ElectrifiedSteelRuleBasedStrategy``'s module docstring.

    "Technically deliverable" is bounded by the block's own consumption, since
    capacity-backed energy can never exceed what is actually being drawn under the
    market-sourcing framing (there is no storage buffer to draw extra power into, by
    design -- that is what makes this framing work for routes without one). The block
    *minimum* is used, not an average: a "retainer" must be deliverable at every
    quarter-hour, not just on average.
    """

    block_frame = pd.DataFrame(
        {
            "block_id": capacity_block_id,
            "total": total,
            "relevant": afrr_system_activation > 1e-12,
            "favorable": afrr_cheaper_than_da,
            "capacity_price": capacity_price,
            "missing_price": capacity_missing_price,
        }
    )
    reserved_mw_by_block: dict[object, float] = {}
    for block_id, block in block_frame.groupby("block_id", sort=False):
        relevant = block["relevant"]
        activation_expected = bool(relevant.any())
        activation_profitable = (
            bool(block.loc[relevant, "favorable"].all()) if activation_expected else False
        )
        block_duration_h = float(len(block) * timestep_hours)
        complete = abs(block_duration_h - capacity_product_duration_h) < 1e-8
        price_ok = (
            not bool(block["missing_price"].any())
            and complete
            and bool((block["capacity_price"] > 0.0).all())
        )
        bid_eligible = activation_expected and activation_profitable and price_ok
        if bid_eligible:
            technical_mw = float(block["total"].min()) / timestep_hours
            reserved_mw_by_block[block_id] = _round_bid_down_to_increment(
                technical_mw, capacity_min_bid_mw, capacity_bid_increment_mw
            )
        else:
            reserved_mw_by_block[block_id] = 0.0
    return reserved_mw_by_block


def _apply_market_timing_rules(
    *,
    plant: SteelPlant,
    physical: pd.DataFrame,
    effective_price: pd.Series,
    da_price: pd.Series,
    afrr_price: pd.Series,
    afrr_available: pd.Series,
    afrr_system_activation: pd.Series,
    capacity_block_id: pd.Series,
    capacity_block_duration: pd.Series,
    capacity_price: pd.Series,
    capacity_missing_price: pd.Series,
    capacity_product_duration_h: float,
    additional_charge: float,
    timestep_hours: float,
    energy_min_bid_mw: float,
    energy_bid_increment_mw: float,
    capacity_min_bid_mw: float,
    capacity_bid_increment_mw: float,
    case_a: bool,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Post-solve, pure-pandas market-sourcing split -- no Pyomo involved."""

    index = physical.index
    total = physical["total_electricity_consumption_MWh"].astype(float)
    if case_a and "electrolyser_electricity_consumption_MWh" in physical.columns:
        electrolyser_mwh = physical["electrolyser_electricity_consumption_MWh"].astype(float)
    else:
        electrolyser_mwh = pd.Series(0.0, index=index)
    aggregate_max_power_mw = sum(
        float(getattr(component, "max_power_mw", 0.0)) for component in plant.components.values()
    )

    # Energy favorability is decided first (independent of capacity): source via aFRR
    # energy wherever it beats day-ahead, regardless of what produced the load.
    afrr_cheaper_than_da = afrr_available & (afrr_price < da_price)

    reserved_mw_by_block = _capacity_reserved_mw_by_block(
        capacity_block_id=capacity_block_id,
        total=total,
        afrr_cheaper_than_da=afrr_cheaper_than_da,
        afrr_system_activation=afrr_system_activation,
        capacity_price=capacity_price,
        capacity_missing_price=capacity_missing_price,
        capacity_product_duration_h=capacity_product_duration_h,
        timestep_hours=timestep_hours,
        capacity_min_bid_mw=capacity_min_bid_mw,
        capacity_bid_increment_mw=capacity_bid_increment_mw,
    )
    reserved_mw = capacity_block_id.map(reserved_mw_by_block).astype(float)
    reserved_mwh = reserved_mw * timestep_hours

    # Energy sourcing: the capacity-backed portion is mandatory (once reserved, it must
    # be offered/delivered regardless of the moment-to-moment price), plus an optional
    # voluntary top-up wherever aFRR energy alone beats day-ahead.
    voluntary_mwh = total.where(afrr_cheaper_than_da, 0.0)
    q_afrr_desired = np.maximum(reserved_mwh, voluntary_mwh)

    bid_mw_raw = q_afrr_desired / timestep_hours
    bid_mw = bid_mw_raw.map(
        lambda mw: _round_bid_down_to_increment(mw, energy_min_bid_mw, energy_bid_increment_mw)
    )
    # Increment rounding must never undercut the mandatory capacity commitment.
    bid_mwh = np.maximum(bid_mw * timestep_hours, reserved_mwh)
    activated = np.minimum(bid_mwh, afrr_system_activation).where(afrr_available, 0.0)
    da_position = total - activated

    capacity_backed_bid_mwh = np.minimum(reserved_mwh, bid_mwh)
    free_bid_mwh = bid_mwh - capacity_backed_bid_mwh
    capacity_backed_activated_mwh = np.minimum(capacity_backed_bid_mwh, activated)
    free_activated_mwh = activated - capacity_backed_activated_mwh

    non_electric_cost = physical["variable_cost_EUR"].astype(float) - total * effective_price
    electricity_market_cost = da_position * da_price + activated * afrr_price
    additional_electricity_charges_cost = total * additional_charge
    gross_operating_cost = (
        non_electric_cost + electricity_market_cost + additional_electricity_charges_cost
    )
    capacity_revenue = reserved_mw * capacity_price * timestep_hours
    net_operating_cost = gross_operating_cost - capacity_revenue

    delivered_afrr_price = afrr_price + additional_charge
    afrr_bid_price = benchmark.where(electrolyser_mwh > 0.0, da_price) if case_a else da_price
    market_spread = afrr_bid_price - afrr_price
    net_spread = afrr_bid_price - delivered_afrr_price
    available_load_headroom = (aggregate_max_power_mw * timestep_hours - da_position).clip(lower=0.0)

    result = physical.copy()
    result["DA_position_MWh"] = da_position
    result["IDC_buy_MWh"] = 0.0
    result["IDC_sell_MWh"] = 0.0
    result["final_planned_electricity_MWh"] = da_position
    result["actual_electricity_consumption_MWh"] = total
    result["day_ahead_price_EUR_per_MWh"] = da_price
    result["day_ahead_delivered_price_EUR_per_MWh"] = da_price + additional_charge
    result["additional_electricity_charge_EUR_per_MWh_el"] = additional_charge
    result["afrr_energy_bid_MW"] = bid_mw
    result["afrr_energy_bid_MWh"] = bid_mwh
    result["afrr_energy_activated_MWh"] = activated
    result["afrr_energy_price_EUR_per_MWh"] = afrr_price
    result["afrr_energy_delivered_price_EUR_per_MWh"] = delivered_afrr_price
    result["afrr_energy_bid_price_EUR_per_MWh"] = afrr_bid_price
    result["afrr_energy_market_spread_EUR_per_MWh"] = market_spread
    result["afrr_energy_net_spread_EUR_per_MWh"] = net_spread
    result["afrr_energy_cost_EUR"] = activated * afrr_price
    result["afrr_energy_savings_vs_benchmark_EUR"] = activated * market_spread
    result["afrr_energy_pay_as_cleared_reward_EUR"] = activated * market_spread
    result["afrr_energy_net_value_after_charges_EUR"] = activated * net_spread
    result["afrr_energy_capacity_backed_bid_MWh"] = capacity_backed_bid_mwh
    result["afrr_energy_free_bid_MWh"] = free_bid_mwh
    result["afrr_energy_capacity_backed_activated_MWh"] = capacity_backed_activated_mwh
    result["afrr_energy_free_activated_MWh"] = free_activated_mwh
    result["afrr_system_activation_MWh"] = afrr_system_activation
    result["afrr_headroom_binding"] = (bid_mwh > 1e-9) & (
        (bid_mwh - available_load_headroom).abs() <= 1e-7
    )
    result["afrr_curtailment_MWh"] = 0.0
    result["afrr_capacity_block_id"] = capacity_block_id
    result["afrr_capacity_block_duration_h"] = capacity_block_duration
    result["afrr_capacity_pricing_rule"] = "pay_as_bid"
    result["afrr_capacity_bid_price_EUR_per_MW_h"] = capacity_price
    result["afrr_capacity_clearing_price_EUR_per_MW_h"] = capacity_price
    result["afrr_capacity_settlement_price_EUR_per_MW_h"] = capacity_price
    result["afrr_capacity_down_price_EUR_per_MW_h"] = capacity_price
    result["afrr_capacity_reserved_MW"] = reserved_mw
    result["afrr_capacity_reserved_MWh"] = reserved_mwh
    result["afrr_capacity_revenue_EUR"] = capacity_revenue
    result["afrr_capacity_opportunity_cost_EUR"] = 0.0
    result["afrr_capacity_market_surplus_EUR"] = 0.0
    result["afrr_capacity_net_value_EUR"] = capacity_revenue
    result["reserved_capacity_headroom_MWh"] = reserved_mwh
    result["available_load_headroom_after_schedule_MWh"] = available_load_headroom
    result["DA_electricity_cost_EUR"] = da_position * da_price
    result["electricity_market_cost_EUR"] = electricity_market_cost
    result["additional_electricity_charges_cost_EUR"] = additional_electricity_charges_cost
    result["gross_operating_cost_EUR"] = gross_operating_cost
    result["net_operating_cost_EUR"] = net_operating_cost
    result["operating_cost_EUR"] = gross_operating_cost
    result["non_electric_variable_cost_EUR"] = non_electric_cost
    result["variable_cost_EUR"] = gross_operating_cost

    missing = set(STEEL_AFRR_RESULT_COLUMNS) - set(result.columns)
    assert not missing, f"rule-based steel dispatch is missing column(s): {sorted(missing)}"
    return result


def _validate_bid_rules(market_name: str, min_bid_mw: float, bid_increment_mw: float) -> None:
    if min_bid_mw < 0:
        raise ValueError(f"{market_name}.product_rules.min_bid_mw must be non-negative")
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
