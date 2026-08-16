# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""aFRR energy market rules and data preparation, and the shared aFRR-down market layer.

This module contains direction-specific aFRR energy products. The implemented
MVP product is aFRR down energy, interpreted for the hybrid ETES case as
additional electricity consumption. aFRR up energy is represented as a clean
placeholder for later industrial cases where upward balancing energy is relevant.

It also owns the aFRR-down market layer shared across plant families: bid prices,
capacity blocks, integer bid sizing, activation, the two-trajectory feasibility check,
and the objective. ``build_afrr_down_market_model`` builds Pyomo structure directly
around a plant - it calls ``plant.afrr_attach_trajectory(...)`` to attach that plant's
own Pyomo blocks into the model - so, unlike the rest of this module, that part is not
Pyomo-free. A plant is reached through the ``AFRRDownPlant`` protocol below and never
through its own module, so the same builder drives steel and cement without either
knowing about the other.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import pandas as pd
import pyomo.environ as pyo

from flexi_mod.markets.base_market import BaseMarket, MarketConfigError


@dataclass(frozen=True)
class AFRRDownEnergyData:
    """Cleaned direction-specific aFRR down energy inputs."""

    frame: pd.DataFrame
    quality_summary: pd.DataFrame


class AFRRDownEnergyMarket(BaseMarket):
    """Configured aFRR down energy product.

    The activation signal is a system-level scenario/proxy cap, not
    plant-specific realised activation. The strategy decides whether the plant
    offers feasible bid potential; this class cleans price and activation inputs
    and validates product rules.
    """

    REQUIRED_SIGNALS = ("price", "system_activation")

    @property
    def interpretation(self) -> dict[str, object]:
        return dict(self.config.get("interpretation", {}))

    def validate_config(self, timestep_minutes: int | None = None) -> None:
        super().validate_config(timestep_minutes=timestep_minutes)
        if not self.enabled:
            return

        activation_unit = self.activation_unit
        if activation_unit not in {"MW", "MWh"}:
            raise MarketConfigError(
                "aFRR energy interpretation.activation_unit must be 'MW' or 'MWh'"
            )
        if timestep_minutes is not None:
            validity_period = int(self.product_rules.get("validity_period_minutes", 0))
            if validity_period != timestep_minutes:
                raise MarketConfigError(
                    "afrr_energy.product_rules.validity_period_minutes must match "
                    "case.timestep_minutes"
                )

    @property
    def activation_unit(self) -> str:
        return str(self.interpretation.get("activation_unit", "MW"))

    def prepare_market_data(
        self,
        forecasts: pd.DataFrame,
        timestep_hours: float,
    ) -> AFRRDownEnergyData:
        self._require_forecast_columns(forecasts)
        self.validate_config()
        return prepare_afrr_down_energy_data(
            forecasts=forecasts,
            price_col=self.signal_column("price"),
            quantity_col=self.signal_column("system_activation"),
            activation_unit=self.activation_unit,
            timestep_hours=timestep_hours,
        )


class AFRRUpEnergyMarket(BaseMarket):
    """Configured aFRR up energy placeholder.

    Upward aFRR energy would represent reduced electricity consumption or
    increased generation for many industrial cases. It is not implemented in
    the current hybrid ETES + gas case.
    """

    REQUIRED_SIGNALS = ("price", "system_activation")

    def prepare_market_data(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        if self.enabled:
            raise NotImplementedError("aFRR up energy is not implemented yet")
        return pd.DataFrame(index=forecasts.index)


def prepare_afrr_down_energy_data(
    forecasts: pd.DataFrame,
    price_col: str,
    quantity_col: str,
    activation_unit: str,
    timestep_hours: float,
) -> AFRRDownEnergyData:
    """Clean direction-specific aFRR down price and system activation data.

    Price values of zero are valid. Missing prices never create bids or
    activation; internally they are replaced by zero only to keep Pyomo
    accounting parameters numeric.
    """

    if activation_unit not in {"MW", "MWh"}:
        raise MarketConfigError("aFRR energy interpretation.activation_unit must be 'MW' or 'MWh'")

    raw_price = forecasts[price_col]
    raw_quantity = forecasts[quantity_col]
    price = pd.to_numeric(raw_price.replace("", pd.NA), errors="coerce")
    quantity = pd.to_numeric(raw_quantity.replace("", pd.NA), errors="coerce")

    missing_price = price.isna()
    missing_quantity = quantity.isna()
    negative_quantity = quantity < 0
    quantity_magnitude = quantity.abs()
    nonzero_quantity = quantity_magnitude.fillna(0.0) > 1e-12

    conversion = timestep_hours if activation_unit == "MW" else 1.0
    raw_activation_mwh = quantity_magnitude.fillna(0.0) * conversion
    clean_activation_mwh = raw_activation_mwh.copy()
    clean_activation_mwh.loc[missing_quantity] = 0.0
    clean_activation_mwh.loc[missing_price] = 0.0

    if missing_price.any():
        warnings.warn(
            "aFRR down price contains missing values. Bids and activations are set to zero "
            f"for {int(missing_price.sum())} timestep(s).",
            stacklevel=2,
        )
    if missing_quantity.any():
        warnings.warn(
            "aFRR down system activation contains missing values. Activation is set to zero "
            f"for {int(missing_quantity.sum())} timestep(s).",
            stacklevel=2,
        )

    activation_without_price = missing_price & nonzero_quantity
    clean = pd.DataFrame(
        {
            "afrr_energy_down_price_EUR_per_MWh": price.fillna(0.0),
            "afrr_price_available": ~missing_price,
            "afrr_system_activation_MWh": clean_activation_mwh.clip(lower=0.0),
            "afrr_activation_without_price": activation_without_price,
        },
        index=forecasts.index,
    )

    valid_activation = (~missing_price) & nonzero_quantity
    price_zero_with_activation = price.eq(0.0) & valid_activation

    summary = pd.DataFrame(
        [
            {
                "aFRR_down_total_rows": int(len(forecasts)),
                "aFRR_down_valid_activation_rows": int(valid_activation.sum()),
                "aFRR_down_zero_activation_rows": int((clean_activation_mwh <= 1e-12).sum()),
                "aFRR_down_missing_price_rows": int(missing_price.sum()),
                "aFRR_down_missing_quantity_rows": int(missing_quantity.sum()),
                "aFRR_down_activation_without_price_rows": int(activation_without_price.sum()),
                "aFRR_down_skipped_activation_MWh_due_to_missing_price": float(
                    raw_activation_mwh.loc[activation_without_price].sum()
                ),
                "aFRR_down_used_system_activation_MWh": float(clean_activation_mwh.sum()),
                "aFRR_down_negative_quantity_rows": int(negative_quantity.fillna(False).sum()),
                "aFRR_down_price_zero_with_activation_rows": int(price_zero_with_activation.sum()),
            }
        ]
    )
    return AFRRDownEnergyData(frame=clean, quality_summary=summary)


# --------------------------------------------------------------------------------------
# Bid-rule arithmetic: pure functions of bid sizes and product durations, shared by every
# strategy that bids into aFRR (energy or capacity) - no Pyomo, no plant coupling.
# --------------------------------------------------------------------------------------


def validate_bid_rules(market_name: str, min_bid_mw: float, bid_increment_mw: float) -> None:
    """The arithmetic of bid sizes, identical whatever is behind the meter."""
    if min_bid_mw < 0:
        raise ValueError(f"{market_name}.product_rules.min_bid_mw must be non-negative")
    if bid_increment_mw <= 0:
        raise ValueError(f"{market_name}.product_rules.bid_increment_mw must be positive")


def round_bid_down_to_increment(
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


def duration_hours(value: str) -> float:
    text = str(value).strip().lower()
    if text.endswith("min"):
        return float(text.removesuffix("min")) / 60.0
    if text.endswith("h"):
        return float(text.removesuffix("h"))
    raise ValueError(f"Unsupported aFRR capacity product_length '{value}'")


# --------------------------------------------------------------------------------------
# The plant contract
# --------------------------------------------------------------------------------------

#: Which of the two trajectories a market model builds. ``actual`` is the plan the plant
#: commits to and the only one reported; ``full_activation`` is the hypothetical "what if
#: the TSO calls the entire bid" twin, which exists solely to prove the bid deliverable.
TrajectoryRole = Literal["actual", "full_activation"]


@dataclass(frozen=True)
class AFRRWindow:
    """One rolling window, with whatever the plant family needs to rebuild its state.

    ``payload`` is opaque to the market layer: steel puts its demand target and backlog
    in there, cement its rolling state. Keeping it opaque is what lets the two families
    share a builder despite having different notions of what a window owes.
    """

    horizon: pd.DataFrame
    commit_steps: int
    is_final_window: bool
    payload: Any = None


@dataclass(frozen=True)
class FuelSubstitution:
    """A plant's breakeven electricity price against its alternative fuel.

    Above the benchmark the fuel route is cheaper; below it, electricity is.

    ``gate_column`` names a prepared boolean column when the substitution is a discrete
    on/off decision - steel's electrolyser either runs or does not. It is ``None`` when
    the plant blends continuously, as a cement stage on a hybrid fuel type does, because
    then feeding the effective price is enough for the LP to choose the split itself.

    ``gated_load_column`` names the dispatch column carrying the load that decision
    governs, used to tell which hours the benchmark actually priced. ``None`` whenever
    there is no discrete gate.
    """

    benchmark_eur_per_mwh_el: pd.Series
    gate_column: str | None = None
    gated_load_column: str | None = None


class AFRRDownPlant(Protocol):
    """The hooks a plant must offer to be driven by the shared aFRR-down market layer."""

    def afrr_attach_commodity_params(
        self,
        model: pyo.ConcreteModel,
        forecasts: pd.DataFrame,
        commodity_signals: Any,
        window: AFRRWindow,
    ) -> None:
        """Attach every root-level Param the plant's own components read.

        Must set ``model.electricity_price`` to zero: electricity is settled once at
        plant level by the market layer, so pricing it inside component operating costs
        as well would double count it.
        """

    def afrr_attach_trajectory(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block,
        role: TrajectoryRole,
        window: AFRRWindow,
        dt_hours: float,
    ) -> None:
        """Attach one complete physical trajectory to ``container``.

        Called twice per model, once per role. A family whose output constraint is an
        equality on the window total must relax it for ``full_activation`` - a different
        power draw fights it. A family constrained by a per-timestep floor should keep
        it on both, since a higher draw cannot violate a floor and dropping it would let
        the twin absorb the bid while producing nothing.
        """

    def afrr_attach_terminal_state(self, model: pyo.ConcreteModel, final_t: int) -> None:
        """Close inventories on ``model.actual`` at the end of the simulation.

        Without it the last window can profit by draining a store it never refills.
        """

    def afrr_fuel_substitution(self, forecasts: pd.DataFrame) -> FuelSubstitution | None:
        """The plant's electricity-versus-fuel breakeven, or ``None`` if it has none.

        Only the plant knows whether a fuel choice exists at all, and what the two routes
        cost per MWh of electricity displaced.
        """

    def afrr_aggregate_max_power_mw(self) -> float:
        """Greatest electric power the plant can draw at once.

        Bounds the bid variables, sets the activation big-M and feeds the reported
        headroom. Over-estimating is safe; under-estimating silently caps every bid.
        """


# --------------------------------------------------------------------------------------
# Signals: where the market layer finds its inputs, and the product rules it respects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AFRRDownMarketSignals:
    """Forecast columns and bid rules for day-ahead plus aFRR-down participation.

    Purely about buying electricity, so it is identical whatever the plant makes. Each
    plant family keeps its own commodity-column record alongside this one - iron ore and
    lime for steel, coal for cement - and the market layer never sees it.
    """

    da_price_col: str
    afrr_energy_price_col: str
    afrr_system_activation_col: str
    afrr_price_available_col: str
    afrr_capacity_block_id_col: str
    afrr_capacity_block_duration_col: str
    afrr_capacity_price_col: str
    afrr_capacity_missing_price_col: str
    #: Levied on metered consumption, so it belongs to the electricity settlement rather
    #: than to any one market leg.
    additional_electricity_charge_eur_per_mwh: float = 0.0
    afrr_energy_min_bid_mw: float = 1.0
    afrr_energy_bid_increment_mw: float = 1.0
    afrr_capacity_min_bid_mw: float = 1.0
    afrr_capacity_bid_increment_mw: float = 1.0
    afrr_capacity_product_duration_h: float = 4.0


# --------------------------------------------------------------------------------------
# Result ontology: what a market dispatch frame carries on top of its plant-specific
# columns. Describes the market, not the process behind the meter, so the same schema
# serves steel, cement and anything added later.
# --------------------------------------------------------------------------------------


def _value(expression: Any) -> float:
    return float(pyo.value(expression))


AFRR_DOWN_RESULT_COLUMNS = (
    "DA_position_MWh",
    "IDC_buy_MWh",
    "IDC_sell_MWh",
    "final_planned_electricity_MWh",
    "actual_electricity_consumption_MWh",
    "day_ahead_price_EUR_per_MWh",
    "day_ahead_delivered_price_EUR_per_MWh",
    "additional_electricity_charge_EUR_per_MWh_el",
    "afrr_energy_bid_MW",
    "afrr_energy_bid_MWh",
    "afrr_energy_activated_MWh",
    "afrr_energy_price_EUR_per_MWh",
    "afrr_energy_delivered_price_EUR_per_MWh",
    "afrr_energy_bid_price_EUR_per_MWh",
    "afrr_energy_market_spread_EUR_per_MWh",
    "afrr_energy_net_spread_EUR_per_MWh",
    "afrr_energy_cost_EUR",
    "afrr_energy_savings_vs_benchmark_EUR",
    "afrr_energy_pay_as_cleared_reward_EUR",
    "afrr_energy_net_value_after_charges_EUR",
    "afrr_energy_capacity_backed_bid_MWh",
    "afrr_energy_free_bid_MWh",
    "afrr_energy_capacity_backed_activated_MWh",
    "afrr_energy_free_activated_MWh",
    "afrr_system_activation_MWh",
    "afrr_headroom_binding",
    "afrr_curtailment_MWh",
    "afrr_capacity_block_id",
    "afrr_capacity_block_duration_h",
    "afrr_capacity_pricing_rule",
    "afrr_capacity_bid_price_EUR_per_MW_h",
    "afrr_capacity_clearing_price_EUR_per_MW_h",
    "afrr_capacity_settlement_price_EUR_per_MW_h",
    "afrr_capacity_down_price_EUR_per_MW_h",
    "afrr_capacity_reserved_MW",
    "afrr_capacity_reserved_MWh",
    "afrr_capacity_revenue_EUR",
    "afrr_capacity_opportunity_cost_EUR",
    "afrr_capacity_market_surplus_EUR",
    "afrr_capacity_net_value_EUR",
    "reserved_capacity_headroom_MWh",
    "available_load_headroom_after_schedule_MWh",
    "DA_electricity_cost_EUR",
    "electricity_market_cost_EUR",
    "additional_electricity_charges_cost_EUR",
    "gross_operating_cost_EUR",
    "net_operating_cost_EUR",
    "operating_cost_EUR",
    "non_electric_variable_cost_EUR",
)


def append_afrr_result_row(
    data: dict[str, list[float] | list[str]],
    model: pyo.ConcreteModel,
    t: int,
    total_electricity_mwh: float,
) -> None:
    block_id = model._afrr_block_by_t[t]
    timestep_hours = float(model._afrr_timestep_hours)
    da_position = _value(model.da_position_mwh[t])
    afrr_bid = _value(model.afrr_energy_bid_mwh[t])
    afrr_activation = _value(model.afrr_energy_activated_mwh[t])
    capacity_bid = _value(model.afrr_energy_capacity_backed_bid_mwh[t])
    free_bid = _value(model.afrr_energy_free_bid_mwh[t])
    capacity_activated = _value(model.afrr_energy_capacity_backed_activated_mwh[t])
    free_activated = _value(model.afrr_energy_free_activated_mwh[t])
    day_ahead_price = _value(model.day_ahead_price[t])
    afrr_price = _value(model.afrr_energy_price[t])
    additional_charge = _value(model.additional_electricity_charge)
    afrr_bid_price = _value(model.afrr_energy_bid_price[t])
    delivered_afrr_price = afrr_price + additional_charge
    market_spread = afrr_bid_price - afrr_price
    net_spread = afrr_bid_price - delivered_afrr_price
    capacity_price = _value(model.afrr_capacity_price[block_id])
    reserved_mw = _value(model.afrr_capacity_reserved_mw[block_id])
    reserved_mwh = reserved_mw * timestep_hours
    capacity_revenue = _value(model.afrr_capacity_revenue[t])
    gross_cost = _value(model.gross_operating_cost[t])
    non_electric_cost = _value(model.actual.variable_cost[t])
    available_load_headroom = max(
        0.0,
        float(model._afrr_aggregate_max_power_mw) * timestep_hours - da_position,
    )

    values: dict[str, float | str | bool] = {
        "DA_position_MWh": da_position,
        "IDC_buy_MWh": 0.0,
        "IDC_sell_MWh": 0.0,
        "final_planned_electricity_MWh": da_position,
        "actual_electricity_consumption_MWh": total_electricity_mwh,
        "day_ahead_price_EUR_per_MWh": day_ahead_price,
        "day_ahead_delivered_price_EUR_per_MWh": day_ahead_price + additional_charge,
        "additional_electricity_charge_EUR_per_MWh_el": additional_charge,
        "afrr_energy_bid_MW": afrr_bid / timestep_hours,
        "afrr_energy_bid_MWh": afrr_bid,
        "afrr_energy_activated_MWh": afrr_activation,
        "afrr_energy_price_EUR_per_MWh": afrr_price,
        "afrr_energy_delivered_price_EUR_per_MWh": delivered_afrr_price,
        "afrr_energy_bid_price_EUR_per_MWh": afrr_bid_price,
        "afrr_energy_market_spread_EUR_per_MWh": market_spread,
        "afrr_energy_net_spread_EUR_per_MWh": net_spread,
        "afrr_energy_cost_EUR": afrr_activation * afrr_price,
        "afrr_energy_savings_vs_benchmark_EUR": afrr_activation * market_spread,
        "afrr_energy_pay_as_cleared_reward_EUR": afrr_activation * market_spread,
        "afrr_energy_net_value_after_charges_EUR": afrr_activation * net_spread,
        "afrr_energy_capacity_backed_bid_MWh": capacity_bid,
        "afrr_energy_free_bid_MWh": free_bid,
        "afrr_energy_capacity_backed_activated_MWh": capacity_activated,
        "afrr_energy_free_activated_MWh": free_activated,
        "afrr_system_activation_MWh": _value(model.afrr_system_activation_mwh[t]),
        "afrr_headroom_binding": bool(
            afrr_bid > 1e-9 and abs(afrr_bid - available_load_headroom) <= 1e-7
        ),
        "afrr_curtailment_MWh": 0.0,
        "afrr_capacity_block_id": block_id,
        "afrr_capacity_block_duration_h": _value(model.afrr_capacity_block_duration[block_id]),
        "afrr_capacity_pricing_rule": "pay_as_bid",
        "afrr_capacity_bid_price_EUR_per_MW_h": capacity_price,
        "afrr_capacity_clearing_price_EUR_per_MW_h": capacity_price,
        "afrr_capacity_settlement_price_EUR_per_MW_h": capacity_price,
        "afrr_capacity_down_price_EUR_per_MW_h": capacity_price,
        "afrr_capacity_reserved_MW": reserved_mw,
        "afrr_capacity_reserved_MWh": reserved_mwh,
        "afrr_capacity_revenue_EUR": capacity_revenue,
        "afrr_capacity_opportunity_cost_EUR": 0.0,
        "afrr_capacity_market_surplus_EUR": 0.0,
        "afrr_capacity_net_value_EUR": capacity_revenue,
        "reserved_capacity_headroom_MWh": reserved_mwh,
        "available_load_headroom_after_schedule_MWh": available_load_headroom,
        "DA_electricity_cost_EUR": da_position * day_ahead_price,
        "electricity_market_cost_EUR": _value(model.electricity_market_cost[t]),
        "additional_electricity_charges_cost_EUR": _value(
            model.additional_electricity_charges_cost[t]
        ),
        "gross_operating_cost_EUR": gross_cost,
        "net_operating_cost_EUR": _value(model.net_operating_cost[t]),
        "operating_cost_EUR": gross_cost,
        "non_electric_variable_cost_EUR": non_electric_cost,
    }
    for column, value in values.items():
        data[column].append(value)


def attach_capacity_opportunity_cost(
    market_result: pd.DataFrame,
    no_capacity_result: pd.DataFrame,
    commit_steps: int,
) -> pd.DataFrame:
    """Allocate cross-market capacity opportunity cost over reserved MW-hours."""

    result = market_result.copy()
    committed_result = result.iloc[:commit_steps]
    committed_baseline = no_capacity_result.iloc[:commit_steps]
    incremental_gross_cost = max(
        0.0,
        float(committed_result["gross_operating_cost_EUR"].sum())
        - float(committed_baseline["gross_operating_cost_EUR"].sum()),
    )
    weights = committed_result["afrr_capacity_reserved_MWh"].clip(lower=0.0)
    total_weight = float(weights.sum())
    result["afrr_capacity_opportunity_cost_EUR"] = 0.0
    if total_weight > 1e-12:
        result.loc[weights.index, "afrr_capacity_opportunity_cost_EUR"] = (
            incremental_gross_cost * weights / total_weight
        )
    result["afrr_capacity_net_value_EUR"] = (
        result["afrr_capacity_revenue_EUR"] - result["afrr_capacity_opportunity_cost_EUR"]
    )
    return result


def capacity_block_summary(dispatch: pd.DataFrame) -> pd.DataFrame:
    """Per-capacity-block reporting, derived from a finished dispatch frame.

    Reads only aFRR columns, so it is independent of the plant family that produced them.
    """
    records: list[dict[str, object]] = []
    for block_id, block in dispatch.groupby("afrr_capacity_block_id", sort=False):
        reserved_mw = float(block["afrr_capacity_reserved_MW"].max())
        duration_h = float(block["afrr_capacity_block_duration_h"].iloc[0])
        opportunity_cost = float(block["afrr_capacity_opportunity_cost_EUR"].sum())
        reserved_mw_h = reserved_mw * duration_h
        available_load_mw = float(
            (block["available_load_headroom_after_schedule_MWh"] / _timestep_hours(block)).min()
        )
        records.append(
            {
                "block_id": str(block_id),
                "block_start": pd.Timestamp(block.index.min()),
                "block_end": pd.Timestamp(block.index.max())
                + pd.Timedelta(hours=_timestep_hours(block)),
                "block_duration_h": duration_h,
                "capacity_pricing_rule": "pay_as_bid",
                "opportunity_cost_EUR_per_MW_h": (
                    opportunity_cost / reserved_mw_h if reserved_mw_h > 1e-12 else 0.0
                ),
                "capacity_bid_price_EUR_per_MW_h": float(
                    block["afrr_capacity_bid_price_EUR_per_MW_h"].iloc[0]
                ),
                "capacity_clearing_price_EUR_per_MW_h": float(
                    block["afrr_capacity_clearing_price_EUR_per_MW_h"].iloc[0]
                ),
                "capacity_settlement_price_EUR_per_MW_h": float(
                    block["afrr_capacity_settlement_price_EUR_per_MW_h"].iloc[0]
                ),
                "capacity_profitable": bool(
                    reserved_mw > 1e-12
                    and float(block["afrr_capacity_net_value_EUR"].sum()) >= -1e-8
                ),
                "technically_feasible": bool(reserved_mw > 1e-12),
                "bid_eligible": bool(reserved_mw > 1e-12),
                "technical_capacity_MW": available_load_mw,
                "compliant_capacity_MW": reserved_mw,
                "reserved_capacity_MW": reserved_mw,
                "capacity_revenue_EUR": float(block["afrr_capacity_revenue_EUR"].sum()),
                "capacity_opportunity_cost_EUR": opportunity_cost,
                "capacity_market_surplus_EUR": float(
                    block["afrr_capacity_market_surplus_EUR"].sum()
                ),
                "capacity_net_value_EUR": float(block["afrr_capacity_net_value_EUR"].sum()),
                "activated_energy_MWh": float(
                    block["afrr_energy_capacity_backed_activated_MWh"].sum()
                ),
                "activation_cost_EUR": float(
                    (
                        block["afrr_energy_capacity_backed_activated_MWh"]
                        * block["afrr_energy_price_EUR_per_MWh"]
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(records)


def _timestep_hours(frame: pd.DataFrame) -> float:
    if len(frame.index) > 1:
        return float((frame.index[1] - frame.index[0]).total_seconds() / 3600.0)
    return float(frame["afrr_capacity_reserved_MWh"].iloc[0]) / max(
        float(frame["afrr_capacity_reserved_MW"].iloc[0]),
        1.0,
    )


# --------------------------------------------------------------------------------------
# The market model builder
# --------------------------------------------------------------------------------------


def build_afrr_down_market_model(
    plant: AFRRDownPlant,
    forecasts: pd.DataFrame,
    market_signals: AFRRDownMarketSignals,
    commodity_signals: Any,
    window: AFRRWindow,
    *,
    dt_hours: float,
    capacity_enabled: bool,
) -> pyo.ConcreteModel:
    """Assemble a day-ahead plus aFRR-down market model around *plant*.

    ``capacity_enabled=False`` builds the otherwise identical baseline used to price the
    capacity commitment's opportunity cost.

    The phases below run in a fixed order, each attaching one coherent group of Pyomo
    Components to ``model``: market prices and the capacity-block catalogue, integer bid
    sizing, the linearised activation logic, the plant's two physical trajectories, and
    finally the cost expressions and objective. Two values computed early
    (``block_by_t``, ``aggregate_max_power_mw``) are needed again in a later phase, so
    they are stashed on ``model`` as soon as they exist rather than threaded through
    every function signature between producer and consumer.
    """
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=range(len(forecasts)), ordered=True)
    model._afrr_timestep_hours = dt_hours

    _define_market_parameters(
        model,
        plant,
        forecasts,
        market_signals,
        commodity_signals,
        window,
        dt_hours,
        capacity_enabled,
    )
    _define_bid_sizing(model, plant, market_signals, dt_hours)
    _define_activation_linearization(model, forecasts, market_signals, dt_hours)
    _attach_trajectories(model, plant, window, dt_hours)
    _define_costs_and_objective(model, dt_hours)
    return model


def _define_market_parameters(
    model: pyo.ConcreteModel,
    plant: AFRRDownPlant,
    forecasts: pd.DataFrame,
    market_signals: AFRRDownMarketSignals,
    commodity_signals: Any,
    window: AFRRWindow,
    dt_hours: float,
    capacity_enabled: bool,
) -> None:
    """Pyomo Components:

    - **Parameters**: the plant's own commodity prices (attached first, so Pyomo's
      declaration order matches the model this was originally built as), day-ahead and
      aFRR-energy prices, the reverse-cummax bid price and free-bid eligibility gate, and
      the capacity-block catalogue (price, duration, availability) keyed by ``model.B``.
    """

    def values(column: str) -> dict[int, float]:
        return {t: float(forecasts[column].iloc[t]) for t in model.T}

    plant.afrr_attach_commodity_params(model, forecasts, commodity_signals, window)

    model.day_ahead_price = pyo.Param(model.T, initialize=values(market_signals.da_price_col))
    model.afrr_energy_price = pyo.Param(
        model.T, initialize=values(market_signals.afrr_energy_price_col)
    )
    model.afrr_system_activation_mwh = pyo.Param(
        model.T, initialize=values(market_signals.afrr_system_activation_col)
    )
    additional_charge = float(market_signals.additional_electricity_charge_eur_per_mwh)
    model.additional_electricity_charge = pyo.Param(initialize=additional_charge)

    delivered_da = forecasts[market_signals.da_price_col].astype(float) + additional_charge
    bid_price = delivered_da.iloc[::-1].cummax().iloc[::-1]
    delivered_afrr = (
        forecasts[market_signals.afrr_energy_price_col].astype(float) + additional_charge
    )
    price_available = forecasts[market_signals.afrr_price_available_col].astype(bool)
    free_bid_allowed = price_available & (delivered_afrr <= bid_price + 1e-9)
    model.afrr_energy_bid_price = pyo.Param(
        model.T, initialize={t: float(bid_price.iloc[t]) for t in model.T}
    )
    model.free_bid_allowed = pyo.Param(
        model.T,
        initialize={t: int(bool(free_bid_allowed.iloc[t])) for t in model.T},
        within=pyo.Binary,
    )

    block_ids = list(
        dict.fromkeys(forecasts[market_signals.afrr_capacity_block_id_col].astype(str))
    )
    model.B = pyo.Set(initialize=block_ids, ordered=True)
    block_by_t = {
        t: str(forecasts[market_signals.afrr_capacity_block_id_col].iloc[t]) for t in model.T
    }
    block_prices = {
        block_id: float(
            forecasts.loc[
                forecasts[market_signals.afrr_capacity_block_id_col].astype(str) == block_id,
                market_signals.afrr_capacity_price_col,
            ].iloc[0]
        )
        for block_id in block_ids
    }
    block_durations = {
        block_id: float(
            forecasts.loc[
                forecasts[market_signals.afrr_capacity_block_id_col].astype(str) == block_id,
                market_signals.afrr_capacity_block_duration_col,
            ].iloc[0]
        )
        for block_id in block_ids
    }
    block_available = {}
    for block_id in block_ids:
        mask = forecasts[market_signals.afrr_capacity_block_id_col].astype(str) == block_id
        missing = bool(forecasts.loc[mask, market_signals.afrr_capacity_missing_price_col].iloc[0])
        complete = (
            abs(float(mask.sum()) * dt_hours - market_signals.afrr_capacity_product_duration_h)
            < 1e-8
        )
        block_available[block_id] = int(
            capacity_enabled and not missing and complete and block_prices[block_id] > 0.0
        )

    model.afrr_capacity_price = pyo.Param(model.B, initialize=block_prices)
    model.afrr_capacity_block_duration = pyo.Param(model.B, initialize=block_durations)
    model.capacity_available = pyo.Param(model.B, initialize=block_available, within=pyo.Binary)
    model._afrr_block_by_t = block_by_t


def _define_bid_sizing(
    model: pyo.ConcreteModel,
    plant: AFRRDownPlant,
    market_signals: AFRRDownMarketSignals,
    dt_hours: float,
) -> None:
    """Pyomo Components:

    - **Variables**: integer bid-step counts for the capacity product
      (``capacity_bid_steps``) and the free energy bid (``free_bid_steps``), each gated
      by its own selection binary.
    - **Constraints**: minimum-bid, selection-limit, and (for capacity) availability
      rules tying the step counts to their selection binaries; the free bid additionally
      gated by ``free_energy_price_gate``.
    """
    aggregate_max_power_mw = plant.afrr_aggregate_max_power_mw()
    capacity_max_steps = int(
        math.floor(aggregate_max_power_mw / market_signals.afrr_capacity_bid_increment_mw + 1e-9)
    )
    energy_max_steps = int(
        math.floor(aggregate_max_power_mw / market_signals.afrr_energy_bid_increment_mw + 1e-9)
    )
    capacity_min_steps = int(
        math.ceil(
            market_signals.afrr_capacity_min_bid_mw / market_signals.afrr_capacity_bid_increment_mw
        )
    )
    energy_min_steps = int(
        math.ceil(
            market_signals.afrr_energy_min_bid_mw / market_signals.afrr_energy_bid_increment_mw
        )
    )

    model.capacity_selected = pyo.Var(model.B, within=pyo.Binary)
    model.capacity_bid_steps = pyo.Var(
        model.B, within=pyo.NonNegativeIntegers, bounds=(0, capacity_max_steps)
    )
    model.afrr_capacity_reserved_mw = pyo.Expression(
        model.B,
        rule=lambda m, b: market_signals.afrr_capacity_bid_increment_mw * m.capacity_bid_steps[b],
    )

    @model.Constraint(model.B)
    def capacity_minimum_bid(m: pyo.ConcreteModel, b: str) -> pyo.Constraint:
        return m.capacity_bid_steps[b] >= capacity_min_steps * m.capacity_selected[b]

    @model.Constraint(model.B)
    def capacity_selection_limit(m: pyo.ConcreteModel, b: str) -> pyo.Constraint:
        return m.capacity_bid_steps[b] <= capacity_max_steps * m.capacity_selected[b]

    @model.Constraint(model.B)
    def capacity_availability(m: pyo.ConcreteModel, b: str) -> pyo.Constraint:
        return m.capacity_selected[b] <= m.capacity_available[b]

    model.free_bid_selected = pyo.Var(model.T, within=pyo.Binary)
    model.free_bid_steps = pyo.Var(
        model.T, within=pyo.NonNegativeIntegers, bounds=(0, energy_max_steps)
    )
    model.afrr_energy_free_bid_mw = pyo.Expression(
        model.T,
        rule=lambda m, t: market_signals.afrr_energy_bid_increment_mw * m.free_bid_steps[t],
    )
    model.afrr_energy_capacity_backed_bid_mwh = pyo.Expression(
        model.T,
        rule=lambda m, t: m.afrr_capacity_reserved_mw[m._afrr_block_by_t[t]] * dt_hours,
    )
    model.afrr_energy_free_bid_mwh = pyo.Expression(
        model.T, rule=lambda m, t: m.afrr_energy_free_bid_mw[t] * dt_hours
    )
    model.afrr_energy_bid_mwh = pyo.Expression(
        model.T,
        rule=lambda m, t: m.afrr_energy_capacity_backed_bid_mwh[t] + m.afrr_energy_free_bid_mwh[t],
    )

    @model.Constraint(model.T)
    def free_energy_minimum_bid(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.free_bid_steps[t] >= energy_min_steps * m.free_bid_selected[t]

    @model.Constraint(model.T)
    def free_energy_selection_limit(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.free_bid_steps[t] <= energy_max_steps * m.free_bid_selected[t]

    @model.Constraint(model.T)
    def free_energy_price_gate(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.free_bid_selected[t] <= m.free_bid_allowed[t]

    model._afrr_aggregate_max_power_mw = aggregate_max_power_mw


def _define_activation_linearization(
    model: pyo.ConcreteModel,
    forecasts: pd.DataFrame,
    market_signals: AFRRDownMarketSignals,
    dt_hours: float,
) -> None:
    """Pyomo Components:

    - **Variables**: total activation (``afrr_energy_activated_mwh``) and the
      capacity-backed share of it (``afrr_energy_capacity_backed_activated_mwh``), each
      with a big-M selection binary linearising a ``min()``.
    - **Constraints**: the two linearised-min blocks - activation against the bid and
      the system request, then the capacity-backed share against the capacity-backed bid
      and total activation.
    """
    max_activation_mwh = max(
        float(model._afrr_aggregate_max_power_mw) * dt_hours,
        float(forecasts[market_signals.afrr_system_activation_col].max()),
        1.0,
    )
    model.afrr_energy_activated_mwh = pyo.Var(
        model.T, within=pyo.NonNegativeReals, bounds=(0.0, max_activation_mwh)
    )
    model.activation_bid_is_minimum = pyo.Var(model.T, within=pyo.Binary)

    @model.Constraint(model.T)
    def activation_not_above_bid(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.afrr_energy_activated_mwh[t] <= m.afrr_energy_bid_mwh[t]

    @model.Constraint(model.T)
    def activation_not_above_request(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.afrr_energy_activated_mwh[t] <= m.afrr_system_activation_mwh[t]

    @model.Constraint(model.T)
    def activation_equals_bid_when_smaller(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.afrr_energy_activated_mwh[t] >= (
            m.afrr_energy_bid_mwh[t] - max_activation_mwh * m.activation_bid_is_minimum[t]
        )

    @model.Constraint(model.T)
    def activation_equals_request_when_smaller(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.afrr_energy_activated_mwh[t] >= (
            m.afrr_system_activation_mwh[t]
            - max_activation_mwh * (1 - m.activation_bid_is_minimum[t])
        )

    model.afrr_energy_capacity_backed_activated_mwh = pyo.Var(
        model.T, within=pyo.NonNegativeReals, bounds=(0.0, max_activation_mwh)
    )
    model.capacity_activation_is_minimum = pyo.Var(model.T, within=pyo.Binary)

    @model.Constraint(model.T)
    def capacity_activation_not_above_bid(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return (
            m.afrr_energy_capacity_backed_activated_mwh[t]
            <= m.afrr_energy_capacity_backed_bid_mwh[t]
        )

    @model.Constraint(model.T)
    def capacity_activation_not_above_total(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.afrr_energy_capacity_backed_activated_mwh[t] <= m.afrr_energy_activated_mwh[t]

    @model.Constraint(model.T)
    def capacity_activation_equals_bid_when_smaller(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.afrr_energy_capacity_backed_activated_mwh[t] >= (
            m.afrr_energy_capacity_backed_bid_mwh[t]
            - max_activation_mwh * m.capacity_activation_is_minimum[t]
        )

    @model.Constraint(model.T)
    def capacity_activation_equals_total_when_smaller(
        m: pyo.ConcreteModel, t: int
    ) -> pyo.Constraint:
        return m.afrr_energy_capacity_backed_activated_mwh[t] >= (
            m.afrr_energy_activated_mwh[t]
            - max_activation_mwh * (1 - m.capacity_activation_is_minimum[t])
        )

    model.afrr_energy_free_activated_mwh = pyo.Expression(
        model.T,
        rule=lambda m, t: (
            m.afrr_energy_activated_mwh[t] - m.afrr_energy_capacity_backed_activated_mwh[t]
        ),
    )


def _attach_trajectories(
    model: pyo.ConcreteModel,
    plant: AFRRDownPlant,
    window: AFRRWindow,
    dt_hours: float,
) -> None:
    """Pyomo Components:

    - **Blocks**: ``actual`` (the committed trajectory) and ``full_activation`` (a
      hypothetical "what if the whole bid is called" twin that is never realised - only
      ``actual`` feeds results and the rolling state - existing purely to prove the bid
      physically deliverable), each populated by the plant's own physics.
    - **Constraints**: the electricity balance tying each trajectory's power draw to the
      day-ahead position plus its own activation or bid.
    """
    model.da_position_mwh = pyo.Var(model.T, within=pyo.NonNegativeReals)

    model.actual = pyo.Block()
    model.full_activation = pyo.Block()
    plant.afrr_attach_trajectory(model, model.actual, "actual", window, dt_hours)
    plant.afrr_attach_trajectory(model, model.full_activation, "full_activation", window, dt_hours)

    if window.is_final_window:
        plant.afrr_attach_terminal_state(model, list(model.T)[-1])

    @model.Constraint(model.T)
    def actual_electricity_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.actual.total_power_input[t] == (
            m.da_position_mwh[t] + m.afrr_energy_activated_mwh[t]
        )

    @model.Constraint(model.T)
    def full_activation_electricity_balance(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        return m.full_activation.total_power_input[t] == (
            m.da_position_mwh[t] + m.afrr_energy_bid_mwh[t]
        )


def _define_costs_and_objective(model: pyo.ConcreteModel, dt_hours: float) -> None:
    """Pyomo Components:

    - **Expressions**: the market-settled electricity cost, additional-charge cost,
      gross and net operating cost, and capacity revenue.
    - **Objective**: minimise net operating cost, plus a negligible tiebreaker that
      favours larger bids among otherwise-equal solutions.
    """
    model.electricity_market_cost = pyo.Expression(
        model.T,
        rule=lambda m, t: (
            m.da_position_mwh[t] * m.day_ahead_price[t]
            + m.afrr_energy_activated_mwh[t] * m.afrr_energy_price[t]
        ),
    )
    model.additional_electricity_charges_cost = pyo.Expression(
        model.T,
        rule=lambda m, t: m.actual.total_power_input[t] * m.additional_electricity_charge,
    )
    model.gross_operating_cost = pyo.Expression(
        model.T,
        rule=lambda m, t: (
            m.actual.variable_cost[t]
            + m.electricity_market_cost[t]
            + m.additional_electricity_charges_cost[t]
        ),
    )
    model.afrr_capacity_revenue = pyo.Expression(
        model.T,
        rule=lambda m, t: (
            m.afrr_capacity_reserved_mw[m._afrr_block_by_t[t]]
            * m.afrr_capacity_price[m._afrr_block_by_t[t]]
            * dt_hours
        ),
    )
    model.net_operating_cost = pyo.Expression(
        model.T,
        rule=lambda m, t: m.gross_operating_cost[t] - m.afrr_capacity_revenue[t],
    )
    bid_tiebreaker = 1e-6 * (
        sum(model.capacity_bid_steps[b] for b in model.B)
        + sum(model.free_bid_steps[t] for t in model.T)
    )
    model.objective = pyo.Objective(
        expr=sum(model.net_operating_cost[t] for t in model.T) + bid_tiebreaker,
        sense=pyo.minimize,
    )
