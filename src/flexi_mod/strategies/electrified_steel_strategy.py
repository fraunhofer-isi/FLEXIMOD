# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Perfect-foresight German DA and aFRR-down strategy for any steel production route."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.markets.afrr_capacity import AFRRCapacityMarket
from flexi_mod.markets.afrr_energy import AFRRDownEnergyMarket
from flexi_mod.markets.day_ahead import DayAheadMarket
from flexi_mod.plants.steel_plant import SteelAFRRDownSignals, SteelPlant

IRON_ORE_PRICE_SIGNAL = "iron_ore_price"
LIME_PRICE_SIGNAL = "lime_price"
CO2_PRICE_SIGNAL = "co2_price"
NATURAL_GAS_PRICE_SIGNAL = "natural_gas_price"
HYDROGEN_PRICE_SIGNAL = "hydrogen_price"

_AFRR_ENERGY_PRICE = "__afrr_energy_price_EUR_per_MWh"
_AFRR_SYSTEM_ACTIVATION = "__afrr_system_activation_MWh"
_AFRR_PRICE_AVAILABLE = "__afrr_price_available"
_AFRR_CAPACITY_BLOCK_ID = "__afrr_capacity_block_id"
_AFRR_CAPACITY_BLOCK_DURATION = "__afrr_capacity_block_duration_h"
_AFRR_CAPACITY_PRICE = "__afrr_capacity_price_EUR_per_MW_h"
_AFRR_CAPACITY_MISSING_PRICE = "__afrr_capacity_missing_price"


class ElectrifiedSteelStrategy:
    """Cross-market next-day DA + aFRR-capacity + aFRR-energy strategy for steelmaking.

    Applies to any steel production route (BF-BOF, DRI+BOF, DRI+EAF; any fuel mix) that
    consumes electricity, not only hydrogen/electrolyser routes.
    """

    def __init__(self, config: CaseConfig):
        self.config = config
        self.afrr_energy_data_quality_summary = pd.DataFrame()
        self.afrr_capacity_block_summary = pd.DataFrame()
        self._validate_configuration()

    def required_forecast_columns(self) -> set[str]:
        return {
            self.config.market_signal("day_ahead", "price"),
            self.config.market_signal("afrr_capacity", "price"),
            self.config.market_signal("afrr_energy", "price"),
            self.config.market_signal("afrr_energy", "system_activation"),
            IRON_ORE_PRICE_SIGNAL,
            LIME_PRICE_SIGNAL,
            CO2_PRICE_SIGNAL,
            NATURAL_GAS_PRICE_SIGNAL,
            HYDROGEN_PRICE_SIGNAL,
        }

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
        capacity_data = capacity.prepare_market_data(
            forecasts,
            timestep_hours=timestep_hours,
        )
        energy_data = energy.prepare_market_data(forecasts, timestep_hours=timestep_hours)
        self.afrr_energy_data_quality_summary = energy_data.quality_summary.copy()

        prepared = forecasts.copy()
        prepared[_AFRR_ENERGY_PRICE] = energy_data.frame["afrr_energy_down_price_EUR_per_MWh"]
        prepared[_AFRR_SYSTEM_ACTIVATION] = energy_data.frame["afrr_system_activation_MWh"]
        prepared[_AFRR_PRICE_AVAILABLE] = energy_data.frame["afrr_price_available"]
        prepared[_AFRR_CAPACITY_BLOCK_ID] = capacity_data.frame["afrr_capacity_block_id"]
        prepared[_AFRR_CAPACITY_BLOCK_DURATION] = capacity_data.frame["block_duration_h"]
        prepared[_AFRR_CAPACITY_PRICE] = capacity_data.frame["capacity_price_EUR_per_MW_h"]
        prepared[_AFRR_CAPACITY_MISSING_PRICE] = capacity_data.frame["missing_capacity_price_flag"]

        energy_rules = energy.product_rules
        capacity_rules = capacity.product_rules
        signals = SteelAFRRDownSignals(
            da_price_col=day_ahead.signal_column("price"),
            afrr_energy_price_col=_AFRR_ENERGY_PRICE,
            afrr_system_activation_col=_AFRR_SYSTEM_ACTIVATION,
            afrr_price_available_col=_AFRR_PRICE_AVAILABLE,
            afrr_capacity_block_id_col=_AFRR_CAPACITY_BLOCK_ID,
            afrr_capacity_block_duration_col=_AFRR_CAPACITY_BLOCK_DURATION,
            afrr_capacity_price_col=_AFRR_CAPACITY_PRICE,
            afrr_capacity_missing_price_col=_AFRR_CAPACITY_MISSING_PRICE,
            iron_ore_price_col=IRON_ORE_PRICE_SIGNAL,
            lime_price_col=LIME_PRICE_SIGNAL,
            co2_price_col=CO2_PRICE_SIGNAL,
            natural_gas_price_col=NATURAL_GAS_PRICE_SIGNAL,
            hydrogen_price_col=HYDROGEN_PRICE_SIGNAL,
            additional_electricity_charge_eur_per_mwh=(
                plant.additional_electricity_charge_eur_per_mwh
            ),
            afrr_energy_min_bid_mw=float(energy_rules.get("min_bid_mw", 0.0)),
            afrr_energy_bid_increment_mw=float(energy_rules.get("bid_increment_mw", 1.0)),
            afrr_capacity_min_bid_mw=float(capacity_rules.get("min_bid_mw", 0.0)),
            afrr_capacity_bid_increment_mw=float(capacity_rules.get("bid_increment_mw", 1.0)),
            afrr_capacity_product_duration_h=_duration_hours(capacity.product_length),
        )
        dispatch = plant.solve_afrr_down_rolling(
            self.config,
            prepared,
            signals,
            progress_callback=progress_callback,
        )
        self.afrr_capacity_block_summary = _capacity_block_summary(dispatch)
        return dispatch

    def _validate_configuration(self) -> None:
        if self.config.country.upper() != "DE":
            raise ValueError("electrified_steel currently implements the German market design")
        if not bool(self.config.dispatch_setting("rolling_horizon_enabled", True)):
            raise ValueError("electrified_steel requires rolling_horizon_enabled=true")
        required = {"afrr_capacity", "day_ahead", "afrr_energy"}
        missing = required - set(self.config.enabled_markets)
        if missing:
            raise ValueError(
                "electrified_steel requires enabled market(s): " + ", ".join(sorted(missing))
            )
        if "intraday_continuous" in self.config.enabled_markets:
            raise ValueError("electrified_steel does not support intraday_continuous")
        sequence = [
            market
            for market in self.config.market_sequence
            if market in self.config.enabled_markets
        ]
        if sequence != ["afrr_capacity", "day_ahead", "afrr_energy"]:
            raise ValueError(
                "electrified_steel market_sequence must be: afrr_capacity, day_ahead, afrr_energy"
            )
        capacity = self.config.market("afrr_capacity")
        day_ahead = self.config.market("day_ahead")
        energy = self.config.market("afrr_energy")
        _require_calendar_gate(
            "afrr_capacity.gate_open",
            capacity.get("gate_open", {}),
            {"day_relation": "D-7", "time": "10:00"},
        )
        _require_calendar_gate(
            "afrr_capacity.gate_close",
            capacity.get("gate_close", {}),
            {"day_relation": "D-1", "time": "09:00"},
        )
        _require_calendar_gate(
            "day_ahead.gate_close",
            day_ahead.get("gate_close", {}),
            {"day_relation": "D-1", "time": "12:00"},
        )
        _require_relative_gate(
            "afrr_energy.gate_close",
            energy.get("gate_close", {}),
            -25,
        )


def _duration_hours(value: str) -> float:
    text = str(value).strip().lower()
    if text.endswith("min"):
        return float(text.removesuffix("min")) / 60.0
    if text.endswith("h"):
        return float(text.removesuffix("h"))
    raise ValueError(f"Unsupported aFRR capacity product_length '{value}'")


def _require_calendar_gate(
    label: str,
    configured: dict[str, object],
    expected: dict[str, str],
) -> None:
    relation = str(configured.get("day_relation", "")).upper()
    time = str(configured.get("time", ""))
    if len(time) == 5 and time[1] == ":":
        time = "0" + time
    if relation != expected["day_relation"] or time != expected["time"]:
        raise ValueError(
            f"electrified_steel requires German {label}="
            f"{expected['day_relation']} {expected['time']}"
        )


def _require_relative_gate(
    label: str,
    configured: dict[str, object],
    expected_minutes: int,
) -> None:
    try:
        minutes = int(configured.get("relative_to_delivery_start_minutes", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"electrified_steel has invalid {label}") from exc
    if minutes != expected_minutes:
        raise ValueError(f"electrified_steel requires German {label}={expected_minutes} minutes")


def _capacity_block_summary(dispatch: pd.DataFrame) -> pd.DataFrame:
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
