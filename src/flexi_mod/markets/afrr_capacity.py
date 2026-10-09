# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""aFRR capacity market rules and block data preparation.

The first implemented capacity product is downward aFRR capacity. It reserves
the ability to increase electricity consumption in fixed product blocks. For
the current German-style setup, blocks are generated internally from the
forecast datetime index, anchored at midnight.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Final

import pandas as pd

from flexi_mod.markets.base_market import BaseMarket, MarketConfigError, MarketResultKind

PRICE_CONSISTENCY_TOLERANCE = 1e-6
SUPPORTED_PRICE_UNITS = {"EUR_per_MW_per_h", "EUR_per_MW_per_product"}


AFRR_CAPACITY_AWARD_RESULT_COLUMNS: Final[tuple[str, ...]] = (
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
)


@dataclass(frozen=True)
class BalancingCapacityAward:
    """A time-aligned balancing-capacity award and its settlement accounting.

    This is a market outcome, rather than a plant or technology object.  It
    deliberately contains no information about storage, boilers, heat demand,
    or another plant's physical feasibility.  A plant may consume
    :meth:`reserved_energy_mwh` to enforce delivery capability, while output
    code may use :meth:`result_frame` to append the common market fields.

    ``reserved_mw`` is the awarded capacity magnitude.  The corresponding
    timestep energy is derived from it, rather than stored independently, so
    MW and MWh cannot become inconsistent.  All optional series are aligned
    by timestamp when read; missing accounting values use the conventional
    settlement identities documented in :meth:`result_frame`.
    """

    reserved_mw: pd.Series
    block_id: pd.Series | None = None
    block_duration_h: pd.Series | None = None
    pricing_rule: pd.Series | None = None
    bid_price_eur_per_mw_h: pd.Series | None = None
    clearing_price_eur_per_mw_h: pd.Series | None = None
    settlement_price_eur_per_mw_h: pd.Series | None = None
    revenue_eur: pd.Series | None = None
    opportunity_cost_eur: pd.Series | None = None
    market_surplus_eur: pd.Series | None = None
    net_value_eur: pd.Series | None = None

    def __post_init__(self) -> None:
        """Reject malformed market data at the boundary of the domain object."""

        _validate_numeric_series(self.reserved_mw, "reserved_mw", non_negative=True)
        for name, values in (
            ("block_id", self.block_id),
            ("block_duration_h", self.block_duration_h),
            ("pricing_rule", self.pricing_rule),
            ("bid_price_eur_per_mw_h", self.bid_price_eur_per_mw_h),
            ("clearing_price_eur_per_mw_h", self.clearing_price_eur_per_mw_h),
            ("settlement_price_eur_per_mw_h", self.settlement_price_eur_per_mw_h),
            ("revenue_eur", self.revenue_eur),
            ("opportunity_cost_eur", self.opportunity_cost_eur),
            ("market_surplus_eur", self.market_surplus_eur),
            ("net_value_eur", self.net_value_eur),
        ):
            if values is not None and not isinstance(values, pd.Series):
                raise TypeError(f"BalancingCapacityAward.{name} must be a pandas Series")

        _validate_optional_numeric_series(
            self.block_duration_h,
            "block_duration_h",
            non_negative=True,
        )
        for name, values in (
            ("bid_price_eur_per_mw_h", self.bid_price_eur_per_mw_h),
            ("clearing_price_eur_per_mw_h", self.clearing_price_eur_per_mw_h),
            ("settlement_price_eur_per_mw_h", self.settlement_price_eur_per_mw_h),
            ("revenue_eur", self.revenue_eur),
            ("opportunity_cost_eur", self.opportunity_cost_eur),
            ("market_surplus_eur", self.market_surplus_eur),
            ("net_value_eur", self.net_value_eur),
        ):
            _validate_optional_numeric_series(values, name)

    @classmethod
    def empty(cls, index: pd.Index) -> BalancingCapacityAward:
        """Return an explicit no-award object for a delivery index."""

        return cls(reserved_mw=pd.Series(0.0, index=index, dtype=float))

    def reserved_energy_mwh(
        self,
        index: pd.Index,
        timestep_hours: float,
    ) -> pd.Series:
        """Return the awarded capacity expressed as energy in each timestep."""

        _validate_timestep_hours(timestep_hours)
        return (
            _aligned_numeric_series(
                self.reserved_mw,
                index,
                "reserved_mw",
                default=0.0,
            )
            * timestep_hours
        )

    def result_frame(
        self,
        index: pd.Index,
        timestep_hours: float,
    ) -> pd.DataFrame:
        """Return stable aFRR-capacity output fields for each delivery timestamp.

        When an accounting series is absent, this method derives it using the
        standard timestep settlement identities:

        * revenue = awarded MW × settlement price × timestep hours;
        * market surplus = awarded MW × (settlement - bid price) × timestep
          hours; and
        * net value = revenue - opportunity cost.

        Explicit accounting series override the derived values.  This supports
        external market-clearing data while preserving a single, reusable
        output schema for strategies and plant result mappers.
        """

        _validate_timestep_hours(timestep_hours)
        reserved_mw = _aligned_numeric_series(
            self.reserved_mw,
            index,
            "reserved_mw",
            default=0.0,
        )
        reserved_mwh = reserved_mw * timestep_hours
        block_duration_h = _aligned_numeric_series(
            self.block_duration_h,
            index,
            "block_duration_h",
            default=0.0,
        )
        bid_price = _aligned_numeric_series(
            self.bid_price_eur_per_mw_h,
            index,
            "bid_price_eur_per_mw_h",
            default=0.0,
        )
        clearing_price = _aligned_numeric_series(
            self.clearing_price_eur_per_mw_h,
            index,
            "clearing_price_eur_per_mw_h",
            default=0.0,
        )
        settlement_price = _aligned_numeric_series(
            self.settlement_price_eur_per_mw_h,
            index,
            "settlement_price_eur_per_mw_h",
            default=float("nan"),
        ).fillna(clearing_price)

        derived_revenue = reserved_mw * settlement_price * timestep_hours
        revenue = _aligned_numeric_series(
            self.revenue_eur,
            index,
            "revenue_eur",
            default=float("nan"),
        ).fillna(derived_revenue)
        opportunity_cost = _aligned_numeric_series(
            self.opportunity_cost_eur,
            index,
            "opportunity_cost_eur",
            default=0.0,
        )
        derived_surplus = reserved_mw * (settlement_price - bid_price) * timestep_hours
        market_surplus = _aligned_numeric_series(
            self.market_surplus_eur,
            index,
            "market_surplus_eur",
            default=float("nan"),
        ).fillna(derived_surplus)
        net_value = _aligned_numeric_series(
            self.net_value_eur,
            index,
            "net_value_eur",
            default=float("nan"),
        ).fillna(revenue - opportunity_cost)

        return pd.DataFrame(
            {
                "afrr_capacity_block_id": _aligned_text_series(
                    self.block_id,
                    index,
                    default="",
                ),
                "afrr_capacity_block_duration_h": block_duration_h,
                "afrr_capacity_pricing_rule": _aligned_text_series(
                    self.pricing_rule,
                    index,
                    default="",
                ),
                "afrr_capacity_bid_price_EUR_per_MW_h": bid_price,
                "afrr_capacity_clearing_price_EUR_per_MW_h": clearing_price,
                "afrr_capacity_settlement_price_EUR_per_MW_h": settlement_price,
                # Retained for the established output schema. The clearing
                # price is the market reference price for the down product.
                "afrr_capacity_down_price_EUR_per_MW_h": clearing_price,
                "afrr_capacity_reserved_MW": reserved_mw,
                "afrr_capacity_reserved_MWh": reserved_mwh,
                "afrr_capacity_revenue_EUR": revenue,
                "afrr_capacity_opportunity_cost_EUR": opportunity_cost,
                "afrr_capacity_market_surplus_EUR": market_surplus,
                "afrr_capacity_net_value_EUR": net_value,
            },
            index=index,
        )

    def result_fields(
        self,
        timestamp: pd.Timestamp,
        timestep_hours: float,
    ) -> dict[str, object]:
        """Return this award's standard output fields for one timestamp."""

        return self.result_frame(pd.Index([timestamp]), timestep_hours).iloc[0].to_dict()


def capacity_award_result_frame(
    award: BalancingCapacityAward | None,
    index: pd.Index,
    timestep_hours: float,
) -> pd.DataFrame:
    """Return common aFRR-capacity output fields, including the no-award case.

    Result mappers can call this helper without special-casing plants that do
    not participate in the capacity product.
    """

    if award is None:
        award = BalancingCapacityAward.empty(index)
    return award.result_frame(index, timestep_hours)


def capacity_award_result_fields(
    award: BalancingCapacityAward | None,
    timestamp: pd.Timestamp,
    timestep_hours: float,
) -> dict[str, object]:
    """Return common aFRR-capacity output fields for one timestamp."""

    return (
        capacity_award_result_frame(
            award,
            pd.Index([timestamp]),
            timestep_hours,
        )
        .iloc[0]
        .to_dict()
    )


def capacity_award_from_result_frame(
    values: pd.DataFrame | None,
    index: pd.Index,
    timestep_hours: float,
) -> BalancingCapacityAward | None:
    """Translate a capacity-stage result table into its market outcome object.

    The capacity strategy currently publishes a table because the sequential
    runner also writes that table to outputs.  This adapter keeps that file
    format at the system boundary while giving plant execution a typed market
    outcome.  It accepts both the established public column names and the
    shorter internal block columns used while preparing an award.
    """

    if values is None or values.empty:
        return None
    _validate_timestep_hours(timestep_hours)
    aligned = values.reindex(index)
    reserved_mw = _frame_numeric_column(
        aligned,
        index,
        "afrr_capacity_reserved_MW",
    )
    if "afrr_capacity_reserved_MW" not in aligned and "afrr_capacity_reserved_MWh" in aligned:
        reserved_mw = (
            _frame_numeric_column(
                aligned,
                index,
                "afrr_capacity_reserved_MWh",
            )
            / timestep_hours
        )

    return BalancingCapacityAward(
        reserved_mw=reserved_mw,
        block_id=_frame_text_column(aligned, index, "afrr_capacity_block_id", ""),
        block_duration_h=_frame_numeric_column(aligned, index, "block_duration_h"),
        pricing_rule=_frame_text_column(aligned, index, "capacity_pricing_rule", ""),
        bid_price_eur_per_mw_h=_frame_numeric_column(
            aligned,
            index,
            "capacity_bid_price_EUR_per_MW_h",
        ),
        clearing_price_eur_per_mw_h=_frame_numeric_column(
            aligned,
            index,
            "capacity_clearing_price_EUR_per_MW_h",
        ),
        settlement_price_eur_per_mw_h=_frame_numeric_column(
            aligned,
            index,
            "capacity_settlement_price_EUR_per_MW_h",
        ),
        revenue_eur=_frame_numeric_column(aligned, index, "afrr_capacity_revenue_EUR"),
        opportunity_cost_eur=_frame_numeric_column(
            aligned,
            index,
            "afrr_capacity_opportunity_cost_EUR",
        ),
        market_surplus_eur=_frame_numeric_column(
            aligned,
            index,
            "afrr_capacity_market_surplus_EUR",
        ),
        net_value_eur=_frame_numeric_column(aligned, index, "afrr_capacity_net_value_EUR"),
    )


def _validate_timestep_hours(timestep_hours: float) -> None:
    if isinstance(timestep_hours, bool):
        raise ValueError("timestep_hours must be a positive number")
    try:
        numeric_timestep_hours = float(timestep_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestep_hours must be a positive number") from exc
    if not math.isfinite(numeric_timestep_hours) or numeric_timestep_hours <= 0.0:
        raise ValueError("timestep_hours must be a positive number")


def _validate_numeric_series(
    values: pd.Series,
    field_name: str,
    *,
    non_negative: bool = False,
) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"BalancingCapacityAward.{field_name} must be a pandas Series")
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if invalid.any():
        raise ValueError(f"BalancingCapacityAward.{field_name} must contain numeric values")
    if not numeric.dropna().map(math.isfinite).all():
        raise ValueError(f"BalancingCapacityAward.{field_name} must contain finite values")
    if non_negative and (numeric.dropna() < -1e-9).any():
        raise ValueError(f"BalancingCapacityAward.{field_name} must be non-negative")


def _validate_optional_numeric_series(
    values: pd.Series | None,
    field_name: str,
    *,
    non_negative: bool = False,
) -> None:
    if values is not None:
        _validate_numeric_series(values, field_name, non_negative=non_negative)


def _aligned_numeric_series(
    values: pd.Series | None,
    index: pd.Index,
    field_name: str,
    *,
    default: float,
) -> pd.Series:
    if values is None:
        return pd.Series(default, index=index, dtype=float)
    _validate_numeric_series(values, field_name)
    numeric = pd.to_numeric(values.reindex(index), errors="coerce")
    return numeric.fillna(default).astype(float)


def _aligned_text_series(
    values: pd.Series | None,
    index: pd.Index,
    *,
    default: str,
) -> pd.Series:
    if values is None:
        return pd.Series(default, index=index, dtype=object)
    return values.reindex(index).fillna(default).astype(str)


def _frame_numeric_column(
    frame: pd.DataFrame,
    index: pd.Index,
    column: str,
) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").reindex(index)
    return values.fillna(0.0).astype(float)


def _frame_text_column(
    frame: pd.DataFrame,
    index: pd.Index,
    column: str,
    default: str,
) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=index, dtype=object)
    return frame[column].reindex(index).fillna(default).astype(str)


@dataclass(frozen=True)
class AFRRCapacityData:
    """Prepared timestep and block-level aFRR down capacity inputs."""

    frame: pd.DataFrame
    block_summary: pd.DataFrame


class AFRRCapacityMarket(BaseMarket):
    """Configured aFRR down capacity product.

    This class understands the capacity product structure and price data. It
    does not decide whether the industrial operator should reserve capacity.
    """

    REQUIRED_SIGNALS = ("price", "quantity")
    RESULT_KIND = MarketResultKind.CAPACITY_AWARD

    @property
    def price_unit(self) -> str:
        return str(self.config.get("price_unit", "EUR_per_MW_per_h"))

    @property
    def product_length(self) -> str:
        return str(self.config.get("product_length", "4h"))

    def validate_config(self, timestep_minutes: int | None = None) -> None:
        super().validate_config(timestep_minutes=timestep_minutes)
        if not self.enabled:
            return
        if str(self.config.get("direction", "down")).lower() not in {"down", "negative"}:
            raise MarketConfigError("afrr_capacity.direction must be 'down' or 'negative'")
        if self.price_unit not in SUPPORTED_PRICE_UNITS:
            raise MarketConfigError(
                "afrr_capacity.price_unit must be one of: "
                + ", ".join(sorted(SUPPORTED_PRICE_UNITS))
            )
        product_length_minutes = _duration_to_minutes(self.product_length)
        if timestep_minutes is not None and product_length_minutes < timestep_minutes:
            raise MarketConfigError(
                "afrr_capacity.product_length must not be shorter than case.timestep_minutes"
            )

    def prepare_market_data(
        self,
        forecasts: pd.DataFrame,
        timestep_hours: float = 0.25,
    ) -> AFRRCapacityData:
        if not self.enabled:
            return AFRRCapacityData(
                frame=pd.DataFrame(index=forecasts.index),
                block_summary=pd.DataFrame(),
            )
        self._require_forecast_columns(forecasts)
        self.validate_config()
        product_length_minutes = _duration_to_minutes(self.product_length)
        # TODO: Move block-generation options to config.yaml if future countries
        # need non-midnight anchors, non-standard calendars, or custom aggregation.
        return prepare_afrr_capacity_blocks(
            forecasts=forecasts,
            price_col=self.signal_column("price"),
            quantity_col=self.signal_column("quantity"),
            product_length_minutes=product_length_minutes,
            timestep_hours=timestep_hours,
            price_unit=self.price_unit,
        )


def prepare_afrr_capacity_blocks(
    forecasts: pd.DataFrame,
    price_col: str,
    quantity_col: str,
    product_length_minutes: int,
    timestep_hours: float,
    price_unit: str = "EUR_per_MW_per_h",
) -> AFRRCapacityData:
    """Generate midnight-anchored capacity blocks and extract block prices."""

    if not isinstance(forecasts.index, pd.DatetimeIndex):
        raise MarketConfigError("aFRR capacity block generation requires a DatetimeIndex")
    if product_length_minutes <= 0:
        raise MarketConfigError("aFRR capacity product_length must be positive")
    if price_unit not in SUPPORTED_PRICE_UNITS:
        raise MarketConfigError(
            "aFRR capacity price_unit must be one of: " + ", ".join(sorted(SUPPORTED_PRICE_UNITS))
        )

    price = pd.to_numeric(forecasts[price_col].replace("", pd.NA), errors="coerce")
    raw_quantity = pd.to_numeric(forecasts[quantity_col].replace("", pd.NA), errors="coerce")
    negative_quantity = raw_quantity.lt(0.0)
    quantity_mw = raw_quantity.abs()
    block_ids = []
    block_starts = []
    block_ends = []
    for timestamp in forecasts.index:
        block_start = _block_start(timestamp, product_length_minutes)
        block_end = block_start + pd.Timedelta(minutes=product_length_minutes)
        block_ids.append(_block_id(block_start, block_end))
        block_starts.append(block_start)
        block_ends.append(block_end)

    frame = pd.DataFrame(
        {
            "afrr_capacity_block_id": block_ids,
            "afrr_capacity_block_start": block_starts,
            "afrr_capacity_block_end": block_ends,
            "afrr_capacity_price_raw": price,
            "afrr_capacity_quantity_raw": raw_quantity,
            "afrr_capacity_quantity_MW": quantity_mw,
            "afrr_capacity_price_input_unit": price_unit,
        },
        index=forecasts.index,
    )

    block_records = []
    for block_id, block_frame in frame.groupby("afrr_capacity_block_id", sort=False):
        non_missing_prices = block_frame["afrr_capacity_price_raw"].dropna()
        missing_price = non_missing_prices.empty
        inconsistent = False
        if missing_price:
            block_price = 0.0
        else:
            raw_block_price = float(non_missing_prices.iloc[0])
            inconsistent = bool(
                (non_missing_prices - raw_block_price).abs().gt(PRICE_CONSISTENCY_TOLERANCE).any()
            )
            if inconsistent:
                warnings.warn(
                    f"aFRR capacity prices differ inside block {block_id}. "
                    "Using the first non-missing price.",
                    stacklevel=2,
                )
            product_duration_h = product_length_minutes / 60.0
            block_price = (
                raw_block_price
                if price_unit == "EUR_per_MW_per_h"
                else raw_block_price / product_duration_h
            )

        block_quantities = block_frame["afrr_capacity_quantity_MW"]
        missing_quantity = bool(block_quantities.isna().any())
        non_missing_quantities = block_quantities.dropna()
        quantity_inconsistent = bool(
            not non_missing_quantities.empty
            and (non_missing_quantities - float(non_missing_quantities.iloc[0]))
            .abs()
            .gt(PRICE_CONSISTENCY_TOLERANCE)
            .any()
        )
        if quantity_inconsistent:
            warnings.warn(
                f"aFRR capacity quantities differ inside block {block_id}. "
                "Using the minimum quantity so the block bid never exceeds market demand.",
                stacklevel=2,
            )
        # A capacity bid applies to the complete product block. A missing interval therefore
        # makes its available market volume unknown and conservatively blocks the bid.
        block_quantity_mw = (
            0.0
            if missing_quantity or non_missing_quantities.empty
            else float(non_missing_quantities.min())
        )

        block_start = pd.Timestamp(block_frame["afrr_capacity_block_start"].iloc[0])
        block_end = pd.Timestamp(block_frame["afrr_capacity_block_end"].iloc[0])
        block_duration_h = len(block_frame) * timestep_hours
        block_records.append(
            {
                "block_id": block_id,
                "block_start": block_start,
                "block_end": block_end,
                "block_duration_h": block_duration_h,
                "capacity_price_input_unit": price_unit,
                "capacity_price_raw": 0.0 if missing_price else raw_block_price,
                "capacity_price_EUR_per_MW_h": block_price,
                "missing_capacity_price_flag": bool(missing_price),
                "price_inconsistency_flag": bool(inconsistent),
                "capacity_quantity_MW": block_quantity_mw,
                "missing_capacity_quantity_flag": missing_quantity,
                "negative_capacity_quantity_flag": bool(
                    negative_quantity.loc[block_frame.index].fillna(False).any()
                ),
                "quantity_inconsistency_flag": quantity_inconsistent,
                "number_of_timesteps": int(len(block_frame)),
            }
        )

    block_summary = pd.DataFrame(block_records)
    frame = frame.join(
        block_summary.set_index("block_id")[
            [
                "block_duration_h",
                "capacity_price_EUR_per_MW_h",
                "missing_capacity_price_flag",
                "price_inconsistency_flag",
                "capacity_quantity_MW",
                "missing_capacity_quantity_flag",
                "negative_capacity_quantity_flag",
                "quantity_inconsistency_flag",
            ]
        ],
        on="afrr_capacity_block_id",
    )
    return AFRRCapacityData(frame=frame, block_summary=block_summary)


def _duration_to_minutes(value: str) -> int:
    text = str(value).strip().lower()
    if text.endswith("min"):
        return int(text.removesuffix("min"))
    if text.endswith("h"):
        return int(float(text.removesuffix("h")) * 60)
    raise MarketConfigError(f"Unsupported aFRR capacity product_length '{value}'")


def _block_start(timestamp: pd.Timestamp, product_length_minutes: int) -> pd.Timestamp:
    day_start = timestamp.normalize()
    minutes_since_midnight = int((timestamp - day_start).total_seconds() // 60)
    block_offset = (minutes_since_midnight // product_length_minutes) * product_length_minutes
    return day_start + pd.Timedelta(minutes=block_offset)


def _block_id(block_start: pd.Timestamp, block_end: pd.Timestamp) -> str:
    return f"{block_start:%Y-%m-%d_%H:%M}_{block_end:%H:%M}"
