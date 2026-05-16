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

import warnings
from dataclasses import dataclass

import pandas as pd

from flexi_mod.markets.base_market import BaseMarket, MarketConfigError

PRICE_CONSISTENCY_TOLERANCE = 1e-6


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

    REQUIRED_SIGNALS = ("price",)

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
        if self.price_unit != "EUR_per_MW_per_h":
            raise MarketConfigError(
                "afrr_capacity.price_unit must be 'EUR_per_MW_per_h' in this implementation"
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
            product_length_minutes=product_length_minutes,
            timestep_hours=timestep_hours,
        )


def prepare_afrr_capacity_blocks(
    forecasts: pd.DataFrame,
    price_col: str,
    product_length_minutes: int,
    timestep_hours: float,
) -> AFRRCapacityData:
    """Generate midnight-anchored capacity blocks and extract block prices."""

    if not isinstance(forecasts.index, pd.DatetimeIndex):
        raise MarketConfigError("aFRR capacity block generation requires a DatetimeIndex")
    if product_length_minutes <= 0:
        raise MarketConfigError("aFRR capacity product_length must be positive")

    price = pd.to_numeric(forecasts[price_col].replace("", pd.NA), errors="coerce")
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
            "afrr_capacity_price_raw_EUR_per_MW_h": price,
        },
        index=forecasts.index,
    )

    block_records = []
    for block_id, block_frame in frame.groupby("afrr_capacity_block_id", sort=False):
        non_missing_prices = block_frame["afrr_capacity_price_raw_EUR_per_MW_h"].dropna()
        missing_price = non_missing_prices.empty
        inconsistent = False
        if missing_price:
            block_price = 0.0
        else:
            block_price = float(non_missing_prices.iloc[0])
            inconsistent = bool(
                (non_missing_prices - block_price).abs().gt(PRICE_CONSISTENCY_TOLERANCE).any()
            )
            if inconsistent:
                warnings.warn(
                    f"aFRR capacity prices differ inside block {block_id}. "
                    "Using the first non-missing price.",
                    stacklevel=2,
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
                "capacity_price_EUR_per_MW_h": block_price,
                "missing_capacity_price_flag": bool(missing_price),
                "price_inconsistency_flag": bool(inconsistent),
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
