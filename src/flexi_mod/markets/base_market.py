# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared base class for configured electricity market designs.

Market classes describe the product structure, configured signal names, timing
and market-rule parameters. They prepare market input data for strategies, but
they do not decide how an industrial operator bids, buys or sells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd


class MarketConfigError(ValueError):
    """Raised when one configured market is incomplete or inconsistent."""


@dataclass(frozen=True)
class BaseMarket:
    """Common interface for market-rule and market-data preparation classes."""

    name: str
    config: dict[str, Any]

    REQUIRED_SIGNALS: ClassVar[tuple[str, ...]] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def product_resolution(self) -> str | None:
        value = self.config.get("product_resolution")
        return str(value) if value is not None else None

    @property
    def gate_open(self) -> dict[str, Any]:
        return dict(self.config.get("gate_open", {}))

    @property
    def gate_close(self) -> dict[str, Any]:
        return dict(self.config.get("gate_close", {}))

    @property
    def signals(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in self.config.get("signals", {}).items()}

    @property
    def product_rules(self) -> dict[str, Any]:
        return dict(self.config.get("product_rules", {}))

    def required_signals(self) -> tuple[str, ...]:
        return self.REQUIRED_SIGNALS

    def signal_column(self, signal_name: str) -> str:
        try:
            return self.signals[signal_name]
        except KeyError as exc:
            raise MarketConfigError(
                f"Market '{self.name}' is missing required signal '{signal_name}'"
            ) from exc

    def validate_config(self, timestep_minutes: int | None = None) -> None:
        if not self.enabled:
            return
        missing = [signal for signal in self.required_signals() if signal not in self.signals]
        if missing:
            raise MarketConfigError(
                f"Market '{self.name}' is missing required signal(s): " + ", ".join(sorted(missing))
            )
        if timestep_minutes is not None:
            self._validate_product_resolution(timestep_minutes)

    def prepare_market_data(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Return market-specific data aligned to the forecast index."""

        raise NotImplementedError

    def _require_forecast_columns(self, forecasts: pd.DataFrame) -> None:
        required_columns = {self.signal_column(signal) for signal in self.required_signals()}
        missing = sorted(required_columns - set(forecasts.columns))
        if missing:
            raise KeyError(
                f"Market '{self.name}' requires missing forecast column(s): " + ", ".join(missing)
            )

    def _numeric_signal(self, forecasts: pd.DataFrame, signal_name: str) -> pd.Series:
        column = self.signal_column(signal_name)
        if column not in forecasts.columns:
            raise KeyError(f"Market '{self.name}' requires missing forecast column '{column}'")
        return pd.to_numeric(forecasts[column].replace("", pd.NA), errors="coerce")

    def _validate_product_resolution(self, timestep_minutes: int) -> None:
        if self.product_resolution is None:
            return
        resolution_minutes = product_resolution_to_minutes(self.product_resolution)
        compatible = (
            resolution_minutes % timestep_minutes == 0 or timestep_minutes % resolution_minutes == 0
        )
        if not compatible:
            raise MarketConfigError(
                f"Market '{self.name}' product_resolution={self.product_resolution!r} is not "
                f"compatible with the {timestep_minutes}-minute model timestep"
            )


def product_resolution_to_minutes(value: str) -> int:
    """Convert compact config values such as ``15min`` or ``1h`` into minutes."""

    text = value.strip().lower()
    if text.endswith("min"):
        return int(text.removesuffix("min"))
    if text.endswith("h"):
        return int(float(text.removesuffix("h")) * 60)
    raise MarketConfigError(f"Unsupported product resolution '{value}'")
