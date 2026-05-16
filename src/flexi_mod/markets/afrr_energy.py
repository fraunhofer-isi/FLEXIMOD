# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""aFRR energy market rules and data preparation.

This module contains direction-specific aFRR energy products. The implemented
MVP product is aFRR down energy, interpreted for the hybrid ETES case as
additional electricity consumption. aFRR up energy is represented as a clean
placeholder for later industrial cases where upward balancing energy is relevant.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

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
