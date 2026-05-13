# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AFRRDownData:
    """Cleaned direction-specific aFRR down energy inputs."""

    frame: pd.DataFrame
    quality_summary: pd.DataFrame


def clean_afrr_down_data(
    forecasts: pd.DataFrame,
    price_col: str,
    quantity_col: str,
    activation_unit: str,
    timestep_hours: float,
) -> AFRRDownData:
    """Clean direction-specific aFRR down price and system activation data.

    The activation signal is a system-level/proxy activation magnitude, not
    plant-specific realised activation. Price values of zero are valid.
    """

    if activation_unit not in {"MW", "MWh"}:
        raise ValueError("aFRR energy interpretation.activation_unit must be 'MW' or 'MWh'")

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

    flags = pd.Series("valid_activation", index=forecasts.index, dtype=object)
    flags.loc[negative_quantity.fillna(False)] = "negative_quantity_converted"
    flags.loc[~nonzero_quantity & ~missing_price] = "no_activation_or_missing_quantity"
    flags.loc[missing_price & ~nonzero_quantity] = "no_activation_or_missing_data"
    flags.loc[missing_price & nonzero_quantity] = "activation_without_price"

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

    clean = pd.DataFrame(
        {
            "afrr_energy_down_price_raw": price,
            "afrr_energy_down_price_clean": price.fillna(0.0),
            "afrr_raw_system_activation": quantity,
            "afrr_raw_system_activation_MWh": raw_activation_mwh,
            "afrr_down_system_activation_MWh_clean": clean_activation_mwh.clip(lower=0.0),
            "afrr_data_quality_flag": flags,
        },
        index=forecasts.index,
    )

    activation_without_price = missing_price & nonzero_quantity
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
    return AFRRDownData(frame=clean, quality_summary=summary)
