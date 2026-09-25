# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Charge-gate and fixed-position helpers shared by steam-plant strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant

ELECTRICITY_PRICE_SAFETY_MARGIN_EUR_PER_MWH = 0.0


def grid_charging_block(
    plant: SteamGenerationPlant,
    forecasts: pd.DataFrame,
) -> pd.Series:
    """Per-timestep mask, True where the grid-fee regulation blocks grid-charging.

    Under atypical grid use (S19(2) StromNEV) the plant avoids drawing grid
    power during DSO high-load windows to keep its billed capacity peak low.
    """

    regulation = getattr(plant, "grid_fee_regulation", None)
    if regulation is None:
        return pd.Series(False, index=forecasts.index)
    mask = regulation.charging_block_mask(forecasts)
    return mask.reindex(forecasts.index).fillna(False).astype(bool)


def charge_gate(
    plant: SteamGenerationPlant,
    electricity_price: pd.Series,
    benchmark: pd.Series,
) -> pd.Series:
    """Mask, True where charging the ETES is economically attractive.

    Charging is allowed when the effective electric heat cost (price divided by
    the round-trip efficiency) is at or below the gas heat benchmark minus a
    small safety margin.
    """

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


def fixed_da_position(
    fixed_positions: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Day-ahead electricity position reindexed onto the forecast index."""

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


def series_from_fixed_positions(
    fixed_positions: pd.DataFrame,
    column: str,
    index: pd.DatetimeIndex,
    default: float | None = 0.0,
) -> pd.Series | None:
    """Reindex a fixed-position column onto the forecast index.

    Returns ``None`` when the column is absent and no default is given; otherwise
    fills missing rows with ``default`` and validates alignment with the index.
    """

    if column in fixed_positions.columns:
        series = fixed_positions[column].astype(float).reindex(index)
    elif default is None:
        return None
    else:
        series = pd.Series(float(default), index=index)
    if series.isna().any():
        raise ValueError(f"Fixed position column '{column}' is not aligned with forecasts")
    return series
