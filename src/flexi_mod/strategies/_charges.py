# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regulation-aware additional-charge helpers.

Read the electricity-tax rate and the dynamic grid-fee charge column from a
plant's ``grid_fee_regulation`` and assemble the per-timestep additional
electricity charge series used by the dispatch and settlement logic.
"""

from __future__ import annotations

import warnings

import pandas as pd

from flexi_mod.data.data_loader import DataValidationError
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant


def get_tax_rate(plant: SteamGenerationPlant) -> float:
    """Read the multiplicative electricity tax rate from the plant's regulation."""

    regulation = getattr(plant, "grid_fee_regulation", None)
    if regulation is None:
        return 0.0
    return float(getattr(regulation, "electricity_tax_rate", 0.0))


def get_dynamic_charge_column(plant: SteamGenerationPlant) -> str | None:
    """Read the dynamic charge column name from the plant's regulation."""

    regulation = getattr(plant, "grid_fee_regulation", None)
    if regulation is None:
        return None
    return getattr(regulation, "dynamic_charge_column", None)


def calculate_additional_charges_t(
    plant: SteamGenerationPlant, forecasts: pd.DataFrame
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

    ac_column = get_dynamic_charge_column(plant)
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
