# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared model-input checks for optimisation model construction.

Model modules declare which profiles and components they require. This module
performs repeated mechanical checks so model definitions can focus on physical
topology, balances, and objectives. Technology-specific physical limits remain
with the technology classes in :mod:`flexi_mod.plants.technologies`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pyomo.environ as pyo


def component_from_row[T](
    subject: str,
    plant_name: str,
    technology: str,
    factory: Callable[[pd.Series], T],
    row: pd.Series,
) -> T:
    """Build one component and add plant/technology context to input errors."""

    try:
        return factory(row)
    except ValueError as exc:
        raise ValueError(f"{subject} '{plant_name}', technology '{technology}': {exc}") from exc


def validate_required_forecasts(
    forecasts: pd.DataFrame,
    required_columns: set[str],
    subject: str,
    operation: str = "dispatch",
) -> None:
    """Check the common forecast-frame contract before Pyomo construction."""

    if forecasts.empty:
        raise ValueError(f"{subject} requires at least one forecast row")
    if not forecasts.index.is_unique:
        raise ValueError(f"{subject} forecast timestamps must be unique")
    missing = sorted(required_columns - set(forecasts.columns))
    if missing:
        raise ValueError(
            f"{subject} cannot build {operation}: forecasts_df.csv is missing required column(s): "
            + ", ".join(missing)
        )


def numeric_forecast(
    forecasts: pd.DataFrame,
    column: str,
    subject: str,
    description: str | None = None,
    require_non_negative: bool = False,
    allow_missing_values: bool = False,
) -> pd.Series:
    """Return a finite numeric profile, with clear input errors when invalid.

    ``allow_missing_values`` is reserved for an explicitly documented model
    behaviour, such as steam intraday prices that represent a no-action period
    when unavailable.  Non-numeric values and infinite values always fail.
    """

    raw_values = forecasts[column]
    values = pd.to_numeric(raw_values, errors="coerce")
    invalid_numeric = values.isna() & ~raw_values.isna()
    missing_values = values.isna()
    if allow_missing_values:
        missing_values = pd.Series(False, index=forecasts.index)
    finite_values = values.dropna()
    suffix = f" ({description})" if description else ""
    if invalid_numeric.any() or missing_values.any() or not np.isfinite(finite_values).all():
        raise ValueError(f"{subject} forecast '{column}'{suffix} must be finite and numeric")
    if require_non_negative and (finite_values < 0.0).any():
        raise ValueError(f"{subject} forecast '{column}'{suffix} cannot be negative")
    return values.astype(float)


def validate_profile_range(
    values: pd.Series,
    subject: str,
    label: str,
    lower: float,
    upper: float,
) -> None:
    """Validate a finite numeric profile against inclusive physical bounds."""

    if (values < lower).any() or (values > upper).any():
        raise ValueError(f"{subject} {label} must be between {lower} and {upper}")


def time_parameter(model: pyo.ConcreteModel, values: pd.Series) -> pyo.Param:
    """Create a time-indexed immutable Pyomo parameter from a numeric profile."""

    return pyo.Param(model.T, initialize={t: float(values.iloc[t]) for t in model.T})
