# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reusable pandas and Pyomo helpers for plant optimization models."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from flexi_mod.config.case_config import CaseConfig


def available_pyomo_solvers(config: CaseConfig) -> list[tuple[str, Any]]:
    """Return configured Pyomo solvers that are available on this machine."""

    configured = [config.solver_name, *config.solver_fallbacks]
    expanded: list[str] = []
    for candidate in configured:
        if candidate == "highs":
            expanded.extend(["highs", "appsi_highs"])
        else:
            expanded.append(candidate)

    seen: set[str] = set()
    solvers: list[tuple[str, Any]] = []
    for candidate in expanded:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not _solver_is_installed(candidate):
            continue
        solver = pyo.SolverFactory(candidate)
        if solver.available(exception_flag=False):
            solvers.append((candidate, solver))

    if not solvers:
        raise RuntimeError("No configured Pyomo solver is available. Tried: " + ", ".join(expanded))
    return solvers


def is_infeasible_termination(termination: TerminationCondition) -> bool:
    """Return whether a Pyomo termination condition proves infeasibility."""

    return termination in {
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.provenInfeasible,
    }


def slice_dataclass_series[SignalType](
    signals: SignalType,
    index: pd.Index,
) -> SignalType:
    """Slice every pandas Series in a dataclass to one optimization window."""

    updates: dict[str, object] = {}
    for signal_field in fields(signals):
        value = getattr(signals, signal_field.name)
        updates[signal_field.name] = value.loc[index] if isinstance(value, pd.Series) else value
    return replace(signals, **updates)


def series_or_zero(
    series: pd.Series | None,
    index: pd.Index,
) -> pd.Series:
    """Align a numeric Series or provide an all-zero Series when absent."""

    if series is None:
        return pd.Series(0.0, index=index)
    return series.astype(float).reindex(index).fillna(0.0)


def forecast_values_or_zero(
    forecasts: pd.DataFrame,
    column: str | None,
) -> np.ndarray:
    """Return one numeric forecast column or zeros when it is not configured."""

    if column and column in forecasts.columns:
        return forecasts[column].astype(float).to_numpy()
    return np.zeros(len(forecasts))


def series_value(
    series: pd.Series | None,
    timestamp: pd.Timestamp,
    default: object,
) -> object:
    """Return one timestamp value, falling back for absent or missing data."""

    if series is None:
        return default
    try:
        value = series.loc[timestamp]
    except KeyError:
        return default
    if pd.isna(value):
        return default
    return value


def series_float_or_nan(series: pd.Series, position: int) -> float:
    """Return a positional Series value as float while preserving missingness."""

    value = series.iloc[position]
    if pd.isna(value):
        return float("nan")
    return float(value)


def pyomo_value(expression: pyo.NumericValue) -> float:
    """Extract a solved Pyomo value as float."""

    value = pyo.value(expression)
    if value is None:
        return 0.0
    return float(value)


def _solver_is_installed(candidate: str) -> bool:
    """Avoid constructing solver interfaces whose executables are unavailable."""

    if candidate in {"highs", "appsi_highs"}:
        return importlib.util.find_spec("highspy") is not None or shutil.which("highs") is not None
    if candidate == "glpk":
        return shutil.which("glpsol") is not None
    if candidate == "cbc":
        return shutil.which("cbc") is not None
    return True
