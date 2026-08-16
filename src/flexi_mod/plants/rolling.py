# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Solver and window helpers shared by the rolling-horizon plant models."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.opt import SolverStatus, TerminationCondition

from flexi_mod.config.case_config import CaseConfig

#: Tolerance on the produced-versus-demanded reconciliation at the end of a rolling run.
ROLLING_DEMAND_TOLERANCE_T = 1e-6

_OPTIMAL_TERMINATIONS = frozenset(
    {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal,
    }
)


def solve_dispatch_model(
    config: CaseConfig,
    forecasts: pd.DataFrame,
    model: pyo.ConcreteModel,
    extract_results: Callable[[pyo.ConcreteModel, pd.DataFrame, str], pd.DataFrame],
    *,
    plant_label: str,
) -> pd.DataFrame:
    """Solve *model* with the configured solver and its fallbacks.

    Each solver is tried in turn, and the first optimal result is handed to
    *extract_results*. If none succeeds, the collected per-solver failures are raised
    together so the caller sees why every candidate was rejected.
    """

    errors: list[str] = []
    for solver_name in dict.fromkeys([config.solver_name, *config.solver_fallbacks]):
        try:
            solver = pyo.SolverFactory(solver_name)
            if solver is None or not solver.available(exception_flag=False):
                errors.append(f"{solver_name}: unavailable")
                continue
            if solver_name.lower() in {"highs", "appsi_highs"}:
                # Thousands of rolling solves feed committed output into the next
                # window's demand balance. Tighten HiGHS' primal tolerance so those
                # per-window residuals remain below ROLLING_DEMAND_TOLERANCE_T.
                solver.options["primal_feasibility_tolerance"] = 1e-9
            result = solver.solve(model, tee=config.solver_tee)
        except (ApplicationError, NoFeasibleSolutionError, RuntimeError) as exc:
            errors.append(f"{solver_name}: {exc}")
            continue
        status = result.solver.status
        termination = result.solver.termination_condition
        if status == SolverStatus.ok and termination in _OPTIMAL_TERMINATIONS:
            return extract_results(model, forecasts, solver_name)
        errors.append(f"{solver_name}: status={status}, termination={termination}")
    raise RuntimeError(f"{plant_label} dispatch could not be solved; " + "; ".join(errors))


def validate_rolling_window(
    timestep_hours: float,
    horizon_hours: float,
    step_hours: float,
    *,
    plant_label: str,
) -> None:
    """Reject window settings that cannot be expressed in whole timesteps."""

    if horizon_hours <= 0 or step_hours <= 0:
        raise ValueError(f"{plant_label} rolling horizon and step hours must be positive")
    if step_hours > horizon_hours:
        raise ValueError("rolling_step_hours must not exceed dispatch_horizon_hours")
    for label, hours in {
        "dispatch_horizon_hours": horizon_hours,
        "rolling_step_hours": step_hours,
    }.items():
        steps = hours / timestep_hours
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"{label} must align with case.timestep_minutes")
