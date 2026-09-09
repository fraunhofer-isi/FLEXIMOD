# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Solver and window helpers shared by the rolling-horizon plant models."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable

import highspy
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

_HIGHS_SOLVER_NAMES = frozenset({"highs", "appsi_highs"})


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
            if solver_name.lower() in _HIGHS_SOLVER_NAMES:
                _solve_with_highs_lp_file(config, model)
            else:
                solver = pyo.SolverFactory(solver_name)
                if solver is None or not solver.available(exception_flag=False):
                    errors.append(f"{solver_name}: unavailable")
                    continue
                result = solver.solve(model, tee=config.solver_tee)
                status = result.solver.status
                termination = result.solver.termination_condition
                if not (status == SolverStatus.ok and termination in _OPTIMAL_TERMINATIONS):
                    errors.append(f"{solver_name}: status={status}, termination={termination}")
                    continue
        except (ApplicationError, NoFeasibleSolutionError, RuntimeError) as exc:
            errors.append(f"{solver_name}: {exc}")
            continue
        return extract_results(model, forecasts, solver_name)
    raise RuntimeError(f"{plant_label} dispatch could not be solved; " + "; ".join(errors))


def _solve_with_highs_lp_file(config: CaseConfig, model: pyo.ConcreteModel) -> None:
    """Solve *model* by writing an LP file and calling HiGHS natively via highspy.

    Pyomo's persistent solver interfaces (both the ``highs`` and ``appsi_highs`` names
    route through one) transfer every variable and constraint to the solver one at a
    time in pure Python. For the rolling horizon's multi-technology, ~200-timestep
    models this incremental transfer dominates wall time far more than the actual
    optimization does. Writing an LP file and handing it to HiGHS directly avoids that
    per-constraint Python loop entirely — the writer and HiGHS's own parser are both
    native code. Solution values are written back onto *model*'s variables in place via
    the symbol map produced by the write, so the rest of the pipeline (extract_results)
    is unaffected. Raises RuntimeError on any failure so the caller's fallback loop
    behaves exactly as it did for the previous solver.solve()-based path.
    """
    fd, lp_path = tempfile.mkstemp(suffix=".lp")
    os.close(fd)
    try:
        _filename, smap_id = model.write(lp_path, io_options={"symbolic_solver_labels": False})
        smap = model.solutions.symbol_map[smap_id]

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", bool(config.solver_tee))
        # Thousands of rolling solves feed committed output into the next window's
        # demand balance. Tighten HiGHS' primal tolerance so those per-window
        # residuals remain below ROLLING_DEMAND_TOLERANCE_T.
        highs.setOptionValue("primal_feasibility_tolerance", 1e-9)
        # When several cases run concurrently (run_case.py --workers), each worker
        # process must stay single-threaded internally or the process-level
        # parallelism oversubscribes the machine's cores. 0 is HiGHS' own "auto"
        # default, matching prior behavior when no override is set.
        worker_threads = os.environ.get("FLEXIMOD_HIGHS_THREADS")
        highs.setOptionValue("threads", int(worker_threads) if worker_threads is not None else 0)

        if highs.readModel(lp_path) != highspy.HighsStatus.kOk:
            raise RuntimeError("highs: failed to read LP file")
        if highs.run() != highspy.HighsStatus.kOk:
            raise RuntimeError("highs: solve call failed")
        model_status = highs.getModelStatus()
        if model_status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"highs: model status {model_status}")

        col_values = highs.getSolution().col_value
        for index in range(highs.getNumCol()):
            name = highs.getColName(index)
            if isinstance(name, tuple):
                name = name[1]
            component = smap.bySymbol.get(name)
            if component is not None:
                component.set_value(col_values[index], skip_validation=True)
    finally:
        os.unlink(lp_path)


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
