# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Common interface for route-specific physical equations of the steam plant."""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo


class SteamRouteProcess:
    """Small common interface for route-specific physical equations."""

    name: str
    required_components: frozenset[str]

    def initial_soc(
        self,
        components: dict[str, object],
        requested_soc_mwh: float | None,
    ) -> float | None:
        """Return rolling state; direct routes have no state."""

        return None

    def add_technology_blocks(
        self,
        model: pyo.ConcreteModel,
        components: dict[str, object],
        timestep_hours: float,
        initial_soc_mwh: float | None,
    ) -> None:
        """Add this route's configured technologies to the shared model."""

        model.technology_blocks = pyo.Block(list(components))
        for technology, component in components.items():
            context = {
                "dt_hours": timestep_hours,
                "initial_soc_mwh": (initial_soc_mwh if technology == "thermal_storage" else None),
            }
            component.add_to_model(
                model,
                model.technology_blocks[technology],
                model.T,
                context,
            )

    def validate_capacity_reservation(
        self,
        reserved_capacity_mwh: pd.Series,
    ) -> None:
        """Reject reserve by default; storage routes override this."""

        if (reserved_capacity_mwh.abs() > 1e-9).any():
            raise ValueError(f"Route '{self.name}' does not support aFRR-capacity reservations")

    def next_soc(self, committed_row: pd.Series) -> float | None:
        """Return the next rolling state; direct routes stay stateless."""

        return None

    def objective_penalty(
        self,
        model: pyo.ConcreteModel,
        position: int,
    ) -> pyo.NumericValue | float:
        """Return a route-only feasibility penalty for the objective."""

        return 0.0

    def warn_after_solve(self, model: pyo.ConcreteModel, stage_label: str) -> None:
        """Report any route-specific feasibility relaxation."""

        return
