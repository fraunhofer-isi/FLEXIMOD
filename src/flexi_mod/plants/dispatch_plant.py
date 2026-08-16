# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared Pyomo-model-building scaffolding for component-based dispatch plants.

``SteelPlant`` and ``CementPlant`` both assemble one trajectory from a dict of named
technology components, then layer plant-level totals and demand semantics on top. This
base class factors out the parts that are genuinely identical between the two -
declaring each component's block and the plant-wide power/cost totals - while leaving
the parts that are not (how components link to each other, what "meeting demand" means)
as named hooks each plant defines for itself, mirroring ASSUME's own
``initialize_process_sequence``/``define_constraints`` split.

Deliberately absent: a shared ``_add_physical_system``. Steel's and cement's signatures
differ in both parameter set and order - carrying steel's rolling-horizon backlog
parameters through cement's signature (which has none) would be exactly the kind of flag
introduced for symmetry rather than need that this refactor is trying to avoid. Each
plant keeps its own ``_add_physical_system`` with its existing external signature; only
its body is rebuilt from the hooks and shared steps below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pyomo.environ as pyo

from flexi_mod.plants.base_plant import BasePlant


class DispatchPlant(BasePlant, ABC):
    """Common scaffolding for a plant whose physics is a dict of named components."""

    components: dict[str, Any]

    #: Electric-load attribute names to sum across every technology block for
    #: ``total_power_input``. Steel's technologies expose only ``power_in``; cement adds
    #: ``aux_power_in`` (every kiln-line stage has a separate auxiliary load) and
    #: ``electricity_consumption`` (the thermal store's own attribute name). No steel
    #: technology ever defines the latter two, so the wider default is a no-op for it -
    #: only ``CementPlant`` needs to override this.
    _power_consuming_attrs: ClassVar[tuple[str, ...]] = ("power_in",)

    @abstractmethod
    def _component_context(
        self,
        technology: str,
        dt_hours: float,
        initial_state: Any,
    ) -> dict[str, Any]:
        """Hook: the ``add_to_model`` context for one component, seeded from carried state."""

    @abstractmethod
    def initialize_process_sequence(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
    ) -> None:
        """Hook: constraints linking one component's output to another's input.

        Named after ASSUME's own phase of the same name. Steel's and cement's bodies
        share no code - each plant's components connect to each other in ways specific
        to that plant - so this stays a pure override with no shared default.
        """

    @abstractmethod
    def define_constraints(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
        **kwargs: Any,
    ) -> None:
        """Hook: plant-level rules, chiefly what "meeting demand" means for this plant.

        Named after ASSUME's own phase of the same name - whose own reference
        implementation conditionally skips its global demand constraint under some
        configurations, the same shape as steel's ``enforce_output_total``. That kind of
        trajectory-dependent branching belongs here, in the plant's own override, not in
        a shared step: cement's demand band is a per-timestep floor no different power
        draw can violate, so it never needs to branch at all.
        """

    def _attach_technology_blocks(
        self,
        model: pyo.ConcreteModel,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
        dt_hours: float,
        initial_state: Any,
    ) -> None:
        """Pyomo Components:

        - **Blocks**: one named ``technology_blocks[name]`` per configured component,
          populated by that component's own ``add_to_model``.
        """
        container.technology_blocks = pyo.Block(list(self.components))
        for technology, component in self.components.items():
            component.add_to_model(
                model,
                container.technology_blocks[technology],
                time_steps,
                self._component_context(technology, dt_hours, initial_state),
            )

    def _declare_plant_totals(
        self,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
    ) -> None:
        """Pyomo Components:

        - **Variables**: ``total_power_input`` and ``variable_cost`` - the two
          plant-wide totals every dispatch trajectory needs, regardless of family.
        """
        container.total_power_input = pyo.Var(time_steps, within=pyo.NonNegativeReals)
        container.variable_cost = pyo.Var(time_steps, within=pyo.Reals)

    def _attach_total_power_input_constraint(
        self,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
    ) -> None:
        """Pyomo Components:

        - **Constraints**: ``total_power_input_constraint``, summing every electric-load
          attribute named in ``_power_consuming_attrs`` across every technology block.
        """
        attrs = self._power_consuming_attrs

        @container.Constraint(time_steps)
        def total_power_input_constraint(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            total = 0.0
            for block in m.technology_blocks.values():
                for attr in attrs:
                    if hasattr(block, attr):
                        total += getattr(block, attr)[t]
            return m.total_power_input[t] == total

    def _attach_variable_cost_constraint(
        self,
        container: pyo.Block | pyo.ConcreteModel,
        time_steps: pyo.Set,
    ) -> None:
        """Pyomo Components:

        - **Constraints**: ``variable_cost_constraint``, summing every component's own
          ``operating_cost`` for the timestep.
        """

        @container.Constraint(time_steps)
        def variable_cost_constraint(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.variable_cost[t] == sum(
                block.operating_cost[t]
                for block in m.technology_blocks.values()
                if hasattr(block, "operating_cost")
            )
