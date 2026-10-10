# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

__all__ = ["SimulationRunner"]


def __getattr__(name: str):
    """Import the runner only when the package-level convenience name is used."""

    if name == "SimulationRunner":
        from flexi_mod.simulation.simulation_runner import SimulationRunner

        return SimulationRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
