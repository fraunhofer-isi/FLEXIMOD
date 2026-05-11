# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backward-compatible plotting import path."""

from __future__ import annotations

from flexi_mod.visualisation.plots import create_all_plots_from_output, create_case_plots

__all__ = ["create_all_plots_from_output", "create_case_plots"]
