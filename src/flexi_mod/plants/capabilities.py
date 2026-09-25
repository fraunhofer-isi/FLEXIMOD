# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Structural protocol declaring the capabilities strategies need from a plant.

``SteamGenerationPlant`` satisfies this protocol by duck typing -- no runtime
``isinstance`` checks are performed.  The protocol exists only to decouple
strategy signatures from the concrete plant class so the strategy layer can be
tested and extended against the interface alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd

    from flexi_mod.config.case_config import CaseConfig
    from flexi_mod.plants.routes.base import SteamRouteProcess
    from flexi_mod.plants.signals.afrr_down_signals import AFRRDownSignals
    from flexi_mod.plants.signals.dispatch_signals import DispatchSignals
    from flexi_mod.plants.signals.idc_adjustment_signals import IDCAdjustmentSignals
    from flexi_mod.plants.technologies import ElectricBoiler, GasBoiler, ThermalStorage


class PlantCapabilities(Protocol):
    """Capabilities a plant must expose to be usable by market strategies."""

    # --- Metadata (provided by BasePlant) --------------------------------

    name: str
    heat_demand_column: str
    gas_emissions_factor_kg_per_mwh: float

    # --- Route and component accessors ----------------------------------

    @property
    def route_process(self) -> SteamRouteProcess: ...

    @property
    def technology_route(self) -> str: ...

    @property
    def has_thermal_storage(self) -> bool: ...

    @property
    def etes(self) -> ThermalStorage: ...

    @property
    def electric_boiler(self) -> ElectricBoiler: ...

    @property
    def gas_boiler(self) -> GasBoiler: ...

    # --- Public solve API ------------------------------------------------

    def solve_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: DispatchSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame: ...

    def solve_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: DispatchSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame: ...

    def solve_intraday_adjustment_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: IDCAdjustmentSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame: ...

    def solve_intraday_adjustment_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: IDCAdjustmentSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame: ...

    def solve_afrr_down_rolling(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: AFRRDownSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame: ...

    def solve_afrr_down_horizon(
        self,
        config: CaseConfig,
        forecasts: pd.DataFrame,
        signals: AFRRDownSignals,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame: ...
