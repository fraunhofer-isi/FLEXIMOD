# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registry mapping component sets to their physical steam-route processes."""

from flexi_mod.plants.routes.base import SteamRouteProcess
from flexi_mod.plants.routes.direct_electric_gas_boiler import (
    DirectElectricGasBoilerProcess,
)
from flexi_mod.plants.routes.thermal_storage_gas_boiler import (
    ThermalStorageGasBoilerProcess,
)

ROUTE_PROCESS_REGISTRY: dict[frozenset[str], SteamRouteProcess] = {
    ThermalStorageGasBoilerProcess.required_components: (ThermalStorageGasBoilerProcess()),
    DirectElectricGasBoilerProcess.required_components: (DirectElectricGasBoilerProcess()),
}


def resolve_steam_route_process(
    components: dict[str, object],
) -> SteamRouteProcess:
    """Resolve an exact component set to its physical steam process."""

    component_set = frozenset(components)
    try:
        return ROUTE_PROCESS_REGISTRY[component_set]
    except KeyError as exc:
        supported = ", ".join(
            " + ".join(sorted(route_components)) for route_components in ROUTE_PROCESS_REGISTRY
        )
        configured = " + ".join(sorted(component_set)) or "<none>"
        raise ValueError(
            f"Unsupported steam-plant component route '{configured}'. Supported routes: {supported}"
        ) from exc
