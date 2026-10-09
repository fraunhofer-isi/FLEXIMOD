# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Construct typed plant models from the loader's common input contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from flexi_mod.data.data_loader import PlantDefinition
from flexi_mod.plants.building import Building
from flexi_mod.plants.cement_plant import CementPlant
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant
from flexi_mod.plants.steel_plant import SteelPlant

Plant = Building | CementPlant | SteelPlant | SteamGenerationPlant
PlantInputs = pd.DataFrame | Mapping[str, list[PlantDefinition]] | Iterable[PlantDefinition]


def build_plants(plants: PlantInputs) -> list[Plant]:
    """Build each plant with the class registered for its ``unit_type``.

    Normal runner code passes ``DataLoader.load_plant_definitions()``. A raw
    dataframe remains accepted as a compatibility adapter for existing
    notebooks and direct model tests.
    """

    if isinstance(plants, pd.DataFrame):
        return _build_from_rows(plants)
    if isinstance(plants, Mapping):
        definitions = [plant for group in plants.values() for plant in group]
    else:
        definitions = list(plants)
    return [_build_from_definition(definition) for definition in definitions]


def _build_from_definition(definition: PlantDefinition) -> Plant:
    unit_type = definition.unit_type
    if unit_type in {"building", "bus_depot", "electric_bus_depot"}:
        return Building.from_definition(definition)
    if unit_type == "cement_plant":
        return CementPlant.from_definition(definition)
    if unit_type == "steam_plant":
        return SteamGenerationPlant.from_definition(definition)
    if unit_type == "steel_plant":
        return SteelPlant.from_definition(definition)
    raise ValueError(f"Plant '{definition.name}' uses unsupported unit_type '{unit_type}'")


def _build_from_rows(plants: pd.DataFrame) -> list[Plant]:
    """Keep the earlier dataframe entry point available while callers migrate."""

    result: list[Plant] = []
    for plant_name, rows in plants.groupby("name", sort=False):
        unit_types = {str(value).strip().lower() for value in rows["unit_type"].dropna().tolist()}
        if len(unit_types) != 1:
            raise ValueError(f"Plant '{plant_name}' must use exactly one unit_type")
        unit_type = unit_types.pop()
        if unit_type in {"building", "bus_depot", "electric_bus_depot"}:
            result.append(Building.from_rows(str(plant_name), rows))
        elif unit_type == "cement_plant":
            result.append(CementPlant.from_rows(str(plant_name), rows))
        elif unit_type == "steam_plant":
            result.append(SteamGenerationPlant.from_rows(str(plant_name), rows))
        elif unit_type == "steel_plant":
            result.append(SteelPlant.from_rows(str(plant_name), rows))
        else:
            raise ValueError(f"Plant '{plant_name}' uses unsupported unit_type '{unit_type}'")
    return result
