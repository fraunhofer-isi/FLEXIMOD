# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Construct typed plant models from the loader's common input contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from flexi_mod.data.data_loader import PlantInput
from flexi_mod.plants.building import Building
from flexi_mod.plants.cement_plant import CementPlant
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant
from flexi_mod.plants.steel_plant import SteelPlant

Plant = Building | CementPlant | SteelPlant | SteamGenerationPlant
PlantInputs = pd.DataFrame | Mapping[str, list[PlantInput]] | Iterable[PlantInput]


def build_plants(plant_inputs: PlantInputs) -> list[Plant]:
    """Build each plant with the class registered for its ``unit_type``.

    Normal runner code passes ``DataLoader.load_plant_inputs()``. A raw
    component table remains accepted as a compatibility adapter for direct
    model work.
    """

    if isinstance(plant_inputs, pd.DataFrame):
        return _build_from_component_table(plant_inputs)
    if isinstance(plant_inputs, Mapping):
        inputs = [plant for group in plant_inputs.values() for plant in group]
    else:
        inputs = list(plant_inputs)
    return [_build_plant(plant_input) for plant_input in inputs]


def _build_plant(plant_input: PlantInput) -> Plant:
    unit_type = plant_input.unit_type
    if unit_type in {"building", "bus_depot", "electric_bus_depot"}:
        return Building.create(plant_input)
    if unit_type == "cement_plant":
        return CementPlant.create(plant_input)
    if unit_type == "steam_plant":
        return SteamGenerationPlant.create(plant_input)
    if unit_type == "steel_plant":
        return SteelPlant.create(plant_input)
    raise ValueError(f"Plant '{plant_input.name}' uses unsupported unit_type '{unit_type}'")


def _build_from_component_table(plants: pd.DataFrame) -> list[Plant]:
    """Build directly from a component table for notebooks and model tests."""

    result: list[Plant] = []
    for plant_name, rows in plants.groupby("name", sort=False):
        unit_types = {str(value).strip().lower() for value in rows["unit_type"].dropna().tolist()}
        if len(unit_types) != 1:
            raise ValueError(f"Plant '{plant_name}' must use exactly one unit_type")
        unit_type = unit_types.pop()
        if unit_type in {"building", "bus_depot", "electric_bus_depot"}:
            result.append(Building._assemble(str(plant_name), rows))
        elif unit_type == "cement_plant":
            result.append(CementPlant._assemble(str(plant_name), rows))
        elif unit_type == "steam_plant":
            result.append(SteamGenerationPlant._assemble(str(plant_name), rows))
        elif unit_type == "steel_plant":
            result.append(SteelPlant._assemble(str(plant_name), rows))
        else:
            raise ValueError(f"Plant '{plant_name}' uses unsupported unit_type '{unit_type}'")
    return result
