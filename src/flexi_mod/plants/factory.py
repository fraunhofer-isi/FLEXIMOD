# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Construct plant models from their ``unit_type`` in plants.csv."""

from __future__ import annotations

import pandas as pd

from flexi_mod.plants.building import Building
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant

Plant = Building | SteamGenerationPlant


def build_plants(plants: pd.DataFrame) -> list[Plant]:
    """Build each connected plant group with the matching plant class."""

    result: list[Plant] = []
    for plant_name, rows in plants.groupby("name", sort=False):
        unit_types = {
            str(value).strip().lower() for value in rows["unit_type"].dropna().tolist()
        }
        if len(unit_types) != 1:
            raise ValueError(f"Plant '{plant_name}' must use exactly one unit_type")
        unit_type = unit_types.pop()
        if unit_type in {"building", "bus_depot", "electric_bus_depot"}:
            result.append(Building.from_rows(str(plant_name), rows))
        elif unit_type == "steam_plant":
            result.append(SteamGenerationPlant.from_rows(str(plant_name), rows))
        else:
            raise ValueError(f"Plant '{plant_name}' uses unsupported unit_type '{unit_type}'")
    return result
