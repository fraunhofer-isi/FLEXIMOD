# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from flexi_mod.plants.base_plant import BasePlant
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant
from flexi_mod.plants.steel_plant import SteelPlant
from flexi_mod.plants.technologies import first_non_empty

PLANT_REGISTRY = {
    "steam_plant": SteamGenerationPlant,
    "steam_generator_plant": SteamGenerationPlant,
    "steel_plant": SteelPlant,
}


def build_plants(plants: pd.DataFrame) -> list[BasePlant]:
    """Build heterogeneous plants from the shared ``plants.csv`` schema."""

    built: list[BasePlant] = []
    for plant_name, rows in plants.groupby("name", sort=False):
        unit_types = {
            str(value).strip().lower()
            for value in rows["unit_type"].dropna().tolist()
            if str(value).strip()
        }
        if len(unit_types) != 1:
            raise ValueError(f"Plant '{plant_name}' must define exactly one consistent unit_type")
        unit_type = first_non_empty(rows, "unit_type").lower()
        plant_class = PLANT_REGISTRY.get(unit_type)
        if plant_class is None:
            raise ValueError(f"Plant '{plant_name}' uses unsupported unit_type '{unit_type}'")
        built.append(plant_class.from_rows(str(plant_name), rows))
    return built
