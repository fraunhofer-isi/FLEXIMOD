# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the central solved-model result-mapper registry."""

from flexi_mod.outputs.result_mappers import RESULT_MAPPERS
from flexi_mod.plants.building import Building
from flexi_mod.plants.cement_plant import CementPlant
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant
from flexi_mod.plants.steel_plant import SteelPlant


def test_result_mapper_registry_covers_supported_physical_plants() -> None:
    assert set(RESULT_MAPPERS) == {
        Building,
        CementPlant,
        SteamGenerationPlant,
        SteelPlant,
    }
