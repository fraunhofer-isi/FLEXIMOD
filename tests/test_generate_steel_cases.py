# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_steel_cases import (
    CO2_FACTOR_COLUMNS,
    add_missing_material_prices,
    load_scenario_forecast,
    write_forecasts,
    write_plants_csv,
)


def test_missing_material_prices_are_added_without_overwriting_sources() -> None:
    frame = pd.DataFrame(
        {
            "datetime": ["2030-01-01 00:00"],
            "DE_DA_price": [50.0],
            "iron_ore_price": [230.0],
        }
    )

    market_cols = add_missing_material_prices(
        frame,
        ["DE_DA_price", "iron_ore_price"],
        iron_ore_price=225.0,
        lime_price=100.0,
    )

    assert market_cols == ["DE_DA_price", "iron_ore_price", "lime_price"]
    assert frame.loc[0, "iron_ore_price"] == pytest.approx(230.0)
    assert frame.loc[0, "lime_price"] == pytest.approx(100.0)


def test_bare_plant_id_demand_header_gets_route_specific_suffix(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "datetime": ["2030-01-01 00:00", "2030-01-01 00:15"],
            "DE_DA_price": [50.0, 51.0],
            "P100000120423": [29.2, 30.1],
            "P100000120426": [79.5, 80.2],
        }
    )
    source.to_csv(tmp_path / "forecasts_df.csv", index=False)

    time_col, market_cols, demand_by_id, frame = load_scenario_forecast(tmp_path)
    output_path = tmp_path / "generated_forecasts.csv"
    plant_name = "P100000120423_dri_eaf_hydrogen_external"
    write_forecasts(
        output_path,
        time_col,
        market_cols,
        demand_by_id,
        [plant_name],
        frame,
    )

    generated = pd.read_csv(output_path)
    demand_column = f"{plant_name}_steel_demand"
    assert market_cols == ["DE_DA_price"]
    assert list(generated.columns) == ["datetime", "DE_DA_price", demand_column]
    assert generated[demand_column].tolist() == pytest.approx([29.2, 30.1])


def test_duplicate_demand_headers_for_same_plant_fail(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "datetime": ["2030-01-01 00:00"],
            "P100000120423": [29.2],
            "P100000120423_steel_demand": [29.2],
        }
    ).to_csv(tmp_path / "forecasts_df.csv", index=False)

    with pytest.raises(ValueError, match="Duplicate steel-demand columns for P100000120423"):
        load_scenario_forecast(tmp_path)


def test_generated_plants_have_explicit_applicable_fuel_co2_factors(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {"name": "steel_coal", "technology": "dri_plant", "fuel_type": "coal"},
            {
                "name": "steel_gas",
                "technology": "dri_plant",
                "fuel_type": "natural_gas",
            },
            {
                "name": "steel_hybrid",
                "technology": "bf_bof",
                "fuel_type": "hybrid_hydrogen_natural_gas",
            },
            {
                "name": "steel_hydrogen",
                "technology": "bf_bof",
                "fuel_type": "hydrogen",
            },
            {"name": "steel_coal", "technology": "eaf", "fuel_type": ""},
        ]
    )
    columns = ["name", "technology", "fuel_type", "steel_demand", *CO2_FACTOR_COLUMNS]
    output_path = tmp_path / "plants.csv"

    write_plants_csv(
        output_path,
        rows,
        columns,
        coal_co2_factor=0.34,
        natural_gas_co2_factor=0.20,
    )

    generated = pd.read_csv(output_path).set_index("name")
    assert generated.loc["steel_gas", "natural_gas_co2_factor"] == pytest.approx(0.20)
    assert generated.loc["steel_hybrid", "natural_gas_co2_factor"] == pytest.approx(0.20)
    coal_rows = generated.loc[["steel_coal"]]
    assert coal_rows.iloc[0]["coal_co2_factor"] == pytest.approx(0.34)
    assert pd.isna(coal_rows.iloc[1]["coal_co2_factor"])
    assert pd.isna(generated.loc["steel_hydrogen", "coal_co2_factor"])
    assert pd.isna(generated.loc["steel_hydrogen", "natural_gas_co2_factor"])


def test_generated_plants_preserve_explicit_master_factor(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "name": "steel_coal",
                "technology": "bf_bof",
                "fuel_type": "coal",
                "coal_co2_factor": 0.341,
            }
        ]
    )
    output_path = tmp_path / "plants.csv"

    write_plants_csv(
        output_path,
        rows,
        ["name", "technology", "fuel_type", *CO2_FACTOR_COLUMNS],
        coal_co2_factor=0.34,
        natural_gas_co2_factor=0.20,
    )

    generated = pd.read_csv(output_path)
    assert generated.loc[0, "coal_co2_factor"] == pytest.approx(0.341)
