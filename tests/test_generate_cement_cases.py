# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_cement_cases import (
    TEMPLATE_CASE,
    demand_workbook_for_scenario,
    scenario_year,
    size_route_for_plant,
    strategy_for_route,
    write_config,
    write_forecasts,
    write_plants_csv,
)


def _stage(
    technology: str,
    fuel_type: str,
    specific_heat_demand: float,
    *,
    eta_electric: float = 0.98,
    eta_fossil: float = 0.9,
) -> dict[str, object]:
    return {
        "route_id": "R-test",
        "technology": technology,
        "fuel_type": fuel_type,
        "name": "<plant_name>",
        "node": "<node>",
        "specific_heat_demand": specific_heat_demand,
        "eta_electric": eta_electric,
        "eta_fossil": eta_fossil,
        "raw_meal_to_clinker_ratio": 1.5,
        "waste_heat_per_t_clinker": 0.22,
        "waste_heat_utilization_efficiency": 0.9,
        "max_heat_out": "<required MW_th>",
        "max_power": "<required MW_el>",
        "min_heat_out": "<required MW_th>",
        "min_power": "<required MW_el>",
        "ramp_up": "<required MW/step>",
        "ramp_down": "<required MW/step>",
    }


def test_scenario_year_strips_assume_base_case_suffix() -> None:
    assert scenario_year("aktuellepolitiken_2030_base_case_2030") == (
        "aktuellepolitiken",
        "2030",
    )


def test_cement_scenario_family_selects_demand_workbook() -> None:
    assert demand_workbook_for_scenario("fokusH2").name == (
        "FokusH2_FokusStrom_Technologiemix_load_profile_15min_timestep_demand.xlsx"
    )


def test_generated_cement_template_starts_on_first_of_january() -> None:
    template = (TEMPLATE_CASE / "config.yaml").read_text(encoding="utf-8")

    assert 'simulation_start: "2030-01-01 00:00"' in template


def test_strategy_mapping_uses_hybrid_strategy_only_for_r6() -> None:
    assert strategy_for_route("R6") == "hybrid_strategy_cement"
    assert strategy_for_route("R1") == "electrified_cement_rule_based"
    assert strategy_for_route("R16") == "electrified_cement_rule_based"


def test_r6_config_uses_hybrid_strategy(tmp_path: Path) -> None:
    output_path = tmp_path / "config.yaml"
    template = (
        "cases:\n"
        "  cement_plant_DE:\n"
        "    name: cement_plant_DE\n"
        "    strategy:\n"
        "      name: electrified_cement\n"
    )

    write_config(output_path, "aktuellepolitiken_2030_R6", "2030", template, route="R6")

    generated = output_path.read_text(encoding="utf-8")
    assert "name: aktuellepolitiken_2030_R6" in generated
    assert "name: hybrid_strategy_cement" in generated


def test_non_hybrid_config_uses_neutral_rule_based_strategy(tmp_path: Path) -> None:
    output_path = tmp_path / "config.yaml"
    template = (
        "cases:\n"
        "  cement_plant_DE:\n"
        "    name: cement_plant_DE\n"
        "    strategy:\n"
        "      name: electrified_cement\n"
    )

    write_config(output_path, "aktuellepolitiken_2030_R1", "2030", template, route="R1")

    generated = output_path.read_text(encoding="utf-8")
    assert "name: aktuellepolitiken_2030_R1" in generated
    assert "name: electrified_cement_rule_based" in generated


def test_route_sizing_uses_peak_clinker_rate_for_heat_and_flexibility() -> None:
    route_rows = pd.DataFrame(
        [
            _stage("preheater", "fossil", 0.3),
            _stage("simple_calciner", "fossil", 0.7),
            _stage("simple_kiln", "fossil", 1.7, eta_electric=0.95),
        ]
    )

    sized = size_route_for_plant(
        route_rows,
        plant_name="P100000124212_R-test",
        clinker_rate_t_per_h=10.0,
        node="south",
    ).set_index("technology")

    assert sized.loc["preheater", "max_heat_out"] == pytest.approx(4.5)
    assert sized.loc["simple_calciner", "max_heat_out"] == pytest.approx(7.0)
    assert sized.loc["simple_kiln", "max_heat_out"] == pytest.approx(17.0)
    assert (sized["max_power"] == 0.0).all()
    assert (sized["min_heat_out"] == 0.0).all()
    assert (sized["min_power"] == 0.0).all()
    assert sized["ramp_up"].tolist() == pytest.approx([4.5, 7.0, 17.0])
    assert sized["ramp_down"].tolist() == pytest.approx([4.5, 7.0, 17.0])


def test_electric_preheater_deducts_recovered_waste_heat_from_power_rating() -> None:
    route_rows = pd.DataFrame([_stage("preheater", "electricity", 0.3)])

    sized = size_route_for_plant(
        route_rows,
        plant_name="P100000124212_R-test",
        clinker_rate_t_per_h=10.0,
        node="south",
    )

    assert sized.loc[0, "max_heat_out"] == pytest.approx(4.5)
    assert sized.loc[0, "max_power"] == pytest.approx((4.5 - 10.0 * 0.22 * 0.9) / 0.98)


def test_electrolyser_is_sized_from_maximum_hydrogen_heat_input() -> None:
    electrolyser = _stage("electrolyser", "", 0.0)
    electrolyser["efficiency"] = 0.709
    route_rows = pd.DataFrame(
        [
            _stage("simple_calciner", "hydrogen", 0.7),
            _stage("simple_kiln", "hydrogen", 1.7),
            electrolyser,
        ]
    )

    sized = size_route_for_plant(
        route_rows,
        plant_name="P100000124212_R-test",
        clinker_rate_t_per_h=10.0,
        node="south",
    ).set_index("technology")

    expected_power = ((7.0 / 0.9) + (17.0 / 0.9)) / 0.709
    assert sized.loc["electrolyser", "max_power"] == pytest.approx(expected_power)
    assert sized.loc["electrolyser", "ramp_up"] == pytest.approx(expected_power)


def test_forecasts_rekey_clinker_demand_columns(tmp_path: Path) -> None:
    forecasts = pd.DataFrame(
        {
            "datetime": ["2030-01-01 00:00", "2030-01-01 00:15"],
            "DE_DA_price": [50.0, 51.0],
            "P100000124212": [10.0, 11.0],
        }
    )
    output_path = tmp_path / "forecasts_df.csv"

    write_forecasts(output_path, forecasts, {"P100000124212": "P100000124212_R2"})

    generated = pd.read_csv(output_path)
    assert list(generated.columns) == [
        "datetime",
        "DE_DA_price",
        "P100000124212_R2_clinker_demand",
    ]
    assert generated["P100000124212_R2_clinker_demand"].tolist() == pytest.approx([10.0, 11.0])


def test_forecasts_keep_the_rdf_price_column(tmp_path: Path) -> None:
    forecasts = pd.DataFrame(
        {
            "datetime": ["2030-01-01 00:00"],
            "rdf_price": [0.0],
            "P100000124212": [10.0],
        }
    )
    output_path = tmp_path / "forecasts_df.csv"

    write_forecasts(output_path, forecasts, {"P100000124212": "P100000124212_R2"})

    generated = pd.read_csv(output_path)
    assert generated["rdf_price"].tolist() == [0.0]


def test_plants_csv_excludes_route_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "plants.csv"
    write_plants_csv(output_path, pd.DataFrame([_stage("simple_kiln", "fossil", 1.7)]))

    generated = pd.read_csv(output_path)
    assert "route_id" not in generated.columns
    assert "route_description" not in generated.columns
