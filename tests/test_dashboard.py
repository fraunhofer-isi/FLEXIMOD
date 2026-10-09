# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dashboard data loading, chart builders, static HTML and case comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flexi_mod.visualisation.dashboard.charts import BUILDERS, ChartContext
from flexi_mod.visualisation.dashboard.comparison import (
    load_case_summaries,
    ranking_chart,
    render_comparison_html,
    tradeoff_chart,
)
from flexi_mod.visualisation.dashboard.data import (
    ALL_PLANTS,
    aggregate_plants,
    auto_resolution,
    build_case,
    column_kind,
    detect_family,
    discover_cases,
    load_case,
    resample,
)
from flexi_mod.visualisation.dashboard.sections import build_tabs
from flexi_mod.visualisation.dashboard.static_html import render_case_html, write_case_dashboard
from flexi_mod.visualisation.dashboard.theme import DARK, LIGHT, color_swap_map


def _steam_dispatch(days: int = 3, plants: tuple[str, ...] = ("plant_1",)) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=days * 96, freq="15min")
    rng = np.random.default_rng(1)
    frames = []
    for plant in plants:
        discharge = rng.uniform(0, 1, len(index))
        frames.append(
            pd.DataFrame(
                {
                    "datetime": index,
                    "plant_name": plant,
                    "heat_demand_MWh": 2.0,
                    "gas_heat_MWh": 2.0 - discharge,
                    "etes_discharge_MWh": discharge,
                    "etes_charge_MWh": discharge,
                    "etes_soc_MWh": rng.uniform(0, 5, len(index)),
                    "day_ahead_price_EUR_per_MWh": rng.uniform(20, 120, len(index)),
                    "DA_position_MWh": discharge,
                    "IDC_buy_MWh": 0.0,
                    "IDC_sell_MWh": 0.0,
                    "afrr_energy_activated_MWh": 0.0,
                    "actual_electricity_consumption_MWh": discharge,
                    "DA_electricity_cost_EUR": discharge * 50,
                    "additional_electricity_charges_cost_EUR": discharge * 10,
                    "gas_cost_EUR": (2.0 - discharge) * 40,
                    "gross_operating_cost_EUR": discharge * 60 + (2.0 - discharge) * 40,
                    "net_operating_cost_EUR": discharge * 60 + (2.0 - discharge) * 40,
                    "electricity_emissions_kg": discharge * 100,
                    "gas_emissions_kg": (2.0 - discharge) * 200,
                    "total_emissions_kg": discharge * 100 + (2.0 - discharge) * 200,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_detect_family_from_columns() -> None:
    assert detect_family(["heat_demand_MWh"]) == "steam"
    assert detect_family(["building_demand_MWh", "heat_demand_MWh"]) == "building"
    assert detect_family(["clinker_output_t"]) == "cement"
    assert detect_family(["steel_output_t"]) == "steel"
    assert detect_family(["something"]) == "generic"


def test_column_kind_separates_flows_levels_and_prices() -> None:
    assert column_kind("gas_heat_MWh") == "flow"
    assert column_kind("etes_soc_MWh") == "level"
    assert column_kind("afrr_capacity_reserved_MW") == "level"
    assert column_kind("day_ahead_price_EUR_per_MWh") == "price"
    assert column_kind("kiln_operational_status") == "price"


def test_resample_sums_flows_and_averages_prices() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="15min")
    frame = pd.DataFrame(
        {
            "gas_heat_MWh": [1.0, 1.0, 1.0, 1.0],
            "etes_soc_MWh": [0.0, 2.0, 4.0, 6.0],
            "day_ahead_price_EUR_per_MWh": [10.0, 20.0, 30.0, 40.0],
        },
        index=index,
    )

    hourly = resample(frame, "1h")

    assert hourly["gas_heat_MWh"].iloc[0] == 4.0
    assert hourly["etes_soc_MWh"].iloc[0] == 3.0
    assert hourly["day_ahead_price_EUR_per_MWh"].iloc[0] == 25.0


def test_aggregate_plants_sums_quantities_and_averages_prices() -> None:
    index = pd.DatetimeIndex(["2025-01-01 00:00"] * 2, name="datetime")
    frame = pd.DataFrame(
        {
            "plant_name": ["a", "b"],
            "gas_heat_MWh": [1.0, 3.0],
            "day_ahead_price_EUR_per_MWh": [40.0, 60.0],
        },
        index=index,
    )

    total = aggregate_plants(frame)

    assert total["gas_heat_MWh"].iloc[0] == 4.0
    assert total["day_ahead_price_EUR_per_MWh"].iloc[0] == 50.0


def test_auto_resolution_keeps_series_short() -> None:
    day = pd.date_range("2025-01-01", periods=96, freq="15min")
    year = pd.date_range("2025-01-01", periods=35040, freq="15min")

    assert auto_resolution(day, 0.25) == "native"
    assert auto_resolution(year, 0.25) in {"6h", "1D"}


def test_load_case_reads_compressed_tables_with_mixed_dst_offsets(tmp_path: Path) -> None:
    # The wall-clock offset changes on 2025-03-30; the loader must not fail on it.
    stamps = ["2025-03-30 01:00:00+01:00", "2025-03-30 03:00:00+02:00"]
    pd.DataFrame(
        {
            "datetime": stamps,
            "plant_name": ["plant_1", "plant_1"],
            "heat_demand_MWh": [1.0, 1.0],
            "gas_heat_MWh": [1.0, 0.5],
        }
    ).to_csv(tmp_path / "dispatch_results.csv.zst", index=False, compression="zstd")

    case = load_case(tmp_path)

    assert case.family == "steam"
    assert case.dispatch.index[-1] == pd.Timestamp("2025-03-30 03:00:00")
    assert case.plants == ["plant_1"]


def test_load_case_requires_dispatch_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_case(tmp_path)


def test_discover_cases_finds_plain_and_compressed_outputs(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "dispatch_results.csv").write_text("datetime\n")
    (tmp_path / "b" / "nested").mkdir(parents=True)
    (tmp_path / "b" / "nested" / "dispatch_results.csv.zst").write_bytes(b"")

    found = discover_cases(tmp_path)

    assert [path.name for path in found] == ["a", "nested"]


def test_builders_skip_charts_without_their_columns() -> None:
    case = build_case(
        "minimal", _steam_dispatch().loc[:, ["datetime", "plant_name", "heat_demand_MWh"]]
    )
    ctx = ChartContext(case)

    assert BUILDERS["heat_and_storage"](ctx) is not None
    assert BUILDERS["markets_overview"](ctx) is None
    assert BUILDERS["afrr_capacity"](ctx) is None
    assert BUILDERS["ev_fleet"](ctx) is None


def test_all_builders_run_on_steam_case() -> None:
    case = build_case("steam", _steam_dispatch(plants=("plant_1", "plant_2")))
    for plant in (ALL_PLANTS, "plant_1"):
        ctx = ChartContext(case, plant)
        for name, builder in BUILDERS.items():
            figure = builder(ctx)
            assert figure is None or len(figure.data) > 0, name


def test_cost_breakdown_adds_up_to_net_cost() -> None:
    case = build_case("steam", _steam_dispatch())
    ctx = ChartContext(case)

    from flexi_mod.visualisation.dashboard.charts import cost_components

    parts = dict(cost_components(ctx))

    assert sum(parts.values()) == pytest.approx(ctx.total("net_operating_cost_EUR"))


def test_charts_use_no_secondary_axis() -> None:
    case = build_case("steam", _steam_dispatch())
    ctx = ChartContext(case)

    for name, builder in BUILDERS.items():
        figure = builder(ctx)
        if figure is None:
            continue
        for axis in figure.layout.to_plotly_json():
            if axis.startswith("yaxis"):
                assert figure.layout[axis].overlaying is None, name


def test_theme_swap_map_moves_light_colours_to_their_dark_steps() -> None:
    to_dark = color_swap_map(LIGHT, DARK)

    for light, dark in zip(LIGHT.categorical, DARK.categorical, strict=True):
        if light != dark:
            assert to_dark[light] == dark
    assert to_dark[LIGHT.surface] == DARK.surface
    assert to_dark[LIGHT.text_primary] == DARK.text_primary


def test_static_dashboard_contains_tabs_plant_selector_and_figures(tmp_path: Path) -> None:
    case = build_case("steam", _steam_dispatch(plants=("plant_1", "plant_2")), output_dir=tmp_path)

    path = write_case_dashboard(case, plotly_js="cdn")
    content = path.read_text(encoding="utf-8")

    assert path == tmp_path / "dashboard.html"
    assert 'id="plant-select"' in content
    assert "All plants (aggregate)" in content
    for label in ("Overview", "Markets", "Operation", "Costs &amp; emissions", "Patterns", "Data"):
        assert label in content
    assert "Heat supply and storage" in content
    assert "cdn.plot.ly" in content
    assert "</script></script>" not in content


def test_static_dashboard_limits_individual_plants() -> None:
    plants = tuple(f"plant_{index}" for index in range(5))
    case = build_case("many", _steam_dispatch(days=1, plants=plants))

    content = render_case_html(case, max_plants=2, plotly_js="cdn")

    assert content.count('<div class="view"') == 3  # aggregate + two plants
    assert "top 2 of 5 plants" in content


def test_period_filter_restricts_charts_and_hides_run_totals() -> None:
    case = build_case("steam", _steam_dispatch(days=3))
    ctx = ChartContext(case, start="2025-01-02", end="2025-01-02")

    assert ctx.raw.index.min() == pd.Timestamp("2025-01-02")
    assert ctx.raw.index.max() < pd.Timestamp("2025-01-03")
    tabs = {tab.id: tab for tab in build_tabs(ctx)}
    assert "grid_fee_breakdown" not in {block.id for block in tabs["costs"].blocks}


def test_build_tabs_provides_kpis_for_steam_overview() -> None:
    case = build_case("steam", _steam_dispatch())

    overview = build_tabs(ChartContext(case), only="overview")[0]

    labels = [kpi.label for kpi in overview.kpis]
    assert {"Net operating cost", "Heat demand", "Gas replaced"} <= set(labels)


def _write_summary(folder: Path, **values: float) -> None:
    folder.mkdir(parents=True)
    pd.DataFrame([{"plant_name": "p", **values}]).to_csv(
        folder / "summary_indicators.csv", index=False
    )


def test_comparison_summarises_cases_and_builds_charts(tmp_path: Path) -> None:
    _write_summary(
        tmp_path / "scenarioa_2030_route_x",
        net_operating_cost_EUR=100.0,
        total_steel_production_t=10.0,
        total_co2_emissions_t=50.0,
        total_electricity_consumption_MWh=5.0,
    )
    _write_summary(
        tmp_path / "scenariob_2035_route_y",
        net_operating_cost_EUR=300.0,
        total_steel_production_t=10.0,
        total_co2_emissions_t=20.0,
        total_electricity_consumption_MWh=9.0,
    )

    frame = load_case_summaries([tmp_path])

    assert sorted(frame["case"]) == ["scenarioa_2030_route_x", "scenariob_2035_route_y"]
    row = frame.set_index("case").loc["scenarioa_2030_route_x"]
    assert (row["scenario"], row["year"], row["family"]) == ("scenarioa", 2030, "steel")
    assert row["specific_cost"] == pytest.approx(10.0)
    assert ranking_chart(frame, "net_cost", "scenario") is not None
    assert tradeoff_chart(frame, "net_cost", "co2", "scenario") is not None
    assert "Case comparison" in render_comparison_html(frame, plotly_js="cdn")


def test_dash_app_builds_with_expected_controls(tmp_path: Path) -> None:
    pytest.importorskip("dash")
    from flexi_mod.visualisation.dashboard.dash_app import create_app

    app = create_app(tmp_path)

    layout = str(app.layout)
    for component_id in ("case-dd", "plant-dd", "date-range", "tab-radio", "cmp-metric"):
        assert component_id in layout
