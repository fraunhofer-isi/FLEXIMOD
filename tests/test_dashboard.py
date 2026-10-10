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
    cost_stack_chart,
    load_case_summaries,
    ranking_chart,
    render_comparison_html,
    tradeoff_chart,
    volume_chart,
)
from flexi_mod.visualisation.dashboard.data import (
    ALL_PLANTS,
    aggregate_plants,
    auto_resolution,
    build_case,
    column_kind,
    detect_family,
    discover_cases,
    investment_costs,
    load_case,
    resample,
)
from flexi_mod.visualisation.dashboard.sections import ALL_BUILDERS, build_tabs, tab_ids
from flexi_mod.visualisation.dashboard.static_html import render_case_html, write_case_dashboard
from flexi_mod.visualisation.dashboard.steam_charts import (
    STEAM_TABLES,
    _active_window,
    atypical_grid_use,
    cashflow_waterfall,
    gas_only_benchmark,
    heat_split,
    plant_sizing,
    system_kpis,
)
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
                    "gas_price_EUR_per_MWh": 40.0,
                    "gas_input_MWh": (2.0 - discharge) / 0.8,
                    "electricity_trading_benchmark_EUR_per_MWh_el": 70.0,
                    "day_ahead_delivered_price_EUR_per_MWh": 60.0,
                    "IDC_delivered_price_EUR_per_MWh": 65.0,
                    "final_planned_electricity_MWh": discharge,
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
        for name, builder in ALL_BUILDERS.items():
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
    for label in (
        "Overview",
        "Heat &amp; storage",
        "Markets",
        "Operation",
        "Costs &amp; financials",
        "Patterns",
        "Data",
    ):
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
    assert {"Net operating cost", "Heat demand", "Electrification rate"} <= set(labels)


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


def _plants_csv(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "name": "plant_1",
                "technology": "thermal_storage",
                "max_capacity": 50.0,
                "max_power_charge": 10.0,
                "max_power_discharge": 10.0,
                "efficiency_charge": 0.98,
                "efficiency_discharge": 0.97,
            },
            {"name": "plant_1", "technology": "boiler", "max_power": 10.0, "efficiency": 0.8},
        ]
    ).to_csv(folder / "plants.csv", index=False)
    return folder


def test_heat_split_separates_pass_through_from_stored_heat() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="15min")
    raw = pd.DataFrame(
        {
            "etes_charge_MWh": [1.0, 0.0, 0.0],
            "etes_discharge_MWh": [1.0, 1.0, 0.0],
            "electric_boiler_heat_MWh": [0.5, 0.0, 0.0],
            "gas_heat_MWh": [0.0, 0.0, 2.0],
        },
        index=index,
    )

    split = heat_split(raw)

    assert split["direct"].tolist() == [1.5, 0.0, 0.0]
    assert split["storage"].tolist() == [0.0, 1.0, 0.0]
    assert split["gas"].tolist() == [0.0, 0.0, 2.0]


def test_gas_only_benchmark_uses_boiler_efficiency_from_dispatch() -> None:
    case = build_case("steam", _steam_dispatch())
    ctx = ChartContext(case)

    heat = ctx.total("heat_demand_MWh")
    # Efficiency 0.8 is implied by gas_input = gas_heat / 0.8; gas price is 40.
    assert gas_only_benchmark(ctx) == pytest.approx(heat / 0.8 * 40.0)


def test_cashflow_waterfall_ends_at_net_cost() -> None:
    case = build_case("steam", _steam_dispatch())
    ctx = ChartContext(case)

    figure = cashflow_waterfall(ctx)

    values = list(figure.data[0].y)
    measures = list(figure.data[0].measure)
    assert measures[0] == "absolute" and measures[-1] == "total"
    assert sum(values[:-1]) == pytest.approx(values[-1])
    net = ctx.total("net_operating_cost_EUR")
    assert values[-1] * 1000 == pytest.approx(net)


def test_investment_costs_follow_plants_csv() -> None:
    config = pd.DataFrame(
        [
            {"technology": "thermal_storage", "max_capacity": 50.0, "max_power_charge": 10.0},
            {"technology": "boiler", "max_power": 10.0},
        ]
    )

    costs = investment_costs(config)

    assert costs["capex"] == pytest.approx(50 * 20000 + 10 * 200000)
    assert costs["opex_annual"] == pytest.approx(costs["capex"] * 0.02)
    assert 0 < costs["annuity"] < costs["capex"]
    assert investment_costs(pd.DataFrame()) is None


def test_system_tab_needs_input_folder_and_reports_sizing(tmp_path: Path) -> None:
    without = build_case("steam", _steam_dispatch())
    with_inputs = build_case("steam", _steam_dispatch(), input_dir=_plants_csv(tmp_path))

    assert "system" not in [tab for tab, _ in tab_ids(ChartContext(without))]
    assert "system" in [tab for tab, _ in tab_ids(ChartContext(with_inputs))]
    ctx = ChartContext(with_inputs)
    labels = {label: value for label, value, _ in system_kpis(ctx)}
    assert labels["E-heater charge power"] == "10.0 MW el"
    assert labels["Storage capacity"] == "50.0 MWh th"
    assert plant_sizing(with_inputs.plant_config)["gas_boiler_efficiency"] == 0.8
    assert STEAM_TABLES["per_mw"](ctx) is not None


def test_steam_overview_reports_savings_against_gas_only(tmp_path: Path) -> None:
    case = build_case("steam", _steam_dispatch(), input_dir=_plants_csv(tmp_path))

    overview = build_tabs(ChartContext(case), only="overview")[0]

    labels = [kpi.label for kpi in overview.kpis]
    assert {
        "Gas-only benchmark",
        "Savings vs gas-only",
        "Net savings",
        "CAPEX (annualised)",
    } <= set(labels)


def test_active_window_picks_the_busiest_week() -> None:
    index = pd.date_range("2025-01-01", periods=21 * 96, freq="15min")
    raw = pd.DataFrame({"DA_position_MWh": 0.0}, index=index)
    raw.loc["2025-01-10":"2025-01-16", "DA_position_MWh"] = 1.0

    window = _active_window(raw, days=7)

    assert window.index.min() == pd.Timestamp("2025-01-10")
    assert window.index.max() < pd.Timestamp("2025-01-17")


def test_atypical_grid_use_needs_tiers_and_draws_high_load_bands() -> None:
    dispatch = _steam_dispatch(days=3)
    tiered = pd.DataFrame({"plant_name": ["plant_1"], "grid_realized_tier": ["low"]})

    with_tiers = ChartContext(build_case("steam", dispatch, summary=tiered))
    without = ChartContext(build_case("steam", dispatch))

    figure = atypical_grid_use(with_tiers)
    assert figure is not None and len(figure.layout.shapes) >= 1
    assert atypical_grid_use(without) is None


def test_comparison_cost_stack_and_volumes_for_steam_cases(tmp_path: Path) -> None:
    for name, gas in (("case_a", 100.0), ("case_b", 200.0)):
        _write_summary(
            tmp_path / name,
            total_heat_demand_MWh=10.0,
            total_gas_cost_EUR=gas,
            total_electricity_market_cost_EUR=50.0,
            IDC_buy_cost_EUR=5.0,
            IDC_sell_revenue_EUR=10.0,
            afrr_energy_cost_EUR=20.0,
            total_additional_electricity_charges_cost_EUR=30.0,
            total_afrr_capacity_revenue_EUR=15.0,
            net_operating_cost_EUR=gas + 50.0 + 30.0 - 15.0,
            total_DA_electricity_MWh=4.0,
            total_afrr_energy_activated_MWh=2.0,
        )

    frame = load_case_summaries([tmp_path]).set_index("case")

    # Day-ahead cost is what is left of the electricity market cost after intraday and aFRR.
    assert frame.loc["case_a", "cost_day_ahead"] == pytest.approx(50 - 5 + 10 - 20)
    assert cost_stack_chart(frame.reset_index()) is not None
    assert volume_chart(frame.reset_index()) is not None


def test_input_folder_is_found_from_the_output_folder_name(tmp_path: Path) -> None:
    from flexi_mod.visualisation.dashboard.data import guess_input_dir

    (tmp_path / "pyproject.toml").write_text("")
    _plants_csv(tmp_path / "data" / "input" / "case_x")
    _plants_csv(tmp_path / "data" / "input" / "case")
    output = tmp_path / "data" / "output" / "case_x_hybrid_etes_gas"
    output.mkdir(parents=True)

    # The longest matching input folder name wins ("case_x", not "case").
    assert guess_input_dir(output) == tmp_path / "data" / "input" / "case_x"
    assert guess_input_dir(tmp_path / "data" / "output" / "unrelated") is None


def test_plant_dashboards_inherit_from_the_generic_dashboard() -> None:
    from flexi_mod.visualisation.dashboard.sections import (
        BuildingDashboard,
        CementDashboard,
        IndustrialDashboard,
        PlantDashboard,
        SteamDashboard,
        SteelDashboard,
    )

    assert isinstance(PlantDashboard.for_family("cement"), CementDashboard)
    assert issubclass(CementDashboard, IndustrialDashboard)
    assert issubclass(SteelDashboard, IndustrialDashboard)
    assert issubclass(SteamDashboard, PlantDashboard)
    assert type(PlantDashboard.for_family("steam")) is SteamDashboard
    assert type(PlantDashboard.for_family("building")) is BuildingDashboard
    assert type(PlantDashboard.for_family("something_new")) is PlantDashboard


def test_a_subclass_gets_the_parent_charts_plus_its_own() -> None:
    from flexi_mod.visualisation.dashboard.sections import (
        BuildingDashboard,
        CementDashboard,
        SteamDashboard,
    )

    steam = SteamDashboard.chart_builders()
    assert "cost_breakdown" in steam and "cashflow_waterfall" in steam  # parent + own
    assert "production" not in steam and "ev_fleet" not in steam
    assert "production" in CementDashboard.chart_builders()
    assert "heat_coverage_annual" not in BuildingDashboard.chart_builders()


def test_every_chart_in_every_layout_has_a_builder() -> None:
    from flexi_mod.visualisation.dashboard.sections import PlantDashboard

    for family, klass in PlantDashboard._registry.items():
        available = set(klass.chart_builders()) | set(klass.table_builders())
        for tab in klass().layout():
            for chart in tab.charts:
                assert chart.key in available, f"{family}/{tab.id}: no builder for {chart.key}"


def test_new_plant_type_is_added_by_subclassing() -> None:
    import plotly.graph_objects as go

    from flexi_mod.visualisation.dashboard.sections import (
        ChartSpec,
        PlantDashboard,
        TabSpec,
    )

    def my_chart(ctx: ChartContext) -> go.Figure:
        return go.Figure(go.Bar(x=["a"], y=[1]))

    class HeatPumpDashboard(PlantDashboard):
        family = "heat_pump_for_test"
        builders = {"my_chart": my_chart}

        def overview_tab(self) -> TabSpec:
            return TabSpec("overview", "Overview", [ChartSpec("my_chart", "My chart")])

    try:
        case = build_case("heat pump", _steam_dispatch())
        case.family = "heat_pump_for_test"
        ctx = ChartContext(case)

        tabs = {tab.id: tab for tab in build_tabs(ctx)}

        assert [block.id for block in tabs["overview"].blocks] == ["my_chart"]
        assert "costs" in tabs  # inherited unchanged from the generic dashboard
    finally:
        PlantDashboard._registry.pop("heat_pump_for_test")
