# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dashboard content per plant type: tabs, KPI tiles and chart/table blocks.

Each plant type is a class. :class:`PlantDashboard` holds the generic tabs and charts and each
plant type inherits from it, overriding only what differs::

    PlantDashboard                  generic: markets, costs, patterns
    |-- SteamDashboard              steam / ETES / boiler plants
    |-- IndustrialDashboard         production-based plants
    |   |-- CementDashboard
    |   `-- SteelDashboard
    `-- BuildingDashboard           building with EV fleet and PV

A class declares its own chart builders (merged with those of its parents), the layout as
:class:`TabSpec` lists, and the headline tiles. Setting ``family`` registers the class for the
plant type detected in the dispatch table. Both front ends (static HTML and Dash) render the
:class:`Tab` objects built here, so they always show the same content.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

import pandas as pd
import plotly.graph_objects as go

from flexi_mod.visualisation.dashboard.charts import (
    BUILDERS,
    BUILDING_BUILDERS,
    GENERIC_BUILDERS,
    INDUSTRIAL_BUILDERS,
    ChartContext,
    FigureBuilder,
    money,
    number,
)
from flexi_mod.visualisation.dashboard.data import column_kind
from flexi_mod.visualisation.dashboard.steam_charts import (
    STEAM_BUILDERS,
    STEAM_TABLES,
    grid_fee_kpis,
    steam_kpis,
    system_kpis,
)

TableBuilder = Callable[[ChartContext], pd.DataFrame | None]

# Every figure builder in one lookup (handy in notebooks and tests).
ALL_BUILDERS = {**BUILDERS, **STEAM_BUILDERS}


@dataclass(frozen=True)
class Kpi:
    label: str
    value: str
    detail: str = ""


@dataclass
class Block:
    """One chart or table on a tab."""

    id: str
    title: str
    description: str = ""
    figure: go.Figure | None = None
    table: pd.DataFrame | None = None
    caption: str = ""
    wide: bool = True


@dataclass
class Tab:
    id: str
    label: str
    kpis: list[Kpi] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)


class ChartSpec(NamedTuple):
    """A chart or table on a tab: the builder key, its heading and whether it is full width."""

    key: str
    title: str
    description: str = ""
    wide: bool = True


@dataclass
class TabSpec:
    id: str
    label: str
    charts: list[ChartSpec] = field(default_factory=list)


# Chart specs shared by several plant types.
MARKETS_OVERVIEW = ChartSpec(
    "markets_overview",
    "Prices and electricity procurement",
    "Stacked bars show where each MWh was bought (day-ahead, intraday, aFRR activation); "
    "bars below zero are intraday volumes sold back. The line is the actual consumption.",
)
AFRR_CAPACITY = ChartSpec(
    "afrr_capacity",
    "aFRR capacity reservation",
    "Downward capacity held back for the balancing market and the price it earned.",
)
SAMPLE_WINDOW = ChartSpec("sample_window", "Representative window", "")
LOAD_HEATMAP = ChartSpec(
    "load_heatmap",
    "Load by hour of day",
    "Each column is one day, each row an hour. Dark cells are high load.",
)
PRICE_RESPONSE = ChartSpec(
    "price_response",
    "Load response to price",
    "Average load in each price decile: a flexible plant leans to the cheap deciles.",
    wide=False,
)
LOAD_DURATION = ChartSpec(
    "load_duration",
    "Load duration curve",
    "Load sorted from highest to lowest. A flat curve means a steady baseload.",
    wide=False,
)
COST_BREAKDOWN = ChartSpec(
    "cost_breakdown",
    "Cost breakdown",
    "Signed contribution of each cost component. Credits (revenues) are shown in green.",
    wide=False,
)
CUMULATIVE_COST = ChartSpec(
    "cumulative_cost",
    "Cumulative operating cost",
    "How the net operating cost builds up over the period.",
    wide=False,
)
GRID_FEES = ChartSpec(
    "grid_fee_breakdown",
    "Grid fees",
    "Grid fee components after the ex-post tier check.",
    wide=False,
)
EMISSIONS = ChartSpec("emissions", "Emissions", "CO₂ emitted per period and accumulated.", False)


class PlantDashboard:
    """Generic dashboard: electricity markets, costs and load patterns.

    Subclass it for a plant type. Override the ``*_tab`` methods to change a tab, ``layout`` to
    add or reorder tabs, ``overview_kpis`` for the headline tiles and ``tab_available`` to hide
    tabs whose inputs are missing. Extend ``builders`` (and ``tables``) with the charts that only
    this plant type needs; the charts of parent classes stay available.
    """

    family: ClassVar[str] = "generic"
    builders: ClassVar[dict[str, FigureBuilder]] = GENERIC_BUILDERS
    tables: ClassVar[dict[str, TableBuilder]] = {}
    _registry: ClassVar[dict[str, type[PlantDashboard]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "family" in cls.__dict__:
            PlantDashboard._registry[cls.family] = cls

    # ---- lookup -------------------------------------------------------------------
    @classmethod
    def for_family(cls, family: str) -> PlantDashboard:
        """The dashboard registered for a plant type (the generic one when unknown)."""

        return PlantDashboard._registry.get(family, PlantDashboard)()

    @classmethod
    def chart_builders(cls) -> dict[str, FigureBuilder]:
        """Figure builders of this class and its parents (the subclass wins on a clash)."""

        merged: dict[str, FigureBuilder] = {}
        for klass in reversed(cls.__mro__):
            merged.update(klass.__dict__.get("builders", {}))
        return merged

    @classmethod
    def table_builders(cls) -> dict[str, TableBuilder]:
        merged: dict[str, TableBuilder] = {}
        for klass in reversed(cls.__mro__):
            merged.update(klass.__dict__.get("tables", {}))
        return merged

    # ---- layout -------------------------------------------------------------------
    def layout(self) -> list[TabSpec]:
        """Tabs in display order (the Data tab is appended automatically)."""

        return [self.overview_tab(), self.costs_tab(), self.patterns_tab()]

    def overview_tab(self) -> TabSpec:
        return TabSpec("overview", "Overview", [MARKETS_OVERVIEW])

    def costs_tab(self) -> TabSpec:
        return TabSpec(
            "costs",
            "Costs & emissions",
            [COST_BREAKDOWN, CUMULATIVE_COST, GRID_FEES, EMISSIONS],
        )

    def patterns_tab(self) -> TabSpec:
        return TabSpec("patterns", "Patterns", [LOAD_HEATMAP, PRICE_RESPONSE, LOAD_DURATION])

    def tab_available(self, ctx: ChartContext, tab_id: str) -> bool:
        """Whether a tab can be offered for this case (override to hide tabs)."""

        return True

    # ---- headline tiles ---------------------------------------------------------------
    def overview_kpis(self, ctx: ChartContext) -> list[Kpi]:
        net, detail = _net_cost(ctx)
        tiles = [Kpi("Net operating cost", money(net), detail)] if net is not None else []
        consumption = ctx.total("actual_electricity_consumption_MWh")
        if consumption is not None:
            tiles.append(Kpi("Electricity used", number(consumption, "MWh")))
        return tiles

    def tab_kpis(self, ctx: ChartContext, tab_id: str) -> list[Kpi]:
        return self.overview_kpis(ctx) if tab_id == "overview" else []

    # ---- assembly ---------------------------------------------------------------------
    def tab_ids(self, ctx: ChartContext) -> list[tuple[str, str]]:
        """Tabs offered for the case. Tabs that need missing inputs are left out."""

        tabs = [(t.id, t.label) for t in self.layout() if self.tab_available(ctx, t.id)]
        return [*tabs, ("data", "Data")]

    def build_tabs(self, ctx: ChartContext, only: str | None = None) -> list[Tab]:
        """Build all tabs (or only ``only``) for the selected case, plant and period."""

        builders, tables = self.chart_builders(), self.table_builders()
        tabs: list[Tab] = []
        for spec in [*self.layout(), TabSpec("data", "Data")]:
            if only is not None and spec.id != only:
                continue
            tab = Tab(id=spec.id, label=spec.label, kpis=self.tab_kpis(ctx, spec.id))
            if spec.id == "data":
                tab.blocks = data_blocks(ctx)
            for chart in spec.charts:
                if chart.key in tables:
                    table = tables[chart.key](ctx)
                    if table is not None and not table.empty:
                        tab.blocks.append(
                            Block(chart.key, chart.title, chart.description, table=table)
                        )
                    continue
                figure = builders[chart.key](ctx)
                if figure is None:
                    continue
                tab.blocks.append(
                    Block(
                        id=chart.key,
                        title=chart.title,
                        description=chart.description,
                        figure=figure,
                        caption=getattr(figure, "_ctx_caption", "") or "",
                        wide=chart.wide,
                    )
                )
            if tab.blocks or tab.kpis:
                tabs.append(tab)
        return tabs


class SteamDashboard(PlantDashboard):
    """Steam, ETES and boiler plants: heat coverage, benchmark savings, markets, grid fees."""

    family = "steam"
    builders = STEAM_BUILDERS
    tables = STEAM_TABLES

    def layout(self) -> list[TabSpec]:
        standard = {tab.id: tab for tab in super().layout()}
        return [
            standard["overview"],
            self.heat_tab(),
            self.markets_tab(),
            self.operation_tab(),
            standard["costs"],
            self.grid_fees_tab(),
            standard["patterns"],
            self.system_tab(),
        ]

    def overview_tab(self) -> TabSpec:
        return TabSpec(
            "overview",
            "Overview",
            [
                ChartSpec(
                    "heat_and_storage",
                    "Heat supply and storage",
                    "How the heat demand is covered and how full the thermal storage is.",
                ),
                MARKETS_OVERVIEW,
            ],
        )

    def heat_tab(self) -> TabSpec:
        return TabSpec(
            "heat",
            "Heat & storage",
            [
                ChartSpec(
                    "heat_coverage_annual",
                    "Heat demand coverage",
                    "Share of the heat delivered by direct electric supply (storage charging "
                    "and discharging at the same time), by stored heat, and by the gas boiler.",
                ),
                ChartSpec(
                    "heat_coverage_monthly",
                    "Monthly heat coverage",
                    "Heat by supply type per month and the electrified share of the heat.",
                ),
                ChartSpec(
                    "gas_replacement",
                    "Gas replacement by market stage",
                    "Gas heat that is still needed after each sequential market stage.",
                    wide=False,
                ),
                ChartSpec(
                    "storage_sources",
                    "Stored heat by procurement market",
                    "Which market the heat in the storage was bought on.",
                    wide=False,
                ),
                ChartSpec(
                    "storage_operation",
                    "Storage operation",
                    "Charging above the axis, discharging below, and the state of charge.",
                ),
            ],
        )

    def markets_tab(self) -> TabSpec:
        return TabSpec(
            "markets",
            "Markets",
            [
                ChartSpec(
                    "monthly_procurement",
                    "Monthly electricity procurement",
                    "Electricity bought per month by market. Day-ahead is shown net of the "
                    "volume sold back in the intraday market.",
                    wide=False,
                ),
                ChartSpec(
                    "monthly_market_value",
                    "Monthly market value",
                    "Savings of each market against the gas-based electricity benchmark: "
                    "day-ahead and intraday purchases below the benchmark, aFRR capacity "
                    "revenue and aFRR energy value.",
                    wide=False,
                ),
                AFRR_CAPACITY,
                ChartSpec(
                    "monthly_price",
                    "Monthly day-ahead price",
                    "Average day-ahead market price per month.",
                    wide=False,
                ),
                ChartSpec(
                    "electricity_balance",
                    "Electricity balance by market channel",
                    "Volumes per market channel; they add up to the total consumption.",
                ),
                ChartSpec(
                    "price_statistics",
                    "Price statistics",
                    "Mean, extremes and the 10th and 90th percentile of each price series.",
                ),
            ],
        )

    def operation_tab(self) -> TabSpec:
        return TabSpec(
            "operation",
            "Operation",
            [
                ChartSpec(
                    "sequential_profile",
                    "Sequential market profile",
                    "The most active week: day-ahead baseline, intraday adjustment, aFRR and "
                    "final electricity, gas boiler heat and storage content.",
                )
            ],
        )

    def costs_tab(self) -> TabSpec:
        return TabSpec(
            "costs",
            "Costs & financials",
            [
                ChartSpec(
                    "cashflow_waterfall",
                    "Cashflow from the gas-only benchmark to net cost",
                    "Starts at the cost of heating with the gas boiler alone. Blue steps add "
                    "cost, green steps are savings or revenue.",
                ),
                COST_BREAKDOWN,
                CUMULATIVE_COST,
                EMISSIONS,
            ],
        )

    def grid_fees_tab(self) -> TabSpec:
        return TabSpec(
            "gridfees",
            "Grid fees",
            [
                ChartSpec(
                    "grid_fee_breakdown",
                    "Grid fee components",
                    "Grid fee components after the ex-post full-load-hour tier check.",
                    wide=False,
                ),
                ChartSpec(
                    "grid_peaks",
                    "Capacity charge peak basis",
                    "Atypical grid use: the capacity charge is billed on the peak inside the "
                    "high-load windows, not on the annual peak.",
                    wide=False,
                ),
                ChartSpec(
                    "atypical_grid_use",
                    "How grid draw avoids the high-load windows",
                    "Average weekday profile by hour for winter and autumn. Shaded bands are "
                    "the DSO high-load windows; draw collapses there while gas and stored "
                    "heat carry the load.",
                ),
            ],
        )

    def patterns_tab(self) -> TabSpec:
        return TabSpec(
            "patterns",
            "Patterns",
            [
                ChartSpec(
                    "heat_heatmap",
                    "Heat demand by hour of day",
                    "Each column is one day, each row an hour. Dark cells are high demand.",
                ),
                ChartSpec(
                    "demand_weekday_heatmap",
                    "Heat demand by weekday and hour",
                    "Average demand for every weekday and hour.",
                    wide=False,
                ),
                ChartSpec(
                    "demand_averages",
                    "Average heat demand",
                    "Average by hour of day and by weekday.",
                    wide=False,
                ),
                ChartSpec(
                    "heat_duration",
                    "Heat load duration curve",
                    "Heat demand sorted from highest to lowest.",
                    wide=False,
                ),
                *super().patterns_tab().charts,
            ],
        )

    def system_tab(self) -> TabSpec:
        return TabSpec(
            "system",
            "System setup",
            [
                ChartSpec(
                    "plant_config",
                    "Plant technology configuration",
                    "Technologies of the case from plants.csv.",
                ),
                ChartSpec(
                    "charges",
                    "Additional electricity charges",
                    "Network charges and levies from additional_charges.csv (EUR per MWh el).",
                ),
                ChartSpec(
                    "per_mw",
                    "Results per installed MW of e-heater",
                    "Cost components divided by the installed e-heater power.",
                ),
            ],
        )

    def tab_available(self, ctx: ChartContext, tab_id: str) -> bool:
        case = ctx.case
        if tab_id == "system":
            return not case.plant_config.empty
        if tab_id == "gridfees":
            return not case.summary.empty and "grid_fee_total_EUR" in case.summary
        return True

    def overview_kpis(self, ctx: ChartContext) -> list[Kpi]:
        return [Kpi(*tile) for tile in steam_kpis(ctx)]

    def tab_kpis(self, ctx: ChartContext, tab_id: str) -> list[Kpi]:
        tiles = {"gridfees": grid_fee_kpis, "system": system_kpis}.get(tab_id)
        if tiles is not None:
            return [Kpi(*tile) for tile in tiles(ctx)]
        return super().tab_kpis(ctx, tab_id)


class IndustrialDashboard(PlantDashboard):
    """Production-based plants. Subclasses name the output column and its label."""

    builders = INDUSTRIAL_BUILDERS
    output_column: ClassVar[str] = ""
    output_label: ClassVar[str] = "Output"

    def layout(self) -> list[TabSpec]:
        standard = {tab.id: tab for tab in super().layout()}
        return [
            standard["overview"],
            TabSpec("markets", "Markets", [AFRR_CAPACITY]),
            self.operation_tab(),
            standard["costs"],
            standard["patterns"],
        ]

    def overview_tab(self) -> TabSpec:
        return TabSpec(
            "overview",
            "Overview",
            [
                ChartSpec(
                    "production",
                    "Production",
                    "Output per period and the demand still to produce.",
                ),
                MARKETS_OVERVIEW,
            ],
        )

    def operation_tab(self) -> TabSpec:
        return TabSpec(
            "operation",
            "Operation",
            [
                ChartSpec("energy_carriers", "Energy carriers", "Energy input by carrier."),
                ChartSpec(
                    "unit_electricity",
                    "Electricity by process unit",
                    "Where the electricity is used in the process chain.",
                ),
                ChartSpec(
                    "unit_status",
                    "Process unit utilisation",
                    "Share of each day a unit is running. Gaps show where flexibility is used.",
                ),
                SAMPLE_WINDOW,
            ],
        )

    def overview_kpis(self, ctx: ChartContext) -> list[Kpi]:
        tiles: list[Kpi] = []
        output = ctx.total(self.output_column) if self.output_column else None
        net, detail = _net_cost(ctx)
        if net is not None:
            tiles.append(Kpi("Net operating cost", money(net), detail))
        if output:
            tiles.append(Kpi(self.output_label, number(output, "t")))
            if net is not None:
                tiles.append(Kpi("Specific cost", f"€{net / output:,.1f}/t", "per tonne produced"))
        electricity = ctx.total("total_electricity_consumption_MWh")
        if electricity is not None:
            detail = f"{electricity / output:,.2f} MWh/t" if output else ""
            tiles.append(Kpi("Electricity used", number(electricity, "MWh"), detail))
        co2 = ctx.total("co2_emissions_t")
        if co2 is not None:
            detail = f"{co2 / output:,.2f} t/t" if output else ""
            tiles.append(Kpi("CO₂ emissions", number(co2, "t"), detail))
        credit = ctx.total("afrr_capacity_net_value_EUR")
        if credit:
            tiles.append(Kpi("aFRR capacity value", money(credit), "net of opportunity cost"))
        return tiles


class CementDashboard(IndustrialDashboard):
    family = "cement"
    output_column = "clinker_output_t"
    output_label = "Clinker produced"


class SteelDashboard(IndustrialDashboard):
    family = "steel"
    output_column = "steel_output_t"
    output_label = "Steel produced"


class BuildingDashboard(PlantDashboard):
    """Buildings with EV fleets and PV: grid exchange, tariff, fleet and regional grid stress."""

    family = "building"
    builders = BUILDING_BUILDERS

    def layout(self) -> list[TabSpec]:
        standard = {tab.id: tab for tab in super().layout()}
        return [
            standard["overview"],
            TabSpec(
                "grid",
                "Grid & PV",
                [
                    ChartSpec(
                        "grid_stress",
                        "Regional grid stress",
                        "Whether imports coincide with high regional grid load.",
                    ),
                    ChartSpec(
                        "pv_use",
                        "PV use",
                        "Where the PV generation ends up each day.",
                        wide=False,
                    ),
                ],
            ),
            TabSpec(
                "operation",
                "Operation",
                [
                    ChartSpec(
                        "ev_fleet",
                        "EV fleet",
                        "Fleet state of charge and the charge/discharge schedule.",
                    ),
                    SAMPLE_WINDOW,
                ],
            ),
            standard["costs"],
            standard["patterns"],
        ]

    def overview_tab(self) -> TabSpec:
        return TabSpec(
            "overview",
            "Overview",
            [
                ChartSpec(
                    "building_balance",
                    "Grid exchange and PV",
                    "Imports above the axis, exports below. The line is the building demand.",
                ),
                ChartSpec(
                    "building_tariff",
                    "Tariff and billing peak",
                    "Energy price and the grid import power against the monthly billing peak.",
                ),
            ],
        )

    def costs_tab(self) -> TabSpec:
        return TabSpec("costs", "Costs", [COST_BREAKDOWN, CUMULATIVE_COST])

    def patterns_tab(self) -> TabSpec:
        return TabSpec(
            "patterns",
            "Patterns",
            [
                ChartSpec(
                    "load_heatmap",
                    "Net grid import by hour of day",
                    "Each column is one day, each row an hour. Dark cells are high import.",
                ),
                LOAD_DURATION,
            ],
        )

    def overview_kpis(self, ctx: ChartContext) -> list[Kpi]:
        tiles: list[Kpi] = []
        currency = ctx.case.currency
        total = ctx.total("total_cost")
        if total is not None:
            tiles.append(Kpi("Total cost", money(total, currency), "energy plus demand charge"))
        imported = ctx.total("grid_import_MWh")
        if imported is not None:
            tiles.append(Kpi("Grid import", number(imported, "MWh")))
        exported = ctx.total("grid_export_MWh")
        if exported:
            tiles.append(Kpi("Grid export", number(exported, "MWh")))
        if "grid_import_MW" in ctx.raw:
            tiles.append(
                Kpi("Peak import", number(float(ctx.raw["grid_import_MW"].max()), "MW", 2))
            )
        generation = ctx.total("pv_generation_MWh")
        if generation:
            own = max(generation - (exported or 0.0), 0.0)
            tiles.append(
                Kpi("PV used on site", f"{own / generation:.0%}", number(generation, "MWh PV"))
            )
        unmet = ctx.total("unmet_trip_energy_MWh")
        if unmet is not None:
            tiles.append(Kpi("Unmet trip energy", number(unmet, "MWh", 2), "should be zero"))
        return tiles


def _net_cost(ctx: ChartContext) -> tuple[float | None, str]:
    net = ctx.total("net_operating_cost_EUR")
    if net is None:
        return None, ""
    extra = (
        None if ctx.sliced else ctx.case.summary_total("grid_fee_ex_post_addition_EUR", ctx.plant)
    )
    if extra:
        return net + extra, "incl. grid fee correction"
    return net, "net of aFRR credits" if ctx.total("afrr_capacity_net_value_EUR") else ""


# ---- module-level API used by the front ends -------------------------------------------


def dashboard_for(ctx: ChartContext) -> PlantDashboard:
    """The dashboard class instance for the plant type of the selected case."""

    return PlantDashboard.for_family(ctx.case.family)


def build_tabs(ctx: ChartContext, only: str | None = None) -> list[Tab]:
    """Build all tabs (or only ``only``) for the selected case, plant and period."""

    return dashboard_for(ctx).build_tabs(ctx, only)


def tab_ids(ctx: ChartContext) -> list[tuple[str, str]]:
    """Tabs offered for the case. Tabs that need missing inputs are left out."""

    return dashboard_for(ctx).tab_ids(ctx)


def tab_kpis(ctx: ChartContext, tab_id: str) -> list[Kpi]:
    """Headline tiles shown at the top of a tab."""

    return dashboard_for(ctx).tab_kpis(ctx, tab_id)


def build_kpis(ctx: ChartContext) -> list[Kpi]:
    return dashboard_for(ctx).overview_kpis(ctx)


def summary_table(ctx: ChartContext) -> pd.DataFrame:
    """Summary indicators of the selected plants as an ``indicator / value`` table."""

    rows = ctx.case.summary_rows(ctx.plant)
    if rows.empty:
        return pd.DataFrame()
    numeric = rows.select_dtypes("number")
    values: dict[str, float] = {}
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            continue
        aggregate = series.mean() if column_kind(column) == "price" else series.sum()
        values[column] = float(aggregate)
    return pd.DataFrame({"indicator": list(values), "value": list(values.values())})


def data_blocks(ctx: ChartContext) -> list[Block]:
    blocks: list[Block] = []
    summary = summary_table(ctx)
    if not summary.empty:
        blocks.append(
            Block(
                "summary",
                "Summary indicators",
                "Totals over the full run for the selected plants (not limited to the period).",
                table=summary,
            )
        )
    blocks_frame = ctx.case.blocks_frame(ctx.plant)
    if not blocks_frame.empty:
        blocks.append(
            Block(
                "afrr_blocks",
                "aFRR capacity blocks",
                "One row per capacity block with prices, quantities and bid decisions.",
                table=blocks_frame.head(500),
            )
        )
    fees = ctx.case.grid_fees
    if not fees.empty:
        blocks.append(Block("grid_fees", "Grid fee summary", table=fees.head(500)))
    return blocks
