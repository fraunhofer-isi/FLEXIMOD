# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dashboard content model: tabs, KPI tiles and chart/table blocks per plant family.

Both front ends (static HTML and Dash) render the :class:`Tab` objects produced here, so the
two dashboards always show the same content.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go

from flexi_mod.visualisation.dashboard.charts import (
    BUILDERS,
    ChartContext,
    money,
    number,
)
from flexi_mod.visualisation.dashboard.data import column_kind


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


# (builder key, title, description, full width)
ChartSpec = tuple[str, str, str, bool]

_ELECTRICITY_MARKETS: list[ChartSpec] = [
    (
        "markets_overview",
        "Prices and electricity procurement",
        "Stacked bars show where each MWh was bought (day-ahead, intraday, aFRR activation); "
        "bars below zero are intraday volumes sold back. The line is the actual consumption.",
        True,
    ),
    (
        "afrr_capacity",
        "aFRR capacity reservation",
        "Downward capacity held back for the balancing market and the price it earned.",
        True,
    ),
]

_PATTERNS: list[ChartSpec] = [
    (
        "load_heatmap",
        "Load by hour of day",
        "Each column is one day, each row an hour. Dark cells are high load.",
        True,
    ),
    (
        "price_response",
        "Load response to price",
        "Average load in each price decile: a flexible plant leans to the cheap deciles.",
        False,
    ),
    (
        "load_duration",
        "Load duration curve",
        "Load sorted from highest to lowest. A flat curve means a steady baseload.",
        False,
    ),
]

_COSTS: list[ChartSpec] = [
    (
        "cost_breakdown",
        "Cost breakdown",
        "Signed contribution of each cost component. Credits (revenues) are shown in green.",
        False,
    ),
    (
        "cumulative_cost",
        "Cumulative operating cost",
        "How the net operating cost builds up over the period.",
        False,
    ),
    (
        "grid_fee_breakdown",
        "Grid fees",
        "Grid fee components after the ex-post tier check.",
        False,
    ),
    ("emissions", "Emissions", "CO₂ emitted per period and accumulated.", False),
]

_LAYOUT: dict[str, list[tuple[str, str, list[ChartSpec]]]] = {
    "steam": [
        (
            "overview",
            "Overview",
            [
                (
                    "heat_and_storage",
                    "Heat supply and storage",
                    "How the heat demand is covered and how full the thermal storage is.",
                    True,
                ),
                _ELECTRICITY_MARKETS[0],
            ],
        ),
        (
            "markets",
            "Markets",
            [
                *_ELECTRICITY_MARKETS[1:],
                (
                    "gas_replacement",
                    "Gas replacement by market stage",
                    "Gas heat that is still needed after each sequential market stage.",
                    False,
                ),
                (
                    "storage_sources",
                    "Stored heat by procurement market",
                    "Which market the heat in the storage was bought on.",
                    False,
                ),
            ],
        ),
        ("operation", "Operation", [("sample_window", "Representative window", "", True)]),
        ("costs", "Costs & emissions", _COSTS),
        ("patterns", "Patterns", _PATTERNS),
    ],
    "cement": [
        (
            "overview",
            "Overview",
            [
                (
                    "production",
                    "Production",
                    "Output per period and the demand still to produce.",
                    True,
                ),
                _ELECTRICITY_MARKETS[0],
            ],
        ),
        ("markets", "Markets", _ELECTRICITY_MARKETS[1:]),
        (
            "operation",
            "Operation",
            [
                ("energy_carriers", "Energy carriers", "Energy input by carrier.", True),
                (
                    "unit_electricity",
                    "Electricity by process unit",
                    "Where the electricity is used in the process chain.",
                    True,
                ),
                (
                    "unit_status",
                    "Process unit utilisation",
                    "Share of each day a unit is running. Gaps show where flexibility is used.",
                    True,
                ),
                ("sample_window", "Representative window", "", True),
            ],
        ),
        ("costs", "Costs & emissions", _COSTS),
        ("patterns", "Patterns", _PATTERNS),
    ],
    "building": [
        (
            "overview",
            "Overview",
            [
                (
                    "building_balance",
                    "Grid exchange and PV",
                    "Imports above the axis, exports below. The line is the building demand.",
                    True,
                ),
                (
                    "building_tariff",
                    "Tariff and billing peak",
                    "Energy price and the grid import power against the monthly billing peak.",
                    True,
                ),
            ],
        ),
        (
            "grid",
            "Grid & PV",
            [
                (
                    "grid_stress",
                    "Regional grid stress",
                    "Whether imports coincide with high regional grid load.",
                    True,
                ),
                ("pv_use", "PV use", "Where the PV generation ends up each day.", False),
            ],
        ),
        (
            "operation",
            "Operation",
            [
                (
                    "ev_fleet",
                    "EV fleet",
                    "Fleet state of charge and the charge/discharge schedule.",
                    True,
                ),
                ("sample_window", "Representative window", "", True),
            ],
        ),
        (
            "costs",
            "Costs",
            [_COSTS[0], _COSTS[1]],
        ),
        (
            "patterns",
            "Patterns",
            [
                (
                    "load_heatmap",
                    "Net grid import by hour of day",
                    "Each column is one day, each row an hour. Dark cells are high import.",
                    True,
                ),
                _PATTERNS[2],
            ],
        ),
    ],
}
_LAYOUT["steel"] = _LAYOUT["cement"]
_LAYOUT["generic"] = [
    ("overview", "Overview", [_ELECTRICITY_MARKETS[0]]),
    ("costs", "Costs & emissions", _COSTS),
    ("patterns", "Patterns", _PATTERNS),
]


def build_tabs(ctx: ChartContext, only: str | None = None) -> list[Tab]:
    """Build all tabs (or only ``only``) for the selected case, plant and period."""

    tabs: list[Tab] = []
    layout = _LAYOUT.get(ctx.case.family, _LAYOUT["generic"])
    for tab_id, label, specs in [*layout, ("data", "Data", [])]:
        if only is not None and tab_id != only:
            continue
        tab = Tab(id=tab_id, label=label)
        if tab_id == "overview":
            tab.kpis = build_kpis(ctx)
        if tab_id == "data":
            tab.blocks = data_blocks(ctx)
        for key, title, description, wide in specs:
            fig = BUILDERS[key](ctx)
            if fig is None:
                continue
            tab.blocks.append(
                Block(
                    id=key,
                    title=title,
                    description=description,
                    figure=fig,
                    caption=getattr(fig, "_ctx_caption", "") or "",
                    wide=wide,
                )
            )
        if tab.blocks or tab.kpis:
            tabs.append(tab)
    return tabs


def tab_ids(ctx: ChartContext) -> list[tuple[str, str]]:
    layout = _LAYOUT.get(ctx.case.family, _LAYOUT["generic"])
    return [(tab_id, label) for tab_id, label, _ in layout] + [("data", "Data")]


def build_kpis(ctx: ChartContext) -> list[Kpi]:
    family = ctx.case.family
    builders = {
        "steam": _steam_kpis,
        "cement": _industrial_kpis,
        "steel": _industrial_kpis,
        "building": _building_kpis,
    }
    return builders.get(family, _generic_kpis)(ctx)


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


def _steam_kpis(ctx: ChartContext) -> list[Kpi]:
    tiles: list[Kpi] = []
    net, detail = _net_cost(ctx)
    demand = ctx.total("heat_demand_MWh")
    if net is not None:
        tiles.append(Kpi("Net operating cost", money(net), detail))
        if demand:
            tiles.append(Kpi("Cost of heat", f"€{net / demand:,.1f}/MWh", "per MWh th delivered"))
    if demand:
        tiles.append(Kpi("Heat demand", number(demand, "MWh th")))
        gas = ctx.total("gas_heat_MWh")
        if gas is not None:
            tiles.append(
                Kpi("Gas replaced", f"{1 - gas / demand:.0%}", f"{number(gas, 'MWh th')} gas")
            )
    consumption = ctx.total("actual_electricity_consumption_MWh") or ctx.total(
        "electricity_consumption_MWh"
    )
    if consumption is not None:
        tiles.append(Kpi("Electricity used", number(consumption, "MWh")))
    emissions = ctx.total("total_emissions_kg")
    if emissions is not None:
        tiles.append(Kpi("Emissions", number(emissions / 1000.0, "t CO₂")))
    credit = ctx.total("afrr_capacity_net_value_EUR")
    if credit:
        tiles.append(Kpi("aFRR capacity value", money(credit), "net of opportunity cost"))
    return tiles


def _industrial_kpis(ctx: ChartContext) -> list[Kpi]:
    tiles: list[Kpi] = []
    output_column, label, unit = {
        "cement": ("clinker_output_t", "Clinker produced", "t"),
        "steel": ("steel_output_t", "Steel produced", "t"),
    }.get(ctx.case.family, ("", "", ""))
    output = ctx.total(output_column) if output_column else None
    net, detail = _net_cost(ctx)
    if net is not None:
        tiles.append(Kpi("Net operating cost", money(net), detail))
    if output:
        tiles.append(Kpi(label, number(output, unit)))
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


def _building_kpis(ctx: ChartContext) -> list[Kpi]:
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
        tiles.append(Kpi("Peak import", number(float(ctx.raw["grid_import_MW"].max()), "MW", 2)))
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


def _generic_kpis(ctx: ChartContext) -> list[Kpi]:
    net, detail = _net_cost(ctx)
    tiles = [Kpi("Net operating cost", money(net), detail)] if net is not None else []
    consumption = ctx.total("actual_electricity_consumption_MWh")
    if consumption is not None:
        tiles.append(Kpi("Electricity used", number(consumption, "MWh")))
    return tiles


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
