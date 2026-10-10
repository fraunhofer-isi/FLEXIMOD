# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plotly figure builders shared by the static and the Dash dashboard.

Every builder takes a :class:`ChartContext` and returns a figure, or ``None`` when the case
does not contain the columns the chart needs. Charts never use a second y-axis: measures
with different units go on stacked rows that share the time axis.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cached_property

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flexi_mod.visualisation.analytics import select_sample_day
from flexi_mod.visualisation.dashboard.data import (
    ALL_PLANTS,
    RESOLUTIONS,
    CaseData,
    auto_resolution,
    resample,
    slice_period,
)
from flexi_mod.visualisation.dashboard.theme import LIGHT, Theme, apply_layout, with_alpha

RESOLUTION_LABELS = {
    "1h": "hour",
    "6h": "6 hours",
    "1D": "day",
    "1W": "week",
}

ELECTRICITY_COLUMNS = (
    "actual_electricity_consumption_MWh",
    "total_electricity_consumption_MWh",
    "electricity_consumption_MWh",
    "net_grid_import_MWh",
)

FigureBuilder = Callable[["ChartContext"], go.Figure | None]


class ChartContext:
    """Selected case, plant, period and resolution plus the derived data frames."""

    def __init__(
        self,
        case: CaseData,
        plant: str = ALL_PLANTS,
        theme: Theme = LIGHT,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        resolution: str = "auto",
    ) -> None:
        self.case = case
        self.plant = plant
        self.theme = theme
        self.start = start
        self.end = end
        self.resolution = resolution

    @property
    def sliced(self) -> bool:
        return self.start is not None or self.end is not None

    @cached_property
    def raw(self) -> pd.DataFrame:
        return slice_period(self.case.frame(self.plant), self.start, self.end)

    @cached_property
    def resolution_key(self) -> str:
        if self.resolution in RESOLUTIONS:
            return self.resolution
        return auto_resolution(self.raw.index, self.case.step_hours)

    @cached_property
    def df(self) -> pd.DataFrame:
        return resample(self.raw, RESOLUTIONS[self.resolution_key])

    @property
    def native(self) -> bool:
        return self.resolution_key == "native"

    @property
    def step_hours(self) -> float:
        hours = {"1h": 1.0, "6h": 6.0, "1D": 24.0, "1W": 168.0}
        return self.case.step_hours if self.native else hours[self.resolution_key]

    @property
    def step_label(self) -> str:
        if self.native:
            minutes = self.case.step_hours * 60
            return f"{minutes:g} min"
        return RESOLUTION_LABELS[self.resolution_key]

    @property
    def line_shape(self) -> str:
        return "hv" if self.native else "linear"

    def total(self, column: str) -> float | None:
        if column not in self.raw:
            return None
        return float(self.raw[column].sum())

    def electricity_column(self) -> str | None:
        return first_column(self.raw, ELECTRICITY_COLUMNS)

    def sample_window(self, days: int = 3) -> pd.DataFrame:
        """Native-resolution window around the most active day."""

        raw = self.raw
        if raw.empty:
            return raw
        day = select_sample_day(raw)
        start = max(day - pd.Timedelta(days=1), raw.index.min().normalize())
        window = raw[(raw.index >= start) & (raw.index < start + pd.Timedelta(days=days))]
        return window if not window.empty else raw.head(int(24 * days / self.case.step_hours))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def first_column(frame: pd.DataFrame, names: tuple[str, ...] | list[str]) -> str | None:
    return next((name for name in names if name in frame), None)


def has_signal(frame: pd.DataFrame, column: str) -> bool:
    return column in frame and bool(frame[column].fillna(0).abs().max() > 1e-9)


def money(value: float, currency: str = "EUR") -> str:
    symbol = "€" if currency == "EUR" else currency + " "
    magnitude = abs(value)
    if magnitude >= 1e9:
        text = f"{magnitude / 1e9:,.2f} bn"
    elif magnitude >= 1e6:
        text = f"{magnitude / 1e6:,.2f} M"
    elif magnitude >= 1e3:
        text = f"{magnitude / 1e3:,.1f} k"
    else:
        text = f"{magnitude:,.0f}"
    sign = "−" if value < 0 else ""
    return f"{sign}{symbol}{text}"


def number(value: float, unit: str = "", digits: int = 1) -> str:
    magnitude = abs(value)
    if magnitude >= 1e9:
        text = f"{value / 1e9:,.2f} G"
    elif magnitude >= 1e6:
        text = f"{value / 1e6:,.2f} M"
    elif magnitude >= 1e4:
        text = f"{value / 1e3:,.1f} k"
    else:
        text = f"{value:,.{digits}f}"
    return f"{text}{(' ' + unit) if unit else ''}"


def _new_figure(
    ctx: ChartContext,
    rows: int = 1,
    titles: list[str] | None = None,
    heights: list[float] | None = None,
    height: int | None = None,
    shared_x: bool = True,
) -> go.Figure:
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=shared_x,
        vertical_spacing=(0.09 if titles else 0.05) + (0 if shared_x else 0.08),
        row_heights=heights,
        subplot_titles=titles,
    )
    fig._ctx_height = height or (150 + 190 * rows)  # type: ignore[attr-defined]
    return fig


def _finish(ctx: ChartContext, fig: go.Figure, caption: str | None = None) -> go.Figure:
    """Apply the shared layout. ``caption`` is shown by the page next to the chart title."""

    apply_layout(fig, ctx.theme, height=getattr(fig, "_ctx_height", 360))
    fig._ctx_caption = caption  # type: ignore[attr-defined]
    return fig


def _yaxis(fig: go.Figure, row: int, title: str, **kwargs: object) -> None:
    fig.update_yaxes(title_text=title, row=row, col=1, **kwargs)


def _time_axis(index: pd.Index) -> dict[str, object]:
    """Describe the x values compactly: ``x0``/``dx`` for a regular index, else explicit ``x``."""

    if len(index) > 2 and isinstance(index, pd.DatetimeIndex):
        steps = np.diff(index.values.astype("datetime64[ms]").astype("int64"))
        if (steps == steps[0]).all() and steps[0] > 0:
            return {"x0": index[0].strftime("%Y-%m-%d %H:%M:%S"), "dx": float(steps[0])}
    return {"x": index}


def _f32(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="float32")


def _add_line(
    fig: go.Figure,
    ctx: ChartContext,
    column: str,
    name: str,
    color: str,
    row: int = 1,
    unit: str = "",
    dash: str | None = None,
    width: float = 2.0,
    series: pd.Series | None = None,
    frame: pd.DataFrame | None = None,
    shape: str | None = None,
    fill: bool = False,
    markers: bool = False,
) -> bool:
    frame = ctx.df if frame is None else frame
    values = series if series is not None else (frame[column] if column in frame else None)
    if values is None or not values.notna().any():
        return False
    fig.add_trace(
        go.Scatter(
            **_time_axis(values.index),
            y=_f32(values),
            name=name,
            mode="lines+markers" if markers else "lines",
            marker={"size": 7, "color": color} if markers else None,
            line={"color": color, "width": width, "dash": dash, "shape": shape or ctx.line_shape},
            fill="tozeroy" if fill else None,
            fillcolor=with_alpha(color, 0.25) if fill else None,
            hovertemplate=f"%{{y:,.2f}} {unit}<extra>{name}</extra>",
            legendgroup=name,
        ),
        row=row,
        col=1,
    )
    return True


def _add_bars(
    fig: go.Figure,
    ctx: ChartContext,
    column: str,
    name: str,
    color: str,
    row: int = 1,
    unit: str = "MWh",
    scale: float = 1.0,
    series: pd.Series | None = None,
    frame: pd.DataFrame | None = None,
) -> bool:
    frame = ctx.df if frame is None else frame
    values = series if series is not None else (frame[column] if column in frame else None)
    if values is None:
        return False
    values = values.fillna(0.0) * scale
    if values.abs().max() < 1e-9:
        return False
    fig.add_trace(
        go.Bar(
            **_time_axis(values.index),
            y=_f32(values),
            name=name,
            marker={"color": color, "line": {"width": 0}},
            hovertemplate=f"%{{y:,.2f}} {unit}<extra>{name}</extra>",
            legendgroup=name,
        ),
        row=row,
        col=1,
    )
    return True


# ---------------------------------------------------------------------------
# Electricity market panels (steam, cement, steel)
# ---------------------------------------------------------------------------


def _price_panel(fig: go.Figure, ctx: ChartContext, row: int, frame: pd.DataFrame) -> None:
    theme = ctx.theme
    _add_line(
        fig,
        ctx,
        "day_ahead_price_EUR_per_MWh",
        "Day-ahead",
        theme.entity("day_ahead"),
        row,
        "€/MWh",
        frame=frame,
    )
    if has_signal(frame, "day_ahead_delivered_price_EUR_per_MWh"):
        _add_line(
            fig,
            ctx,
            "day_ahead_delivered_price_EUR_per_MWh",
            "Day-ahead incl. charges",
            theme.entity("day_ahead"),
            row,
            "€/MWh",
            dash="dot",
            width=1.5,
            frame=frame,
        )
    if has_signal(frame, "IDC_price_EUR_per_MWh"):
        _add_line(
            fig,
            ctx,
            "IDC_price_EUR_per_MWh",
            "Intraday",
            theme.entity("intraday"),
            row,
            "€/MWh",
            width=1.5,
            frame=frame,
        )
    if has_signal(frame, "afrr_energy_price_EUR_per_MWh"):
        _add_line(
            fig,
            ctx,
            "afrr_energy_price_EUR_per_MWh",
            "aFRR energy (down)",
            theme.entity("afrr_energy"),
            row,
            "€/MWh",
            width=1.5,
            frame=frame,
        )
    if has_signal(frame, "gas_based_heat_benchmark_EUR_per_MWh_th"):
        _add_line(
            fig,
            ctx,
            "gas_based_heat_benchmark_EUR_per_MWh_th",
            "Gas-based heat cost",
            theme.neutral_series,
            row,
            "€/MWh th",
            dash="dash",
            width=1.5,
            frame=frame,
        )
    _yaxis(fig, row, "€/MWh")


def _procurement_panel(fig: go.Figure, ctx: ChartContext, row: int, frame: pd.DataFrame) -> None:
    theme = ctx.theme
    _add_bars(
        fig,
        ctx,
        "DA_position_MWh",
        "Day-ahead schedule",
        theme.entity("day_ahead"),
        row,
        frame=frame,
    )
    _add_bars(
        fig, ctx, "IDC_buy_MWh", "Intraday bought", theme.entity("intraday"), row, frame=frame
    )
    _add_bars(
        fig,
        ctx,
        "IDC_sell_MWh",
        "Intraday sold back",
        theme.entity("intraday"),
        row,
        scale=-1.0,
        frame=frame,
    )
    _add_bars(
        fig,
        ctx,
        "afrr_energy_activated_MWh",
        "aFRR energy activated",
        theme.entity("afrr_energy"),
        row,
        frame=frame,
    )
    column = first_column(frame, ELECTRICITY_COLUMNS)
    if column and (len(fig.data) > 0):
        _add_line(
            fig,
            ctx,
            column,
            "Actual consumption",
            theme.text_primary,
            row,
            "MWh",
            width=1.2,
            frame=frame,
            shape="linear",
        )
    fig.update_layout(barmode="relative")
    _yaxis(fig, row, f"MWh per {ctx.step_label}")


def markets_overview(ctx: ChartContext) -> go.Figure | None:
    """Market prices on top of the electricity procurement by market."""

    frame = ctx.df
    if "day_ahead_price_EUR_per_MWh" not in frame and "DA_position_MWh" not in frame:
        return None
    fig = _new_figure(
        ctx, 2, ["Prices", "Electricity procurement by market"], [0.42, 0.58], height=560
    )
    _price_panel(fig, ctx, 1, frame)
    _procurement_panel(fig, ctx, 2, frame)
    return _finish(ctx, fig)


def afrr_capacity(ctx: ChartContext) -> go.Figure | None:
    frame = ctx.df
    theme = ctx.theme
    reserved = "afrr_capacity_reserved_MW"
    price = first_column(
        frame,
        ["afrr_capacity_clearing_price_EUR_per_MW_h", "afrr_capacity_down_price_EUR_per_MW_h"],
    )
    has_reserved = has_signal(frame, reserved)
    has_price = bool(price and has_signal(frame, price))
    if not has_reserved and not has_price:
        return None
    titles = (["Reserved aFRR capacity (down)"] if has_reserved else []) + ["Capacity price"]
    rows = len(titles)
    fig = _new_figure(
        ctx, rows, titles, [0.5, 0.5] if rows == 2 else None, height=470 if rows == 2 else 330
    )
    price_row = rows
    if has_reserved:
        _add_line(
            fig,
            ctx,
            reserved,
            "Reserved capacity",
            theme.entity("afrr_capacity"),
            1,
            "MW",
            fill=True,
            shape="hv",
        )
        _yaxis(fig, 1, "MW")
    if has_price:
        _add_line(
            fig,
            ctx,
            price,
            "Capacity clearing price",
            theme.entity("afrr_capacity"),
            price_row,
            "€/MW/h",
            shape="hv",
        )
    market_price = "afrr_capacity_down_price_EUR_per_MW_h"
    if has_signal(frame, market_price) and price != market_price:
        _add_line(
            fig,
            ctx,
            market_price,
            "Market capacity price",
            theme.neutral_series,
            price_row,
            "€/MW/h",
            dash="dash",
            width=1.5,
            shape="hv",
        )
    _yaxis(fig, price_row, "€/MW/h")
    return _finish(ctx, fig)


def price_response(ctx: ChartContext) -> go.Figure | None:
    """Average electricity use per day-ahead price decile."""

    raw = ctx.raw
    column = ctx.electricity_column()
    price = first_column(raw, ["day_ahead_price_EUR_per_MWh", "electricity_import_price_per_MWh"])
    if not column or not price or len(raw) < 40:
        return None
    data = pd.DataFrame({"price": raw[price], "load": raw[column] / ctx.case.step_hours}).dropna()
    if data["price"].nunique() < 4:
        return None
    data["bin"] = pd.qcut(data["price"], 10, duplicates="drop")
    grouped = data.groupby("bin", observed=True).agg(load=("load", "mean"), price=("price", "mean"))
    labels = [f"{interval.left:,.0f}…{interval.right:,.0f}" for interval in grouped.index]
    fig = _new_figure(ctx, height=380)
    fig.add_trace(
        go.Bar(
            x=labels,
            y=grouped["load"].to_numpy(),
            marker={"color": ctx.theme.entity("electricity"), "line": {"width": 0}},
            name="Average load",
            hovertemplate="%{y:,.2f} MW average<extra>price %{x}</extra>",
        )
    )
    fig.update_xaxes(title_text=f"Price decile ({ctx.case.currency}/MWh)", type="category")
    _yaxis(fig, 1, "Average load (MW)")
    _finish(ctx, fig)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def load_heatmap(ctx: ChartContext) -> go.Figure | None:
    """Hour-of-day by date heat map of the average electric load."""

    column = ctx.electricity_column()
    if not column:
        return None
    hourly = ctx.raw[column].resample("1h").sum()
    if hourly.empty or hourly.index.normalize().nunique() < 2:
        return None
    table = (
        hourly.rename("mw")
        .to_frame()
        .assign(day=lambda f: f.index.normalize(), hour=lambda f: f.index.hour)
        .pivot_table(index="hour", columns="day", values="mw", aggfunc="mean")
    )
    fig = _new_figure(ctx, height=380)
    fig.add_trace(
        go.Heatmap(
            x=table.columns,
            y=table.index,
            z=_f32(table.to_numpy()),
            colorscale=ctx.theme.colorscale(),
            colorbar={
                "title": {"text": "MW", "font": {"color": ctx.theme.text_secondary}},
                "tickfont": {"color": ctx.theme.text_muted},
                "thickness": 12,
                "outlinewidth": 0,
            },
            hovertemplate="%{x|%d %b %Y} %{y}:00<br>%{z:,.2f} MW<extra></extra>",
            xgap=0,
            ygap=0,
        )
    )
    fig.update_yaxes(title_text="Hour of day", dtick=6, autorange="reversed")
    _finish(ctx, fig)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def load_duration(ctx: ChartContext) -> go.Figure | None:
    column = ctx.electricity_column()
    if not column or len(ctx.raw) < 10:
        return None
    load = np.sort((ctx.raw[column] / ctx.case.step_hours).to_numpy())[::-1]
    share = np.linspace(0, 100, len(load))
    fig = _new_figure(ctx, height=380)
    fig.add_trace(
        go.Scatter(
            x=_f32(share),
            y=_f32(load),
            mode="lines",
            line={"color": ctx.theme.entity("electricity"), "width": 2},
            name="Load",
            hovertemplate="%{y:,.2f} MW<extra>exceeded %{x:.0f}% of the time</extra>",
        )
    )
    fig.update_xaxes(title_text="Share of time the load is exceeded (%)")
    _yaxis(fig, 1, "Electric load (MW)")
    _finish(ctx, fig)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Costs and emissions
# ---------------------------------------------------------------------------


def cost_components(ctx: ChartContext) -> list[tuple[str, float]]:
    """Return signed cost components of the selected plants (revenues are negative)."""

    raw = ctx.raw
    parts: list[tuple[str, float]] = []

    def add(label: str, value: float | None) -> None:
        if value is not None and abs(value) > 1e-6:
            parts.append((label, value))

    if ctx.case.family == "building":
        add("Energy cost", ctx.total("energy_cost"))
        add("Demand charge", ctx.total("demand_charge_cost"))
        return parts

    add("Day-ahead electricity", ctx.total("DA_electricity_cost_EUR"))
    if "IDC_buy_cost_EUR" in raw or "IDC_sell_revenue_EUR" in raw:
        add(
            "Intraday (net)",
            (ctx.total("IDC_buy_cost_EUR") or 0.0) - (ctx.total("IDC_sell_revenue_EUR") or 0.0),
        )
    add("aFRR energy", ctx.total("afrr_energy_cost_EUR"))
    add("Network charges & levies", ctx.total("additional_electricity_charges_cost_EUR"))
    add("Electricity tax", ctx.total("tax_cost_EUR"))
    add("Gas", ctx.total("gas_cost_EUR"))
    add("CO₂", ctx.total("co2_cost_EUR"))
    if "gas_cost_EUR" not in raw:
        add("Fuels, materials & other", ctx.total("non_electric_variable_cost_EUR"))
    gross, net = ctx.total("gross_operating_cost_EUR"), ctx.total("net_operating_cost_EUR")
    if gross is not None and net is not None:
        add("aFRR capacity (credit)", net - gross)
    if not ctx.sliced:
        add(
            "Grid fee ex-post correction",
            ctx.case.summary_total("grid_fee_ex_post_addition_EUR", ctx.plant),
        )
    return parts


def cost_breakdown(ctx: ChartContext) -> go.Figure | None:
    parts = cost_components(ctx)
    if not parts:
        return None
    theme = ctx.theme
    currency = ctx.case.currency
    total = sum(value for _, value in parts)
    labels = [label for label, _ in parts][::-1]
    values = [value for _, value in parts][::-1]
    colors = [theme.entity("cost") if value >= 0 else theme.entity("revenue") for value in values]
    fig = _new_figure(ctx, height=max(300, 90 + 44 * len(parts)))
    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[money(value, currency) for value in values],
            textposition="outside",
            textfont={"color": theme.text_secondary, "size": 11},
            cliponaxis=False,
            hovertemplate="%{x:,.0f} " + currency + "<extra>%{y}</extra>",
            name="Cost",
        )
    )
    low, high = min(0.0, min(values)), max(0.0, max(values))
    pad = (high - low) * 0.22 or 1.0
    fig.update_xaxes(
        title_text=currency,
        tickformat=".3s",
        range=[low - (pad if low < 0 else 0), high + pad],
    )
    fig.update_yaxes(automargin=True)
    _finish(ctx, fig, f"Net {money(total, currency)}")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def cumulative_cost(ctx: ChartContext) -> go.Figure | None:
    frame = ctx.df
    column = first_column(frame, ["net_operating_cost_EUR", "total_cost", "operating_cost_EUR"])
    if not column:
        return None
    currency = ctx.case.currency
    fig = _new_figure(ctx, height=380)
    _add_line(
        fig,
        ctx,
        column,
        "Cumulative net cost",
        ctx.theme.entity("cost"),
        1,
        currency,
        series=frame[column].cumsum(),
        shape="linear",
        fill=True,
    )
    _yaxis(fig, 1, currency, tickformat=".3s")
    _finish(ctx, fig)
    fig.update_layout(showlegend=False)
    return fig


def grid_fee_breakdown(ctx: ChartContext) -> go.Figure | None:
    if ctx.sliced:
        return None
    fees = {
        "Energy charge": "grid_fee_energy_charge_EUR",
        "Capacity charge": "grid_fee_capacity_charge_EUR",
        "Special network use": "grid_fee_special_network_use_EUR",
        "Levies": "grid_fee_levies_EUR",
        "Electricity tax": "grid_fee_electricity_tax_EUR",
    }
    values = {
        label: ctx.case.summary_total(column, ctx.plant) or 0.0 for label, column in fees.items()
    }
    values = {label: value for label, value in values.items() if abs(value) > 1e-6}
    if not values:
        return None
    theme = ctx.theme
    labels = list(values)[::-1]
    amounts = list(values.values())[::-1]
    fig = _new_figure(ctx, height=max(300, 90 + 44 * len(values)))
    fig.add_trace(
        go.Bar(
            y=labels,
            x=amounts,
            orientation="h",
            marker={"color": theme.entity("cost"), "line": {"width": 0}},
            text=[money(value) for value in amounts],
            textposition="outside",
            textfont={"color": theme.text_secondary, "size": 11},
            cliponaxis=False,
            hovertemplate="%{x:,.0f} EUR<extra>%{y}</extra>",
        )
    )
    fig.update_xaxes(title_text="EUR", tickformat=".3s", range=[0, max(amounts) * 1.22])
    total = sum(values.values())
    _finish(ctx, fig, f"Total {money(total)}")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def emissions(ctx: ChartContext) -> go.Figure | None:
    frame = ctx.df
    theme = ctx.theme
    if "total_emissions_kg" in frame:
        fig = _new_figure(ctx, 2, ["Emissions", "Cumulative emissions"], [0.55, 0.45], height=520)
        electricity = _add_bars(
            fig,
            ctx,
            "electricity_emissions_kg",
            "Electricity",
            theme.entity("electricity"),
            1,
            "t CO₂",
            scale=1e-3,
        )
        gas = _add_bars(
            fig, ctx, "gas_emissions_kg", "Gas", theme.entity("gas"), 1, "t CO₂", scale=1e-3
        )
        if not (electricity or gas):
            return None
        total = frame["total_emissions_kg"] / 1000.0
    elif "co2_emissions_t" in frame:
        fig = _new_figure(ctx, 2, ["Emissions", "Cumulative emissions"], [0.55, 0.45], height=520)
        if not _add_bars(fig, ctx, "co2_emissions_t", "CO₂", theme.entity("emissions"), 1, "t CO₂"):
            return None
        total = frame["co2_emissions_t"]
    else:
        return None
    fig.update_layout(barmode="stack")
    _yaxis(fig, 1, f"t CO₂ per {ctx.step_label}")
    _add_line(
        fig,
        ctx,
        "cum",
        "Cumulative CO₂",
        theme.entity("emissions"),
        2,
        "t CO₂",
        series=total.cumsum(),
        shape="linear",
        fill=True,
    )
    _yaxis(fig, 2, "t CO₂", tickformat=".3s")
    return _finish(ctx, fig)


# ---------------------------------------------------------------------------
# Steam / heat plants (ETES, gas boiler, electric boiler)
# ---------------------------------------------------------------------------


def heat_and_storage(ctx: ChartContext) -> go.Figure | None:
    frame = ctx.df
    theme = ctx.theme
    if "heat_demand_MWh" not in frame:
        return None
    has_storage = has_signal(frame, "etes_soc_MWh")
    rows = 2 if has_storage else 1
    fig = _new_figure(
        ctx,
        rows,
        ["Heat supply", "Thermal storage content"] if has_storage else ["Heat supply"],
        [0.62, 0.38] if has_storage else None,
        height=520 if has_storage else 380,
    )
    _add_bars(fig, ctx, "gas_heat_MWh", "Gas boiler", theme.entity("gas"), 1, "MWh th")
    _add_bars(
        fig, ctx, "etes_discharge_MWh", "Storage discharge", theme.entity("storage"), 1, "MWh th"
    )
    _add_bars(
        fig,
        ctx,
        "electric_boiler_heat_MWh",
        "Electric boiler",
        theme.entity("electric_boiler"),
        1,
        "MWh th",
    )
    _add_line(
        fig,
        ctx,
        "heat_demand_MWh",
        "Heat demand",
        theme.text_primary,
        1,
        "MWh th",
        width=1.2,
        shape="linear",
    )
    fig.update_layout(barmode="stack")
    _yaxis(fig, 1, f"MWh th per {ctx.step_label}")
    if has_storage:
        _add_line(
            fig,
            ctx,
            "etes_soc_MWh",
            "Storage content",
            theme.entity("storage"),
            2,
            "MWh th",
            fill=True,
            shape="linear",
        )
        _yaxis(fig, 2, "MWh th")
    return _finish(ctx, fig)


def storage_sources(ctx: ChartContext) -> go.Figure | None:
    """Stored heat by the market it was procured on."""

    storage = ctx.case.storage_frame(ctx.plant)
    columns = {
        "thermal_inventory_day_ahead_MWh_th": ("Day-ahead", "day_ahead"),
        "thermal_inventory_intraday_continuous_MWh_th": ("Intraday", "intraday"),
        "thermal_inventory_afrr_energy_MWh_th": ("aFRR energy", "afrr_energy"),
    }
    if storage.empty or not any(has_signal(storage, column) for column in columns):
        return None
    storage = slice_period(storage, ctx.start, ctx.end)
    storage = resample(storage, RESOLUTIONS[ctx.resolution_key])
    fig = _new_figure(ctx, height=380)
    for column, (label, entity) in columns.items():
        if has_signal(storage, column):
            fig.add_trace(
                go.Scatter(
                    **_time_axis(storage.index),
                    y=_f32(storage[column]),
                    name=label,
                    mode="lines",
                    stackgroup="inventory",
                    line={"color": ctx.theme.entity(entity), "width": 0.8},
                    fillcolor=ctx.theme.entity(entity),
                    hovertemplate=f"%{{y:,.2f}} MWh th<extra>{label}</extra>",
                )
            )
    _yaxis(fig, 1, "MWh th")
    return _finish(ctx, fig)


def gas_replacement(ctx: ChartContext) -> go.Figure | None:
    """Gas heat that remains after each market stage."""

    raw = ctx.raw
    if "heat_demand_MWh" not in raw:
        return None
    stages = [
        ("No flexibility", "heat_demand_MWh"),
        ("After day-ahead", "gas_heat_after_day_ahead_MWh"),
        ("After intraday", "gas_heat_after_intraday_MWh"),
        ("After aFRR energy", "gas_heat_after_afrr_energy_MWh"),
        ("Final", "gas_heat_MWh"),
    ]
    stages = [(label, column) for label, column in stages if column in raw]
    if len(stages) < 2:
        return None
    totals = [float(raw[column].sum()) for _, column in stages]
    baseline = totals[0] or 1.0
    theme = ctx.theme
    fig = _new_figure(ctx, height=380)
    fig.add_trace(
        go.Bar(
            x=[label for label, _ in stages],
            y=totals,
            marker={"color": theme.entity("gas"), "line": {"width": 0}},
            text=[f"{value:,.0f} MWh<br>{1 - value / baseline:.0%} replaced" for value in totals],
            textposition="outside",
            textfont={"color": theme.text_secondary, "size": 11},
            cliponaxis=False,
            hovertemplate="%{y:,.0f} MWh th<extra>%{x}</extra>",
        )
    )
    fig.update_xaxes(type="category")
    _yaxis(fig, 1, "Gas heat (MWh th)", range=[0, max(totals) * 1.25])
    _finish(ctx, fig)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Industrial plants (cement, steel)
# ---------------------------------------------------------------------------

_UNIT_COLUMN = re.compile(r"^(?P<unit>[a-z0-9_]+?)_electricity_consumption_MWh$")
_STATUS_COLUMN = re.compile(r"^(?P<unit>[a-z0-9_]+?)_operational_status$")
_OUTPUT_COLUMNS = {
    "cement": ("clinker_output_t", "remaining_clinker_demand_t", "Clinker"),
    "steel": ("steel_output_t", "remaining_steel_demand_t", "Steel"),
}


def production(ctx: ChartContext) -> go.Figure | None:
    spec = _OUTPUT_COLUMNS.get(ctx.case.family)
    frame = ctx.df
    if not spec or spec[0] not in frame:
        return None
    output, remaining, label = spec
    theme = ctx.theme
    has_remaining = remaining in frame
    fig = _new_figure(
        ctx,
        2 if has_remaining else 1,
        [f"{label} output", "Remaining demand"] if has_remaining else [f"{label} output"],
        [0.6, 0.4] if has_remaining else None,
        height=500 if has_remaining else 380,
    )
    _add_bars(fig, ctx, output, f"{label} output", theme.entity("electricity"), 1, "t")
    _yaxis(fig, 1, f"t per {ctx.step_label}")
    if has_remaining:
        _add_line(
            fig,
            ctx,
            remaining,
            "Remaining demand",
            theme.neutral_series,
            2,
            "t",
            shape="linear",
            fill=True,
        )
        _yaxis(fig, 2, "t", tickformat=".3s")
    return _finish(ctx, fig)


def energy_carriers(ctx: ChartContext) -> go.Figure | None:
    theme = ctx.theme
    carriers = [
        ("total_electricity_consumption_MWh", "Electricity", theme.entity("electricity")),
        ("natural_gas_consumption_MWh", "Natural gas", theme.entity("gas")),
        ("hydrogen_consumption_MWh", "Hydrogen", theme.entity("hydrogen")),
        ("coal_consumption_MWh", "Coal", theme.neutral_series),
    ]
    fig = _new_figure(ctx, height=400)
    for column, label, color in carriers:
        _add_bars(fig, ctx, column, label, color, 1, "MWh")
    if not len(fig.data):
        return None
    fig.update_layout(barmode="stack")
    _yaxis(fig, 1, f"MWh per {ctx.step_label}")
    return _finish(ctx, fig)


def unit_electricity(ctx: ChartContext) -> go.Figure | None:
    frame = ctx.df
    if ctx.case.family not in _OUTPUT_COLUMNS:
        return None
    units = sorted(
        match["unit"]
        for column in frame.columns
        if (match := _UNIT_COLUMN.match(column)) and match["unit"] not in {"total", "actual"}
    )
    units = [unit for unit in units if has_signal(frame, f"{unit}_electricity_consumption_MWh")]
    if not units:
        return None
    fig = _new_figure(ctx, height=400)
    for index, unit in enumerate(units[:8]):
        _add_bars(
            fig,
            ctx,
            f"{unit}_electricity_consumption_MWh",
            unit.replace("_", " ").title(),
            ctx.theme.categorical[index],
            1,
            "MWh",
        )
    fig.update_layout(barmode="stack")
    _yaxis(fig, 1, f"MWh per {ctx.step_label}")
    return _finish(ctx, fig)


def unit_status(ctx: ChartContext) -> go.Figure | None:
    """Share of each day in which a process unit is operating."""

    raw = ctx.raw
    units = [match["unit"] for column in raw.columns if (match := _STATUS_COLUMN.match(column))]
    units = [unit for unit in units if raw[f"{unit}_operational_status"].notna().any()]
    if not units:
        return None
    daily = pd.DataFrame({u: raw[f"{u}_operational_status"].resample("1D").mean() for u in units})
    if len(daily) < 2:
        return None
    fig = _new_figure(ctx, height=120 + 56 * len(units))
    fig.add_trace(
        go.Heatmap(
            x=daily.index,
            y=[u.replace("_", " ").title() for u in daily.columns],
            z=_f32(daily.to_numpy().T),
            zmin=0,
            zmax=1,
            colorscale=ctx.theme.colorscale(),
            colorbar={
                "title": {"text": "Share on", "font": {"color": ctx.theme.text_secondary}},
                "tickfont": {"color": ctx.theme.text_muted},
                "thickness": 12,
                "outlinewidth": 0,
            },
            hovertemplate="%{x|%d %b %Y}<br>%{y}: %{z:.0%} of the day running<extra></extra>",
        )
    )
    _finish(ctx, fig)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Buildings with EV fleets, PV and grid tariff
# ---------------------------------------------------------------------------


def building_balance(ctx: ChartContext) -> go.Figure | None:
    frame = ctx.df
    theme = ctx.theme
    if "grid_import_MWh" not in frame:
        return None
    fig = _new_figure(
        ctx, 2, ["Grid exchange and building demand", "PV generation"], [0.6, 0.4], height=520
    )
    _add_bars(fig, ctx, "grid_import_MWh", "Grid import", theme.entity("import"), 1)
    _add_bars(fig, ctx, "grid_export_MWh", "Grid export", theme.entity("export"), 1, scale=-1.0)
    _add_line(
        fig,
        ctx,
        "building_demand_MWh",
        "Building demand",
        theme.text_primary,
        1,
        "MWh",
        width=1.2,
        shape="linear",
    )
    fig.update_layout(barmode="relative")
    _yaxis(fig, 1, f"MWh per {ctx.step_label}")
    _add_line(
        fig,
        ctx,
        "pv_generation_MWh",
        "PV generation",
        theme.entity("pv"),
        2,
        "MWh",
        fill=True,
        shape="linear",
    )
    _add_line(
        fig,
        ctx,
        "pv_curtailment_MWh",
        "PV curtailed",
        theme.neutral_series,
        2,
        "MWh",
        dash="dot",
        width=1.5,
        shape="linear",
    )
    _yaxis(fig, 2, f"MWh per {ctx.step_label}")
    return _finish(ctx, fig)


def building_tariff(ctx: ChartContext) -> go.Figure | None:
    frame = ctx.df
    theme = ctx.theme
    import_price = first_column(frame, ["electricity_import_price_per_MWh"])
    if not import_price:
        return None
    currency = ctx.case.currency
    peak = has_signal(frame, "grid_import_MW")
    fig = _new_figure(
        ctx,
        2 if peak else 1,
        ["Electricity price", "Grid import power vs billing peak"]
        if peak
        else ["Electricity price"],
        [0.5, 0.5] if peak else None,
        height=500 if peak else 380,
    )
    _add_line(fig, ctx, import_price, "Import price", theme.entity("import"), 1, f"{currency}/MWh")
    _add_line(
        fig,
        ctx,
        "electricity_export_price_per_MWh",
        "Export price",
        theme.entity("export"),
        1,
        f"{currency}/MWh",
    )
    _yaxis(fig, 1, f"{currency}/MWh")
    if peak:
        _add_line(fig, ctx, "grid_import_MW", "Grid import power", theme.entity("import"), 2, "MW")
        _add_line(
            fig,
            ctx,
            "billing_peak_MW",
            "Billing peak so far (month)",
            theme.neutral_series,
            2,
            "MW",
            dash="dash",
            width=1.5,
        )
        _yaxis(fig, 2, "MW")
    return _finish(ctx, fig)


def ev_fleet(ctx: ChartContext) -> go.Figure | None:
    frame = ctx.df
    theme = ctx.theme
    routes = [
        column for column in frame.columns if re.fullmatch(r"bus_route_.+_soc_fraction", column)
    ]
    soc = routes or [c for c in ["bus_soc_fraction"] if c in frame]
    if not soc:
        return None
    fig = _new_figure(
        ctx, 2, ["State of charge", "Charging and discharging"], [0.5, 0.5], height=520
    )
    for index, column in enumerate(soc[:8]):
        label = column.replace("_soc_fraction", "").replace("_", " ").title()
        _add_line(
            fig,
            ctx,
            column,
            label,
            theme.categorical[index],
            1,
            "of capacity",
            shape="linear",
            width=1.5,
        )
    _yaxis(fig, 1, "Fraction of capacity", range=[0, 1.02], tickformat=".0%")
    _add_bars(fig, ctx, "bus_charge_MWh", "Charging", theme.entity("ev"), 2)
    _add_bars(
        fig,
        ctx,
        "bus_discharge_MWh",
        "Discharging (V2G / V2B)",
        theme.entity("export"),
        2,
        scale=-1.0,
    )
    fig.update_layout(barmode="relative")
    _yaxis(fig, 2, f"MWh per {ctx.step_label}")
    return _finish(ctx, fig)


def grid_stress(ctx: ChartContext) -> go.Figure | None:
    frame = ctx.df
    theme = ctx.theme
    if not has_signal(frame, "regional_grid_load_fraction"):
        return None
    fig = _new_figure(
        ctx, 2, ["Regional grid load", "Building grid import"], [0.5, 0.5], height=500
    )
    _add_line(
        fig,
        ctx,
        "regional_grid_load_fraction",
        "Regional load (share of peak)",
        theme.neutral_series,
        1,
        "",
        shape="linear",
    )
    if "grid_stress_threshold" in frame and frame["grid_stress_threshold"].notna().any():
        _add_line(
            fig,
            ctx,
            "grid_stress_threshold",
            "Stress threshold",
            theme.entity("emissions"),
            1,
            "",
            dash="dash",
            width=1.5,
            shape="linear",
        )
    _yaxis(fig, 1, "Share of peak", tickformat=".0%")
    _add_bars(fig, ctx, "grid_import_MWh", "Grid import", theme.entity("import"), 2)
    _yaxis(fig, 2, f"MWh per {ctx.step_label}")
    return _finish(ctx, fig)


def pv_use(ctx: ChartContext) -> go.Figure | None:
    raw = ctx.raw
    theme = ctx.theme
    if "pv_generation_MWh" not in raw:
        return None
    daily = (
        pd.DataFrame(
            {
                "generation": raw["pv_generation_MWh"],
                "export": raw.get("grid_export_MWh", pd.Series(0.0, index=raw.index)),
                "curtailed": raw.get("pv_curtailment_MWh", pd.Series(0.0, index=raw.index)),
            }
        )
        .resample("1D")
        .sum()
    )
    if daily["generation"].max() < 1e-9:
        return None
    exported = daily[["export", "generation"]].min(axis=1)
    own = (daily["generation"] - exported).clip(lower=0)
    fig = _new_figure(ctx, height=380)
    _add_bars(fig, ctx, "own", "Used on site", theme.entity("pv"), 1, series=own)
    _add_bars(fig, ctx, "exp", "Exported", theme.entity("export"), 1, series=exported)
    _add_bars(fig, ctx, "cur", "Curtailed", theme.neutral_series, 1, series=daily["curtailed"])
    fig.update_layout(barmode="stack")
    _yaxis(fig, 1, "MWh per day")
    return _finish(ctx, fig)


# ---------------------------------------------------------------------------
# Representative window at native resolution
# ---------------------------------------------------------------------------


def sample_window(ctx: ChartContext) -> go.Figure | None:
    window = ctx.sample_window()
    if window.empty:
        return None
    family = ctx.case.family
    theme = ctx.theme
    native = ChartContext(ctx.case, ctx.plant, theme, resolution="native")
    native.__dict__["raw"] = window
    native.__dict__["df"] = window
    day_label = window.index.min().strftime("%d %b %Y")
    if family == "building":
        fig = _new_figure(
            ctx,
            3,
            ["Electricity price", "Grid exchange", "EV state of charge"],
            [0.3, 0.4, 0.3],
            height=660,
        )
        currency = ctx.case.currency
        _add_line(
            fig,
            native,
            "electricity_import_price_per_MWh",
            "Import price",
            theme.entity("import"),
            1,
            f"{currency}/MWh",
            frame=window,
        )
        _yaxis(fig, 1, f"{currency}/MWh")
        _add_bars(
            fig, native, "grid_import_MWh", "Grid import", theme.entity("import"), 2, frame=window
        )
        _add_bars(
            fig,
            native,
            "grid_export_MWh",
            "Grid export",
            theme.entity("export"),
            2,
            scale=-1.0,
            frame=window,
        )
        fig.update_layout(barmode="relative")
        _yaxis(fig, 2, "MWh per step")
        _add_line(
            fig,
            native,
            "bus_soc_fraction",
            "Fleet state of charge",
            theme.entity("ev"),
            3,
            "of capacity",
            frame=window,
            shape="linear",
        )
        _yaxis(fig, 3, "Fraction", tickformat=".0%")
    else:
        third = "Heat supply" if family == "steam" else "Production"
        fig = _new_figure(
            ctx,
            3,
            ["Prices", "Electricity procurement by market", third],
            [0.3, 0.38, 0.32],
            height=700,
        )
        _price_panel(fig, native, 1, window)
        _procurement_panel(fig, native, 2, window)
        if family == "steam":
            _add_bars(
                fig,
                native,
                "gas_heat_MWh",
                "Gas boiler",
                theme.entity("gas"),
                3,
                "MWh th",
                frame=window,
            )
            _add_bars(
                fig,
                native,
                "etes_discharge_MWh",
                "Storage discharge",
                theme.entity("storage"),
                3,
                "MWh th",
                frame=window,
            )
            _add_bars(
                fig,
                native,
                "electric_boiler_heat_MWh",
                "Electric boiler",
                theme.entity("electric_boiler"),
                3,
                "MWh th",
                frame=window,
            )
            _add_line(
                fig,
                native,
                "heat_demand_MWh",
                "Heat demand",
                theme.text_primary,
                3,
                "MWh th",
                width=1.2,
                frame=window,
                shape="linear",
            )
            _yaxis(fig, 3, "MWh th per step")
        else:
            spec = _OUTPUT_COLUMNS.get(family)
            if spec:
                _add_bars(
                    fig,
                    native,
                    spec[0],
                    f"{spec[2]} output",
                    theme.entity("electricity"),
                    3,
                    "t",
                    frame=window,
                )
            _yaxis(fig, 3, "t per step")
        fig.update_layout(barmode="relative")
    _finish(ctx, fig, f"Window starting {day_label}")
    return fig


# Chart builders grouped by who needs them. Each plant dashboard class (see sections.py) starts
# from the generic set and adds its own group, so a chart is registered exactly once.
GENERIC_BUILDERS: dict[str, FigureBuilder] = {
    "markets_overview": markets_overview,
    "afrr_capacity": afrr_capacity,
    "price_response": price_response,
    "load_heatmap": load_heatmap,
    "load_duration": load_duration,
    "cost_breakdown": cost_breakdown,
    "cumulative_cost": cumulative_cost,
    "grid_fee_breakdown": grid_fee_breakdown,
    "emissions": emissions,
    "sample_window": sample_window,
}

STEAM_CORE_BUILDERS: dict[str, FigureBuilder] = {
    "heat_and_storage": heat_and_storage,
    "storage_sources": storage_sources,
    "gas_replacement": gas_replacement,
}

INDUSTRIAL_BUILDERS: dict[str, FigureBuilder] = {
    "production": production,
    "energy_carriers": energy_carriers,
    "unit_electricity": unit_electricity,
    "unit_status": unit_status,
}

BUILDING_BUILDERS: dict[str, FigureBuilder] = {
    "building_balance": building_balance,
    "building_tariff": building_tariff,
    "ev_fleet": ev_fleet,
    "grid_stress": grid_stress,
    "pv_use": pv_use,
}

# Every builder of this module in one lookup (handy in notebooks and tests).
BUILDERS: dict[str, FigureBuilder] = {
    **GENERIC_BUILDERS,
    **STEAM_CORE_BUILDERS,
    **INDUSTRIAL_BUILDERS,
    **BUILDING_BUILDERS,
}
