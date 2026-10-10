# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Steam / ETES report analyses: benchmark savings, demand coverage, market value, grid fees.

These charts and tables carry over the analyses of ``notebooks/fleximod_report_analysis.ipynb``
into the shared dashboard code, so the static HTML and the Dash app both show them. Builders have
the same contract as in ``charts.py``: they return ``None`` when the case lacks the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from flexi_mod.regulations.grid_fees import GermanGridFeeRegulation
from flexi_mod.visualisation.dashboard.charts import (
    STEAM_CORE_BUILDERS,
    ChartContext,
    FigureBuilder,
    _add_bars,
    _add_line,
    _f32,
    _finish,
    _new_figure,
    _yaxis,
    cost_components,
    first_column,
    has_signal,
    money,
    number,
)
from flexi_mod.visualisation.dashboard.data import investment_costs
from flexi_mod.visualisation.dashboard.theme import with_alpha

DEFAULT_BOILER_EFFICIENCY = 0.9
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Shared calculations
# ---------------------------------------------------------------------------


def boiler_efficiency(ctx: ChartContext) -> float:
    """Gas boiler efficiency, derived from the dispatch (heat out / gas in)."""

    raw = ctx.raw
    if "gas_heat_MWh" in raw and "gas_input_MWh" in raw and raw["gas_input_MWh"].sum() > 0:
        return float(raw["gas_heat_MWh"].sum() / raw["gas_input_MWh"].sum())
    config = ctx.case.plant_config
    if not config.empty and {"technology", "efficiency"} <= set(config.columns):
        boiler = config[config["technology"].astype(str).str.lower() == "boiler"]
        values = pd.to_numeric(boiler["efficiency"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return DEFAULT_BOILER_EFFICIENCY


def gas_only_benchmark(ctx: ChartContext) -> float | None:
    """Cost of covering the whole heat demand with the gas boiler alone."""

    raw = ctx.raw
    if "heat_demand_MWh" not in raw or "gas_price_EUR_per_MWh" not in raw:
        return None
    cost = raw["heat_demand_MWh"] / boiler_efficiency(ctx) * raw["gas_price_EUR_per_MWh"]
    return float(cost.sum())


def period_fraction(ctx: ChartContext) -> float:
    """Share of a year covered by the selected period (annual costs are pro-rated)."""

    raw = ctx.raw
    if raw.empty:
        return 0.0
    days = (raw.index.max().normalize() - raw.index.min().normalize()).days + 1
    return min(days, 365) / 365.0


def investment(ctx: ChartContext) -> dict[str, float] | None:
    """CAPEX annuity and OPEX for the simulated period, from the case input ``plants.csv``."""

    costs = investment_costs(ctx.case.plant_config)
    if costs is None:
        return None
    fraction = period_fraction(ctx)
    return {key: value * fraction for key, value in costs.items()}


def net_cost_with_investment(ctx: ChartContext) -> float | None:
    parts = cost_components(ctx)
    if not parts:
        return None
    total = sum(value for _, value in parts)
    extra = investment(ctx)
    return total + (extra["annuity"] + extra["opex_annual"] if extra else 0.0)


def heat_split(raw: pd.DataFrame) -> pd.DataFrame:
    """Split heat into direct electric supply, storage discharge and gas boiler heat.

    Discharge while the storage charges at the same time is direct pass-through of electricity
    ("direct supply"); discharge from stored heat alone is "storage discharge".
    """

    zero = pd.Series(0.0, index=raw.index)
    charge = raw.get("etes_charge_MWh", zero)
    discharge = raw.get("etes_discharge_MWh", zero)
    boiler = raw.get("electric_boiler_heat_MWh", zero)
    through = (charge > 1e-9) & (discharge > 1e-9)
    stored = (charge <= 1e-9) & (discharge > 1e-9)
    return pd.DataFrame(
        {
            "direct": boiler + discharge.where(through, 0.0),
            "storage": discharge.where(stored, 0.0),
            "gas": raw.get("gas_heat_MWh", zero),
        }
    )


def _monthly(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.resample("MS").sum()


def _month_axis(fig: go.Figure, index: pd.DatetimeIndex) -> None:
    multi_year = index.year.nunique() > 1
    fig.update_xaxes(tickformat="%b %Y" if multi_year else "%b", dtick="M1")


def _active_window(raw: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """The ``days``-long window with the most market activity."""

    columns = [
        c
        for c in (
            "DA_position_MWh",
            "IDC_buy_MWh",
            "IDC_sell_MWh",
            "afrr_energy_activated_MWh",
            "afrr_capacity_reserved_MWh",
        )
        if c in raw
    ]
    if not columns:
        return raw.head(0)
    daily = sum(raw[c].abs() for c in columns).resample("1D").sum()
    if len(daily) <= days:
        return raw
    end_day = daily.rolling(days).sum().idxmax()
    start = end_day - pd.Timedelta(days=days - 1)
    return raw[(raw.index >= start) & (raw.index < start + pd.Timedelta(days=days))]


def _mw(raw: pd.DataFrame, column: str, step_hours: float) -> pd.Series:
    return raw[column] / step_hours


# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------


def steam_kpis(ctx: ChartContext) -> list[tuple[str, str, str]]:
    """Notebook executive summary: heat, savings against gas-only, CAPEX and OPEX."""

    tiles: list[tuple[str, str, str]] = []
    parts = cost_components(ctx)
    net = sum(value for _, value in parts) if parts else None
    demand = ctx.total("heat_demand_MWh")
    benchmark = gas_only_benchmark(ctx)
    extra = investment(ctx)
    if net is not None:
        detail = "incl. grid fee correction" if any("ex-post" in p for p, _ in parts) else ""
        tiles.append(("Net operating cost", money(net), detail))
    if benchmark is not None and net is not None:
        savings = benchmark - net
        tiles.append(("Gas-only benchmark", money(benchmark), "all heat from the gas boiler"))
        tiles.append(
            (
                "Savings vs gas-only",
                money(savings),
                f"{savings / benchmark:.1%} of the benchmark" if benchmark else "",
            )
        )
        if extra:
            net_savings = savings - extra["annuity"] - extra["opex_annual"]
            tiles.append(
                (
                    "Net savings",
                    money(net_savings),
                    f"{net_savings / benchmark:.1%}, after CAPEX and OPEX",
                )
            )
    if net is not None and demand:
        tiles.append(("Cost of heat", f"€{net / demand:,.1f}/MWh", "per MWh th delivered"))
    if demand:
        tiles.append(("Heat demand", number(demand, "MWh th"), ""))
        electric = float(heat_split(ctx.raw)[["direct", "storage"]].to_numpy().sum())
        tiles.append(
            ("Electrification rate", f"{electric / demand:.0%}", f"{number(electric, 'MWh th')}")
        )
    consumption = ctx.total("actual_electricity_consumption_MWh") or ctx.total(
        "electricity_consumption_MWh"
    )
    if consumption is not None:
        tiles.append(("Electricity used", number(consumption, "MWh"), ""))
    emissions = ctx.total("total_emissions_kg")
    if emissions is not None:
        tiles.append(("Emissions", number(emissions / 1000.0, "t CO₂"), ""))
    if extra:
        tiles.append(("CAPEX (annualised)", money(extra["annuity"]), "annuity over the period"))
        tiles.append(("OPEX", money(extra["opex_annual"]), "maintenance and operations"))
    credit = ctx.total("afrr_capacity_net_value_EUR")
    if credit:
        tiles.append(("aFRR capacity value", money(credit), "net of opportunity cost"))
    return tiles


def grid_fee_kpis(ctx: ChartContext) -> list[tuple[str, str, str]]:
    """Grid fee tiles from the ex-post settlement."""

    if ctx.sliced:
        return []
    total = ctx.case.summary_total("grid_fee_total_EUR", ctx.plant)
    if total is None:
        return []

    def val(column: str) -> float:
        return ctx.case.summary_total(column, ctx.plant) or 0.0

    tiles = [
        ("Total grid fee", money(total), "settled ex-post"),
        ("Energy charge", money(val("grid_fee_energy_charge_EUR")), tier_text(ctx)),
        (
            "Capacity charge",
            money(val("grid_fee_capacity_charge_EUR")),
            f"billed peak {val('grid_billed_peak_MW'):,.2f} MW",
        ),
    ]
    if val("grid_fee_special_network_use_EUR"):
        tiles.append(("Special network use", money(val("grid_fee_special_network_use_EUR")), ""))
    tiles.append(("Levies", money(val("grid_fee_levies_EUR")), ""))
    if val("grid_fee_electricity_tax_EUR"):
        tiles.append(("Electricity tax", money(val("grid_fee_electricity_tax_EUR")), ""))
    net_incl = val("net_operating_cost_incl_grid_fees_EUR")
    if net_incl:
        tiles.append(("Net cost incl. grid fees", money(net_incl), "operating cost + grid fees"))
    return tiles


def _summary_text(ctx: ChartContext, column: str) -> str:
    rows = ctx.case.summary_rows(ctx.plant)
    if column not in rows or rows.empty:
        return ""
    return str(rows[column].iloc[0])


def has_tiers(ctx: ChartContext) -> bool:
    """German-style full-load-hour tiers exist (the only market with high-load windows)."""

    return _summary_text(ctx, "grid_realized_tier") in {"high", "low"}


def tier_text(ctx: ChartContext) -> str:
    """Realised full-load-hour tier and whether the assumed tier held."""

    if not has_tiers(ctx):
        return ""
    realized = _summary_text(ctx, "grid_realized_tier")
    assumed = _summary_text(ctx, "grid_assumed_tier")
    hours = ctx.case.summary_total("grid_full_load_hours", ctx.plant) or 0.0
    held = _summary_text(ctx, "grid_tier_assumption_held").lower() in {"true", "1", "1.0"}
    text = f"tier '{realized}' ({hours:,.0f} full-load h/a)"
    if not held:
        text += f"; assumed '{assumed}' did not hold, re-run with --assumed-grid-tier {realized}"
    return text


# ---------------------------------------------------------------------------
# Heat demand coverage and storage
# ---------------------------------------------------------------------------


def heat_coverage_annual(ctx: ChartContext) -> go.Figure | None:
    if "heat_demand_MWh" not in ctx.raw:
        return None
    split = heat_split(ctx.raw).sum()
    total = float(split.sum())
    if total <= 0:
        return None
    theme = ctx.theme
    segments = [
        ("Direct supply by e-heater", "direct", theme.entity("electric_boiler")),
        ("Storage discharge", "storage", theme.entity("storage")),
        ("Gas boiler", "gas", theme.entity("gas")),
    ]
    fig = _new_figure(ctx, height=220)
    for label, key, color in segments:
        value = float(split[key])
        fig.add_trace(
            go.Bar(
                y=["Heat demand"],
                x=[value],
                orientation="h",
                name=label,
                marker={"color": color, "line": {"width": 0}},
                hovertemplate=f"%{{x:,.0f}} MWh th ({value / total:.1%})<extra>{label}</extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="MWh th")
    fig.update_yaxes(showticklabels=False, showgrid=False, linecolor="rgba(0,0,0,0)")
    caption = " · ".join(f"{label} {split[key] / total:.0%}" for label, key, _ in segments)
    _finish(ctx, fig, caption)
    fig.update_layout(hovermode="closest")
    return fig


def heat_coverage_monthly(ctx: ChartContext) -> go.Figure | None:
    if "heat_demand_MWh" not in ctx.raw:
        return None
    monthly = _monthly(heat_split(ctx.raw))
    total = monthly.sum(axis=1)
    if total.sum() <= 0:
        return None
    theme = ctx.theme
    fig = _new_figure(
        ctx, 2, ["Heat supplied per month", "Electrified share of heat"], [0.62, 0.38], height=560
    )
    for label, key, color in [
        ("Direct supply by e-heater", "direct", theme.entity("electric_boiler")),
        ("Storage discharge", "storage", theme.entity("storage")),
        ("Gas boiler", "gas", theme.entity("gas")),
    ]:
        _add_bars(fig, ctx, key, label, color, 1, "MWh th", series=monthly[key])
    fig.update_layout(barmode="stack")
    _yaxis(fig, 1, "MWh th")
    share = ((monthly["direct"] + monthly["storage"]) / total.replace(0, np.nan) * 100.0).fillna(0)
    _add_line(
        fig, ctx, "share", "Electrified share", theme.entity("storage"), 2, "%", series=share,
        shape="linear", width=2.5, markers=True,
    )  # fmt: skip
    _yaxis(fig, 2, "% of heat", range=[0, 100], ticksuffix="%")
    _month_axis(fig, monthly.index)
    return _finish(ctx, fig)


def storage_operation(ctx: ChartContext) -> go.Figure | None:
    frame = ctx.df
    if not has_signal(frame, "etes_soc_MWh"):
        return None
    theme = ctx.theme
    fig = _new_figure(
        ctx, 2, ["Storage charging and discharging", "Storage state of charge"], [0.55, 0.45],
        height=520,
    )  # fmt: skip
    _add_bars(fig, ctx, "etes_charge_MWh", "Charging (electricity in)", theme.entity("electricity"),
              1, "MWh")  # fmt: skip
    _add_bars(fig, ctx, "etes_discharge_MWh", "Discharging (heat out)", theme.entity("storage"), 1,
              "MWh th", scale=-1.0)  # fmt: skip
    fig.update_layout(barmode="relative")
    _yaxis(fig, 1, f"MWh per {ctx.step_label}")
    _add_line(fig, ctx, "etes_soc_MWh", "State of charge", theme.entity("storage"), 2, "MWh th",
              fill=True, shape="linear")  # fmt: skip
    _yaxis(fig, 2, "MWh th")
    return _finish(ctx, fig)


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------


def monthly_procurement(ctx: ChartContext) -> go.Figure | None:
    raw = ctx.raw
    if "DA_position_MWh" not in raw:
        return None
    zero = pd.Series(0.0, index=raw.index)
    frame = pd.DataFrame(
        {
            "da": raw["DA_position_MWh"] - raw.get("IDC_sell_MWh", zero),
            "idc": raw.get("IDC_buy_MWh", zero),
            "afrr": raw.get("afrr_energy_activated_MWh", zero),
        }
    )
    monthly = _monthly(frame)
    total = float(monthly.sum().sum())
    if total <= 0:
        return None
    theme = ctx.theme
    fig = _new_figure(ctx, height=400)
    _add_bars(fig, ctx, "da", "Day-ahead (net of intraday sales)", theme.entity("day_ahead"), 1,
              series=monthly["da"])  # fmt: skip
    _add_bars(fig, ctx, "idc", "Intraday bought", theme.entity("intraday"), 1,
              series=monthly["idc"])  # fmt: skip
    _add_bars(fig, ctx, "afrr", "aFRR energy", theme.entity("afrr_energy"), 1,
              series=monthly["afrr"])  # fmt: skip
    fig.update_layout(barmode="relative")
    _yaxis(fig, 1, "MWh el")
    _month_axis(fig, monthly.index)
    return _finish(ctx, fig, f"Total {number(total, 'MWh el')}")


def monthly_market_value(ctx: ChartContext) -> go.Figure | None:
    """Monthly value of each market against the gas-based electricity benchmark."""

    raw = ctx.raw
    benchmark = raw.get("electricity_trading_benchmark_EUR_per_MWh_el")
    if benchmark is None or "DA_position_MWh" not in raw:
        return None
    zero = pd.Series(0.0, index=raw.index)
    da_price = raw.get("day_ahead_delivered_price_EUR_per_MWh", zero)
    idc_price = raw.get("IDC_delivered_price_EUR_per_MWh", zero)
    value = pd.DataFrame(
        {
            "da": raw["DA_position_MWh"] * (benchmark - da_price).clip(lower=0),
            "idc": raw.get("IDC_buy_MWh", zero) * (benchmark - idc_price).clip(lower=0)
            + raw.get("IDC_sell_MWh", zero) * (idc_price - benchmark).clip(lower=0),
            "capacity": raw.get("afrr_capacity_revenue_EUR", zero),
            "energy": raw.get("afrr_energy_net_value_after_charges_EUR", zero).clip(lower=0),
        }
    )
    monthly = _monthly(value)
    total = float(monthly.sum().sum())
    if abs(total) < 1e-6:
        return None
    theme = ctx.theme
    fig = _new_figure(ctx, height=400)
    for key, label, entity in [
        ("da", "Day-ahead", "day_ahead"),
        ("idc", "Intraday", "intraday"),
        ("capacity", "aFRR capacity", "afrr_capacity"),
        ("energy", "aFRR energy", "afrr_energy"),
    ]:
        _add_bars(fig, ctx, key, label, theme.entity(entity), 1, ctx.case.currency,
                  series=monthly[key])  # fmt: skip
    fig.update_layout(barmode="relative")
    _yaxis(fig, 1, f"Savings vs gas ({ctx.case.currency})", tickformat=".3s")
    _month_axis(fig, monthly.index)
    return _finish(ctx, fig, f"Total {money(total, ctx.case.currency)}")


def monthly_price(ctx: ChartContext) -> go.Figure | None:
    raw = ctx.raw
    if "day_ahead_price_EUR_per_MWh" not in raw:
        return None
    monthly = raw["day_ahead_price_EUR_per_MWh"].resample("MS").mean()
    fig = _new_figure(ctx, height=400)
    _add_bars(fig, ctx, "price", "Monthly average", ctx.theme.entity("day_ahead"), 1, "€/MWh",
              series=monthly)  # fmt: skip
    _yaxis(fig, 1, "€/MWh")
    _month_axis(fig, monthly.index)
    _finish(ctx, fig)
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def cashflow_waterfall(ctx: ChartContext) -> go.Figure | None:
    """From the gas-only benchmark to the net cost, one step per market and charge."""

    benchmark = gas_only_benchmark(ctx)
    gas_cost = ctx.total("gas_cost_EUR")
    if benchmark is None or gas_cost is None:
        return None
    total = ctx.total

    def amount(column: str) -> float:
        return total(column) or 0.0

    gross, net = total("gross_operating_cost_EUR"), total("net_operating_cost_EUR")
    steps: list[tuple[str, float]] = [
        ("Gas displacement", gas_cost - benchmark),
        ("Day-ahead", amount("DA_electricity_cost_EUR")),
        ("Intraday buy", amount("IDC_buy_cost_EUR")),
        ("Intraday sell", -amount("IDC_sell_revenue_EUR")),
        ("aFRR energy", amount("afrr_energy_cost_EUR")),
        ("aFRR capacity", (net - gross) if gross is not None and net is not None else 0.0),
        ("Network charges", amount("additional_electricity_charges_cost_EUR")),
        ("Electricity tax", amount("tax_cost_EUR")),
        ("CO₂", amount("co2_cost_EUR")),
    ]
    if not ctx.sliced:
        steps.append(
            (
                "Grid fee correction",
                ctx.case.summary_total("grid_fee_ex_post_addition_EUR", ctx.plant) or 0.0,
            )
        )
    extra = investment(ctx)
    if extra:
        steps += [("CAPEX", extra["annuity"]), ("OPEX", extra["opex_annual"])]
    steps = [(label, value) for label, value in steps if abs(value) > 1e-6]
    final = benchmark + sum(value for _, value in steps)
    theme = ctx.theme
    labels = ["Gas-only benchmark", *[label for label, _ in steps], "Net cost"]
    values = [benchmark, *[value for _, value in steps], final]
    fig = _new_figure(ctx, height=520)
    fig.add_trace(
        go.Waterfall(
            x=labels,
            y=[v / 1000.0 for v in values],
            measure=["absolute", *["relative"] * len(steps), "total"],
            increasing={"marker": {"color": theme.entity("cost")}},
            decreasing={"marker": {"color": theme.entity("revenue")}},
            totals={"marker": {"color": theme.neutral_series}},
            connector={"line": {"color": theme.text_muted, "width": 1}},
            texttemplate="%{y:,.0f}",
            textposition="outside",
            textfont={"color": theme.text_secondary, "size": 11},
            hovertemplate="%{y:,.1f} k" + ctx.case.currency + "<extra>%{x}</extra>",
            cliponaxis=False,
        )
    )
    fig.update_yaxes(title_text=f"k{ctx.case.currency}")
    fig.update_xaxes(tickangle=-30)
    saving = benchmark - final
    basis = "Net incl. CAPEX and OPEX" if extra else "Net"
    _finish(
        ctx,
        fig,
        f"{basis} {money(final, ctx.case.currency)} · saving {money(saving, ctx.case.currency)}",
    )
    fig.update_layout(hovermode="closest", showlegend=False, margin={"b": 90})
    return fig


# ---------------------------------------------------------------------------
# Operation profile of the most active week
# ---------------------------------------------------------------------------


def sequential_profile(ctx: ChartContext) -> go.Figure | None:
    window = _active_window(ctx.raw)
    if window.empty or "DA_position_MWh" not in window:
        return None
    theme = ctx.theme
    native = ChartContext(ctx.case, ctx.plant, theme, resolution="native")
    native.__dict__["raw"] = window
    native.__dict__["df"] = window
    titles = [
        "Day-ahead baseline",
        "Intraday adjustment",
        "aFRR energy and final electricity",
        "Gas boiler heat",
        "Storage state of charge",
    ]
    fig = _new_figure(ctx, 5, titles, [0.2, 0.2, 0.22, 0.19, 0.19], height=1000)
    kwargs = {"frame": window}
    _add_bars(fig, native, "DA_position_MWh", "Day-ahead schedule", theme.entity("day_ahead"), 1,
              **kwargs)  # fmt: skip
    _add_bars(fig, native, "IDC_buy_MWh", "Intraday buy", theme.entity("intraday"), 2, **kwargs)
    _add_bars(fig, native, "IDC_sell_MWh", "Intraday sell / reduction", theme.entity("intraday"), 2,
              scale=-1.0, **kwargs)  # fmt: skip
    _add_line(fig, native, "final_planned_electricity_MWh", "Scheduled day-ahead + intraday",
              theme.text_primary, 2, "MWh", width=1.2, **kwargs)  # fmt: skip
    _add_bars(fig, native, "afrr_energy_activated_MWh", "aFRR energy activated",
              theme.entity("afrr_energy"), 3, **kwargs)  # fmt: skip
    _add_bars(fig, native, "afrr_capacity_reserved_MWh", "Reserved headroom",
              theme.entity("afrr_capacity"), 3, **kwargs)  # fmt: skip
    column = first_column(window, ["actual_electricity_consumption_MWh"])
    if column:
        _add_line(fig, native, column, "Actual electricity", theme.text_primary, 3, "MWh",
                  dash="dash", width=1.4, **kwargs)  # fmt: skip
    _add_bars(fig, native, "gas_heat_MWh", "Gas boiler heat", theme.entity("gas"), 4, "MWh th",
              **kwargs)  # fmt: skip
    _add_line(fig, native, "etes_soc_MWh", "Storage state of charge", theme.entity("storage"), 5,
              "MWh th", fill=True, shape="linear", **kwargs)  # fmt: skip
    fig.update_layout(barmode="relative")
    for row, label in enumerate(["MWh el", "MWh el", "MWh el", "MWh th", "MWh th"], start=1):
        _yaxis(fig, row, label)
    first, last = window.index.min(), window.index.max()
    return _finish(ctx, fig, f"{first:%d %b} – {last:%d %b %Y}")


# ---------------------------------------------------------------------------
# Grid fees
# ---------------------------------------------------------------------------


def grid_peaks(ctx: ChartContext) -> go.Figure | None:
    """Peak basis of the capacity charge (atypical grid use)."""

    if ctx.sliced or not has_tiers(ctx):
        return None
    annual = ctx.case.summary_total("grid_annual_peak_MW", ctx.plant)
    window = ctx.case.summary_total("grid_high_load_window_peak_MW", ctx.plant)
    billed = ctx.case.summary_total("grid_billed_peak_MW", ctx.plant)
    if annual is None or window is None or billed is None:
        return None
    theme = ctx.theme
    labels = ["Annual peak", "High-load-window peak", "Billed peak"]
    values = [annual, window, billed]
    fig = _new_figure(ctx, height=380)
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker={
                "color": [theme.neutral_fill, theme.neutral_series, theme.entity("cost")],
                "line": {"width": 0},
            },
            text=[f"{v:,.2f} MW" for v in values],
            textposition="outside",
            textfont={"color": theme.text_secondary, "size": 11},
            hovertemplate="%{y:,.2f} MW<extra>%{x}</extra>",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(type="category")
    _yaxis(fig, 1, "MW", range=[0, max(values + [1e-9]) * 1.2])
    _finish(ctx, fig)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def _window_bands(date: str, step_hours: float) -> list[tuple[float, float]]:
    """High-load-window hour bands of one weekday, from the regulation used by the model."""

    steps = int(round(24 / step_hours))
    index = pd.date_range(date, periods=steps, freq=f"{int(round(step_hours * 60))}min")
    flag = GermanGridFeeRegulation.compute_high_load_window(index).to_numpy() == 1
    bands, start = [], None
    for position, inside in enumerate(flag):
        hour = position * step_hours
        if inside and start is None:
            start = hour
        elif not inside and start is not None:
            bands.append((start, hour))
            start = None
    if start is not None:
        bands.append((start, 24.0))
    return bands


def atypical_grid_use(ctx: ChartContext) -> go.Figure | None:
    """Average weekday profile around the German high-load windows (winter and autumn)."""

    raw = ctx.raw
    if not has_tiers(ctx) or "actual_electricity_consumption_MWh" not in raw:
        return None
    step = ctx.case.step_hours
    frame = pd.DataFrame(
        {
            "grid": _mw(raw, "actual_electricity_consumption_MWh", step),
            "gas": _mw(raw, "gas_heat_MWh", step) if "gas_heat_MWh" in raw else 0.0,
            "discharge": _mw(raw, "etes_discharge_MWh", step)
            if "etes_discharge_MWh" in raw
            else 0.0,
        }
    )
    frame["hour"] = frame.index.hour + frame.index.minute / 60.0
    weekday = frame.index.weekday < 5
    seasons = [
        ("Winter weekdays (Jan, Feb, Dec)", [1, 2, 12], "2025-01-15"),
        ("Autumn weekdays (Sep, Oct, Nov)", [9, 10, 11], "2025-10-15"),
    ]
    seasons = [s for s in seasons if (weekday & frame.index.month.isin(s[1])).any()]
    if not seasons:
        return None
    theme = ctx.theme
    fig = _new_figure(ctx, len(seasons), [s[0] for s in seasons], height=260 + 250 * len(seasons))
    for row, (_, months, date) in enumerate(seasons, start=1):
        subset = frame[weekday & frame.index.month.isin(months)]
        profile = subset.groupby("hour")[["grid", "gas", "discharge"]].mean()
        for key, label, color, dash, fill in [
            ("grid", "Grid draw (e-heater)", theme.entity("electricity"), None, True),
            ("gas", "Gas boiler heat", theme.entity("gas"), None, False),
            ("discharge", "Storage discharge", theme.entity("storage"), "dot", False),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=profile.index,
                    y=_f32(profile[key]),
                    mode="lines",
                    name=label,
                    legendgroup=key,
                    showlegend=row == 1,
                    line={"color": color, "width": 2, "dash": dash},
                    fill="tozeroy" if fill else None,
                    fillcolor=with_alpha(color, 0.25) if fill else None,
                    hovertemplate=f"%{{y:,.2f}} MW<extra>{label}</extra>",
                ),
                row=row,
                col=1,
            )
        # Bands go in after the traces: Plotly ignores shapes on a subplot without traces.
        for start, end in _window_bands(date, step):
            fig.add_vrect(
                x0=start, x1=end, fillcolor=with_alpha(theme.entity("emissions"), 0.12),
                line_width=0, row=row, col=1,
            )  # fmt: skip
        _yaxis(fig, row, "MW")
    fig.update_xaxes(range=[0, 24], dtick=2)
    fig.update_xaxes(title_text="Hour of day (local)", row=len(seasons), col=1)
    inside = GermanGridFeeRegulation.compute_high_load_window(frame.index).to_numpy() == 1
    energy_in = float(raw.loc[inside, "actual_electricity_consumption_MWh"].sum())
    energy_total = float(raw["actual_electricity_consumption_MWh"].sum())
    peak_in = float(np.nan_to_num(frame.loc[inside, "grid"].max())) if inside.any() else 0.0
    peak_all = float(np.nan_to_num(frame["grid"].max()))
    share = energy_in / energy_total if energy_total else 0.0
    _finish(
        ctx,
        fig,
        f"In-window draw {number(energy_in, 'MWh')} ({share:.2%}) · "
        f"peak {peak_in:,.2f} MW in-window vs {peak_all:,.2f} MW overall",
    )
    return fig


# ---------------------------------------------------------------------------
# Heat demand patterns
# ---------------------------------------------------------------------------


def _heat_mw(ctx: ChartContext) -> pd.Series | None:
    if "heat_demand_MWh" not in ctx.raw or ctx.raw.empty:
        return None
    return ctx.raw["heat_demand_MWh"] / ctx.case.step_hours


def heat_heatmap(ctx: ChartContext) -> go.Figure | None:
    heat = _heat_mw(ctx)
    if heat is None:
        return None
    hourly = heat.resample("1h").mean()
    if hourly.index.normalize().nunique() < 2:
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
                "title": {"text": "MW th", "font": {"color": ctx.theme.text_secondary}},
                "tickfont": {"color": ctx.theme.text_muted},
                "thickness": 12,
                "outlinewidth": 0,
            },
            hovertemplate="%{x|%d %b %Y} %{y}:00<br>%{z:,.2f} MW th<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Hour of day", dtick=6, autorange="reversed")
    _finish(ctx, fig)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def demand_weekday_heatmap(ctx: ChartContext) -> go.Figure | None:
    heat = _heat_mw(ctx)
    if heat is None or heat.index.normalize().nunique() < 7:
        return None
    frame = pd.DataFrame({"mw": heat, "weekday": heat.index.dayofweek, "hour": heat.index.hour})
    table = frame.groupby(["weekday", "hour"])["mw"].mean().unstack("hour")
    fig = _new_figure(ctx, height=380)
    fig.add_trace(
        go.Heatmap(
            x=table.columns,
            y=[WEEKDAYS[i] for i in table.index],
            z=_f32(table.to_numpy()),
            colorscale=ctx.theme.colorscale(),
            colorbar={
                "title": {"text": "MW th", "font": {"color": ctx.theme.text_secondary}},
                "tickfont": {"color": ctx.theme.text_muted},
                "thickness": 12,
                "outlinewidth": 0,
            },
            hovertemplate="%{y} %{x}:00<br>%{z:,.2f} MW th<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Hour of day", dtick=2)
    fig.update_yaxes(autorange="reversed")
    _finish(ctx, fig)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def demand_averages(ctx: ChartContext) -> go.Figure | None:
    heat = _heat_mw(ctx)
    if heat is None or len(heat) < 48:
        return None
    theme = ctx.theme
    hourly = heat.groupby(heat.index.hour).mean()
    weekday = heat.groupby(heat.index.dayofweek).mean()
    fig = _new_figure(ctx, 2, ["Average by hour of day", "Average by weekday"], height=520,
                      shared_x=False)  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=hourly.index,
            y=_f32(hourly),
            mode="lines+markers",
            name="Hour of day",
            line={"color": theme.neutral_series, "width": 2},
            marker={"size": 6, "color": theme.neutral_series},
            hovertemplate="%{y:,.2f} MW th<extra>hour %{x}</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[WEEKDAYS[i] for i in weekday.index],
            y=_f32(weekday),
            name="Weekday",
            marker={"color": theme.neutral_series, "line": {"width": 0}},
            hovertemplate="%{y:,.2f} MW th<extra>%{x}</extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Hour of day", dtick=2, row=1, col=1)
    fig.update_xaxes(type="category", row=2, col=1)
    _yaxis(fig, 1, "MW th")
    _yaxis(fig, 2, "MW th")
    _finish(ctx, fig)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def heat_duration(ctx: ChartContext) -> go.Figure | None:
    heat = _heat_mw(ctx)
    if heat is None or len(heat) < 10:
        return None
    load = np.sort(heat.to_numpy())[::-1]
    hours = np.arange(1, len(load) + 1) * ctx.case.step_hours
    color = ctx.theme.neutral_series
    fig = _new_figure(ctx, height=380)
    fig.add_trace(
        go.Scatter(
            x=_f32(hours),
            y=_f32(load),
            mode="lines",
            name="Heat demand",
            line={"color": color, "width": 2},
            fill="tozeroy",
            fillcolor=with_alpha(color, 0.25),
            hovertemplate="%{y:,.2f} MW th<extra>%{x:,.0f} h</extra>",
        )
    )
    fig.update_xaxes(title_text="Hours")
    _yaxis(fig, 1, "MW th")
    _finish(ctx, fig)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def electricity_balance_table(ctx: ChartContext) -> pd.DataFrame | None:
    raw = ctx.raw
    if "DA_position_MWh" not in raw:
        return None
    da = float(raw["DA_position_MWh"].sum())
    buy = float(raw["IDC_buy_MWh"].sum()) if "IDC_buy_MWh" in raw else 0.0
    sell = float(raw["IDC_sell_MWh"].sum()) if "IDC_sell_MWh" in raw else 0.0
    afrr = (
        float(raw["afrr_energy_activated_MWh"].sum()) if "afrr_energy_activated_MWh" in raw else 0.0
    )
    total = da + buy - sell + afrr
    if total <= 0:
        return None
    rows = [
        ("Day-ahead procurement", da),
        ("Intraday bought", buy),
        ("Intraday sold back (reduction)", -sell),
        ("aFRR energy activated", afrr),
        ("Total consumption", total),
    ]
    return pd.DataFrame(
        {
            "Market channel": [label for label, _ in rows],
            "Volume (MWh el)": [value for _, value in rows],
            "Share of consumption (%)": [100.0 * value / total for _, value in rows],
        }
    )


def price_statistics_table(ctx: ChartContext) -> pd.DataFrame | None:
    raw = ctx.raw
    rows = []
    for column, label in [
        ("day_ahead_price_EUR_per_MWh", "Day-ahead market price"),
        ("day_ahead_delivered_price_EUR_per_MWh", "Day-ahead delivered price"),
        ("IDC_delivered_price_EUR_per_MWh", "Intraday delivered price"),
        ("afrr_energy_delivered_price_EUR_per_MWh", "aFRR delivered price"),
        ("electricity_trading_benchmark_EUR_per_MWh_el", "Gas-based electricity benchmark"),
    ]:
        if column in raw:
            values = raw[column].replace([np.inf, -np.inf], np.nan).dropna()
            if not values.empty:
                rows.append(
                    {
                        "Series (EUR/MWh el)": label,
                        "mean": values.mean(),
                        "min": values.min(),
                        "p10": values.quantile(0.1),
                        "p90": values.quantile(0.9),
                        "max": values.max(),
                    }
                )
    return pd.DataFrame(rows) if rows else None


def plant_config_table(ctx: ChartContext) -> pd.DataFrame | None:
    config = ctx.case.plant_config
    if config.empty:
        return None
    keep = [
        c
        for c in (
            "name", "unit_type", "technology", "fuel_type", "max_power", "efficiency",
            "max_capacity", "max_power_charge", "max_power_discharge", "initial_soc",
            "efficiency_charge", "efficiency_discharge",
        )
        if c in config.columns
    ]  # fmt: skip
    return config[keep]


def charges_table(ctx: ChartContext) -> pd.DataFrame | None:
    return ctx.case.charges if not ctx.case.charges.empty else None


def plant_sizing(config: pd.DataFrame) -> dict[str, float]:
    """Installed power, storage size and efficiencies from ``plants.csv``."""

    def num(row: pd.Series, column: str) -> float:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        return float(value) if pd.notna(value) else float("nan")

    sizing = dict.fromkeys(
        [
            "eheater_power_MW",
            "discharge_power_MW",
            "storage_capacity_MWh",
            "gas_boiler_power_MW",
            "gas_boiler_efficiency",
            "charge_efficiency",
            "discharge_efficiency",
        ],
        float("nan"),
    )
    if config.empty or "technology" not in config:
        return sizing
    tech = config["technology"].astype(str).str.lower()
    storage = config[tech.str.contains("storage")]
    eboiler = config[tech.eq("electric_boiler")]
    boiler = config[tech.eq("boiler")]
    if not storage.empty:
        row = storage.iloc[0]
        charge = num(row, "max_power_charge")
        sizing["eheater_power_MW"] = charge if np.isfinite(charge) else num(row, "max_power")
        sizing["discharge_power_MW"] = num(row, "max_power_discharge")
        sizing["storage_capacity_MWh"] = num(row, "max_capacity")
        sizing["charge_efficiency"] = num(row, "efficiency_charge")
        sizing["discharge_efficiency"] = num(row, "efficiency_discharge")
    if not eboiler.empty:
        row = eboiler.iloc[0]
        sizing["eheater_power_MW"] = num(row, "max_power")
        sizing["charge_efficiency"] = num(row, "efficiency")
    if not boiler.empty:
        row = boiler.iloc[0]
        sizing["gas_boiler_power_MW"] = num(row, "max_power")
        sizing["gas_boiler_efficiency"] = num(row, "efficiency")
    return sizing


def system_kpis(ctx: ChartContext) -> list[tuple[str, str, str]]:
    """System setup tiles: demand profile type, installed power and storage, efficiencies."""

    config = ctx.case.plant_config
    if config.empty:
        return []
    size = plant_sizing(config)
    tiles: list[tuple[str, str, str]] = []
    heat = ctx.raw.get("heat_demand_MWh")
    if heat is not None and heat.mean() > 0:
        cv = float(heat.std() / heat.mean())
        kind = "Constant" if cv < 0.05 else "Time-variant"
        peak = float(heat.max())
        tiles.append(("Thermal profile", kind, f"CV {cv:.0%} · peak {peak:,.1f} MWh/step"))

    def show(label: str, key: str, unit: str, detail: str = "", pct: bool = False) -> None:
        value = size[key]
        if np.isfinite(value):
            tiles.append((label, f"{value:.0%}" if pct else f"{value:,.1f} {unit}", detail))

    show("E-heater charge power", "eheater_power_MW", "MW el", "storage charging capacity")
    show("E-heater discharge power", "discharge_power_MW", "MW th", "storage discharging capacity")
    capacity, power = size["storage_capacity_MWh"], size["eheater_power_MW"]
    if np.isfinite(capacity):
        hours = f"~{capacity / power:,.1f} h at full e-heater power" if power > 0 else ""
        tiles.append(("Storage capacity", f"{capacity:,.1f} MWh th", hours))
    show("Charge efficiency", "charge_efficiency", "", "electricity → stored heat", pct=True)
    show("Discharge efficiency", "discharge_efficiency", "", "stored heat → process heat", pct=True)
    eta_c, eta_d = size["charge_efficiency"], size["discharge_efficiency"]
    if np.isfinite(eta_c) and np.isfinite(eta_d):
        tiles.append(("Round-trip efficiency", f"{eta_c * eta_d:.0%}", "charge × discharge"))
    show("Gas boiler efficiency", "gas_boiler_efficiency", "", "gas → heat", pct=True)
    show("Gas boiler backup power", "gas_boiler_power_MW", "MW th", "hybrid-mode fallback supply")
    return tiles


def per_mw_table(ctx: ChartContext) -> pd.DataFrame | None:
    """Cost components per installed MW of e-heater power."""

    power = plant_sizing(ctx.case.plant_config)["eheater_power_MW"]
    parts = cost_components(ctx)
    if not (np.isfinite(power) and power > 0 and parts):
        return None
    net = sum(value for _, value in parts)
    rows = [*parts, ("Net operating cost", net)]
    benchmark = gas_only_benchmark(ctx)
    if benchmark is not None:
        rows.append(("Savings vs gas-only", benchmark - net))
    return pd.DataFrame(
        {
            "Component": [label for label, _ in rows],
            ctx.case.currency: [value for _, value in rows],
            f"{ctx.case.currency} per MW el e-heater ({power:g} MW)": [v / power for _, v in rows],
        }
    )


STEAM_BUILDERS: dict[str, FigureBuilder] = {
    **STEAM_CORE_BUILDERS,
    "heat_coverage_annual": heat_coverage_annual,
    "heat_coverage_monthly": heat_coverage_monthly,
    "storage_operation": storage_operation,
    "monthly_procurement": monthly_procurement,
    "monthly_market_value": monthly_market_value,
    "monthly_price": monthly_price,
    "cashflow_waterfall": cashflow_waterfall,
    "sequential_profile": sequential_profile,
    "grid_peaks": grid_peaks,
    "atypical_grid_use": atypical_grid_use,
    "heat_heatmap": heat_heatmap,
    "demand_weekday_heatmap": demand_weekday_heatmap,
    "demand_averages": demand_averages,
    "heat_duration": heat_duration,
}

STEAM_TABLES = {
    "electricity_balance": electricity_balance_table,
    "price_statistics": price_statistics_table,
    "plant_config": plant_config_table,
    "charges": charges_table,
    "per_mw": per_mw_table,
}
