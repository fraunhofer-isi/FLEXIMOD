# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from flexi_mod.visualisation.analytics import (
    calculate_summary_indicators,
    create_output_dir,
    derive_gas_benchmark,
    ensure_datetime_index,
    load_results,
    require_columns,
    save_summary_indicators,
    select_sample_day,
    storage_content_by_source,
    warn_missing,
)

REPORT_STYLE = {
    "figure.figsize": (12, 5.5),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": "#3A3A3A",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

_MUTED = sns.color_palette("muted", 10)
COLOR_GAS = _MUTED[3]
COLOR_DA = _MUTED[0]
COLOR_IDC = _MUTED[1]
COLOR_AFRR = _MUTED[4]
COLOR_STORAGE = _MUTED[2]
COLOR_BENCHMARK = _MUTED[3]
COLOR_CAPACITY = _MUTED[6]
COLOR_NEUTRAL = "#4A4A4A"
COLOR_LIGHT_NEUTRAL = "#8A8A8A"


def create_all_plots_from_output(
    output_dir: str | Path,
    file_format: str = "png",
    show: bool = False,
    sample_day: str | None = None,
) -> list[Path]:
    """Load output CSV files, refresh analytics and create all report plots."""

    results = load_results(output_dir)
    summary = calculate_summary_indicators(
        results.dispatch_results,
        market_ledger=results.market_ledger,
        storage_cost_ledger=results.storage_cost_ledger,
        afrr_energy_data_quality_summary=results.afrr_energy_data_quality_summary,
    )
    if not summary.empty:
        save_summary_indicators(summary, results.output_dir)

    return create_case_plots(
        dispatch_results=results.dispatch_results,
        summary_indicators=summary if not summary.empty else results.summary_indicators,
        output_dir=results.output_dir,
        market_ledger=results.market_ledger,
        storage_cost_ledger=results.storage_cost_ledger,
        file_format=file_format,
        show=show,
        sample_day=sample_day,
    )


def create_case_plots(
    dispatch_results: pd.DataFrame,
    summary_indicators: pd.DataFrame | None,
    output_dir: str | Path,
    market_ledger: pd.DataFrame | None = None,
    storage_cost_ledger: pd.DataFrame | None = None,
    file_format: str = "png",
    show: bool = False,
    sample_day: str | None = None,
) -> list[Path]:
    """Create the standard FlexIMOD plotting suite."""

    with (
        sns.axes_style("whitegrid"),
        sns.plotting_context("notebook"),
        plt.rc_context(REPORT_STYLE),
    ):
        output_dir = Path(output_dir)
        plot_dir = create_output_dir(output_dir)
        dispatch = ensure_datetime_index(dispatch_results)
        market = ensure_datetime_index(market_ledger)
        storage = ensure_datetime_index(storage_cost_ledger)
        summary = summary_indicators if summary_indicators is not None else pd.DataFrame()
        if summary.empty and not dispatch.empty:
            summary = calculate_summary_indicators(dispatch, market, storage)

        created: list[Path] = []
        created.extend(plot_operation_and_storage_dynamics(dispatch, plot_dir, file_format, show))
        created.extend(
            plot_market_prices_and_benchmark(dispatch, market, plot_dir, file_format, show)
        )
        created.extend(
            plot_electricity_procurement_by_market(market, dispatch, plot_dir, file_format, show)
        )
        created.extend(
            plot_storage_content_by_market_source(storage, dispatch, plot_dir, file_format, show)
        )
        created.extend(
            plot_sample_day_detailed_operation(
                dispatch,
                market,
                plot_dir,
                file_format,
                show,
                sample_day=sample_day,
            )
        )
        created.extend(plot_cost_breakdown(summary, plot_dir, file_format, show))
        created.extend(plot_heat_supply_share(dispatch, plot_dir, file_format, show))
        created.extend(plot_electricity_market_share(market, dispatch, plot_dir, file_format, show))
        created.extend(plot_price_response_storage_charging(dispatch, plot_dir, file_format, show))
        created.extend(
            plot_sequential_market_position_evolution(
                market,
                dispatch,
                plot_dir,
                file_format,
                show,
            )
        )
        created.extend(
            plot_idc_sell_source_and_compensation(
                market,
                dispatch,
                plot_dir,
                file_format,
                show,
            )
        )
        created.extend(plot_stagewise_gas_replacement(dispatch, plot_dir, file_format, show))
        created.extend(plot_afrr_capacity_price_and_reserve(market, plot_dir, file_format, show))
        created.extend(plot_procurement_and_capacity_headroom(market, plot_dir, file_format, show))
        created.extend(plot_afrr_capacity_and_energy(market, plot_dir, file_format, show))
        return created


def plot_operation_and_storage_dynamics(
    dispatch_results: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    dispatch = _aggregate_by_datetime(dispatch_results)
    required = require_columns(
        dispatch,
        ["heat_demand_MWh", "gas_heat_MWh", "etes_discharge_MWh"],
        "operation and storage dynamics plot",
    )
    if len(required) < 3:
        return []

    fig, (ax, storage_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.25]},
    )
    ax.plot(dispatch.index, dispatch["heat_demand_MWh"], color=COLOR_NEUTRAL, label="Heat demand")
    gas_heat = dispatch["gas_heat_MWh"].fillna(0.0)
    storage_discharge = dispatch["etes_discharge_MWh"].fillna(0.0)

    stack_values = [gas_heat, storage_discharge]
    labels = ["Gas boiler heat", "Storage discharge heat"]

    ax.stackplot(
        dispatch.index,
        *stack_values,
        labels=labels,
        colors=[COLOR_GAS, COLOR_STORAGE],
        alpha=0.78,
    )
    ax.set_title("Plant operation and storage dynamics")
    ax.set_ylabel("Heat per time step [MWh_th]")
    ax.legend(loc="upper right", ncol=2)

    _draw_storage_operation_panel(storage_ax, dispatch)
    storage_ax.set_xlabel("Time")

    _format_datetime_axis(storage_ax)
    fig.tight_layout()
    return _save_figure(fig, plot_dir, "01_operation_and_storage_dynamics", file_format, show)


def plot_storage_operation_soc(
    dispatch_results: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    dispatch = _aggregate_by_datetime(dispatch_results)
    required = require_columns(
        dispatch,
        ["etes_charge_MWh", "etes_discharge_MWh", "etes_soc_MWh"],
        "storage operation and SoC plot",
    )
    if len(required) < 3:
        return []

    fig, ax = plt.subplots()
    width = _bar_width(dispatch.index)
    ax.bar(
        dispatch.index,
        dispatch["etes_charge_MWh"],
        width=width,
        label="ETES charge",
        color=COLOR_DA,
        alpha=0.65,
    )
    ax.bar(
        dispatch.index,
        -dispatch["etes_discharge_MWh"],
        width=width,
        label="ETES discharge",
        color=COLOR_STORAGE,
        alpha=0.65,
    )
    ax.axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)
    ax.set_title("Storage operation and state of charge")
    ax.set_ylabel("Charge/discharge per time step [MWh]")
    ax.set_xlabel("Time")

    ax2 = ax.twinx()
    ax2.plot(dispatch.index, dispatch["etes_soc_MWh"], color=COLOR_STORAGE, label="ETES SoC")
    ax2.set_ylabel("State of charge [MWh_th]")
    _combined_legend(ax, ax2)
    _format_datetime_axis(ax)
    return _save_figure(fig, plot_dir, "02_storage_operation_soc", file_format, show)


def plot_market_prices_and_benchmark(
    dispatch_results: pd.DataFrame,
    market_ledger: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    dispatch = _price_frame(dispatch_results, market_ledger)
    if dispatch.empty:
        return []

    fig, ax = plt.subplots()
    plotted = False
    day_ahead_column = _first_existing(
        dispatch,
        ["day_ahead_delivered_price_EUR_per_MWh_el", "day_ahead_price_EUR_per_MWh_el"],
    )
    if day_ahead_column:
        ax.plot(
            dispatch.index,
            dispatch[day_ahead_column],
            label=_price_label("Day-ahead", day_ahead_column),
            color=COLOR_DA,
        )
        plotted = True
    else:
        warn_missing("day_ahead_price_EUR_per_MWh_el", "market prices plot")

    for column, label, color in [
        (
            _first_existing(
                dispatch,
                ["intraday_delivered_price_EUR_per_MWh_el", "intraday_price_EUR_per_MWh_el"],
            ),
            "IDC",
            COLOR_IDC,
        ),
        (
            _first_existing(
                dispatch,
                [
                    "afrr_energy_delivered_price_EUR_per_MWh_el",
                    "afrr_energy_price_EUR_per_MWh_el",
                ],
            ),
            "aFRR energy",
            COLOR_AFRR,
        ),
    ]:
        if column in dispatch.columns and dispatch[column].notna().any():
            ax.plot(
                dispatch.index,
                dispatch[column],
                label=_price_label(label, column),
                color=color,
                alpha=0.9,
            )
        else:
            warn_missing(f"{label} price", "market prices plot")

    benchmark = derive_gas_benchmark(dispatch_results)
    if benchmark is not None:
        benchmark = benchmark.groupby(benchmark.index).mean()
        ax.plot(
            benchmark.index,
            benchmark,
            label="Gas-based heat benchmark",
            color=COLOR_BENCHMARK,
            linewidth=1.8,
        )
        plotted = True
    else:
        warn_missing("gas_based_heat_benchmark", "market prices plot")

    if not plotted:
        plt.close(fig)
        return []

    ax.set_title("Market prices and gas-based heat benchmark")
    ax.set_ylabel("Price [EUR/MWh]")
    ax.set_xlabel("Time")
    ax.legend(loc="best")
    _format_datetime_axis(ax)
    return _save_figure(fig, plot_dir, "03_market_prices_and_benchmark", file_format, show)


def plot_electricity_procurement_by_market(
    market_ledger: pd.DataFrame,
    dispatch_results: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    market = _market_frame(market_ledger, dispatch_results)
    if market.empty:
        return []

    fig, ax = plt.subplots()
    positive_columns = [
        ("day_ahead_position_MWh_el", "Day-ahead", COLOR_DA),
        ("intraday_buy_MWh_el", "IDC buy", COLOR_IDC),
        ("afrr_energy_activated_MWh_el", "aFRR activated", COLOR_AFRR),
    ]
    bottom = pd.Series(0.0, index=market.index)
    plotted = False
    for column, label, color in positive_columns:
        if column in market.columns:
            values = market[column].fillna(0.0).astype(float)
            _fill_between_series(
                ax,
                market.index,
                bottom + values,
                lower=bottom,
                color=color,
                alpha=0.62,
                label=label,
            )
            bottom += values
            plotted = True
        else:
            warn_missing(column, "electricity procurement plot")

    if "intraday_sell_MWh_el" in market.columns:
        idc_sell = market["intraday_sell_MWh_el"].fillna(0.0).astype(float)
        _fill_between_series(
            ax,
            market.index,
            -idc_sell,
            color=COLOR_GAS,
            alpha=0.58,
            label="IDC sell/reduction",
        )
    else:
        warn_missing("intraday_sell_MWh_el", "electricity procurement plot")

    if not plotted:
        plt.close(fig)
        return []

    ax.axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)
    ax.set_title("Electricity procurement by market over time")
    ax.set_ylabel("Electricity per time step [MWh_el]")
    ax.set_xlabel("Time")
    ax.legend(loc="best", ncol=2)
    _format_datetime_axis(ax)
    return _save_figure(
        fig,
        plot_dir,
        "04_electricity_procurement_by_market",
        file_format,
        show,
    )


def plot_storage_content_by_market_source(
    storage_cost_ledger: pd.DataFrame,
    dispatch_results: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    content = storage_content_by_source(storage_cost_ledger, dispatch_results)
    if content.empty:
        return []

    total_soc = None
    if "Total ETES SoC" in content.columns:
        total_soc = content.pop("Total ETES SoC")

    fig, ax = plt.subplots()
    if not content.empty:
        ax.stackplot(
            content.index,
            *[content[column].fillna(0.0) for column in content.columns],
            labels=list(content.columns),
            colors=sns.color_palette("muted", len(content.columns)),
            alpha=0.82,
        )
    if total_soc is not None:
        ax.plot(
            total_soc.index,
            total_soc,
            color=COLOR_NEUTRAL,
            linewidth=1.2,
            label="Total ETES SoC",
        )

    ax.set_title("Storage content by source market")
    ax.set_ylabel("Stored heat [MWh_th]")
    ax.set_xlabel("Time")
    ax.legend(loc="best")
    _format_datetime_axis(ax)
    return _save_figure(
        fig,
        plot_dir,
        "05_storage_content_by_market_source",
        file_format,
        show,
    )


def plot_sample_day_detailed_operation(
    dispatch_results: pd.DataFrame,
    market_ledger: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
    sample_day: str | None = None,
) -> list[Path]:
    dispatch = _aggregate_by_datetime(dispatch_results)
    if dispatch.empty:
        return []

    day = select_sample_day(dispatch, sample_day)
    end = day + pd.Timedelta(days=1)
    day_dispatch = dispatch.loc[(dispatch.index >= day) & (dispatch.index < end)]
    day_market = _market_frame(market_ledger, dispatch_results)
    day_market = day_market.loc[(day_market.index >= day) & (day_market.index < end)]
    if day_dispatch.empty:
        warnings.warn(f"No dispatch rows found for sample day {day.date()}.", stacklevel=2)
        return []

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle(f"Detailed operation on {day.date()}", fontsize=15)
    _sample_day_prices_panel(axes[0], day_dispatch, day_market, dispatch_results)
    _sample_day_procurement_panel(axes[1], day_market)
    _sample_day_heat_panel(axes[2], day_dispatch)
    _sample_day_storage_panel(axes[3], day_dispatch)
    for ax in axes:
        ax.legend(loc="best", ncol=2)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time")
    _format_datetime_axis(axes[-1])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(fig, plot_dir, "06_sample_day_detailed_operation", file_format, show)


def plot_cost_breakdown(
    summary_indicators: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    if summary_indicators is None or summary_indicators.empty:
        warnings.warn(
            "No summary indicators available. Skipping cost breakdown plot.",
            stacklevel=2,
        )
        return []

    summary = _aggregate_summary(summary_indicators)
    labels_and_columns = [
        ("Electricity market cost", "total_electricity_market_cost_EUR"),
        ("Additional electricity charges", "total_additional_electricity_charges_cost_EUR"),
        ("Gas cost", "total_gas_cost_EUR"),
        ("CO2 cost", "total_co2_cost_EUR"),
        ("IDC trading value", "total_IDC_trading_value_EUR"),
        ("aFRR energy value", "total_afrr_energy_value_EUR"),
        ("aFRR capacity revenue", "total_afrr_capacity_revenue_EUR"),
        ("Net operating cost", "total_net_operating_cost_EUR"),
    ]
    values = []
    labels = []
    for label, column in labels_and_columns:
        if column in summary:
            values.append(float(summary[column]))
            labels.append(label)
        else:
            warn_missing(column, "cost breakdown plot")
    if not values:
        return []

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [COLOR_DA if value >= 0 else COLOR_STORAGE for value in values]
    ax.bar(labels, values, color=colors, alpha=0.8)
    ax.set_title("Cost and value breakdown")
    ax.set_ylabel("EUR")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save_figure(fig, plot_dir, "07_cost_breakdown", file_format, show)


def plot_heat_supply_share(
    dispatch_results: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    dispatch = _aggregate_by_datetime(dispatch_results)
    columns = [
        ("gas_heat_MWh", "Gas boiler heat"),
        ("etes_discharge_MWh", "Storage discharge heat"),
    ]
    values = [
        (label, _sum_column(dispatch, column)) for column, label in columns if column in dispatch
    ]
    if not values:
        return []

    fig, ax = plt.subplots(figsize=(10, 3.5))
    left = 0.0
    for label, value in values:
        color = COLOR_GAS if "Gas" in label else COLOR_STORAGE
        ax.barh(["Heat supply"], [value], left=left, label=label, color=color, alpha=0.8)
        left += value
    ax.set_title("Heat supply share")
    ax.set_xlabel("Heat supplied [MWh_th]")
    ax.legend(loc="best")
    return _save_figure(fig, plot_dir, "08_heat_supply_share", file_format, show)


def plot_electricity_market_share(
    market_ledger: pd.DataFrame,
    dispatch_results: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    market = _market_frame(market_ledger, dispatch_results)
    if market.empty:
        return []

    components = [
        ("day_ahead_position_MWh_el", "Day-ahead"),
        ("intraday_buy_MWh_el", "IDC buy"),
        ("intraday_sell_MWh_el", "IDC sell/reduction"),
        ("afrr_energy_activated_MWh_el", "aFRR activated"),
    ]
    values = []
    for column, label in components:
        if column in market.columns:
            sign = -1.0 if column == "intraday_sell_MWh_el" else 1.0
            values.append((label, sign * _sum_column(market, column)))
        else:
            warn_missing(column, "electricity market share plot")

    if not values:
        return []

    actual = _sum_column(market, "actual_electricity_consumption_MWh_el")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    left = 0.0
    for label, value in values:
        color = {
            "Day-ahead": COLOR_DA,
            "IDC buy": COLOR_IDC,
            "IDC sell/reduction": COLOR_GAS,
            "aFRR activated": COLOR_AFRR,
        }.get(label, COLOR_LIGHT_NEUTRAL)
        ax.barh(["Market procurement"], [value], left=left, label=label, color=color, alpha=0.8)
        left += value
    if actual:
        ax.axvline(actual, color=COLOR_NEUTRAL, linewidth=1.5, label="Actual consumption")
    ax.set_title("Electricity market share")
    ax.set_xlabel("Electricity [MWh_el]")
    ax.legend(loc="best")
    return _save_figure(fig, plot_dir, "09_electricity_market_share", file_format, show)


def plot_price_response_storage_charging(
    dispatch_results: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    dispatch = _aggregate_by_datetime(dispatch_results)
    price_col = _first_existing(
        dispatch,
        [
            "day_ahead_delivered_price_EUR_per_MWh",
            "day_ahead_delivered_price_EUR_per_MWh_el",
            "day_ahead_price_EUR_per_MWh",
            "day_ahead_price_EUR_per_MWh_el",
        ],
    )
    if price_col is None:
        warn_missing("day_ahead_price_EUR_per_MWh", "price response plot")
        return []
    if "etes_charge_MWh" not in dispatch.columns:
        warn_missing("etes_charge_MWh", "price response plot")
        return []

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(
        dispatch[price_col],
        dispatch["etes_charge_MWh"],
        s=18,
        alpha=0.65,
        color=COLOR_DA,
        label="ETES charging",
    )
    benchmark = derive_gas_benchmark(dispatch_results)
    if benchmark is not None:
        ax.axvline(
            float(benchmark.mean()),
            color=COLOR_BENCHMARK,
            linestyle="--",
            label="Average gas benchmark",
        )
    else:
        warn_missing("gas_based_heat_benchmark", "price response plot")

    ax.set_title("Price response of storage charging")
    ax.set_xlabel("Delivered electricity price [EUR/MWh]")
    ax.set_ylabel("ETES charging per time step [MWh_el]")
    ax.legend(loc="best")
    return _save_figure(fig, plot_dir, "10_price_response_storage_charging", file_format, show)


def plot_sequential_market_position_evolution(
    market_ledger: pd.DataFrame,
    dispatch_results: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    """Show how electricity positions evolve through DA, IDC and aFRR energy."""

    market = _market_frame(market_ledger, dispatch_results)
    if market.empty:
        return []

    if "day_ahead_position_MWh_el" not in market.columns:
        warn_missing("day_ahead_position_MWh_el", "sequential market position plot")
        return []

    market = market.sort_index()
    day_ahead = _column_or_zero(market, "day_ahead_position_MWh_el")
    intraday_buy = _column_or_zero(market, "intraday_buy_MWh_el")
    intraday_sell = _column_or_zero(market, "intraday_sell_MWh_el")
    scheduled = _column_or_default(
        market,
        "scheduled_electricity_procurement_MWh_el",
        day_ahead + intraday_buy - intraday_sell,
    )
    afrr_activated = _column_or_zero(market, "afrr_energy_activated_MWh_el")
    actual = _column_or_default(
        market,
        "actual_electricity_consumption_MWh_el",
        scheduled + afrr_activated,
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 11.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.25, 1.0]},
    )
    fig.suptitle("Sequential market positions and gas boiler operation", fontsize=15)

    gas_heat = _gas_heat_output_series(market, dispatch_results)
    _fill_between_series(
        axes[0],
        market.index,
        day_ahead,
        color=COLOR_DA,
        alpha=0.22,
        label="Day-ahead position",
    )
    axes[0].step(
        market.index,
        day_ahead,
        where="post",
        color=COLOR_DA,
        linewidth=1.3,
        label="DA position",
    )
    axes[0].set_title("Day-ahead position")
    axes[0].set_ylabel("MWh_el")
    axes[0].axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)

    _fill_between_series(
        axes[1],
        market.index,
        day_ahead,
        color=COLOR_DA,
        alpha=0.18,
        label="DA baseline",
    )
    _fill_between_series(
        axes[1],
        market.index,
        day_ahead + intraday_buy,
        lower=day_ahead,
        color=COLOR_IDC,
        alpha=0.34,
        label="IDC buy addition",
    )
    _fill_between_series(
        axes[1],
        market.index,
        -intraday_sell,
        color=COLOR_GAS,
        alpha=0.28,
        label="IDC sell/reduction",
    )
    axes[1].step(
        market.index,
        day_ahead,
        where="post",
        color=COLOR_DA,
        linewidth=1.0,
        alpha=0.65,
        label="DA baseline line",
    )
    axes[1].step(
        market.index,
        scheduled,
        where="post",
        color=COLOR_NEUTRAL,
        linewidth=1.3,
        label="Scheduled DA+IDC",
    )
    axes[1].axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)
    axes[1].set_title("After intraday adjustment")
    axes[1].set_ylabel("MWh_el")

    _fill_between_series(
        axes[2],
        market.index,
        scheduled,
        color=COLOR_DA,
        alpha=0.18,
        label="Scheduled DA+IDC",
    )
    _fill_between_series(
        axes[2],
        market.index,
        actual,
        lower=scheduled,
        color=COLOR_AFRR,
        alpha=0.32,
        label="aFRR down activation",
    )
    axes[2].step(
        market.index,
        scheduled,
        where="post",
        color=COLOR_DA,
        linewidth=1.0,
        alpha=0.7,
        label="Scheduled DA+IDC line",
    )
    axes[2].step(
        market.index,
        actual,
        where="post",
        color=COLOR_NEUTRAL,
        linewidth=1.3,
        label="Actual consumption",
    )
    axes[2].axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)
    axes[2].set_title("After aFRR down energy activation")
    axes[2].set_ylabel("MWh_el")
    axes[2].set_xlabel("Time")

    axes[3].set_title("Gas boiler operation")
    axes[3].set_ylabel("MWh_th")
    axes[3].set_xlabel("Time")
    if gas_heat is not None and gas_heat.notna().any():
        _fill_between_series(
            axes[3],
            gas_heat.index,
            gas_heat,
            color=COLOR_GAS,
            alpha=0.2,
            label="Gas boiler heat",
        )
        axes[3].step(
            gas_heat.index,
            gas_heat,
            where="post",
            color=COLOR_GAS,
            linestyle="--",
            linewidth=1.1,
            label="Gas boiler heat line",
        )
    else:
        warn_missing("gas_heat_output_MWh_th", "sequential market position plot")

    for ax in axes:
        _legend_if_present(ax)
    _format_datetime_axis(axes[-1])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(
        fig,
        plot_dir,
        "11_sequential_market_position_evolution",
        file_format,
        show,
    )


def plot_idc_sell_source_and_compensation(
    market_ledger: pd.DataFrame,
    dispatch_results: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    """Show which electricity is sold in IDC and how operation is covered afterwards."""

    market = _market_frame(market_ledger, dispatch_results)
    if market.empty:
        return []
    if "intraday_sell_MWh_el" not in market.columns:
        warn_missing("intraday_sell_MWh_el", "IDC sell source and compensation plot")
        return []

    market = market.sort_index()
    idc_sell = _column_or_zero(market, "intraday_sell_MWh_el")
    sell_mask = idc_sell > 1e-9
    if not sell_mask.any():
        warnings.warn(
            "No IDC sell/reduction volumes found. Skipping IDC sell source and compensation plot.",
            stacklevel=2,
        )
        return []

    da_position = _column_or_zero(market, "day_ahead_position_MWh_el")
    da_sold_in_idc = idc_sell.clip(upper=da_position)
    remaining_da = (da_position - da_sold_in_idc).clip(lower=0.0)
    idc_buy = _column_or_zero(market, "intraday_buy_MWh_el")
    afrr_activated = _column_or_zero(market, "afrr_energy_activated_MWh_el")
    dispatch = _aggregate_by_datetime(dispatch_results).reindex(market.index)
    gas_heat = _gas_heat_output_series(market, dispatch_results)
    if gas_heat is None:
        gas_heat = pd.Series(0.0, index=market.index)
        warn_missing("gas_heat_output_MWh_th", "IDC sell source and compensation plot")
    else:
        gas_heat = gas_heat.reindex(market.index).fillna(0.0)
    storage_discharge = _dispatch_column_or_zero(dispatch, "etes_discharge_MWh")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.25]},
    )
    fig.suptitle("IDC sell source and final operational compensation", fontsize=15)

    _fill_between_series(
        axes[0],
        market.index,
        _only_when(da_position, sell_mask),
        color=COLOR_DA,
        alpha=0.16,
        label="Available DA position during IDC sell",
    )
    _fill_between_series(
        axes[0],
        market.index,
        _only_when(da_sold_in_idc, sell_mask),
        color=COLOR_GAS,
        alpha=0.38,
        label="DA electricity sold/reduced in IDC",
    )
    axes[0].step(
        market.index,
        _only_when(da_sold_in_idc, sell_mask),
        where="post",
        color=COLOR_GAS,
        linewidth=1.2,
        label="IDC sell volume",
    )
    axes[0].set_title("Source of IDC sell/reduction")
    axes[0].set_ylabel("MWh_el")

    _fill_between_series(
        axes[1],
        market.index,
        _only_when(remaining_da, sell_mask),
        color=COLOR_DA,
        alpha=0.18,
        label="Remaining DA electricity",
    )
    _fill_between_series(
        axes[1],
        market.index,
        _only_when(remaining_da + idc_buy, sell_mask),
        lower=_only_when(remaining_da, sell_mask),
        color=COLOR_IDC,
        alpha=0.28,
        label="IDC buy electricity",
    )
    _fill_between_series(
        axes[1],
        market.index,
        _only_when(remaining_da + idc_buy + afrr_activated, sell_mask),
        lower=_only_when(remaining_da + idc_buy, sell_mask),
        color=COLOR_AFRR,
        alpha=0.3,
        label="aFRR activated electricity",
    )
    axes[1].set_title("Electricity still available from markets during IDC sell timesteps")
    axes[1].set_ylabel("MWh_el")

    _fill_between_series(
        axes[2],
        market.index,
        _only_when(gas_heat, sell_mask),
        color=COLOR_GAS,
        alpha=0.25,
        label="Gas boiler heat",
    )
    _fill_between_series(
        axes[2],
        market.index,
        _only_when(gas_heat + storage_discharge, sell_mask),
        lower=_only_when(gas_heat, sell_mask),
        color=COLOR_STORAGE,
        alpha=0.22,
        label="Storage discharge heat",
    )
    axes[2].set_title("Final compensation during IDC sell timesteps")
    axes[2].set_ylabel("MWh_th")
    axes[2].set_xlabel("Time")

    for ax in axes:
        ax.axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)
        _legend_if_present(ax)
    _format_datetime_axis(axes[-1])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(
        fig,
        plot_dir,
        "12_idc_sell_source_and_compensation",
        file_format,
        show,
    )


def plot_stagewise_gas_replacement(
    dispatch_results: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    """Show how each sequential market stage changes gas boiler heat dispatch."""

    dispatch = _aggregate_by_datetime(dispatch_results)
    required = require_columns(
        dispatch,
        ["heat_demand_MWh", "gas_heat_MWh"],
        "stagewise gas replacement plot",
    )
    if len(required) < 2:
        return []

    dispatch = dispatch.sort_index()
    heat_demand = dispatch["heat_demand_MWh"].fillna(0.0).astype(float)
    gas_after_da = _column_or_default(
        dispatch,
        "gas_heat_after_day_ahead_MWh",
        dispatch["gas_heat_MWh"],
    )
    gas_after_idc = _column_or_default(
        dispatch,
        "gas_heat_after_intraday_MWh",
        gas_after_da,
    )
    gas_final = dispatch["gas_heat_MWh"].fillna(0.0).astype(float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
    )
    fig.suptitle("Stagewise gas replacement by electricity markets", fontsize=15)

    _draw_gas_replacement_panel(
        axes[0],
        dispatch.index,
        baseline_gas=heat_demand,
        stage_gas=gas_after_da,
        baseline_label="Gas-only reference heat demand",
        stage_label="Gas after day-ahead",
        replacement_label="Gas replaced by DA electricity/storage",
        replacement_color=COLOR_DA,
        title="Day-ahead stage",
    )
    _draw_gas_replacement_panel(
        axes[1],
        dispatch.index,
        baseline_gas=gas_after_da,
        stage_gas=gas_after_idc,
        baseline_label="Gas after day-ahead",
        stage_label="Gas after intraday",
        replacement_label="Additional gas replaced by IDC",
        replacement_color=COLOR_IDC,
        title="Intraday adjustment stage",
        increase_label="Gas restored by IDC sell/reduction",
    )
    _draw_gas_replacement_panel(
        axes[2],
        dispatch.index,
        baseline_gas=gas_after_idc,
        stage_gas=gas_final,
        baseline_label="Gas after DA+IDC",
        stage_label="Final gas after aFRR down",
        replacement_label="Additional gas replaced by aFRR down",
        replacement_color=COLOR_AFRR,
        title="aFRR down energy stage",
    )
    axes[2].set_xlabel("Time")

    for ax in axes:
        ax.set_ylabel("Heat per time step [MWh_th]")
        ax.axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)
        _legend_if_present(ax)
    _format_datetime_axis(axes[-1])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(
        fig,
        plot_dir,
        "13_stagewise_gas_replacement",
        file_format,
        show,
    )


def plot_afrr_capacity_price_and_reserve(
    market_ledger: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    market = _market_frame(market_ledger, pd.DataFrame())
    if market.empty or "afrr_capacity_reserved_MW" not in market.columns:
        return []
    if market["afrr_capacity_reserved_MW"].fillna(0.0).abs().sum() <= 1e-12:
        return []

    fig, ax = plt.subplots()
    reserve = market["afrr_capacity_reserved_MW"].fillna(0.0)
    ax.step(
        market.index,
        reserve,
        where="post",
        label="Reserved aFRR down capacity",
        color=COLOR_CAPACITY,
    )
    ax.set_ylabel("Reserved capacity [MW]")
    ax.set_title("aFRR down capacity price and reservation")
    ax2 = ax.twinx()
    if "afrr_capacity_down_price_EUR_per_MW_h" in market.columns:
        ax2.plot(
            market.index,
            market["afrr_capacity_down_price_EUR_per_MW_h"],
            color=COLOR_IDC,
            label="Capacity price",
        )
    ax2.set_ylabel("Capacity price [EUR/MW/h]")
    _combined_legend(ax, ax2)
    _format_datetime_axis(ax)
    return _save_figure(fig, plot_dir, "14_afrr_capacity_price_and_reserve", file_format, show)


def plot_procurement_and_capacity_headroom(
    market_ledger: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    market = _market_frame(market_ledger, pd.DataFrame())
    if market.empty or "afrr_capacity_reserved_MWh" not in market.columns:
        return []
    if market["afrr_capacity_reserved_MWh"].fillna(0.0).abs().sum() <= 1e-12:
        return []

    fig, ax = plt.subplots()
    scheduled = _column_or_zero(market, "scheduled_electricity_procurement_MWh_el")
    reserve = _column_or_zero(market, "afrr_capacity_reserved_MWh")
    actual = _column_or_zero(market, "actual_electricity_consumption_MWh_el")
    _fill_between_series(
        ax,
        market.index,
        scheduled,
        color=COLOR_DA,
        alpha=0.2,
        label="Scheduled DA+IDC",
    )
    _fill_between_series(
        ax,
        market.index,
        scheduled + reserve,
        lower=scheduled,
        color=COLOR_CAPACITY,
        alpha=0.28,
        label="Reserved aFRR capacity headroom",
    )
    ax.step(market.index, actual, where="post", color=COLOR_NEUTRAL, label="Actual electricity")
    ax.set_title("Electricity procurement and reserved aFRR headroom")
    ax.set_ylabel("Electricity per time step [MWh_el]")
    ax.set_xlabel("Time")
    _legend_if_present(ax)
    _format_datetime_axis(ax)
    return _save_figure(
        fig,
        plot_dir,
        "15_electricity_procurement_and_capacity_headroom",
        file_format,
        show,
    )


def plot_afrr_capacity_and_energy(
    market_ledger: pd.DataFrame,
    plot_dir: Path,
    file_format: str = "png",
    show: bool = False,
) -> list[Path]:
    market = _market_frame(market_ledger, pd.DataFrame())
    if market.empty or "afrr_capacity_reserved_MW" not in market.columns:
        return []
    if market["afrr_capacity_reserved_MW"].fillna(0.0).abs().sum() <= 1e-12:
        return []

    fig, ax = plt.subplots()
    reserve = _column_or_zero(market, "afrr_capacity_reserved_MW")
    activation_mw = _column_or_zero(market, "afrr_energy_activated_MWh_el") / _plot_timestep_hours(
        market.index
    )
    system_mw = _column_or_zero(market, "afrr_system_activation_MWh_el") / _plot_timestep_hours(
        market.index
    )
    ax.step(market.index, reserve, where="post", label="Reserved capacity", color=COLOR_CAPACITY)
    ax.step(market.index, activation_mw, where="post", label="Plant activation", color=COLOR_AFRR)
    ax.step(
        market.index,
        system_mw,
        where="post",
        label="System activation proxy",
        color=COLOR_LIGHT_NEUTRAL,
        alpha=0.8,
    )
    ax.set_title("aFRR down capacity and energy activation")
    ax.set_ylabel("Power [MW]")
    ax.set_xlabel("Time")
    ax2 = ax.twinx()
    if "afrr_energy_price_EUR_per_MWh_el" in market.columns:
        ax2.plot(
            market.index,
            market["afrr_energy_price_EUR_per_MWh_el"],
            color=COLOR_IDC,
            alpha=0.8,
            label="aFRR energy price",
        )
    ax2.set_ylabel("Energy price [EUR/MWh]")
    _combined_legend(ax, ax2)
    _format_datetime_axis(ax)
    return _save_figure(fig, plot_dir, "16_afrr_capacity_and_energy", file_format, show)


def _sample_day_prices_panel(
    ax: plt.Axes,
    dispatch: pd.DataFrame,
    market: pd.DataFrame,
    full_dispatch: pd.DataFrame,
) -> None:
    price_frame = _price_frame(dispatch, market)
    day_ahead_column = _first_existing(
        price_frame,
        ["day_ahead_delivered_price_EUR_per_MWh_el", "day_ahead_price_EUR_per_MWh_el"],
    )
    if day_ahead_column:
        ax.plot(
            price_frame.index,
            price_frame[day_ahead_column],
            label=_price_label("DA", day_ahead_column),
            color=COLOR_DA,
        )
    for column, label, color in [
        (
            _first_existing(
                price_frame,
                ["intraday_delivered_price_EUR_per_MWh_el", "intraday_price_EUR_per_MWh_el"],
            ),
            "IDC",
            COLOR_IDC,
        ),
        (
            _first_existing(
                price_frame,
                [
                    "afrr_energy_delivered_price_EUR_per_MWh_el",
                    "afrr_energy_price_EUR_per_MWh_el",
                ],
            ),
            "aFRR energy",
            COLOR_AFRR,
        ),
    ]:
        if column in price_frame.columns and price_frame[column].notna().any():
            ax.plot(
                price_frame.index,
                price_frame[column],
                label=_price_label(label, column),
                color=color,
            )
    benchmark = derive_gas_benchmark(full_dispatch)
    if benchmark is not None:
        benchmark = benchmark.loc[dispatch.index.min() : dispatch.index.max()]
        ax.plot(benchmark.index, benchmark, label="Gas benchmark", color=COLOR_BENCHMARK)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Market prices and benchmark")


def _sample_day_procurement_panel(ax: plt.Axes, market: pd.DataFrame) -> None:
    if market.empty:
        warn_missing("market ledger rows", "sample day electricity procurement panel")
        return
    width = _bar_width(market.index)
    bottom = pd.Series(0.0, index=market.index)
    for column, label in [
        ("day_ahead_position_MWh_el", "DA"),
        ("intraday_buy_MWh_el", "IDC buy"),
        ("afrr_energy_activated_MWh_el", "aFRR activated"),
    ]:
        if column in market.columns:
            values = market[column].fillna(0.0).astype(float)
            color = {"DA": COLOR_DA, "IDC buy": COLOR_IDC, "aFRR activated": COLOR_AFRR}.get(
                label, COLOR_LIGHT_NEUTRAL
            )
            ax.bar(
                market.index,
                values,
                width=width,
                bottom=bottom,
                label=label,
                color=color,
                alpha=0.7,
            )
            bottom += values
    if "intraday_sell_MWh_el" in market.columns:
        ax.bar(
            market.index,
            -market["intraday_sell_MWh_el"].fillna(0.0).astype(float),
            width=width,
            label="IDC sell",
            color=COLOR_GAS,
            alpha=0.55,
        )
    if "actual_electricity_consumption_MWh_el" in market.columns:
        ax.plot(
            market.index,
            market["actual_electricity_consumption_MWh_el"],
            color=COLOR_NEUTRAL,
            label="Actual consumption",
        )
    ax.set_ylabel("MWh_el")
    ax.set_title("Electricity procurement by market")


def _sample_day_heat_panel(ax: plt.Axes, dispatch: pd.DataFrame) -> None:
    if "heat_demand_MWh" in dispatch.columns:
        ax.plot(
            dispatch.index,
            dispatch["heat_demand_MWh"],
            color=COLOR_NEUTRAL,
            label="Heat demand",
        )
    stack_columns = [
        column for column in ["gas_heat_MWh", "etes_discharge_MWh"] if column in dispatch
    ]
    labels = {
        "gas_heat_MWh": "Gas heat",
        "etes_discharge_MWh": "Storage discharge",
    }
    if stack_columns:
        ax.stackplot(
            dispatch.index,
            *[dispatch[column].fillna(0.0) for column in stack_columns],
            labels=[labels[column] for column in stack_columns],
            colors=[
                COLOR_GAS if column == "gas_heat_MWh" else COLOR_STORAGE for column in stack_columns
            ],
            alpha=0.78,
        )
    ax.set_ylabel("MWh_th")
    ax.set_title("Plant heat operation")


def _sample_day_storage_panel(ax: plt.Axes, dispatch: pd.DataFrame) -> None:
    width = _bar_width(dispatch.index)
    if "etes_charge_MWh" in dispatch.columns:
        ax.bar(
            dispatch.index,
            dispatch["etes_charge_MWh"],
            width=width,
            label="Charge",
            color=COLOR_DA,
            alpha=0.7,
        )
    if "etes_discharge_MWh" in dispatch.columns:
        ax.bar(
            dispatch.index,
            -dispatch["etes_discharge_MWh"],
            width=width,
            label="Discharge",
            color=COLOR_STORAGE,
            alpha=0.7,
        )
    ax.set_ylabel("MWh")
    ax.set_title("Storage operation")
    if "etes_soc_MWh" in dispatch.columns:
        ax2 = ax.twinx()
        ax2.plot(dispatch.index, dispatch["etes_soc_MWh"], color=COLOR_STORAGE, label="SoC")
        ax2.set_ylabel("SoC [MWh_th]")
        _combined_legend(ax, ax2)


def _draw_storage_operation_panel(ax: plt.Axes, dispatch: pd.DataFrame) -> None:
    has_storage_columns = any(
        column in dispatch.columns
        for column in ["etes_charge_MWh", "etes_discharge_MWh", "etes_soc_MWh"]
    )
    if not has_storage_columns:
        warnings.warn(
            "No storage operation columns found. Skipping lower storage panel.",
            stacklevel=2,
        )
        ax.set_axis_off()
        return

    if "etes_discharge_MWh" in dispatch.columns:
        ax.plot(
            dispatch.index,
            dispatch["etes_discharge_MWh"].fillna(0.0),
            color=COLOR_STORAGE,
            linewidth=1.5,
            label="Storage discharge",
        )
    else:
        warn_missing("etes_discharge_MWh", "storage operation panel")

    if "etes_charge_MWh" in dispatch.columns:
        ax.plot(
            dispatch.index,
            -dispatch["etes_charge_MWh"].fillna(0.0),
            color=COLOR_DA,
            linewidth=1.5,
            label="Storage charge",
        )
    else:
        warn_missing("etes_charge_MWh", "storage operation panel")

    ax.axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)
    ax.set_ylabel("Storage flow [MWh]\ncharge below zero")
    ax.set_title("Storage operation")

    if "etes_soc_MWh" in dispatch.columns:
        ax2 = ax.twinx()
        soc = dispatch["etes_soc_MWh"].fillna(0.0)
        ax2.fill_between(
            dispatch.index,
            0,
            soc,
            color=COLOR_STORAGE,
            alpha=0.18,
            label="Storage SoC",
        )
        ax2.plot(dispatch.index, soc, color=COLOR_STORAGE, linewidth=1.0, alpha=0.75)
        ax2.set_ylabel("SoC [MWh_th]")
        _combined_legend(ax, ax2)
    else:
        warn_missing("etes_soc_MWh", "storage operation panel")
        ax.legend(loc="best")


def _draw_gas_replacement_panel(
    ax: plt.Axes,
    index: pd.Index,
    baseline_gas: pd.Series,
    stage_gas: pd.Series,
    baseline_label: str,
    stage_label: str,
    replacement_label: str,
    replacement_color: str,
    title: str,
    increase_label: str | None = None,
) -> None:
    baseline = baseline_gas.reindex(index).fillna(0.0).astype(float)
    stage = stage_gas.reindex(index).fillna(0.0).astype(float)
    replacement_mask = stage < baseline
    replacement_upper = baseline
    replacement_lower = stage.where(replacement_mask, baseline)

    _fill_between_series(
        ax,
        index,
        replacement_upper,
        lower=replacement_lower,
        color=replacement_color,
        alpha=0.32,
        label=replacement_label,
    )
    if increase_label is not None:
        increase_mask = stage > baseline
        increase_upper = stage.where(increase_mask, baseline)
        increase_lower = baseline
        _fill_between_series(
            ax,
            index,
            increase_upper,
            lower=increase_lower,
            color=COLOR_GAS,
            alpha=0.2,
            label=increase_label,
        )

    ax.step(
        index,
        baseline,
        where="post",
        color=COLOR_LIGHT_NEUTRAL,
        linestyle="--",
        linewidth=1.1,
        label=baseline_label,
    )
    ax.step(
        index,
        stage,
        where="post",
        color=COLOR_GAS,
        linewidth=1.3,
        label=stage_label,
    )
    ax.set_title(title)


def _aggregate_by_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_datetime_index(frame)
    if frame.empty:
        return frame
    numeric = frame.select_dtypes(include="number").copy()
    grouped = numeric.groupby(numeric.index).sum()
    for column in [
        "day_ahead_price_EUR_per_MWh",
        "day_ahead_delivered_price_EUR_per_MWh",
        "IDC_price_EUR_per_MWh",
        "IDC_delivered_price_EUR_per_MWh",
        "afrr_energy_price_EUR_per_MWh",
        "afrr_energy_delivered_price_EUR_per_MWh",
        "gas_based_heat_benchmark_EUR_per_MWh_th",
        "gas_price_EUR_per_MWh",
        "co2_price_EUR_per_t",
    ]:
        if column in frame.columns:
            grouped[column] = frame.groupby(frame.index)[column].mean()
    return grouped


def _market_frame(market_ledger: pd.DataFrame, dispatch_results: pd.DataFrame) -> pd.DataFrame:
    market = ensure_datetime_index(market_ledger)
    if not market.empty:
        numeric = market.select_dtypes(include="number").copy()
        grouped = numeric.groupby(numeric.index).sum()
        price_columns = [
            "day_ahead_price_EUR_per_MWh_el",
            "day_ahead_delivered_price_EUR_per_MWh_el",
            "intraday_price_EUR_per_MWh_el",
            "intraday_delivered_price_EUR_per_MWh_el",
            "afrr_energy_price_EUR_per_MWh_el",
            "afrr_energy_delivered_price_EUR_per_MWh_el",
            "afrr_capacity_down_price_EUR_per_MW_h",
        ]
        for column in price_columns:
            if column in market.columns:
                grouped[column] = market.groupby(market.index)[column].mean()
        return grouped

    dispatch = _aggregate_by_datetime(dispatch_results)
    if dispatch.empty:
        return pd.DataFrame()
    fallback = pd.DataFrame(index=dispatch.index)
    if "electricity_consumption_MWh" in dispatch.columns:
        fallback["day_ahead_position_MWh_el"] = dispatch["electricity_consumption_MWh"]
        fallback["actual_electricity_consumption_MWh_el"] = dispatch["electricity_consumption_MWh"]
    if "day_ahead_price_EUR_per_MWh" in dispatch.columns:
        fallback["day_ahead_price_EUR_per_MWh_el"] = dispatch["day_ahead_price_EUR_per_MWh"]
    if "day_ahead_delivered_price_EUR_per_MWh" in dispatch.columns:
        fallback["day_ahead_delivered_price_EUR_per_MWh_el"] = dispatch[
            "day_ahead_delivered_price_EUR_per_MWh"
        ]
    return fallback


def _price_frame(dispatch_results: pd.DataFrame, market_ledger: pd.DataFrame) -> pd.DataFrame:
    dispatch = _aggregate_by_datetime(dispatch_results)
    market = _market_frame(market_ledger, dispatch_results)
    price = pd.DataFrame(index=dispatch.index if not dispatch.empty else market.index)
    _copy_price_column(
        price,
        dispatch,
        market,
        target="day_ahead_price_EUR_per_MWh_el",
        dispatch_column="day_ahead_price_EUR_per_MWh",
        market_column="day_ahead_price_EUR_per_MWh_el",
    )
    _copy_price_column(
        price,
        dispatch,
        market,
        target="day_ahead_delivered_price_EUR_per_MWh_el",
        dispatch_column="day_ahead_delivered_price_EUR_per_MWh",
        market_column="day_ahead_delivered_price_EUR_per_MWh_el",
    )
    _copy_price_column(
        price,
        dispatch,
        market,
        target="intraday_price_EUR_per_MWh_el",
        dispatch_column="IDC_price_EUR_per_MWh",
        market_column="intraday_price_EUR_per_MWh_el",
    )
    _copy_price_column(
        price,
        dispatch,
        market,
        target="intraday_delivered_price_EUR_per_MWh_el",
        dispatch_column="IDC_delivered_price_EUR_per_MWh",
        market_column="intraday_delivered_price_EUR_per_MWh_el",
    )
    _copy_price_column(
        price,
        dispatch,
        market,
        target="afrr_energy_price_EUR_per_MWh_el",
        dispatch_column="afrr_energy_price_EUR_per_MWh",
        market_column="afrr_energy_price_EUR_per_MWh_el",
    )
    _copy_price_column(
        price,
        dispatch,
        market,
        target="afrr_energy_delivered_price_EUR_per_MWh_el",
        dispatch_column="afrr_energy_delivered_price_EUR_per_MWh",
        market_column="afrr_energy_delivered_price_EUR_per_MWh_el",
    )
    return price


def _copy_price_column(
    target_frame: pd.DataFrame,
    dispatch: pd.DataFrame,
    market: pd.DataFrame,
    target: str,
    dispatch_column: str,
    market_column: str,
) -> None:
    if dispatch_column in dispatch.columns:
        target_frame[target] = dispatch[dispatch_column]
    elif market_column in market.columns:
        target_frame[target] = market[market_column]


def _price_label(market_name: str, column: str) -> str:
    suffix = "delivered price" if "delivered" in column else "market price"
    return f"{market_name} {suffix}"


def _aggregate_summary(summary: pd.DataFrame) -> dict[str, float]:
    numeric = summary.select_dtypes(include="number")
    return {column: float(numeric[column].sum()) for column in numeric.columns}


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(frame[column].fillna(0.0).astype(float).sum())


def _fill_between_series(
    ax: plt.Axes,
    index: pd.Index,
    upper: pd.Series,
    color: str,
    alpha: float,
    label: str,
    lower: pd.Series | float = 0.0,
) -> None:
    if isinstance(lower, pd.Series):
        lower_values = lower.fillna(0.0).astype(float).to_numpy()
    else:
        lower_values = lower
    ax.fill_between(
        index,
        lower_values,
        upper.fillna(0.0).astype(float).to_numpy(),
        color=color,
        alpha=alpha,
        step="post",
        label=label,
    )


def _legend_if_present(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", ncol=2)


def _column_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return frame[column].fillna(0.0).astype(float)


def _column_or_default(
    frame: pd.DataFrame,
    column: str,
    default: pd.Series,
) -> pd.Series:
    if column not in frame.columns:
        return default.fillna(0.0).astype(float)
    return frame[column].fillna(default).fillna(0.0).astype(float)


def _dispatch_column_or_zero(dispatch: pd.DataFrame, column: str) -> pd.Series:
    if column not in dispatch.columns:
        return pd.Series(0.0, index=dispatch.index)
    return dispatch[column].fillna(0.0).astype(float)


def _only_when(series: pd.Series, mask: pd.Series) -> pd.Series:
    return series.reindex(mask.index).fillna(0.0).astype(float).where(mask, 0.0)


def _gas_heat_output_series(
    market_ledger: pd.DataFrame,
    dispatch_results: pd.DataFrame,
) -> pd.Series | None:
    if "gas_heat_output_MWh_th" in market_ledger.columns:
        return market_ledger["gas_heat_output_MWh_th"].fillna(0.0).astype(float)

    dispatch = _aggregate_by_datetime(dispatch_results)
    if "gas_heat_MWh" in dispatch.columns:
        return dispatch["gas_heat_MWh"].fillna(0.0).astype(float)
    return None


def _first_existing(frame: pd.DataFrame, columns: list[str]) -> str | None:
    return next((column for column in columns if column in frame.columns), None)


def _format_datetime_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.figure.autofmt_xdate()


def _bar_width(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 0.02
    diffs = index.to_series().diff().dropna().dt.total_seconds().div(86400)
    if diffs.empty:
        return 0.02
    return float(diffs.median() * 0.8)


def _plot_timestep_hours(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 1.0
    diffs = index.to_series().diff().dropna().dt.total_seconds().div(3600)
    if diffs.empty:
        return 1.0
    return float(diffs.mode().iloc[0])


def _combined_legend(ax1: plt.Axes, ax2: plt.Axes) -> None:
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", ncol=2)


def _save_figure(
    fig: plt.Figure,
    plot_dir: Path,
    stem: str,
    file_format: str,
    show: bool,
) -> list[Path]:
    formats = ["png", "pdf"] if file_format == "both" else [file_format]
    created = []
    for fmt in formats:
        path = plot_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=180 if fmt == "png" else None, bbox_inches="tight")
        created.append(path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return created
