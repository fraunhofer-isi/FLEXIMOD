# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

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
}


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

    with plt.rc_context(REPORT_STYLE):
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
    ax.plot(dispatch.index, dispatch["heat_demand_MWh"], color="black", label="Heat demand")
    stack_columns = ["gas_heat_MWh", "etes_discharge_MWh"]
    labels = ["Gas boiler heat", "Storage discharge heat"]
    if "unmet_heat_MWh" in dispatch.columns and dispatch["unmet_heat_MWh"].sum() > 0:
        stack_columns.append("unmet_heat_MWh")
        labels.append("Unmet heat")
    elif "unmet_heat_MWh" not in dispatch.columns:
        warn_missing("unmet_heat_MWh", "operation dynamics unmet heat series")

    ax.stackplot(
        dispatch.index,
        *[dispatch[column].fillna(0.0) for column in stack_columns],
        labels=labels,
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
        color="tab:blue",
        alpha=0.65,
    )
    ax.bar(
        dispatch.index,
        -dispatch["etes_discharge_MWh"],
        width=width,
        label="ETES discharge",
        color="tab:orange",
        alpha=0.65,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Storage operation and state of charge")
    ax.set_ylabel("Charge/discharge per time step [MWh]")
    ax.set_xlabel("Time")

    ax2 = ax.twinx()
    ax2.plot(dispatch.index, dispatch["etes_soc_MWh"], color="tab:green", label="ETES SoC")
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
    if "DA_price" in dispatch.columns:
        ax.plot(dispatch.index, dispatch["DA_price"], label="Day-ahead price", color="tab:blue")
        plotted = True
    else:
        warn_missing("DA_price", "market prices plot")

    for column, label, color in [
        ("IDC_price", "IDC price", "tab:orange"),
        ("afrr_energy_price", "aFRR energy price", "tab:purple"),
    ]:
        if column in dispatch.columns and dispatch[column].notna().any():
            ax.plot(dispatch.index, dispatch[column], label=label, color=color, alpha=0.9)
        else:
            warn_missing(column, "market prices plot")

    benchmark = derive_gas_benchmark(dispatch_results)
    if benchmark is not None:
        benchmark = benchmark.groupby(benchmark.index).mean()
        ax.plot(
            benchmark.index,
            benchmark,
            label="Gas-based heat benchmark",
            color="tab:red",
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
    width = _bar_width(market.index)
    positive_columns = [
        ("DA_position_MWh", "Day-ahead"),
        ("IDC_buy_MWh", "IDC buy"),
        ("afrr_energy_activated_MWh", "aFRR activated"),
    ]
    bottom = pd.Series(0.0, index=market.index)
    plotted = False
    for column, label in positive_columns:
        if column in market.columns:
            values = market[column].fillna(0.0).astype(float)
            ax.bar(market.index, values, width=width, bottom=bottom, label=label, alpha=0.72)
            bottom += values
            plotted = True
        else:
            warn_missing(column, "electricity procurement plot")

    if "IDC_sell_MWh" in market.columns:
        ax.bar(
            market.index,
            -market["IDC_sell_MWh"].fillna(0.0).astype(float),
            width=width,
            label="IDC sell/reduction",
            color="tab:red",
            alpha=0.62,
        )
    else:
        warn_missing("IDC_sell_MWh", "electricity procurement plot")

    if "actual_electricity_consumption_MWh" in market.columns:
        ax.plot(
            market.index,
            market["actual_electricity_consumption_MWh"],
            color="black",
            linewidth=1.4,
            label="Actual consumption",
        )
        plotted = True
    else:
        warn_missing("actual_electricity_consumption_MWh", "electricity procurement plot")

    if not plotted:
        plt.close(fig)
        return []

    ax.axhline(0, color="black", linewidth=0.8)
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
            alpha=0.82,
        )
    if total_soc is not None:
        ax.plot(total_soc.index, total_soc, color="black", linewidth=1.2, label="Total ETES SoC")

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
        ("Electricity procurement cost", "total_electricity_procurement_cost_EUR"),
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
    colors = ["tab:blue" if value >= 0 else "tab:green" for value in values]
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
        ("unmet_heat_MWh", "Unmet heat"),
    ]
    values = [
        (label, _sum_column(dispatch, column)) for column, label in columns if column in dispatch
    ]
    if not values:
        return []

    fig, ax = plt.subplots(figsize=(10, 3.5))
    left = 0.0
    for label, value in values:
        ax.barh(["Heat supply"], [value], left=left, label=label, alpha=0.8)
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
        ("DA_position_MWh", "Day-ahead"),
        ("IDC_buy_MWh", "IDC buy"),
        ("IDC_sell_MWh", "IDC sell/reduction"),
        ("afrr_energy_activated_MWh", "aFRR activated"),
    ]
    values = []
    for column, label in components:
        if column in market.columns:
            sign = -1.0 if column == "IDC_sell_MWh" else 1.0
            values.append((label, sign * _sum_column(market, column)))
        else:
            warn_missing(column, "electricity market share plot")

    if not values:
        return []

    actual = _sum_column(market, "actual_electricity_consumption_MWh")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    left = 0.0
    for label, value in values:
        ax.barh(["Market procurement"], [value], left=left, label=label, alpha=0.8)
        left += value
    if actual:
        ax.axvline(actual, color="black", linewidth=1.5, label="Actual consumption")
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
    price_col = _first_existing(dispatch, ["day_ahead_price_EUR_per_MWh", "DA_price"])
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
        color="tab:blue",
        label="ETES charging",
    )
    benchmark = derive_gas_benchmark(dispatch_results)
    if benchmark is not None:
        ax.axvline(
            float(benchmark.mean()),
            color="tab:red",
            linestyle="--",
            label="Average gas benchmark",
        )
    else:
        warn_missing("gas_based_heat_benchmark", "price response plot")

    ax.set_title("Price response of storage charging")
    ax.set_xlabel("Electricity price [EUR/MWh]")
    ax.set_ylabel("ETES charging per time step [MWh_el]")
    ax.legend(loc="best")
    return _save_figure(fig, plot_dir, "10_price_response_storage_charging", file_format, show)


def _sample_day_prices_panel(
    ax: plt.Axes,
    dispatch: pd.DataFrame,
    market: pd.DataFrame,
    full_dispatch: pd.DataFrame,
) -> None:
    price_frame = _price_frame(dispatch, market)
    if "DA_price" in price_frame.columns:
        ax.plot(price_frame.index, price_frame["DA_price"], label="DA price", color="tab:blue")
    for column, label, color in [
        ("IDC_price", "IDC price", "tab:orange"),
        ("afrr_energy_price", "aFRR energy price", "tab:purple"),
    ]:
        if column in price_frame.columns and price_frame[column].notna().any():
            ax.plot(price_frame.index, price_frame[column], label=label, color=color)
    benchmark = derive_gas_benchmark(full_dispatch)
    if benchmark is not None:
        benchmark = benchmark.loc[dispatch.index.min() : dispatch.index.max()]
        ax.plot(benchmark.index, benchmark, label="Gas benchmark", color="tab:red")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Market prices and benchmark")


def _sample_day_procurement_panel(ax: plt.Axes, market: pd.DataFrame) -> None:
    if market.empty:
        warn_missing("market ledger rows", "sample day electricity procurement panel")
        return
    width = _bar_width(market.index)
    bottom = pd.Series(0.0, index=market.index)
    for column, label in [
        ("DA_position_MWh", "DA"),
        ("IDC_buy_MWh", "IDC buy"),
        ("afrr_energy_activated_MWh", "aFRR activated"),
    ]:
        if column in market.columns:
            values = market[column].fillna(0.0).astype(float)
            ax.bar(market.index, values, width=width, bottom=bottom, label=label, alpha=0.7)
            bottom += values
    if "IDC_sell_MWh" in market.columns:
        ax.bar(
            market.index,
            -market["IDC_sell_MWh"].fillna(0.0).astype(float),
            width=width,
            label="IDC sell",
            alpha=0.55,
        )
    if "actual_electricity_consumption_MWh" in market.columns:
        ax.plot(
            market.index,
            market["actual_electricity_consumption_MWh"],
            color="black",
            label="Actual consumption",
        )
    ax.set_ylabel("MWh_el")
    ax.set_title("Electricity procurement by market")


def _sample_day_heat_panel(ax: plt.Axes, dispatch: pd.DataFrame) -> None:
    if "heat_demand_MWh" in dispatch.columns:
        ax.plot(dispatch.index, dispatch["heat_demand_MWh"], color="black", label="Heat demand")
    stack_columns = [
        column
        for column in ["gas_heat_MWh", "etes_discharge_MWh", "unmet_heat_MWh"]
        if column in dispatch
    ]
    labels = {
        "gas_heat_MWh": "Gas heat",
        "etes_discharge_MWh": "Storage discharge",
        "unmet_heat_MWh": "Unmet heat",
    }
    if stack_columns:
        ax.stackplot(
            dispatch.index,
            *[dispatch[column].fillna(0.0) for column in stack_columns],
            labels=[labels[column] for column in stack_columns],
            alpha=0.78,
        )
    ax.set_ylabel("MWh_th")
    ax.set_title("Plant heat operation")


def _sample_day_storage_panel(ax: plt.Axes, dispatch: pd.DataFrame) -> None:
    width = _bar_width(dispatch.index)
    if "etes_charge_MWh" in dispatch.columns:
        ax.bar(dispatch.index, dispatch["etes_charge_MWh"], width=width, label="Charge", alpha=0.7)
    if "etes_discharge_MWh" in dispatch.columns:
        ax.bar(
            dispatch.index,
            -dispatch["etes_discharge_MWh"],
            width=width,
            label="Discharge",
            alpha=0.7,
        )
    ax.set_ylabel("MWh")
    ax.set_title("Storage operation")
    if "etes_soc_MWh" in dispatch.columns:
        ax2 = ax.twinx()
        ax2.plot(dispatch.index, dispatch["etes_soc_MWh"], color="tab:green", label="SoC")
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
            color="tab:orange",
            linewidth=1.5,
            label="Storage discharge",
        )
    else:
        warn_missing("etes_discharge_MWh", "storage operation panel")

    if "etes_charge_MWh" in dispatch.columns:
        ax.plot(
            dispatch.index,
            -dispatch["etes_charge_MWh"].fillna(0.0),
            color="tab:blue",
            linewidth=1.5,
            label="Storage charge",
        )
    else:
        warn_missing("etes_charge_MWh", "storage operation panel")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Storage flow [MWh]\ncharge below zero")
    ax.set_title("Storage operation")

    if "etes_soc_MWh" in dispatch.columns:
        ax2 = ax.twinx()
        soc = dispatch["etes_soc_MWh"].fillna(0.0)
        ax2.fill_between(
            dispatch.index,
            0,
            soc,
            color="tab:green",
            alpha=0.18,
            label="Storage SoC",
        )
        ax2.plot(dispatch.index, soc, color="tab:green", linewidth=1.0, alpha=0.75)
        ax2.set_ylabel("SoC [MWh_th]")
        _combined_legend(ax, ax2)
    else:
        warn_missing("etes_soc_MWh", "storage operation panel")
        ax.legend(loc="best")


def _aggregate_by_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_datetime_index(frame)
    if frame.empty:
        return frame
    numeric = frame.select_dtypes(include="number").copy()
    grouped = numeric.groupby(numeric.index).sum()
    for column in [
        "day_ahead_price_EUR_per_MWh",
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
        for column in ["DA_price", "IDC_price", "afrr_energy_price", "afrr_capacity_price"]:
            if column in market.columns:
                grouped[column] = market.groupby(market.index)[column].mean()
        return grouped

    dispatch = _aggregate_by_datetime(dispatch_results)
    if dispatch.empty:
        return pd.DataFrame()
    fallback = pd.DataFrame(index=dispatch.index)
    if "electricity_consumption_MWh" in dispatch.columns:
        fallback["DA_position_MWh"] = dispatch["electricity_consumption_MWh"]
        fallback["actual_electricity_consumption_MWh"] = dispatch["electricity_consumption_MWh"]
    if "day_ahead_price_EUR_per_MWh" in dispatch.columns:
        fallback["DA_price"] = dispatch["day_ahead_price_EUR_per_MWh"]
    return fallback


def _price_frame(dispatch_results: pd.DataFrame, market_ledger: pd.DataFrame) -> pd.DataFrame:
    dispatch = _aggregate_by_datetime(dispatch_results)
    market = _market_frame(market_ledger, dispatch_results)
    price = pd.DataFrame(index=dispatch.index if not dispatch.empty else market.index)
    if "day_ahead_price_EUR_per_MWh" in dispatch.columns:
        price["DA_price"] = dispatch["day_ahead_price_EUR_per_MWh"]
    elif "DA_price" in market.columns:
        price["DA_price"] = market["DA_price"]
    for column in ["IDC_price", "afrr_energy_price"]:
        if column in market.columns:
            price[column] = market[column]
    return price


def _aggregate_summary(summary: pd.DataFrame) -> dict[str, float]:
    numeric = summary.select_dtypes(include="number")
    return {column: float(numeric[column].sum()) for column in numeric.columns}


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(frame[column].fillna(0.0).astype(float).sum())


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
