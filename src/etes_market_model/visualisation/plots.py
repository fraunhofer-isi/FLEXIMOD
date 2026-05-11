# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_case_plots(
    dispatch_results: pd.DataFrame,
    summary_indicators: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Create the basic MVP plots and return their file paths."""

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for plant_name, plant_dispatch in dispatch_results.groupby("plant_name"):
        created.append(_plot_heat_supply_stack(plant_name, plant_dispatch, plot_dir))
        created.append(_plot_etes_operation(plant_name, plant_dispatch, plot_dir))
        created.append(_plot_prices_and_benchmark(plant_name, plant_dispatch, plot_dir))

    created.append(_plot_cost_summary(summary_indicators, plot_dir))
    return created


def _plot_heat_supply_stack(plant_name: str, dispatch: pd.DataFrame, plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dispatch.index, dispatch["heat_demand_MWh"], color="black", label="Heat demand")
    ax.stackplot(
        dispatch.index,
        dispatch["gas_heat_MWh"],
        dispatch["etes_discharge_MWh"],
        labels=["Gas boiler heat", "ETES discharge"],
        alpha=0.8,
    )
    ax.set_ylabel("Heat per time step [MWh_th]")
    ax.legend(loc="upper right")
    ax.set_title(f"{plant_name}: heat supply stack")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = plot_dir / f"{plant_name}_heat_supply_stack.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_etes_operation(plant_name: str, dispatch: pd.DataFrame, plot_dir: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(dispatch.index, dispatch["etes_charge_MWh"], label="Charge", color="tab:blue")
    ax1.plot(
        dispatch.index,
        dispatch["etes_discharge_MWh"],
        label="Discharge",
        color="tab:orange",
    )
    ax1.set_ylabel("Energy per time step [MWh]")
    ax2 = ax1.twinx()
    ax2.plot(dispatch.index, dispatch["etes_soc_MWh"], label="SoC", color="tab:green")
    ax2.set_ylabel("SoC [MWh_th]")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax1.set_title(f"{plant_name}: ETES operation")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = plot_dir / f"{plant_name}_etes_operation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_prices_and_benchmark(plant_name: str, dispatch: pd.DataFrame, plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        dispatch.index,
        dispatch["day_ahead_price_EUR_per_MWh"],
        label="DA price",
        color="tab:blue",
    )
    ax.plot(
        dispatch.index,
        dispatch["gas_based_heat_benchmark_EUR_per_MWh_th"],
        label="Gas-based heat benchmark",
        color="tab:red",
    )
    ax.set_ylabel("Price [EUR/MWh]")
    ax.legend(loc="upper right")
    ax.set_title(f"{plant_name}: market price and benchmark")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = plot_dir / f"{plant_name}_prices_and_benchmark.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_cost_summary(summary: pd.DataFrame, plot_dir: Path) -> Path:
    cost_columns = [
        "total_electricity_cost_EUR",
        "total_gas_cost_EUR",
        "total_co2_cost_EUR",
        "total_operating_cost_EUR",
    ]
    available = [column for column in cost_columns if column in summary.columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    summary.set_index("plant_name")[available].plot(kind="bar", ax=ax)
    ax.set_ylabel("EUR")
    ax.set_title("Cost summary")
    ax.legend(loc="best")
    fig.tight_layout()
    path = plot_dir / "cost_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
