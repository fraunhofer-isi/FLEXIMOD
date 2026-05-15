# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RESULT_FILES = {
    "dispatch_results": "dispatch_results.csv",
    "market_ledger": "market_ledger.csv",
    "storage_cost_ledger": "storage_cost_ledger.csv",
    "summary_indicators": "summary_indicators.csv",
    "afrr_energy_data_quality_summary": "afrr_energy_data_quality_summary.csv",
}


@dataclass(frozen=True)
class CaseResults:
    """Loaded output tables for one FlexIMOD case."""

    output_dir: Path
    dispatch_results: pd.DataFrame
    market_ledger: pd.DataFrame
    storage_cost_ledger: pd.DataFrame
    summary_indicators: pd.DataFrame
    afrr_energy_data_quality_summary: pd.DataFrame


def load_results(output_dir: str | Path) -> CaseResults:
    """Load available output CSV files from a case output directory."""

    output_dir = Path(output_dir)
    dispatch = _load_csv(output_dir / RESULT_FILES["dispatch_results"], required=True)
    market = _load_csv(output_dir / RESULT_FILES["market_ledger"], required=False)
    storage = _load_csv(output_dir / RESULT_FILES["storage_cost_ledger"], required=False)
    summary = _load_csv(
        output_dir / RESULT_FILES["summary_indicators"],
        required=False,
        datetime_index=False,
    )
    afrr_quality = _load_csv(
        output_dir / RESULT_FILES["afrr_energy_data_quality_summary"],
        required=False,
        datetime_index=False,
    )
    return CaseResults(
        output_dir=output_dir,
        dispatch_results=dispatch,
        market_ledger=market,
        storage_cost_ledger=storage,
        summary_indicators=summary,
        afrr_energy_data_quality_summary=afrr_quality,
    )


def save_summary_indicators(summary: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save analytics summary indicators to ``summary_indicators.csv``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RESULT_FILES["summary_indicators"]
    summary.to_csv(path, index=False)
    return path


def calculate_summary_indicators(
    dispatch_results: pd.DataFrame,
    market_ledger: pd.DataFrame | None = None,
    storage_cost_ledger: pd.DataFrame | None = None,
    afrr_energy_data_quality_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate operational, market, economic and storage indicators.

    The function is intentionally tolerant of missing future-market columns so
    DA-only MVP outputs can be analysed with the same API as later cases.
    """

    dispatch = ensure_datetime_index(dispatch_results)
    market = ensure_datetime_index(market_ledger) if market_ledger is not None else pd.DataFrame()
    storage = (
        ensure_datetime_index(storage_cost_ledger)
        if storage_cost_ledger is not None
        else pd.DataFrame()
    )
    afrr_quality = (
        afrr_energy_data_quality_summary
        if afrr_energy_data_quality_summary is not None
        else pd.DataFrame()
    )

    if dispatch.empty:
        return pd.DataFrame()

    records = []
    for plant_name, plant_dispatch in _group_by_plant(dispatch):
        plant_market = _filter_plant(market, plant_name)
        plant_storage = _filter_plant(storage, plant_name)
        records.append(
            {
                "plant_name": plant_name,
                **_operational_indicators(plant_dispatch),
                **_market_indicators(plant_dispatch, plant_market),
                **_economic_indicators(plant_dispatch, plant_market, plant_storage),
                **_storage_source_indicators(plant_storage),
                **_afrr_data_quality_indicators(afrr_quality),
            }
        )
    return pd.DataFrame(records)


def select_sample_day(
    dispatch_results: pd.DataFrame,
    sample_day: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Return a representative day, preferring highest storage activity."""

    dispatch = ensure_datetime_index(dispatch_results)
    if dispatch.empty:
        raise ValueError("Cannot select a sample day from empty dispatch results")

    if sample_day is not None:
        return pd.Timestamp(sample_day).normalize()

    activity_columns = [
        column for column in ["etes_charge_MWh", "etes_discharge_MWh"] if column in dispatch
    ]
    if activity_columns:
        daily_activity = dispatch[activity_columns].fillna(0.0).sum(axis=1).resample("D").sum()
        if daily_activity.max() > 0:
            return pd.Timestamp(daily_activity.idxmax()).normalize()

    price_column = _first_existing(
        dispatch,
        [
            "day_ahead_price_EUR_per_MWh",
            "day_ahead_price_EUR_per_MWh_el",
            "intraday_price_EUR_per_MWh_el",
            "afrr_energy_price_EUR_per_MWh_el",
        ],
    )
    if price_column:
        daily_spread = (
            dispatch[price_column].resample("D").agg(lambda values: values.max() - values.min())
        )
        return pd.Timestamp(daily_spread.idxmax()).normalize()

    return pd.Timestamp(dispatch.index.min()).normalize()


def ensure_datetime_index(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return a copy with a DatetimeIndex when a datetime column is present."""

    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.copy()
    if isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
        return result.sort_index()

    if "datetime" not in result.columns:
        warn_missing("datetime", "result table")
        return result

    result["datetime"] = pd.to_datetime(result["datetime"], errors="raise")
    return result.set_index("datetime").sort_index()


def require_columns(frame: pd.DataFrame, columns: list[str], context: str) -> list[str]:
    """Return available required columns and warn for missing ones."""

    available = []
    for column in columns:
        if column in frame.columns:
            available.append(column)
        else:
            warn_missing(column, context)
    return available


def warn_missing(column: str, context: str) -> None:
    warnings.warn(
        f"Column {column} not found. Skipping {context}.",
        stacklevel=2,
    )


def create_output_dir(output_dir: str | Path, subdir: str = "plots") -> Path:
    plot_dir = Path(output_dir) / subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def derive_gas_benchmark(dispatch_results: pd.DataFrame) -> pd.Series | None:
    """Return a gas-based heat benchmark if it exists or can be approximated."""

    dispatch = ensure_datetime_index(dispatch_results)
    existing = _first_existing(
        dispatch,
        ["gas_based_heat_benchmark_EUR_per_MWh_th", "gas_based_heat_benchmark"],
    )
    if existing:
        return dispatch[existing].astype(float)

    required = {"gas_price_EUR_per_MWh", "gas_input_MWh", "gas_heat_MWh"}
    if required.issubset(dispatch.columns):
        denominator = dispatch["gas_heat_MWh"].replace(0, pd.NA).astype(float)
        gas_input_per_heat = dispatch["gas_input_MWh"].astype(float) / denominator
        benchmark = dispatch["gas_price_EUR_per_MWh"].astype(float) * gas_input_per_heat
        benchmark.name = "gas_based_heat_benchmark_EUR_per_MWh_th"
        return benchmark.ffill().bfill()

    return None


def storage_content_by_source(
    storage_cost_ledger: pd.DataFrame,
    dispatch_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return storage content columns by source market.

    Current ledgers include economics-oriented source inventory columns.
    """

    storage = ensure_datetime_index(storage_cost_ledger)
    if storage.empty:
        return pd.DataFrame()

    source_columns = [
        "thermal_inventory_day_ahead_MWh_th",
        "thermal_inventory_intraday_continuous_MWh_th",
        "thermal_inventory_afrr_energy_MWh_th",
        "thermal_inventory_other_MWh_th",
    ]
    available = [column for column in source_columns if column in storage.columns]
    if available:
        return storage[available].rename(columns=_inventory_column_label).fillna(0.0)

    if "thermal_inventory_MWh_th" not in storage.columns:
        warn_missing("thermal_inventory_MWh_th", "storage source plot")
        return pd.DataFrame()

    procurement_market = ""
    if "procurement_market" in storage.columns:
        sources = sorted(
            {
                str(value)
                for value in storage["procurement_market"].dropna().unique()
                if str(value).strip()
            }
        )
        if len(sources) == 1:
            procurement_market = sources[0]

    label = _market_label(procurement_market) if procurement_market else "Unknown or mixed source"
    result = pd.DataFrame(index=storage.index)
    result[label] = storage["thermal_inventory_MWh_th"].fillna(0.0).astype(float)

    if dispatch_results is not None and not dispatch_results.empty:
        dispatch = ensure_datetime_index(dispatch_results)
        if "etes_soc_MWh" in dispatch.columns:
            result["Total ETES SoC"] = dispatch.groupby(dispatch.index)["etes_soc_MWh"].sum()

    return result


def _load_csv(path: Path, required: bool, datetime_index: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required output file not found: {path}")
        warnings.warn(f"Output file not found: {path}. Continuing without it.", stacklevel=2)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return ensure_datetime_index(frame) if datetime_index else frame


def _operational_indicators(dispatch: pd.DataFrame) -> dict[str, float]:
    soc = _series(dispatch, "etes_soc_MWh")
    timestep_hours = _infer_timestep_hours(dispatch.index)
    max_soc = float(soc.max()) if not soc.empty else 0.0
    full_threshold = max_soc * 0.999 if max_soc > 0 else float("inf")
    empty_threshold = max(max_soc * 0.001, 1e-9) if max_soc > 0 else 1e-9
    total_storage_discharge = _sum(dispatch, "etes_discharge_MWh")
    total_etes_charge = _sum(dispatch, "etes_charge_MWh")
    return {
        "total_heat_demand_MWh": _sum(dispatch, "heat_demand_MWh"),
        "total_gas_heat_MWh": _sum(dispatch, "gas_heat_MWh"),
        "total_storage_discharge_heat_MWh": total_storage_discharge,
        "total_etes_charging_electricity_MWh": total_etes_charge,
        "total_etes_discharging_heat_MWh": total_storage_discharge,
        "total_electric_heat_MWh": total_storage_discharge,
        "total_etes_charged_MWh": total_etes_charge,
        "total_etes_discharged_MWh": total_storage_discharge,
        "final_etes_soc_MWh": float(soc.iloc[-1]) if not soc.empty else 0.0,
        "max_etes_soc_MWh": max_soc,
        "hours_storage_full": float((soc >= full_threshold).sum() * timestep_hours),
        "hours_storage_empty": float((soc <= empty_threshold).sum() * timestep_hours),
        "total_unmet_heat_MWh": _sum(dispatch, "unmet_heat_MWh"),
    }


def _market_indicators(dispatch: pd.DataFrame, market: pd.DataFrame) -> dict[str, float]:
    da = _sum(
        market,
        "day_ahead_position_MWh_el",
        fallback=_sum(dispatch, "electricity_consumption_MWh"),
    )
    idc_buy = _sum(market, "intraday_buy_MWh_el")
    idc_sell = _sum(market, "intraday_sell_MWh_el")
    afrr = _sum(market, "afrr_energy_activated_MWh_el")
    afrr_bid = _sum(market, "afrr_energy_bid_MWh_el")
    final_planned = _sum(
        market,
        "scheduled_electricity_procurement_MWh_el",
        fallback=da + idc_buy - idc_sell,
    )
    actual = _sum(
        market,
        "actual_electricity_consumption_MWh_el",
        fallback=_sum(dispatch, "electricity_consumption_MWh"),
    )
    denominator = actual if abs(actual) > 1e-12 else 1.0
    return {
        "total_DA_electricity_MWh": da,
        "total_IDC_buy_MWh": idc_buy,
        "total_IDC_sell_MWh": idc_sell,
        "total_final_planned_electricity_MWh": final_planned,
        "total_afrr_energy_bid_MWh": afrr_bid,
        "total_afrr_energy_activated_MWh": afrr,
        "total_actual_electricity_consumption_MWh": actual,
        "total_intraday_buy_MWh_el": idc_buy,
        "total_intraday_sell_MWh_el": idc_sell,
        "total_scheduled_electricity_procurement_MWh_el": final_planned,
        "total_afrr_energy_bid_MWh_el": afrr_bid,
        "total_afrr_energy_activated_MWh_el": afrr,
        "total_IDC_buy_electricity_MWh": idc_buy,
        "total_IDC_sell_electricity_MWh": idc_sell,
        "total_afrr_activated_electricity_MWh": afrr,
        "total_actual_electricity_consumption_MWh_el": actual,
        "share_DA_electricity": da / denominator,
        "share_IDC_net_electricity": (idc_buy - idc_sell) / denominator,
        "share_afrr_activated_electricity": afrr / denominator,
    }


def _economic_indicators(
    dispatch: pd.DataFrame,
    market: pd.DataFrame,
    storage: pd.DataFrame,
) -> dict[str, float]:
    heat_demand = _sum(dispatch, "heat_demand_MWh")
    total_operating_cost = _sum(dispatch, "operating_cost_EUR")
    total_electricity_cost = _sum(dispatch, "electricity_cost_EUR")
    idc_buy_cost, idc_sell_revenue = _trading_cashflows(
        market,
        buy_col="intraday_buy_MWh_el",
        sell_col="intraday_sell_MWh_el",
        price_col="intraday_price_EUR_per_MWh_el",
    )
    idc_value = _trading_value(
        market,
        buy_col="intraday_buy_MWh_el",
        sell_col="intraday_sell_MWh_el",
        price_col="intraday_price_EUR_per_MWh_el",
    )
    afrr_energy_value = _energy_value(
        market,
        "afrr_energy_activated_MWh_el",
        "afrr_energy_price_EUR_per_MWh_el",
    )
    afrr_energy_cost = _sum(
        dispatch,
        "afrr_energy_cost_EUR",
        fallback=_energy_value(
            market,
            "afrr_energy_activated_MWh_el",
            "afrr_energy_price_EUR_per_MWh_el",
        ),
    )
    afrr_savings = _sum(dispatch, "afrr_energy_savings_vs_benchmark_EUR")
    afrr_capacity_revenue = 0.0
    average_stored_heat_cost = _last_existing_non_missing(
        storage,
        ["weighted_average_inventory_cost_EUR_per_MWh_th"],
    )
    total_net_operating_cost = (
        total_operating_cost - idc_value - afrr_energy_value - afrr_capacity_revenue
    )
    return {
        "total_electricity_procurement_cost_EUR": total_electricity_cost,
        "total_electricity_cost_EUR": total_electricity_cost,
        "total_gas_cost_EUR": _sum(dispatch, "gas_cost_EUR"),
        "total_co2_cost_EUR": _sum(dispatch, "co2_cost_EUR"),
        "total_unmet_heat_penalty_EUR": _sum(dispatch, "unmet_heat_penalty_EUR"),
        "IDC_buy_cost_EUR": idc_buy_cost,
        "IDC_sell_revenue_EUR": idc_sell_revenue,
        "IDC_net_cashflow_EUR": idc_sell_revenue - idc_buy_cost,
        "afrr_energy_cost_EUR": afrr_energy_cost,
        "afrr_energy_savings_vs_benchmark_EUR": afrr_savings,
        "total_IDC_trading_value_EUR": idc_value,
        "total_afrr_energy_value_EUR": afrr_energy_value,
        "total_afrr_capacity_revenue_EUR": afrr_capacity_revenue,
        "total_operating_cost_EUR": total_operating_cost,
        "total_net_operating_cost_EUR": total_net_operating_cost,
        "average_cost_of_heat_EUR_per_MWh": total_operating_cost / heat_demand
        if heat_demand > 1e-12
        else 0.0,
        "average_cost_of_stored_heat_EUR_per_MWh_th": average_stored_heat_cost,
    }


def _storage_source_indicators(storage: pd.DataFrame) -> dict[str, float]:
    if storage.empty:
        return {
            "share_stored_heat_from_DA": 0.0,
            "share_stored_heat_from_IDC": 0.0,
            "share_stored_heat_from_afrr_energy": 0.0,
            "weighted_average_stored_heat_cost_EUR_per_MWh_th": 0.0,
        }

    da_added = _charged_heat_from_market(storage, "day_ahead")
    idc_added = _charged_heat_from_market(storage, "intraday_continuous")
    afrr_added = _charged_heat_from_market(storage, "afrr_energy")
    total_added = da_added + idc_added + afrr_added
    denominator = total_added if total_added > 1e-12 else 1.0
    return {
        "share_stored_heat_from_DA": da_added / denominator,
        "share_stored_heat_from_IDC": idc_added / denominator,
        "share_stored_heat_from_afrr_energy": afrr_added / denominator,
        "weighted_average_stored_heat_cost_EUR_per_MWh_th": _last_existing_non_missing(
            storage,
            ["weighted_average_inventory_cost_EUR_per_MWh_th"],
        ),
    }


def _afrr_data_quality_indicators(quality_summary: pd.DataFrame) -> dict[str, float]:
    empty = {
        "aFRR_down_total_rows": 0.0,
        "aFRR_down_valid_activation_rows": 0.0,
        "aFRR_down_zero_activation_rows": 0.0,
        "aFRR_down_missing_price_rows": 0.0,
        "aFRR_down_missing_quantity_rows": 0.0,
        "aFRR_down_activation_without_price_rows": 0.0,
        "aFRR_down_skipped_activation_MWh_due_to_missing_price": 0.0,
        "aFRR_down_used_system_activation_MWh": 0.0,
        "aFRR_down_negative_quantity_rows": 0.0,
        "aFRR_down_price_zero_with_activation_rows": 0.0,
    }
    if quality_summary.empty:
        return empty

    result = empty.copy()
    numeric = quality_summary.select_dtypes(include="number")
    for column in result:
        if column in numeric.columns:
            result[column] = float(numeric[column].sum())
    return result


def _group_by_plant(frame: pd.DataFrame):
    if "plant_name" in frame.columns:
        yield from frame.groupby("plant_name", sort=False)
    else:
        yield "all_plants", frame


def _filter_plant(frame: pd.DataFrame, plant_name: str) -> pd.DataFrame:
    if frame.empty or "plant_name" not in frame.columns:
        return frame
    return frame[frame["plant_name"] == plant_name]


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    return frame[column].fillna(0.0).astype(float)


def _sum(frame: pd.DataFrame, column: str, fallback: float = 0.0) -> float:
    if frame.empty or column not in frame.columns:
        return float(fallback)
    return float(frame[column].fillna(0.0).astype(float).sum())


def _trading_value(
    frame: pd.DataFrame,
    buy_col: str,
    sell_col: str,
    price_col: str,
) -> float:
    if frame.empty or price_col not in frame.columns:
        return 0.0
    price = frame[price_col].fillna(0.0).astype(float)
    buys = frame[buy_col].fillna(0.0).astype(float) if buy_col in frame.columns else 0.0
    sells = frame[sell_col].fillna(0.0).astype(float) if sell_col in frame.columns else 0.0
    return float((sells * price - buys * price).sum())


def _trading_cashflows(
    frame: pd.DataFrame,
    buy_col: str,
    sell_col: str,
    price_col: str,
) -> tuple[float, float]:
    if frame.empty or price_col not in frame.columns:
        return 0.0, 0.0
    price = frame[price_col].fillna(0.0).astype(float)
    buys = frame[buy_col].fillna(0.0).astype(float) if buy_col in frame.columns else 0.0
    sells = frame[sell_col].fillna(0.0).astype(float) if sell_col in frame.columns else 0.0
    return float((buys * price).sum()), float((sells * price).sum())


def _energy_value(frame: pd.DataFrame, volume_col: str, price_col: str) -> float:
    if frame.empty or volume_col not in frame.columns or price_col not in frame.columns:
        return 0.0
    return float(
        (
            frame[volume_col].fillna(0.0).astype(float) * frame[price_col].fillna(0.0).astype(float)
        ).sum()
    )


def _last_non_missing(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = frame[column].dropna().astype(float)
    return float(values.iloc[-1]) if not values.empty else 0.0


def _last_existing_non_missing(frame: pd.DataFrame, columns: list[str]) -> float:
    column = _first_existing(frame, columns)
    return _last_non_missing(frame, column) if column else 0.0


def _charged_heat_from_market(storage: pd.DataFrame, procurement_market: str) -> float:
    if "procurement_market" not in storage.columns or "charged_heat_MWh_th" not in storage.columns:
        return 0.0
    normalised = storage["procurement_market"].fillna("").map(_normalise_procurement_market)
    mask = normalised == procurement_market
    return float(storage.loc[mask, "charged_heat_MWh_th"].fillna(0.0).astype(float).sum())


def _first_existing(frame: pd.DataFrame, columns: list[str]) -> str | None:
    return next((column for column in columns if column in frame.columns), None)


def _infer_timestep_hours(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 0.0
    diffs = index.to_series().diff().dropna().dt.total_seconds().div(3600)
    if diffs.empty:
        return 0.0
    return float(diffs.mode().iloc[0])


def _inventory_column_label(column: str) -> str:
    labels = {
        "thermal_inventory_day_ahead_MWh_th": "Day-ahead",
        "thermal_inventory_intraday_continuous_MWh_th": "IDC",
        "thermal_inventory_afrr_energy_MWh_th": "aFRR energy",
        "thermal_inventory_other_MWh_th": "Other/unknown",
    }
    return labels[column]


def _market_label(procurement_market: str) -> str:
    labels = {
        "day_ahead": "Day-ahead",
        "intraday_continuous": "IDC",
        "afrr_energy": "aFRR energy",
    }
    return labels.get(_normalise_procurement_market(procurement_market), "Other/unknown")


def _normalise_procurement_market(procurement_market: str) -> str:
    value = str(procurement_market).strip().lower()
    if value in {"da", "day-ahead", "day_ahead"}:
        return "day_ahead"
    if value in {"idc", "intraday", "intraday_continuous"}:
        return "intraday_continuous"
    if value in {"afrr", "afrr_energy", "afrr energy"}:
        return "afrr_energy"
    if value in {"no_charge", "none", ""}:
        return "other"
    return value
