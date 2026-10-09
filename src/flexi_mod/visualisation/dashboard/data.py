# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load one case output folder into a plant-family aware view for the dashboards."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from flexi_mod.visualisation.analytics import (
    RESULT_FILES,
    parse_datetimes,
    resolve_table_path,
)

ALL_PLANTS = "__all__"
ALL_PLANTS_LABEL = "All plants (aggregate)"

FAMILY_LABELS = {
    "steam": "Steam / heat plant",
    "building": "Building with EV fleet",
    "cement": "Cement plant",
    "steel": "Steel plant",
    "generic": "Plant",
}

# Numeric dispatch columns are recognised by their unit suffix. Everything else
# (solver names, labels, flags) is skipped, which keeps large industrial runs light.
_NUMERIC_COLUMN = re.compile(
    r"(_MWh|_MW|_mw|_t|_kg|_EUR|_THB|_fraction|_km|_status|_soc|_th|_el|_per_t|_MW_h|_weight"
    r"|_threshold)$"
)
_ALWAYS = {
    "datetime",
    "plant_name",
    "rolling_window",
    "energy_cost",
    "demand_charge_cost",
    "total_cost",
}
_STRING_COLUMNS = {"datetime", "plant_name"}


@dataclass
class CaseData:
    """Tables of one case output folder, aggregated lazily per plant selection."""

    name: str
    dispatch: pd.DataFrame
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    storage: pd.DataFrame = field(default_factory=pd.DataFrame)
    grid_fees: pd.DataFrame = field(default_factory=pd.DataFrame)
    afrr_blocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    output_dir: Path | None = None
    family: str = "generic"
    step_hours: float = 0.25
    currency: str = "EUR"
    plants: list[str] = field(default_factory=list)
    _cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    @property
    def family_label(self) -> str:
        return FAMILY_LABELS.get(self.family, FAMILY_LABELS["generic"])

    @property
    def period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.dispatch.index.min(), self.dispatch.index.max()

    def plant_options(self) -> list[tuple[str, str]]:
        """Return ``(value, label)`` pairs for a plant selector."""

        if len(self.plants) <= 1:
            return [(ALL_PLANTS, self.plants[0] if self.plants else "plant")]
        return [(ALL_PLANTS, ALL_PLANTS_LABEL), *((plant, plant) for plant in self.plants)]

    def frame(self, plant: str = ALL_PLANTS) -> pd.DataFrame:
        """Return the dispatch time series of one plant, or all plants aggregated."""

        key = plant if plant in self.plants else ALL_PLANTS
        if key not in self._cache:
            if key == ALL_PLANTS:
                self._cache[key] = aggregate_plants(self.dispatch)
            else:
                subset = self.dispatch[self.dispatch["plant_name"] == key]
                self._cache[key] = subset.drop(columns="plant_name")
        return self._cache[key]

    def storage_frame(self, plant: str = ALL_PLANTS) -> pd.DataFrame:
        if self.storage.empty:
            return self.storage
        frame = self.storage
        if plant in self.plants and "plant_name" in frame:
            frame = frame[frame["plant_name"] == plant]
        numeric = frame.drop(columns=["plant_name", "procurement_market"], errors="ignore")
        return numeric.select_dtypes("number").groupby(level=0).sum()

    def summary_rows(self, plant: str = ALL_PLANTS) -> pd.DataFrame:
        if self.summary.empty or plant not in self.plants or "plant_name" not in self.summary:
            return self.summary
        return self.summary[self.summary["plant_name"] == plant]

    def summary_total(self, column: str, plant: str = ALL_PLANTS) -> float | None:
        """Sum one summary indicator over the selected plants (``None`` when absent)."""

        rows = self.summary_rows(plant)
        if column not in rows or rows.empty:
            return None
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        return float(values.sum()) if not values.empty else None

    def blocks_frame(self, plant: str = ALL_PLANTS) -> pd.DataFrame:
        frame = self.afrr_blocks
        if frame.empty or plant not in self.plants or "plant_name" not in frame:
            return frame
        return frame[frame["plant_name"] == plant]


def load_case(output_dir: str | Path) -> CaseData:
    """Load a case output folder (``.csv`` or ``.csv.zst`` tables)."""

    output_dir = Path(output_dir)
    dispatch_path = resolve_table_path(output_dir / RESULT_FILES["dispatch_results"])
    if not dispatch_path.exists():
        raise FileNotFoundError(f"No dispatch_results table found in {output_dir}")

    dispatch = _read_dispatch(dispatch_path)
    storage = _read_small(output_dir / RESULT_FILES["storage_cost_ledger"], datetime_index=True)
    return build_case(
        name=output_dir.name,
        dispatch=dispatch,
        summary=_read_small(output_dir / RESULT_FILES["summary_indicators"]),
        storage=storage,
        grid_fees=_read_small(output_dir / "grid_fee_summary.csv"),
        afrr_blocks=_read_small(output_dir / RESULT_FILES["afrr_capacity_block_summary"]),
        output_dir=output_dir,
    )


def build_case(
    name: str,
    dispatch: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    storage: pd.DataFrame | None = None,
    grid_fees: pd.DataFrame | None = None,
    afrr_blocks: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> CaseData:
    """Build a :class:`CaseData` from in-memory tables (as produced by the runner)."""

    dispatch = _normalise_dispatch(dispatch)
    if dispatch.empty:
        raise ValueError("Cannot build a dashboard from empty dispatch results")
    summary = summary if summary is not None else pd.DataFrame()
    plants = sorted(dispatch["plant_name"].dropna().unique().tolist())
    currency = "EUR"
    if not summary.empty and "currency" in summary and summary["currency"].notna().any():
        currency = str(summary["currency"].dropna().iloc[0])
    return CaseData(
        name=name,
        dispatch=dispatch,
        summary=summary,
        storage=_normalise_ledger(storage),
        grid_fees=grid_fees if grid_fees is not None else pd.DataFrame(),
        afrr_blocks=afrr_blocks if afrr_blocks is not None else pd.DataFrame(),
        output_dir=output_dir,
        family=detect_family(dispatch.columns),
        step_hours=infer_step_hours(dispatch.index),
        currency=currency,
        plants=plants,
    )


def discover_cases(
    root: str | Path,
    max_depth: int = 3,
    table: str = "dispatch_results",
) -> list[Path]:
    """Return output folders below ``root`` that contain the given result table."""

    root = Path(root)
    if not root.exists():
        return []
    found: list[Path] = []
    for pattern in (f"{table}.csv", f"{table}.csv.zst"):
        for path in root.rglob(pattern):
            if len(path.relative_to(root).parts) <= max_depth + 1:
                found.append(path.parent)
    return sorted(set(found), key=lambda path: str(path).lower())


def detect_family(columns: pd.Index | list[str]) -> str:
    names = set(columns)
    if "building_demand_MWh" in names:
        return "building"
    if "clinker_output_t" in names:
        return "cement"
    if "steel_output_t" in names or "steel_demand_total_t" in names:
        return "steel"
    if "heat_demand_MWh" in names:
        return "steam"
    return "generic"


def column_kind(column: str) -> str:
    """Classify a column for aggregation.

    ``flow``  quantities per timestep (energy, cost, mass): summed over time and plants.
    ``level`` stocks and power: averaged over time, summed over plants.
    ``price`` prices, shares and status flags: averaged over both.
    """

    name = column.lower()
    if any(token in name for token in ("price", "_per_", "fraction", "weight", "benchmark")):
        return "price"
    if "status" in name or "threshold" in name or name == "rolling_window":
        return "price"
    if "soc" in name or "inventory" in name or name.endswith(("_mw", "_mw_h")):
        return "level"
    return "flow"


def aggregate_plants(dispatch: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all plants per timestamp (sums for quantities, means for prices)."""

    numeric = dispatch.drop(columns="plant_name").select_dtypes("number")
    if dispatch["plant_name"].nunique() <= 1:
        return numeric.astype("float64")
    rules = {
        column: "mean" if column_kind(column) == "price" else "sum" for column in numeric.columns
    }
    return numeric.astype("float64").groupby(level=0).agg(rules)


def resample(frame: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    """Resample a dispatch frame, summing flows and averaging levels and prices."""

    if rule is None or frame.empty:
        return frame
    rules = {column: "sum" if column_kind(column) == "flow" else "mean" for column in frame.columns}
    return frame.resample(rule).agg(rules)


RESOLUTIONS = {"native": None, "1h": "1h", "6h": "6h", "1D": "1D", "1W": "7D"}


def auto_resolution(index: pd.DatetimeIndex, step_hours: float, max_points: int = 2200) -> str:
    """Pick the finest resolution that keeps a series under ``max_points``."""

    if len(index) == 0:
        return "native"
    span_hours = max((index.max() - index.min()).total_seconds() / 3600.0, step_hours)
    for label, hours in (("native", step_hours), ("1h", 1.0), ("6h", 6.0), ("1D", 24.0)):
        if hours >= step_hours and span_hours / hours <= max_points:
            return label
    return "1W"


def infer_step_hours(index: pd.Index) -> float:
    if len(index) < 2:
        return 1.0
    deltas = pd.Series(index).diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    return float(deltas.median().total_seconds() / 3600.0) if not deltas.empty else 1.0


def slice_period(
    frame: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    if frame.empty or (start is None and end is None):
        return frame
    lower = pd.Timestamp(start) if start is not None else frame.index.min()
    upper = pd.Timestamp(end) + pd.Timedelta(days=1) if end is not None else frame.index.max()
    return frame[(frame.index >= lower) & (frame.index < upper)]


def _read_dispatch(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    keep = [column for column in header if column in _ALWAYS or _NUMERIC_COLUMN.search(column)]
    dtypes = {
        column: ("str" if column in _STRING_COLUMNS else "float32")
        for column in keep
        if column != "rolling_window"
    }
    try:
        return pd.read_csv(path, usecols=keep, dtype=dtypes)
    except (ValueError, TypeError):
        # A matched column was not numeric; fall back to inferred dtypes.
        return pd.read_csv(path, usecols=keep)


def _read_small(path: Path, datetime_index: bool = False) -> pd.DataFrame:
    path = resolve_table_path(path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if datetime_index:
        return _normalise_ledger(frame)
    return frame


def _normalise_dispatch(dispatch: pd.DataFrame) -> pd.DataFrame:
    frame = dispatch.copy()
    if "datetime" not in frame.columns:
        frame = frame.reset_index()
        if "datetime" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "datetime"})
    keep = [
        column for column in frame.columns if column in _ALWAYS or _NUMERIC_COLUMN.search(column)
    ]
    frame = frame[keep]
    frame["datetime"] = parse_datetimes(frame["datetime"])
    if "plant_name" not in frame.columns:
        frame["plant_name"] = "plant"
    frame["plant_name"] = frame["plant_name"].astype(str)
    return frame.set_index("datetime").sort_index().replace([np.inf, -np.inf], np.nan)


def _normalise_ledger(ledger: pd.DataFrame | None) -> pd.DataFrame:
    if ledger is None or ledger.empty:
        return pd.DataFrame()
    frame = ledger.copy()
    if "datetime" in frame.columns:
        frame["datetime"] = parse_datetimes(frame["datetime"])
        frame = frame.set_index("datetime")
    elif isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    return frame.sort_index()
