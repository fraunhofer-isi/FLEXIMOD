# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate FLEXIMOD steel-plant input cases per (scenario, year, technology route).

Source data lives OUTSIDE the repository (a PhD dataset):

* Scenario forecasts:
  ``<EXTERNAL_OUTPUT_DIR>/<scenario>_<year>/forecasts_df.csv`` — 15-min market/commodity
  prices plus one ``<plant>_steel_demand`` column per plant. The technology subfolders inside
  each scenario folder are output sinks and define the route list.
* Plant master:
  ``industrial_dsm_units_steel.xlsx`` (sheet ``industrial_dsm_units``), keyed by route in the
  ``name`` column (``P<id>_<route>``); the ``technology`` column is the process step.

Key fact: steel demand is **year-specific and per physical plant, not route-specific**. Every
route uses the same 9 physical plant IDs, so each (scenario, year, route) case reuses that year's
demand, re-keyed by physical plant id onto the route's plant names.

For each (scenario, year, route) the script writes a self-contained case folder
``data/input/<scenario>_<year>_<route>/`` with ``config.yaml`` (template, only name+year changed),
``plants.csv`` (Excel rows for the route, ``demand`` -> blanked ``steel_demand``),
``additional_charges.csv`` (template values, one column per plant) and ``forecasts_df.csv``
(market columns verbatim + re-keyed demand columns), plus REUSE ``.license`` sidecars.

The generated ``forecasts_df.csv`` embed proprietary market data — they are git-ignored and must
not be committed.

Usage:
    python scripts/generate_steel_cases.py                 # generate everything
    python scripts/generate_steel_cases.py --scenario fokusH2_2030
    python scripts/generate_steel_cases.py --route dri_eaf_coal_external --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- config
# Machine-specific external source paths — adjust if the dataset moves.
EXTERNAL_BASE = Path(
    r"C:/Users/khm/ownCloud/Dropbox/Ph.D/My publications/Journal paper/4/Data"
)
EXTERNAL_OUTPUT_DIR = EXTERNAL_BASE / "Assume" / "Output"
STEEL_DB_XLSX = EXTERNAL_BASE / "steel plant database" / "industrial_dsm_units_steel.xlsx"
EXCEL_SHEET = "industrial_dsm_units"

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_CASE = REPO_ROOT / "data" / "input" / "steel_plant_DE"
TARGET_INPUT_DIR = REPO_ROOT / "data" / "input"

LICENSE_TEXT = (
    "SPDX-FileCopyrightText: FLEXIMOD Developers\n\n"
    "SPDX-License-Identifier: AGPL-3.0-or-later\n"
)
TEMPLATE_CASE_NAME = "steel_plant_DE"
TEMPLATE_YEAR = "2045"
DEMAND_SUFFIX = "_steel_demand"
NAME_RE = re.compile(r"^(P\d+)_(.+)$")


# --------------------------------------------------------------------------- helpers
def parse_plant_name(name: object) -> tuple[str | None, str | None]:
    """Split ``P<id>_<route>`` into (physical_id, route)."""
    match = NAME_RE.match(str(name).strip())
    return (match.group(1), match.group(2)) if match else (None, None)


def fmt_cell(value: object) -> str:
    """Format a plants.csv cell: NaN/None -> '', integer-valued floats -> int, else str."""
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def write_license(path: Path) -> None:
    path.write_text(LICENSE_TEXT, encoding="utf-8")


# --------------------------------------------------------------------------- template data
def load_template_plants_columns() -> list[str]:
    with (TEMPLATE_CASE / "plants.csv").open(encoding="utf-8") as handle:
        return next(csv.reader(handle))


def load_template_charge_specs() -> list[tuple[str, str, str]]:
    """Return [(component, unit, value)] from the template additional_charges.csv.

    All plant columns share one value per component, so the first plant column is used.
    """
    frame = pd.read_csv(TEMPLATE_CASE / "additional_charges.csv", dtype=str, keep_default_na=False)
    value_col = frame.columns[2]
    return [(r["component"], r["unit"], r[value_col]) for _, r in frame.iterrows()]


def load_template_config_text() -> str:
    return (TEMPLATE_CASE / "config.yaml").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- excel
def load_excel_plants() -> pd.DataFrame:
    frame = pd.read_excel(STEEL_DB_XLSX, sheet_name=EXCEL_SHEET, engine="openpyxl")
    routes = frame["name"].map(parse_plant_name)
    frame = frame.copy()
    frame["_pid"] = [pid for pid, _ in routes]
    frame["_route"] = [route for _, route in routes]
    return frame


# --------------------------------------------------------------------------- forecast
def load_scenario_forecast(folder: Path):
    """Return (time_col_name, market_cols, demand_by_physical_id, cleaned_frame)."""
    frame = pd.read_csv(folder / "forecasts_df.csv", dtype=str, keep_default_na=False)
    time_col = "datetime" if "datetime" in frame.columns else frame.columns[0]
    frame = frame[frame[time_col].str.strip() != ""].reset_index(drop=True)  # drop trailing blanks
    demand_cols = [c for c in frame.columns if c.endswith(DEMAND_SUFFIX)]
    market_cols = [c for c in frame.columns if c != time_col and c not in demand_cols]
    demand_by_id: dict[str, pd.Series] = {}
    for column in demand_cols:
        pid, _ = parse_plant_name(column[: -len(DEMAND_SUFFIX)])
        if pid:
            demand_by_id[pid] = frame[column]
    return time_col, market_cols, demand_by_id, frame


# --------------------------------------------------------------------------- case writers
def write_plants_csv(path: Path, route_rows: pd.DataFrame, template_cols: list[str]) -> None:
    out = [template_cols]
    for _, row in route_rows.iterrows():
        out.append(
            ["" if col == "steel_demand" else fmt_cell(row.get(col)) for col in template_cols]
        )
    write_csv(path, out)


def write_additional_charges(
    path: Path, plant_names: list[str], charge_specs: list[tuple[str, str, str]]
) -> None:
    out = [["component", "unit", *plant_names]]
    for component, unit, value in charge_specs:
        out.append([component, unit, *([value] * len(plant_names))])
    write_csv(path, out)


def write_forecasts(
    path: Path,
    time_col: str,
    market_cols: list[str],
    demand_by_id: dict[str, pd.Series],
    plant_names: list[str],
    frame: pd.DataFrame,
) -> None:
    out = pd.DataFrame({"datetime": frame[time_col].to_numpy()})
    for column in market_cols:
        out[column] = frame[column].to_numpy()
    for plant_name in plant_names:
        pid, _ = parse_plant_name(plant_name)
        out[f"{plant_name}{DEMAND_SUFFIX}"] = demand_by_id[pid].to_numpy()
    out.to_csv(path, index=False)


def write_config(path: Path, case_name: str, year: str, template_text: str) -> None:
    text = template_text.replace(TEMPLATE_CASE_NAME, case_name).replace(TEMPLATE_YEAR, year)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- driver
def scenario_year(folder_name: str) -> tuple[str, str]:
    scenario, _, year = folder_name.rpartition("_")
    return scenario, year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="Only this scenario_year folder (e.g. fokusH2_2030).")
    parser.add_argument("--route", help="Only this technology route (e.g. dri_eaf_coal_external).")
    parser.add_argument("--dry-run", action="store_true", help="List cases without writing files.")
    args = parser.parse_args()

    if not EXTERNAL_OUTPUT_DIR.exists():
        raise SystemExit(f"External scenario dir not found: {EXTERNAL_OUTPUT_DIR}")
    if not STEEL_DB_XLSX.exists():
        raise SystemExit(f"Steel plant Excel not found: {STEEL_DB_XLSX}")

    excel = load_excel_plants()
    excel_routes = set(excel["_route"].dropna().unique())
    template_cols = load_template_plants_columns()
    charge_specs = load_template_charge_specs()
    config_text = load_template_config_text()

    scenario_folders = sorted(
        p for p in EXTERNAL_OUTPUT_DIR.iterdir() if p.is_dir() and "_" in p.name
    )
    if args.scenario:
        scenario_folders = [p for p in scenario_folders if p.name == args.scenario]

    written = 0
    skipped: list[str] = []
    for folder in scenario_folders:
        forecast_path = folder / "forecasts_df.csv"
        if not forecast_path.exists():
            skipped.append(f"{folder.name}: no forecasts_df.csv")
            continue
        _, year = scenario_year(folder.name)
        time_col, market_cols, demand_by_id, frame = load_scenario_forecast(folder)

        # Routes = technology subfolders that are also real Excel routes.
        routes = sorted(
            d.name for d in folder.iterdir() if d.is_dir() and d.name in excel_routes
        )
        if args.route:
            routes = [r for r in routes if r == args.route]

        for route in routes:
            route_rows = excel[excel["_route"] == route]
            plant_names = list(dict.fromkeys(route_rows["name"].astype(str)))
            missing = [n for n in plant_names if parse_plant_name(n)[0] not in demand_by_id]
            if missing:
                skipped.append(f"{folder.name}/{route}: no demand for {missing}")
                continue

            case_name = f"{folder.name}_{route}"
            case_dir = TARGET_INPUT_DIR / case_name
            print(f"{'[dry-run] ' if args.dry_run else ''}{case_name}  "
                  f"({len(plant_names)} plants, {len(route_rows)} rows)")
            if args.dry_run:
                written += 1
                continue

            case_dir.mkdir(parents=True, exist_ok=True)
            write_config(case_dir / "config.yaml", case_name, year, config_text)
            write_plants_csv(case_dir / "plants.csv", route_rows, template_cols)
            write_additional_charges(case_dir / "additional_charges.csv", plant_names, charge_specs)
            write_forecasts(
                case_dir / "forecasts_df.csv",
                time_col, market_cols, demand_by_id, plant_names, frame,
            )
            for data_file in ("plants.csv", "additional_charges.csv", "forecasts_df.csv"):
                write_license(case_dir / f"{data_file}.license")
            written += 1

    print(f"\n{'Would generate' if args.dry_run else 'Generated'} {written} case(s).")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
