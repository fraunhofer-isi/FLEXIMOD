# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate FLEXIMOD steel-plant input cases per (scenario, year, technology route).

Source data lives OUTSIDE the repository (a PhD dataset):

* Scenario forecasts:
  ``<EXTERNAL_OUTPUT_DIR>/<scenario>_<year>/forecasts_df.csv`` — 15-min market/commodity
  prices plus one demand column per physical plant. Demand headers may be bare IDs such as
  ``P100000120423`` or already end in ``_steel_demand``.
* Scenario-sized plant masters:
  ``industrial_dsm_units_steel_scenarios_sized/*.xlsx`` (sheet
  ``industrial_dsm_units``), keyed by route in the ``name`` column
  (``P<id>_<route>``); the ``technology`` column is the process step. Each
  scenario family is mapped explicitly to its correctly sized workbook.

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

Fuel-consuming DRI and BF-BOF rows receive explicit ``coal_co2_factor`` or
``natural_gas_co2_factor`` values in ``plants.csv``. The command-line defaults
are scenario assumptions, not model fallbacks, and can be overridden when a
case requires different literature-grounded factors.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- config
# Machine-specific external source paths — adjust if the dataset moves.
EXTERNAL_BASE = Path(r"C:/Users/khm/ownCloud (2)/Dropbox/Ph.D/My publications/Journal paper/4/Data")
EXTERNAL_OUTPUT_DIR = EXTERNAL_BASE / "Assume" / "Output"
STEEL_DB_DIR = EXTERNAL_BASE / "steel plant database" / "industrial_dsm_units_steel_scenarios_sized"
STEEL_DB_BY_SCENARIO = {
    "aktuellepolitiken": STEEL_DB_DIR / "industrial_dsm_units_steel_AktuellePolitiken_sized.xlsx",
    "hohenachfrage": STEEL_DB_DIR / "industrial_dsm_units_steel_HoheNachfrage_sized.xlsx",
    "niedrigenachfrage": (STEEL_DB_DIR / "industrial_dsm_units_steel_Niedrigenachfrage_sized.xlsx"),
    "fokush2": (
        STEEL_DB_DIR / "industrial_dsm_units_steel_fokusStrom_fokusH2_technologiemix_sized.xlsx"
    ),
    "fokusstrom": (
        STEEL_DB_DIR / "industrial_dsm_units_steel_fokusStrom_fokusH2_technologiemix_sized.xlsx"
    ),
    "technologiemix": (
        STEEL_DB_DIR / "industrial_dsm_units_steel_fokusStrom_fokusH2_technologiemix_sized.xlsx"
    ),
}
EXCEL_SHEET = "industrial_dsm_units"

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_CASE = REPO_ROOT / "data" / "input" / "steel_plant_DE"
TARGET_INPUT_DIR = REPO_ROOT / "data" / "input"

LICENSE_TEXT = (
    "SPDX-FileCopyrightText: FLEXIMOD Developers\n\nSPDX-License-Identifier: AGPL-3.0-or-later\n"
)
TEMPLATE_CASE_NAME = "steel_plant_DE"
TEMPLATE_YEAR = "2045"
DEMAND_SUFFIX = "_steel_demand"
NAME_RE = re.compile(r"^(P\d+)_(.+)$")
PHYSICAL_ID_RE = re.compile(r"^P\d+$")
COAL_CO2_FACTOR_COLUMN = "coal_co2_factor"
NATURAL_GAS_CO2_FACTOR_COLUMN = "natural_gas_co2_factor"
CO2_FACTOR_COLUMNS = (COAL_CO2_FACTOR_COLUMN, NATURAL_GAS_CO2_FACTOR_COLUMN)
DEFAULT_COAL_CO2_FACTOR_T_PER_MWH = 0.34
DEFAULT_NATURAL_GAS_CO2_FACTOR_T_PER_MWH = 0.20
DEFAULT_IRON_ORE_PRICE_EUR_PER_T = 225.0
DEFAULT_LIME_PRICE_EUR_PER_T = 100.0
FUEL_TECHNOLOGIES = {"dri_plant", "bf_bof"}
HYBRID_HYDROGEN_NATURAL_GAS = "hybrid_hydrogen_natural_gas"
TECHNOLOGY_ROUTES = (
    "bf_bof_coal_external",
    "bf_bof_hybrid_hydrogen_natural_gas_electrolyser",
    "bf_bof_hybrid_hydrogen_natural_gas_external",
    "bf_bof_hydrogen_electrolyser",
    "bf_bof_hydrogen_external",
    "bf_bof_natural_gas_external",
    "dri_bof_coal_external",
    "dri_bof_hybrid_hydrogen_natural_gas_electrolyser",
    "dri_bof_hybrid_hydrogen_natural_gas_external",
    "dri_bof_hydrogen_electrolyser",
    "dri_bof_hydrogen_external",
    "dri_bof_natural_gas_external",
    "dri_eaf_coal_external",
    "dri_eaf_hybrid_hydrogen_natural_gas_electrolyser",
    "dri_eaf_hybrid_hydrogen_natural_gas_external",
    "dri_eaf_hydrogen_electrolyser",
    "dri_eaf_hydrogen_external",
    "dri_eaf_natural_gas_external",
)


# --------------------------------------------------------------------------- helpers
def parse_plant_name(name: object) -> tuple[str | None, str | None]:
    """Split ``P<id>_<route>`` into (physical_id, route)."""
    match = NAME_RE.match(str(name).strip())
    return (match.group(1), match.group(2)) if match else (None, None)


def demand_column_physical_id(column: object) -> str | None:
    """Return the physical plant ID represented by a steel-demand column.

    Source forecasts may use a bare physical ID (``P123``), a physical ID
    with ``_steel_demand``, or a route-specific generated demand header.
    """
    name = str(column).strip()
    base = name[: -len(DEMAND_SUFFIX)] if name.endswith(DEMAND_SUFFIX) else name
    if PHYSICAL_ID_RE.fullmatch(base):
        return base
    if name.endswith(DEMAND_SUFFIX):
        physical_id, _ = parse_plant_name(base)
        return physical_id
    return None


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
        columns = next(csv.reader(handle))
    for column in CO2_FACTOR_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


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
def steel_database_for_scenario(scenario: str) -> Path:
    """Return the explicitly configured plant workbook for a scenario family."""
    scenario_key = scenario.strip().casefold()
    try:
        return STEEL_DB_BY_SCENARIO[scenario_key]
    except KeyError as exc:
        supported = ", ".join(sorted(STEEL_DB_BY_SCENARIO))
        raise ValueError(
            f"No steel-plant workbook is configured for scenario family '{scenario}'. "
            f"Supported families: {supported}"
        ) from exc


def load_excel_plants(workbook: Path) -> pd.DataFrame:
    """Load and annotate one scenario-sized steel-plant workbook."""
    frame = pd.read_excel(workbook, sheet_name=EXCEL_SHEET, engine="openpyxl")
    required = {"name", "technology"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Steel-plant workbook '{workbook}' is missing required column(s): "
            + ", ".join(sorted(missing))
        )
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
    demand_columns: dict[str, str] = {}
    for column in frame.columns:
        if column == time_col:
            continue
        physical_id = demand_column_physical_id(column)
        if physical_id is None:
            continue
        if physical_id in demand_columns:
            previous = demand_columns[physical_id]
            raise ValueError(
                f"Duplicate steel-demand columns for {physical_id}: '{previous}' and '{column}'"
            )
        demand_columns[physical_id] = column

    demand_source_columns = set(demand_columns.values())
    market_cols = [
        column
        for column in frame.columns
        if column != time_col and column not in demand_source_columns
    ]
    demand_by_id = {physical_id: frame[column] for physical_id, column in demand_columns.items()}
    return time_col, market_cols, demand_by_id, frame


def add_missing_material_prices(
    frame: pd.DataFrame,
    market_cols: list[str],
    *,
    iron_ore_price: float,
    lime_price: float,
) -> list[str]:
    """Add explicit steel-material price assumptions when source data omit them."""
    updated_market_cols = list(market_cols)
    for column, value in (
        ("iron_ore_price", iron_ore_price),
        ("lime_price", lime_price),
    ):
        if column in frame.columns:
            continue
        frame[column] = value
        updated_market_cols.append(column)
    return updated_market_cols


# --------------------------------------------------------------------------- case writers
def _co2_factor_value(
    row: pd.Series,
    column: str,
    *,
    coal_factor: float,
    natural_gas_factor: float,
) -> str:
    """Return an explicit factor for applicable fuel-consuming steel rows."""

    technology = str(row.get("technology", "")).strip().lower()
    fuel_type = str(row.get("fuel_type", "")).strip().lower()
    if technology not in FUEL_TECHNOLOGIES:
        return ""

    required = (
        column == COAL_CO2_FACTOR_COLUMN
        and fuel_type == "coal"
        or column == NATURAL_GAS_CO2_FACTOR_COLUMN
        and fuel_type in {"natural_gas", HYBRID_HYDROGEN_NATURAL_GAS}
    )
    if not required:
        return ""

    source_value = fmt_cell(row.get(column))
    value = (
        float(source_value)
        if source_value
        else (coal_factor if column == COAL_CO2_FACTOR_COLUMN else natural_gas_factor)
    )
    if value <= 0:
        raise ValueError(f"{row.get('name', '<unnamed plant>')} requires a positive {column}")
    return fmt_cell(value)


def write_plants_csv(
    path: Path,
    route_rows: pd.DataFrame,
    template_cols: list[str],
    *,
    coal_co2_factor: float,
    natural_gas_co2_factor: float,
) -> None:
    out = [template_cols]
    for _, row in route_rows.iterrows():
        values: list[str] = []
        for column in template_cols:
            if column == "steel_demand":
                values.append("")
            elif column in CO2_FACTOR_COLUMNS:
                values.append(
                    _co2_factor_value(
                        row,
                        column,
                        coal_factor=coal_co2_factor,
                        natural_gas_factor=natural_gas_co2_factor,
                    )
                )
            else:
                values.append(fmt_cell(row.get(column)))
        out.append(values)
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
    parser.add_argument(
        "--route",
        choices=TECHNOLOGY_ROUTES,
        help="Only this configured technology route (e.g. dri_eaf_coal_external).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List cases without writing files.")
    parser.add_argument(
        "--coal-co2-factor",
        type=float,
        default=DEFAULT_COAL_CO2_FACTOR_T_PER_MWH,
        help="Coal factor written to applicable plants.csv rows [tCO2/MWh_fuel].",
    )
    parser.add_argument(
        "--natural-gas-co2-factor",
        type=float,
        default=DEFAULT_NATURAL_GAS_CO2_FACTOR_T_PER_MWH,
        help="Natural-gas factor written to applicable plants.csv rows [tCO2/MWh_fuel].",
    )
    parser.add_argument(
        "--iron-ore-price",
        type=float,
        default=DEFAULT_IRON_ORE_PRICE_EUR_PER_T,
        help="Fallback iron-ore price when the source forecast omits it [EUR/t].",
    )
    parser.add_argument(
        "--lime-price",
        type=float,
        default=DEFAULT_LIME_PRICE_EUR_PER_T,
        help="Fallback lime price when the source forecast omits it [EUR/t].",
    )
    args = parser.parse_args()

    if args.coal_co2_factor <= 0:
        parser.error("--coal-co2-factor must be positive")
    if args.natural_gas_co2_factor <= 0:
        parser.error("--natural-gas-co2-factor must be positive")
    if args.iron_ore_price < 0:
        parser.error("--iron-ore-price must be non-negative")
    if args.lime_price < 0:
        parser.error("--lime-price must be non-negative")

    if not EXTERNAL_OUTPUT_DIR.exists():
        raise SystemExit(f"External scenario dir not found: {EXTERNAL_OUTPUT_DIR}")
    if not STEEL_DB_DIR.exists():
        raise SystemExit(f"Scenario-sized steel plant directory not found: {STEEL_DB_DIR}")

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
    excel_cache: dict[Path, pd.DataFrame] = {}
    for folder in scenario_folders:
        forecast_path = folder / "forecasts_df.csv"
        if not forecast_path.exists():
            skipped.append(f"{folder.name}: no forecasts_df.csv")
            continue
        scenario, year = scenario_year(folder.name)
        try:
            workbook = steel_database_for_scenario(scenario)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not workbook.exists():
            raise SystemExit(
                f"Steel plant Excel for scenario family '{scenario}' not found: {workbook}"
            )
        if workbook not in excel_cache:
            excel = load_excel_plants(workbook)
            excel_routes = set(excel["_route"].dropna().unique())
            missing_excel_routes = sorted(set(TECHNOLOGY_ROUTES) - excel_routes)
            if missing_excel_routes:
                raise SystemExit(
                    f"Steel plant Excel '{workbook}' is missing configured technology routes: "
                    + ", ".join(missing_excel_routes)
                )
            excel_cache[workbook] = excel
        excel = excel_cache[workbook]
        print(f"{folder.name}: plant workbook = {workbook.name}")

        time_col, market_cols, demand_by_id, frame = load_scenario_forecast(folder)
        missing_material_prices = [
            column for column in ("iron_ore_price", "lime_price") if column not in frame.columns
        ]
        market_cols = add_missing_material_prices(
            frame,
            market_cols,
            iron_ore_price=args.iron_ore_price,
            lime_price=args.lime_price,
        )
        if missing_material_prices:
            supplied = ", ".join(missing_material_prices)
            print(f"{folder.name}: supplied fallback material price(s): {supplied}")

        routes = list(TECHNOLOGY_ROUTES)
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
            print(
                f"{'[dry-run] ' if args.dry_run else ''}{case_name}  "
                f"({len(plant_names)} plants, {len(route_rows)} rows)"
            )
            if args.dry_run:
                written += 1
                continue

            case_dir.mkdir(parents=True, exist_ok=True)
            write_config(case_dir / "config.yaml", case_name, year, config_text)
            write_plants_csv(
                case_dir / "plants.csv",
                route_rows,
                template_cols,
                coal_co2_factor=args.coal_co2_factor,
                natural_gas_co2_factor=args.natural_gas_co2_factor,
            )
            write_additional_charges(case_dir / "additional_charges.csv", plant_names, charge_specs)
            write_forecasts(
                case_dir / "forecasts_df.csv",
                time_col,
                market_cols,
                demand_by_id,
                plant_names,
                frame,
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
