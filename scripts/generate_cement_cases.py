# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate scenario-, year-, and route-specific cement FLEXIMOD input cases.

For each ASSUME cement scenario/year, the generator expands ``market_meta.csv`` to
the model's 15-minute forecasts inside this workflow, merges the matching cement
clinker-demand workbook, and writes one case per cement route. Every physical plant
uses the maximum demand across the four model years as its installed design rate.

The generated folders are self-contained: ``config.yaml``, ``plants.csv``,
``additional_charges.csv``, and ``forecasts_df.csv``. The latter is derived from
proprietary scenario data and is git-ignored.

Usage:
    python scripts/generate_cement_cases.py --dry-run
    python scripts/generate_cement_cases.py --scenario aktuellepolitiken_2030 --route R1
    python scripts/generate_cement_cases.py --rdf-price 25.0
"""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path

import pandas as pd

try:
    from scripts.create_forecasts_from_assume_market_meta import (
        convert_assume_market_meta_to_forecasts,
    )
except ModuleNotFoundError:  # Direct ``python scripts/generate_cement_cases.py`` execution.
    from create_forecasts_from_assume_market_meta import convert_assume_market_meta_to_forecasts

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DATA_DIR = Path(
    r"C:/Users/khm/ownCloud (2)/Dropbox/Ph.D/My publications/Journal paper/4/Data"
)
ASSUME_OUTPUT_DIR = EXTERNAL_DATA_DIR / "Assume" / "Output" / "cement"
ASSUME_INPUT_DIR = EXTERNAL_DATA_DIR / "Assume" / "Input" / "cement"
CEMENT_DATABASE_DIR = EXTERNAL_DATA_DIR / "cement plant database"
CEMENT_CASES_DIR = CEMENT_DATABASE_DIR / "plant case"

DEMAND_WORKBOOKS = {
    "aktuellepolitiken": CEMENT_DATABASE_DIR / "AktuellePolitiken_demand_series_15min.xlsx",
    "hohenachfrage": CEMENT_DATABASE_DIR / "HoheNachfrage_demand_series_15min.xlsx",
    "niedrigenachfrage": CEMENT_DATABASE_DIR / "Niedrigenachfrage_demand_series_15min.xlsx",
    "fokush2": (
        CEMENT_DATABASE_DIR
        / "FokusH2_FokusStrom_Technologiemix_load_profile_15min_timestep_demand.xlsx"
    ),
    "fokusstrom": (
        CEMENT_DATABASE_DIR
        / "FokusH2_FokusStrom_Technologiemix_load_profile_15min_timestep_demand.xlsx"
    ),
    "technologiemix": (
        CEMENT_DATABASE_DIR
        / "FokusH2_FokusStrom_Technologiemix_load_profile_15min_timestep_demand.xlsx"
    ),
}

TEMPLATE_CASE = REPO_ROOT / "data" / "input" / "cement_plant_DE"
ROUTE_TEMPLATE = TEMPLATE_CASE / "cement_route_parameter_template.xlsx"
TEMPLATE_SHEET = "plants_template"
TARGET_INPUT_DIR = CEMENT_CASES_DIR
TEMPLATE_CASE_NAME = "cement_plant_DE"
TEMPLATE_YEAR = "2030"
DEMAND_SUFFIX = "_clinker_demand"
ROUTE_METADATA_COLUMNS = ("route_id", "route_description")
HYBRID_STRATEGY_CEMENT_ROUTES = frozenset({"R6"})
ELECTRIFIED_CEMENT_STRATEGY = "electrified_cement"
HYBRID_CEMENT_STRATEGY = "hybrid_strategy_cement"
LICENSE_TEXT = (
    "SPDX-FileCopyrightText: FLEXIMOD Developers\n\nSPDX-License-Identifier: AGPL-3.0-or-later\n"
)

_BASE_CASE_SUFFIX_RE = re.compile(r"_base_case_\d+$")


def scenario_year(folder_name: str) -> tuple[str, str]:
    """Split an ASSUME folder name into its scenario family and model year."""
    stripped = _BASE_CASE_SUFFIX_RE.sub("", folder_name)
    scenario, separator, year = stripped.rpartition("_")
    if not separator or not year.isdigit():
        raise ValueError(f"Cannot infer scenario/year from '{folder_name}'")
    return scenario, year


def demand_workbook_for_scenario(scenario: str) -> Path:
    """Return the explicit cement demand workbook for a scenario family."""
    try:
        return DEMAND_WORKBOOKS[scenario.casefold()]
    except KeyError as exc:
        supported = ", ".join(sorted(DEMAND_WORKBOOKS))
        raise ValueError(
            f"No cement demand workbook is configured for scenario family '{scenario}'. "
            f"Supported families: {supported}"
        ) from exc


def load_route_template(path: Path = ROUTE_TEMPLATE) -> pd.DataFrame:
    """Load the route master, excluding optional components and blank rows."""
    rows = pd.read_excel(path, sheet_name=TEMPLATE_SHEET, engine="openpyxl", dtype=object)
    required = {"route_id", "technology", "fuel_type"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Route template is missing column(s): {', '.join(sorted(missing))}")
    rows = rows[rows["route_id"].notna() & rows["technology"].notna()].copy()
    rows = rows[rows["route_id"].astype(str).str.upper() != "OPTIONAL"].copy()
    rows["route_id"] = rows["route_id"].astype(str)
    rows["technology"] = rows["technology"].astype(str)
    rows["fuel_type"] = rows["fuel_type"].fillna("").astype(str)
    return rows.reset_index(drop=True)


def route_ids(rows: pd.DataFrame) -> list[str]:
    """Route IDs in the worksheet's modeler-defined order."""
    return list(dict.fromkeys(rows["route_id"].tolist()))


def strategy_for_route(route: str) -> str:
    """Return the cross-market strategy selected by a cement technology route.

    R6 continuously substitutes electric heat for its fossil fuel blend, so it uses the
    strike-price-based hybrid strategy. All remaining routes use the exact joint
    DA/aFRR capacity/aFRR energy optimisation.
    """
    if route in HYBRID_STRATEGY_CEMENT_ROUTES:
        return HYBRID_CEMENT_STRATEGY
    return ELECTRIFIED_CEMENT_STRATEGY


def load_design_clinker_rates(workbook: Path) -> dict[str, float]:
    """Return the maximum hourly clinker rate for every plant across all year sheets."""
    peaks: dict[str, float] = {}
    expected_columns: list[str] | None = None
    excel = pd.ExcelFile(workbook, engine="openpyxl")
    for year in excel.sheet_names:
        demand = pd.read_excel(excel, sheet_name=year, engine="openpyxl")
        if "datetime" not in demand.columns:
            raise ValueError(f"{workbook.name} sheet '{year}' is missing 'datetime'")
        columns = [str(column) for column in demand.columns if column != "datetime"]
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(f"{workbook.name} sheet '{year}' has different plant columns")
        for plant_id in columns:
            values = pd.to_numeric(demand[plant_id], errors="coerce")
            if values.isna().any():
                raise ValueError(f"{workbook.name}.{year}.{plant_id} contains non-numeric demand")
            peaks[plant_id] = max(peaks.get(plant_id, 0.0), float(values.max()))
    if not peaks:
        raise ValueError(f"{workbook.name} contains no plant demand columns")
    return {plant_id: demand_per_step / 0.25 for plant_id, demand_per_step in peaks.items()}


def _number(row: pd.Series, column: str) -> float:
    value = row.get(column)
    if pd.isna(value) or str(value).strip() == "" or str(value).startswith("<required"):
        raise ValueError(
            f"Route {row['route_id']} technology {row['technology']} requires '{column}'"
        )
    return float(value)


def _fuel_type(row: pd.Series) -> str:
    return str(row.get("fuel_type", "")).strip().lower()


def _stage_heat_rating_mw(row: pd.Series, clinker_rate_t_per_h: float) -> float:
    specific_heat = _number(row, "specific_heat_demand")
    if row["technology"] == "preheater":
        return clinker_rate_t_per_h * _number(row, "raw_meal_to_clinker_ratio") * specific_heat
    return clinker_rate_t_per_h * specific_heat


def _stage_max_power_mw(
    row: pd.Series,
    clinker_rate_t_per_h: float,
    max_heat_out_mw: float,
) -> float:
    if _fuel_type(row) not in {"electricity", "hybrid_electricity_fossil"}:
        return 0.0
    if row["technology"] == "preheater":
        recovered_heat_mw = (
            clinker_rate_t_per_h
            * _number(row, "waste_heat_per_t_clinker")
            * _number(row, "waste_heat_utilization_efficiency")
        )
        max_heat_out_mw = max(0.0, max_heat_out_mw - recovered_heat_mw)
    return max_heat_out_mw / _number(row, "eta_electric")


def _electrolyser_max_power_mw(route_rows: pd.DataFrame, clinker_rate_t_per_h: float) -> float:
    hydrogen_input_mw = 0.0
    efficiency: float | None = None
    for _, row in route_rows.iterrows():
        technology = str(row["technology"])
        if technology == "electrolyser":
            efficiency = _number(row, "efficiency")
            continue
        if technology not in {
            "preheater",
            "simple_calciner",
            "leilac_calciner",
            "oxyfuel_calciner",
            "simple_kiln",
            "oxyfuel_kiln",
        }:
            continue
        if _fuel_type(row) != "hydrogen":
            continue
        max_heat = _stage_heat_rating_mw(row, clinker_rate_t_per_h)
        if technology == "preheater":
            recovered_heat_mw = (
                clinker_rate_t_per_h
                * _number(row, "waste_heat_per_t_clinker")
                * _number(row, "waste_heat_utilization_efficiency")
            )
            max_heat = max(0.0, max_heat - recovered_heat_mw)
        hydrogen_input_mw += max_heat / _number(row, "eta_fossil")
    if efficiency is None:
        return 0.0
    return hydrogen_input_mw / efficiency


def size_route_for_plant(
    route_rows: pd.DataFrame,
    *,
    plant_name: str,
    clinker_rate_t_per_h: float,
    node: str,
) -> pd.DataFrame:
    """Set a route's installed stage ratings for one physical cement plant."""
    sized = route_rows.copy()
    for column in (
        "max_heat_out",
        "max_power",
        "min_heat_out",
        "min_power",
        "ramp_up",
        "ramp_down",
    ):
        sized[column] = sized[column].astype(object)
    stage_technologies = {
        "preheater",
        "simple_calciner",
        "leilac_calciner",
        "oxyfuel_calciner",
        "simple_kiln",
        "oxyfuel_kiln",
    }
    for index, row in sized.iterrows():
        technology = str(row["technology"])
        sized.at[index, "name"] = plant_name
        sized.at[index, "node"] = node
        if technology not in stage_technologies:
            continue
        max_heat_out_mw = _stage_heat_rating_mw(row, clinker_rate_t_per_h)
        sized.at[index, "max_heat_out"] = max_heat_out_mw
        sized.at[index, "max_power"] = _stage_max_power_mw(
            row, clinker_rate_t_per_h, max_heat_out_mw
        )
        sized.at[index, "min_heat_out"] = 0.0
        sized.at[index, "min_power"] = 0.0
        sized.at[index, "ramp_up"] = max_heat_out_mw
        sized.at[index, "ramp_down"] = max_heat_out_mw

    electrolyser_power_mw = _electrolyser_max_power_mw(sized, clinker_rate_t_per_h)
    for index, row in sized.iterrows():
        if row["technology"] != "electrolyser":
            continue
        sized.at[index, "max_power"] = electrolyser_power_mw
        sized.at[index, "min_power"] = 0.0
        sized.at[index, "ramp_up"] = electrolyser_power_mw
        sized.at[index, "ramp_down"] = electrolyser_power_mw
    return sized


def fmt_cell(value: object) -> str:
    """Format a CSV cell, preserving deliberate blanks."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def write_license(path: Path) -> None:
    path.write_text(LICENSE_TEXT, encoding="utf-8")


def write_plants_csv(path: Path, route_rows: pd.DataFrame) -> None:
    columns = [column for column in route_rows.columns if column not in ROUTE_METADATA_COLUMNS]
    rows = [columns]
    rows.extend(
        [[fmt_cell(row.get(column)) for column in columns] for _, row in route_rows.iterrows()]
    )
    write_csv(path, rows)


def load_template_charge_specs() -> list[tuple[str, str, str]]:
    """Use one template value per charge component for every generated cement plant."""
    frame = pd.read_csv(TEMPLATE_CASE / "additional_charges.csv", dtype=str, keep_default_na=False)
    value_column = frame.columns[2]
    return [(row["component"], row["unit"], row[value_column]) for _, row in frame.iterrows()]


def write_additional_charges(
    path: Path,
    plant_names: list[str],
    charge_specs: list[tuple[str, str, str]],
) -> None:
    rows = [["component", "unit", *plant_names]]
    rows.extend(
        [
            [component, unit, *([value] * len(plant_names))]
            for component, unit, value in charge_specs
        ]
    )
    write_csv(path, rows)


def write_forecasts(
    path: Path,
    forecasts: pd.DataFrame,
    plant_names_by_id: dict[str, str],
) -> None:
    """Re-key bare physical-ID demand columns onto generated cement plant names."""
    if "datetime" not in forecasts.columns:
        raise ValueError("Converted forecasts are missing 'datetime'")
    demand_columns = set(plant_names_by_id)
    market_columns = [
        column
        for column in forecasts.columns
        if column != "datetime" and column not in demand_columns
    ]
    output = forecasts[["datetime", *market_columns]].copy()
    for plant_id, plant_name in plant_names_by_id.items():
        if plant_id not in forecasts.columns:
            raise ValueError(f"Converted forecasts are missing demand for {plant_id}")
        output[f"{plant_name}{DEMAND_SUFFIX}"] = forecasts[plant_id].to_numpy()
    output.to_csv(path, index=False)


def write_config(path: Path, case_name: str, year: str, template_text: str, *, route: str) -> None:
    text = template_text.replace(TEMPLATE_CASE_NAME, case_name).replace(TEMPLATE_YEAR, year)
    strategy_name = strategy_for_route(route)
    template_strategy = "      name: electrified_cement\n"
    if template_strategy not in text:
        raise ValueError(f"Cement config strategy target not found: {template_strategy!r}")
    text = text.replace(template_strategy, f"      name: {strategy_name}\n", 1)
    path.write_text(text, encoding="utf-8")


def fuel_prices_path(folder: Path) -> Path:
    """Locate the ASSUME input fuel-price file matching an output folder."""
    scenario, year = scenario_year(folder.name)
    return ASSUME_INPUT_DIR / f"{scenario}_{year}" / "fuel_prices_df.csv"


def convert_forecasts(
    folder: Path,
    *,
    scenario: str,
    year: str,
    demand_workbook: Path,
) -> pd.DataFrame:
    """Expand ASSUME markets and merge the cement demand workbook within this generator."""
    source = folder / "market_meta.csv"
    with tempfile.TemporaryDirectory(prefix="fleximod_cement_forecasts_") as temporary:
        output_path = Path(temporary) / "forecasts_df.csv"
        convert_assume_market_meta_to_forecasts(
            source,
            output_path,
            timestep_minutes=15,
            activation_unit="MW",
            fuel_prices_path=fuel_prices_path(folder),
            include_fuel_prices=True,
            include_demand=True,
            demand_scenario=scenario,
            demand_year=year,
            demand_workbook_path=demand_workbook,
            pad_to_year_end=True,
        )
        return pd.read_csv(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="Only one scenario_year, e.g. aktuellepolitiken_2030.")
    parser.add_argument("--route", help="Only one route ID, e.g. R1 or R11b.")
    parser.add_argument("--dry-run", action="store_true", help="List cases without writing them.")
    parser.add_argument(
        "--rdf-price",
        type=float,
        default=0.0,
        help=("Constant EUR/MWh_th RDF price written to the R2 forecast column (default: 0.0)."),
    )
    parser.add_argument(
        "--node", default="south", help="Network node assigned to generated plants."
    )
    parser.add_argument("--output-dir", type=Path, default=TARGET_INPUT_DIR)
    parser.add_argument("--assume-output-dir", type=Path, default=ASSUME_OUTPUT_DIR)
    parser.add_argument("--route-template", type=Path, default=ROUTE_TEMPLATE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rdf_price is not None and args.rdf_price < 0.0:
        raise SystemExit("--rdf-price must be non-negative")
    if not args.assume_output_dir.exists():
        raise SystemExit(f"ASSUME cement output directory not found: {args.assume_output_dir}")
    if not args.route_template.exists():
        raise SystemExit(f"Cement route template not found: {args.route_template}")

    template = load_route_template(args.route_template)
    available_routes = route_ids(template)
    if args.route and args.route not in available_routes:
        raise SystemExit(f"Unknown route '{args.route}'. Available: {', '.join(available_routes)}")
    selected_routes = [args.route] if args.route else available_routes
    charge_specs = load_template_charge_specs()
    config_text = (TEMPLATE_CASE / "config.yaml").read_text(encoding="utf-8")
    folders = sorted(path for path in args.assume_output_dir.iterdir() if path.is_dir())
    if args.scenario:
        folders = [
            folder for folder in folders if "_".join(scenario_year(folder.name)) == args.scenario
        ]
    written = 0
    skipped: list[str] = []
    design_cache: dict[Path, dict[str, float]] = {}
    for folder in folders:
        market_meta = folder / "market_meta.csv"
        if not market_meta.exists():
            skipped.append(f"{folder.name}: no market_meta.csv")
            continue
        scenario, year = scenario_year(folder.name)
        try:
            workbook = demand_workbook_for_scenario(scenario)
        except ValueError as exc:
            skipped.append(f"{folder.name}: {exc}")
            continue
        if not workbook.exists():
            skipped.append(f"{folder.name}: demand workbook not found: {workbook}")
            continue
        if workbook not in design_cache:
            design_cache[workbook] = load_design_clinker_rates(workbook)
        design_rates = design_cache[workbook]
        print(f"{folder.name}: demand workbook = {workbook.name}")

        if args.dry_run:
            for route in selected_routes:
                print(f"[dry-run] {scenario}_{year}_{route} ({len(design_rates)} plants)")
                written += 1
            continue

        forecasts = convert_forecasts(
            folder,
            scenario=scenario,
            year=year,
            demand_workbook=workbook,
        )
        for route in selected_routes:
            route_rows = template[template["route_id"] == route]
            plant_names_by_id = {
                plant_id: f"{plant_id}_{route}" for plant_id in sorted(design_rates)
            }
            plant_names = list(plant_names_by_id.values())
            if route == "R2":
                forecasts = forecasts.copy()
                forecasts["rdf_price"] = args.rdf_price
            case_name = f"{scenario}_{year}_{route}"
            case_dir = args.output_dir / case_name
            print(
                f"{case_name} ({len(plant_names)} plants, {len(route_rows)} technology rows each)"
            )
            case_dir.mkdir(parents=True, exist_ok=True)
            all_rows = pd.concat(
                [
                    size_route_for_plant(
                        route_rows,
                        plant_name=plant_names_by_id[plant_id],
                        clinker_rate_t_per_h=design_rates[plant_id],
                        node=args.node,
                    )
                    for plant_id in sorted(design_rates)
                ],
                ignore_index=True,
            )
            write_config(case_dir / "config.yaml", case_name, year, config_text, route=route)
            write_plants_csv(case_dir / "plants.csv", all_rows)
            write_additional_charges(case_dir / "additional_charges.csv", plant_names, charge_specs)
            write_forecasts(case_dir / "forecasts_df.csv", forecasts, plant_names_by_id)
            for filename in ("plants.csv", "additional_charges.csv", "forecasts_df.csv"):
                write_license(case_dir / f"{filename}.license")
            written += 1

    print(f"\n{'Would generate' if args.dry_run else 'Generated'} {written} case(s).")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
