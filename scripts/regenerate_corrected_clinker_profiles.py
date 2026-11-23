"""Refresh 2030 for two corrected AURIS cement pathways in all six worlds."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import os

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv

ROOT = Path(__file__).resolve().parents[1]
CORRECTED = ROOT / "data/input/archive/cement_cases_investment/corrected_clinker_technology_pathways"
RESULTS = ROOT / "data/output/cement_results"
OUTPUT = ROOT / "data/output/auris_inputs_15min/cement"
CASES = ("FokusH2_risk_neutral", "WorstCaseMean_risk_averse")
WORLDS = ("aktuellepolitiken", "fokusstrom", "fokusH2", "technologiemix", "hohenachfrage", "niedrigenachfrage")
MAPPING = {
    "DA_position_MWh": "da_electricity_MWh",
    "day_ahead_price_EUR_per_MWh": "da_bid_price_EUR_per_MWh",
    "afrr_energy_activated_MWh": "afrr_neg_energy_MWh",
    "afrr_energy_bid_MWh": "afrr_neg_bid_MWh",
    "afrr_energy_bid_price_EUR_per_MWh": "afrr_neg_bid_price_EUR_per_MWh",
    "afrr_energy_price_EUR_per_MWh": "afrr_neg_clearing_price_EUR_per_MWh",
}
BACKUP = OUTPUT.parent / ("cement_corrected_2030_backup_" + datetime.now().strftime("%Y%m%dT%H%M%S"))


def regenerate(world):
    pathways = {case: pd.read_csv(CORRECTED / case / "technology_pathway.csv").query("year == 2030") for case in CASES}
    for case, pathway in pathways.items():
        if len(pathway) != 32 or pathway.plant_id.duplicated().any():
            raise ValueError(f"Invalid 2030 plant list for {case}")
    routes = sorted(set().union(*(set(p.active_technology_id) for p in pathways.values())))
    route_frames = {}
    for route in routes:
        folder = RESULTS / f"{world}_2030_{route}_electrified_cement_rule_based"
        columns = ["datetime", "plant_name", *MAPPING]
        with pa.input_stream(str(folder / "dispatch_results.csv.zst"), compression="zstd") as stream:
            frame = csv.read_csv(stream, convert_options=csv.ConvertOptions(include_columns=columns)).to_pandas()
        frame = frame.rename(columns=MAPPING)
        frame["plant_id"] = frame.pop("plant_name").str.removesuffix("_" + route)
        frame["active_technology_id"] = route
        route_frames[route] = frame
    reports = []
    for case, pathway in pathways.items():
        target = OUTPUT / f"{case}__{world}.csv.zst"
        with pa.input_stream(str(target), compression="zstd") as stream:
            schema = csv.open_csv(stream).schema
        parts = []
        for route, plants in pathway.groupby("active_technology_id"):
            frame = route_frames[route]
            parts.append(frame[frame.plant_id.isin(plants.plant_id)].copy())
        fresh = pd.concat(parts, ignore_index=True)
        fresh["case"], fresh["macro_scenario"], fresh["year"] = case, world, 2030
        fresh = fresh.sort_values(["plant_id", "datetime"])
        if len(fresh) != 32 * 34944 or fresh.duplicated(["plant_id", "datetime"]).any():
            raise ValueError(f"Incomplete or duplicate 2030 rows for {case}/{world}")
        for row in pathway.itertuples():
            actual = fresh[fresh.plant_id == row.plant_id]
            if set(actual.active_technology_id) != {row.active_technology_id}:
                raise ValueError(f"Route mismatch: {row.plant_id}")
        for route, plants in pathway.groupby("active_technology_id"):
            summary = pd.read_csv(RESULTS / f"{world}_2030_{route}_electrified_cement_rule_based/summary_indicators.csv.zst")
            selected = summary[summary.plant_name.isin(plants.plant_id + "_" + route)]
            actual = fresh[fresh.active_technology_id == route]
            for source, exported in (("total_DA_electricity_MWh", "da_electricity_MWh"), ("total_afrr_energy_activated_MWh", "afrr_neg_energy_MWh")):
                if abs(selected[source].sum() - actual[exported].sum()) > max(0.001, abs(selected[source].sum()) * 1e-9):
                    raise ValueError(f"Annual energy mismatch for {case}/{world}/{route}/{source}")
        table = pa.Table.from_pandas(fresh[schema.names], schema=schema, preserve_index=False)
        temporary = target.with_suffix(".zst.corrected.tmp")
        counts = {2030: len(fresh)}
        with pa.output_stream(str(temporary), compression="zstd") as stream:
            with csv.CSVWriter(stream, schema) as writer:
                writer.write_table(table)
                with pa.input_stream(str(target), compression="zstd") as old:
                    reader = csv.open_csv(old, convert_options=csv.ConvertOptions(column_types=schema))
                    for batch in reader:
                        keep = pa.Table.from_batches([batch]).filter(pc.not_equal(batch.column("year"), 2030))
                        for year in pc.unique(keep["year"]).to_pylist():
                            counts[year] = counts.get(year, 0) + pc.sum(pc.cast(pc.equal(keep["year"], year), pa.int64())).as_py()
                        writer.write_table(keep)
        # Read the completed compressed file back before replacing its predecessor.
        observed = {}
        with pa.input_stream(str(temporary), compression="zstd") as stream:
            reader = csv.open_csv(stream, convert_options=csv.ConvertOptions(column_types=schema))
            for batch in reader:
                for year in pc.unique(batch.column("year")).to_pylist():
                    observed[year] = observed.get(year, 0) + pc.sum(pc.cast(pc.equal(batch.column("year"), year), pa.int64())).as_py()
        if observed != counts:
            raise ValueError(f"Readback counts differ for {target}: {observed} != {counts}")
        BACKUP.mkdir(parents=True, exist_ok=True)
        os.replace(target, BACKUP / target.name)
        os.replace(temporary, target)
        reports.append({"file": target.name, "corrected_2030_rows": len(fresh), "total_rows": sum(counts.values()), "years": str(counts), "da_2030_MWh": fresh.da_electricity_MWh.sum(), "afrr_2030_MWh": fresh.afrr_neg_energy_MWh.sum()})
        print(f"Published {target.name}: {counts}", flush=True)
    return reports


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        report = [row for rows in pool.map(regenerate, WORLDS) for row in rows]
    pd.DataFrame(report).to_csv(BACKUP / "regeneration_report.csv", index=False)
    print(f"Completed {len(report)} files. Original files and report: {BACKUP}", flush=True)
