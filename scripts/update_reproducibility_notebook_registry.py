"""Point the final-study notebook at the dedicated reproducible case inputs."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "building_use_case_analysis.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Expected notebook text was not found: {old}")
    return source.replace(old, new, 1)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])
    source = replace_once(
        source,
        "| I02b first tested export-onset tariff | I01 | 3.15 THB/kWh tariff; same 5 kW limit; 1.437 MWh/year export in the completed run |",
        "| I02b first tested export-onset tariff | I01 | Dedicated 3.15 THB/kWh tariff input; same 5 kW limit; canonical annual rerun pending |",
    )
    source = replace_once(
        source,
        "| I07 reference: PV + unidirectional charging | I01 | Adds rooftop PV, with no scheduled bus discharge or feed-in; isolates the direct PV value |",
        "| I07 reference: PV + unidirectional charging | I01 | Dedicated physically unidirectional input; no bus discharge or feed-in; isolates direct PV value |",
    )
    notebook["cells"][0]["source"] = source.splitlines(keepends=True)

    for cell in notebook["cells"]:
        cell_source = "".join(cell.get("source", []))
        if "CORE_CASES = {" not in cell_source:
            continue
        cell_source = replace_once(
            cell_source,
            "'I07 reference': {'input': 'building_v2b_pv_self_consumption_annual', 'study_case': 'building_v2b_pv_self_consumption', 'output': 'v2b_rooftop_pv'},",
            "'I07 reference': {'input': 'building_i07_reference_annual', 'study_case': 'building_i07_reference', 'output': 'i07_reference'},",
        )
        cell_source = replace_once(
            cell_source,
            "'I02b economically viable V2G export': {'input': 'building_v2g_tariff_sweep_annual', 'study_case': 'building_v2g_sweep_plus_1000', 'output': 'tariff_plus_1000'},",
            "'I02b economically viable V2G export': {'input': 'building_v2g_export_onset_3150_annual', 'study_case': 'building_v2g_export_onset_3150', 'output': 'i02_export_onset_3150'},",
        )
        cell["source"] = cell_source.splitlines(keepends=True)
        break
    else:
        raise ValueError("CORE_CASES registry was not found")

    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
