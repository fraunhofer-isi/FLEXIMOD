"""Remove dashboard and retired-sensitivity cells superseded by family analysis."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "building_use_case_analysis.ipynb"

REMOVE_PREFIXES = (
    "# Complete peak-demand family, including the no-export V2G case.",
    "# Complete renewable-responsive family, including the no-export case.",
    "# Complete rooftop-PV family, including the higher-tariff sensitivity.",
    "## Appendix A. Cross-case annual comparison",
    "### Cross-case grid and renewable indicators",
    "# Power-system peak-demand import-relief indicators are shown as absolute grid exchange",
    "def concise_case_label(name):",
    "## Appendix D. Supplementary tariff and export-limit sensitivities",
    "def sensitivity_frame(cases: dict, parameter: str)",
)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    retained: list[dict] = []
    removed: list[str] = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        matched = next((prefix for prefix in REMOVE_PREFIXES if source.startswith(prefix)), None)
        if matched is None:
            retained.append(cell)
        else:
            removed.append(matched)
    if len(removed) != len(REMOVE_PREFIXES):
        missing = set(REMOVE_PREFIXES) - set(removed)
        raise ValueError(f"Expected superseded cells not found: {sorted(missing)}")

    for cell in retained:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])
        source = source.replace("## Appendix B. Completed-case results", "## Appendix A. Completed-case results")
        source = source.replace("## Appendix C. Site, route, and roof context", "## Appendix B. Site, route, and roof context")
        cell["source"] = source.splitlines(keepends=True)

    notebook["cells"] = retained
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Removed {len(removed)} superseded notebook cells; retained {len(retained)} cells.")


if __name__ == "__main__":
    main()
