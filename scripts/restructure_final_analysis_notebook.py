"""Give the final-study notebook the workshop's family-by-family decision flow."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "building_use_case_analysis.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def replace_cell(cells: list[dict], starts_with: str, replacement: str) -> None:
    for cell in cells:
        if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith(starts_with):
            cell["source"] = replacement.splitlines(keepends=True)
            return
    raise ValueError(f"Markdown cell not found: {starts_with}")


def insert_after(cells: list[dict], starts_with: str, new_cell: dict) -> None:
    for index, cell in enumerate(cells):
        if "".join(cell.get("source", [])).startswith(starts_with):
            cells.insert(index + 1, new_cell)
            return
    raise ValueError(f"Cell not found: {starts_with}")


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    setup = "".join(cells[2]["source"])
    setup = setup.replace("import matplotlib.pyplot as plt\n", "import matplotlib.pyplot as plt\nimport seaborn as sns\n")
    setup = setup.replace(
        "from IPython.display import display\n",
        "from IPython.display import display\n\n"
        "sns.set_theme(context='notebook', style='whitegrid', font_scale=1.0)\n"
        "plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300, 'axes.titleweight': 'semibold'})\n",
    )
    cells[2]["source"] = setup.splitlines(keepends=True)

    replace_cell(
        cells,
        "## 4. Economic V2G",
        """## 4. Economic V2G: feed-in tariff and export onset

### Comparison and decision framing

**Reference:** V1G managed charging. **Cases:** I02a applies the current 2.20 THB/kWh feed-in tariff and the 5 kW regulatory limit; I02b holds the same physical limit but applies the first tested export-onset tariff, 3.15 THB/kWh.

This is an energy-arbitrage test. It asks whether the depot operator can buy, store and later feed electricity back to the system without compromising passenger service. The annual profile shows the direction and timing of electricity exchange; the annual charts separate extra imported electricity, feed-in revenue and the resulting operating-cost change.
""",
    )
    replace_cell(
        cells,
        "**Interpretation.** The current feed-in tariff",
        """### Benefit, cost and stakeholder conclusion

| Perspective | What this comparison tests | Resulting implication |
| --- | --- | --- |
| Bus-depot operator | Whether feed-in revenue covers purchased energy and conversion losses | At the current tariff, least-cost dispatch does not support routine bus discharge. |
| Power-system buyer / utility | Whether exported bus energy can be procured as an energy product | A payment above the depot's marginal cost is necessary, but this study does not set a tariff. |
| ERC / policy maker | The settlement and export rule needed for a pilot | A pilot would need interval metering, a defined feed-in product and a limit consistent with the connection agreement. |

**Conclusion.** The 3.15 THB/kWh case is a threshold sensitivity, not a tariff recommendation. Its dedicated annual rerun is intentionally marked pending; until then, the robust finding is that the current 2.20 THB/kWh feed-in tariff does not create an energy-arbitrage business case.
""",
    )

    replace_cell(
        cells,
        "## 5. Power-system peak-demand import reduction",
        """## 5. Power-system peak-demand import reduction

### Comparison and decision framing

**Reference:** V1G managed charging. **Cases:** I04a applies the current 5 kW regulatory export limit; I04b is a 450 kW charger-and-fleet technical sensitivity.

The service is reduced depot electricity consumption during selected power-system peak-demand intervals—not relief of a measured feeder or substation. The profile compares V1G with the technical sensitivity on a selected day. The annual results quantify energy reduction during the selected intervals, short-duration power reduction and the associated operator cost gap.
""",
    )
    replace_cell(
        cells,
        "**Interpretation.** The peak-demand signal",
        """### Benefit, cost and stakeholder conclusion

| Perspective | Potential benefit | Condition before it can be claimed |
| --- | --- | --- |
| Depot operator | A defined flexibility payment can compensate controlled charging or discharge | Mobility limits, connection capability and payment must be contractually defined. |
| PEA / MEA / EGAT | Lower coincident depot demand at a time when the power system is under pressure | The depot location and response must be shown to coincide with an operational need. |
| ERC / EPPO | Evidence for a targeted flexibility-pilot design | The technical 450 kW result is not an approved export capacity or a present market product. |

**Conclusion.** The cases demonstrate a conditional depot contribution to power-system peak demand. They do not demonstrate congestion relief at a particular local network asset; the reported cost gap is an operator-compensation indicator, not a proposed tariff.
""",
    )

    replace_cell(
        cells,
        "## 6. Renewable-responsive operation",
        """## 6. Renewable-responsive operation

### Comparison and decision framing

**Reference:** V1G managed charging. **Cases:** I06a uses the current 5 kW export limit; I06b is a 450 kW technical sensitivity.

Charging is preferentially moved into intervals with a higher value of the common 2024 renewable-availability signal; discharge or export is permitted in lower-availability intervals. The profile shows the signal, bus action and SOC on one representative day. The annual charts report the amount of charging in higher-availability intervals, lower-availability depot-to-grid energy and the operating-cost effect.
""",
    )
    replace_cell(
        cells,
        "**Interpretation.** Renewable-responsive dispatch",
        """### Benefit, cost and stakeholder conclusion

| Perspective | Potential benefit | Requirement |
| --- | --- | --- |
| Depot operator | Participation in a time-specific flexibility service | Compensation must cover the additional traction operating cost. |
| Renewable producer / system operator | Demand can be shifted toward higher renewable availability | A published, measurable signal and clear response window are needed. |
| ERC / EPPO | A basis for testing settlement of renewable-responsive demand | The proxy is not proof of traced renewable electricity and cannot be used as a renewable certificate. |

**Conclusion.** The result demonstrates controllable timing, not physical delivery of renewable electrons. A Thai service would need a transparent signal, verification method and payment rule; the cost comparison shows why timing value does not automatically accrue to the depot operator.
""",
    )

    replace_cell(
        cells,
        "## 7. Rooftop PV operation and power-system peak-demand management",
        """## 7. Rooftop PV operation and power-system peak-demand management

### Comparison and decision framing

**Reference:** I07 reference—rooftop PV with physically unidirectional bus charging and no feed-in. **Cases:** I07b adds the current 2.20 THB/kWh feed-in tariff and 5 kW limit; I07d adds peak-demand-aware charging timing within the same tariff and limit.

The representative profile separates PV availability, PV used by the depot, grid exchange and any bus discharge. The annual comparison isolates direct PV value first, then the incremental effect of limited surplus-PV export and charging timing. This avoids calling ordinary PV self-consumption a bus-discharge service.
""",
    )
    replace_cell(
        cells,
        "**Interpretation.** PV creates immediate operating value.",
        """### Benefit, cost and stakeholder conclusion

| Perspective | Immediate value | Limit of the current arrangement |
| --- | --- | --- |
| Depot / PV owner | On-site PV can reduce traction-grid imports and curtailment | PV investment cost is outside this operating-cost study. |
| Bus operator | Charging can be scheduled around PV output and power-system peak-demand periods | The 5 kW limit gives little incentive for material bus discharge. |
| Utility / system operator | Lower coincident depot imports can be useful where it matches an operational need | A bus-discharge service needs a defined product, connection arrangement and compensation. |

**Conclusion.** PV has an immediate depot-level value even without bus discharge. The dedicated I07 reference annual rerun is pending, so its archived result is not presented as final; the core policy finding remains that surplus-PV export and charge timing are not the same as a material bus-V2G service.
""",
    )

    replace_cell(
        cells,
        "## 8. Emergency V2G transfer sensitivity",
        """## 8. Emergency V2G transfer sensitivity

### Comparison and decision framing

I08 examines a synthetic islanded event selected from high values of the 2024 power-system peak-demand proxy. The 4-hour and 8-hour inherited-SOC cases are the primary sensitivities: buses begin with the SOC delivered by normal V1G operation. Prepared-SOC variants remain appendix checks.

During the selected outage intervals, both grid import and export are forced to zero. The result is therefore a technical upper bound on connected-bus fleet-to-fleet transfer inside an islanded depot—not a claim of external critical-load supply, installed grid-forming equipment or regulatory approval.
""",
    )

    # Make the workshop profile captions and labels match the permanent I02b definition.
    for cell in cells:
        source = "".join(cell.get("source", []))
        if "Economic V2G operating profile" in source:
            source = source.replace("Illustrative feed-in tariff\n3.20 THB/kWh", "First tested feed-in tariff\n3.15 THB/kWh")
            source = source.replace("illustrative higher level of 3.20", "first tested export-onset level of 3.15")
            cell["source"] = source.splitlines(keepends=True)

    insert_after(
        cells,
        "# Build the shared service-cost table here",
        markdown(
            """### Benefit, cost and stakeholder conclusion

| Perspective | What I08 can show | What it cannot show |
| --- | --- | --- |
| Bus operator and passengers | The energy available from connected buses while retaining route and terminal-SOC constraints | Guaranteed public-service continuity without an installed islanding design. |
| Depot owner / emergency planner | The size of a potential internal resilience envelope and recovery-energy cost | External critical-load supply, because no building or external load is modelled. |
| Utility / regulator | Input to a future resilience-pilot specification | Permission for islanding, protection requirements or a resilience payment. |

**Conclusion.** I08 is a technical sensitivity. Its recovery-energy cost gap and potential demand-charge exposure identify the operational burden that a resilience arrangement would need to address; they are not a price for an unmodelled emergency-load service.
"""
        ),
    )

    replace_cell(
        cells,
        "## 9. Cross-case annual comparison",
        """## Appendix A. Cross-case annual comparison

The family sections above are the primary decision analysis. The following plots retain a common-view audit of all available annual results; they are not a substitute for the family-specific references and service definitions.
""",
    )
    replace_cell(cells, "## 10. Completed-case results", "## Appendix B. Completed-case results, service cost gaps, and beneficiaries\n")
    replace_cell(cells, "## Appendix A. Site", "## Appendix C. Site, route, and roof context\n")
    replace_cell(cells, "## Appendix B. Supplementary", "## Appendix D. Supplementary tariff and export-limit sensitivities\n")

    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
