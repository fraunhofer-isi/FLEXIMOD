#!/usr/bin/env python3
"""Insert the supplementary I04c ten-bus peak-event sensitivity into the notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "building_use_case_analysis.ipynb"
TAG = "i04c-ten-bus-peak-event"


MARKDOWN = """### Supplementary I04c: illustrative ten-bus annual technical sensitivity

This supplementary sensitivity scales the peak-demand response to an illustrative future fleet of ten buses—three on Route 7ก, four on Route 79 and three on Route 101—with ten 150 kW bidirectional chargers, a nominal 1.5 MW bidirectional connection and 3.0 MWh of nameplate battery capacity.

The annual result retains route, availability, charger-power and SOC constraints. It uses a single maximum-service solve for each 48-hour rolling horizon, rather than the later throughput- and cost-minimising tie-breaks used in the three-bus cases. It is therefore a technical-potential result, not an economic dispatch or firm capacity commitment. The fleet allocation and connection are assumptions for scale testing; they are not verified BMTA fleet counts or approved connection capacity.

The charts separate two effects: energy delivered from buses during selected power-system peak-demand periods, and depot charging avoided by moving charging away from those periods. For the ten-bus bar, charging shift is calculated against a matched ten-bus V1G reference; it is not compared directly with the smaller three-bus fleet.
"""


CODE = """ten_bus_annual_path = OUTPUT_ROOT / 'ten_bus_annual_peak_demand_summary.csv'
if not ten_bus_annual_path.exists():
    print(f'Missing supplementary I04c result: {ten_bus_annual_path}')
else:
    ten_bus_annual = pd.read_csv(ten_bus_annual_path).set_index('case')
    annual_display = ten_bus_annual.loc[:, [
        'bus_energy_delivered_in_peak_periods_MWh_year',
        'charging_shifted_out_of_peak_periods_MWh_year',
        'maximum_charging_avoided_in_peak_periods_kW',
        'net_depot_import_reduction_in_peak_periods_MWh_year',
    ]].rename(columns={
        'bus_energy_delivered_in_peak_periods_MWh_year': 'V2G energy delivered in selected peak periods (MWh/year)',
        'charging_shifted_out_of_peak_periods_MWh_year': 'Charging shifted out of selected peak periods (MWh/year)',
        'maximum_charging_avoided_in_peak_periods_kW': 'Maximum depot charging avoided in selected peak periods (kW)',
        'net_depot_import_reduction_in_peak_periods_MWh_year': 'Net depot-import reduction in selected peak periods (MWh/year)',
    })
    display(annual_display.round(3))

    labels = ['V1G\\nmanaged charging\\n(3 buses)', 'Technical V2G\\n(3 buses)', 'Technical V2G\\n(10-bus sensitivity)']
    delivered = ten_bus_annual['bus_energy_delivered_in_peak_periods_MWh_year'].to_numpy()
    shifted_energy = ten_bus_annual['charging_shifted_out_of_peak_periods_MWh_year'].to_numpy()
    colours = ['#6B7280', '#4C78A8', '#009E73']
    fig, (energy_axis, shift_axis) = plt.subplots(1, 2, figsize=(12.4, 4.5), constrained_layout=True)
    fig.suptitle('Figure 17 — Annual fleet-scale power-system peak-demand response', fontsize=14, fontweight='semibold')
    energy_axis.bar(labels, delivered, color=colours, width=0.58)
    energy_axis.set_ylabel('Bus energy delivered during selected peak periods (MWh/year)')
    energy_axis.set_title('Annual V2G contribution during high demand')
    energy_axis.grid(axis='y', alpha=0.28)
    energy_axis.set_axisbelow(True)
    for x, value in enumerate(delivered):
        energy_axis.text(x, value, f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='semibold')

    shift_axis.bar(labels, shifted_energy, color=colours, width=0.58)
    shift_axis.set_ylabel('Charging shifted out of selected peak periods (MWh/year)')
    shift_axis.set_title('Annual peak-period depot demand avoided by shifting')
    shift_axis.grid(axis='y', alpha=0.28)
    shift_axis.set_axisbelow(True)
    for x, value in enumerate(shifted_energy):
        shift_axis.text(x, value, f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='semibold')
    plt.show()
    display(Markdown(
        '**Figure 17. Annual fleet-scale power-system peak-demand response.** The left panel shows '
        'bus energy delivered during selected high-demand periods; the right panel shows charging shifted '
        'out of those periods. The ten-bus shifting result is measured against a matched ten-bus V1G reference. '
        'The ten-bus case is a route-constrained technical sensitivity, not an economic-dispatch or firm-capacity result.'
    ))

    v2g_3 = ten_bus_annual.loc['Technical V2G (3 buses)']
    v2g_10 = ten_bus_annual.loc['Technical V2G (10 buses)']
    display(Markdown(
        f'**Supplementary I04c result.** The three-bus technical case delivers '
        f'{v2g_3["bus_energy_delivered_in_peak_periods_MWh_year"]:.3f} MWh/year during selected peak periods; '
        f'the ten-bus technical sensitivity delivers {v2g_10["bus_energy_delivered_in_peak_periods_MWh_year"]:.3f} MWh/year. '
        'The right panel isolates avoided charging demand; it does not count bus discharge as charging shift. '
        'Neither bar demonstrates a firm capacity product or relief of a specific local-network asset.'
    ))
"""


def tagged(cell: nbformat.NotebookNode) -> bool:
    return TAG in cell.metadata.get("tags", [])


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if not tagged(cell)]

    family_table = notebook.cells[0]
    family_table.source = family_table.source.replace(
        "supplementary ten-bus peak-event test",
        "supplementary ten-bus annual technical sensitivity",
    )

    framing_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and cell.source.startswith("## 5. Power-system peak-demand import reduction")
    )
    framing = notebook.cells[framing_index]
    annual_sentence = "The annual results quantify energy reduction during the selected intervals, short-duration power reduction and the associated operator cost gap."
    old_supplement = (
        " The supplementary I04c sensitivity uses an illustrative ten-bus future fleet for the representative 6 May peak-demand event; "
        "it is an event-capability result, not an annual service or verified fleet count."
    )
    new_supplement = (
        " The supplementary I04c sensitivity uses an illustrative ten-bus future fleet and reports annual technical potential; "
        "it is not an economic dispatch, firm capacity commitment or verified fleet count."
    )
    if old_supplement in framing.source:
        framing.source = framing.source.replace(old_supplement, new_supplement)
    elif new_supplement.strip() not in framing.source:
        framing.source = framing.source.replace(annual_sentence, annual_sentence + new_supplement)

    insertion_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and cell.source.startswith("### Benefit, cost and stakeholder conclusion") and index > framing_index
    )
    markdown_cell = new_markdown_cell(MARKDOWN)
    code_cell = new_code_cell(CODE)
    markdown_cell.metadata["tags"] = [TAG]
    code_cell.metadata["tags"] = [TAG]
    notebook.cells[insertion_index:insertion_index] = [markdown_cell, code_cell]
    if not notebook.metadata.get("i04c_figure_renumbered", False):
        # I04c is Figure 17, so renumber the later report figures without
        # altering Figures 12–16 in the preceding sections.
        for cell in notebook.cells[insertion_index + 2:]:
            for old, new in ((22, 23), (21, 22), (20, 21), (19, 20), (18, 19), (17, 18)):
                cell.source = cell.source.replace(f"Figure {old}", f"Figure {new}")
                cell.source = cell.source.replace(f"Figures {old}", f"Figures {new}")
        notebook.metadata["i04c_figure_renumbered"] = True
    nbformat.write(notebook, NOTEBOOK)
    print(f"Updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
