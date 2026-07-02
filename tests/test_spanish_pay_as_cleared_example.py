# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.simulation.run_case import resolve_example_paths
from flexi_mod.simulation.simulation_runner import OutputOptions, SimulationRunner

EXAMPLE_NAME = "hybrid_ETES_DA_ID_aFRR_energy_capacity_spain"
CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / EXAMPLE_NAME


def test_spanish_pay_as_cleared_example_runs(tmp_path: Path) -> None:
    paths = resolve_example_paths(EXAMPLE_NAME)
    assert paths["case_dir"] == CASE_DIR
    assert paths["study_case"] == EXAMPLE_NAME

    runner = SimulationRunner(
        case_dir=CASE_DIR,
        input_dir=CASE_DIR,
        output_dir=tmp_path / "output",
        study_case=EXAMPLE_NAME,
        output_options=OutputOptions(create_plots=False),
    )
    outputs = runner.run()

    market = pd.read_csv(outputs["market_ledger"])
    blocks = pd.read_csv(outputs["afrr_capacity_block_summary"])

    assert market["afrr_capacity_pricing_rule"].eq("pay_as_cleared").all()
    assert blocks["capacity_price_input_unit"].eq("EUR_per_MW_per_product").all()
    assert blocks["capacity_clearing_price_EUR_per_MW_h"].to_numpy() == pytest.approx(
        (blocks["capacity_price_raw"] / 0.25).to_numpy()
    )
    assert blocks["capacity_bid_price_EUR_per_MW_h"].gt(0.0).all()
    assert (
        blocks["capacity_clearing_price_EUR_per_MW_h"]
        > blocks["capacity_bid_price_EUR_per_MW_h"]
    ).all()
    assert market["afrr_capacity_revenue_EUR"].to_numpy() == pytest.approx(
        (
            market["afrr_capacity_reserved_MW"]
            * market["afrr_capacity_settlement_price_EUR_per_MW_h"]
            * 0.25
        ).to_numpy()
    )
    assert (
        market["afrr_energy_activated_MWh_el"]
        <= market["afrr_capacity_reserved_MWh"] + 1e-8
    ).all()
    assert market["actual_electricity_consumption_MWh_el"].to_numpy() == pytest.approx(
        (
            market["scheduled_electricity_procurement_MWh_el"]
            + market["afrr_energy_activated_MWh_el"]
        ).to_numpy()
    )
