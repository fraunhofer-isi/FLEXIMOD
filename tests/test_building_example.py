# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.plants import Building
from flexi_mod.simulation.run_case import resolve_example_paths
from flexi_mod.simulation.simulation_runner import OutputOptions, SimulationRunner
from flexi_mod.strategies import build_strategy
from flexi_mod.strategies.building_strategy import BuildingStrategy

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "building_v2g_example"


def test_building_example_optimizes_with_rolling_horizon(
    tmp_path: Path,
) -> None:
    paths = resolve_example_paths("building_v2g_example")
    assert paths["case_dir"] == EXAMPLE_DIR

    config = CaseConfig.from_case_dir(EXAMPLE_DIR)
    loader = DataLoader(config, input_dir=EXAMPLE_DIR)
    plants = loader.load_plants()
    building = Building.from_plants_dataframe(plants)[0]
    strategy = build_strategy(config.strategy_name, config)
    assert isinstance(strategy, BuildingStrategy)

    runner = SimulationRunner(
        case_dir=EXAMPLE_DIR,
        input_dir=EXAMPLE_DIR,
        output_dir=tmp_path / "output",
        study_case="building_v2g_example",
        output_options=OutputOptions(
            create_plots=False,
        ),
    )
    outputs = runner.run()
    result = pd.read_csv(outputs["dispatch_results"], parse_dates=["datetime"])
    market = pd.read_csv(outputs["market_ledger"])
    storage = pd.read_csv(outputs["storage_cost_ledger"])
    summary = pd.read_csv(outputs["summary_indicators"])

    assert len(result) == 24
    assert result["rolling_window"].nunique() == 4
    assert market["grid_export_MWh"].sum() == pytest.approx(result["grid_export_MWh"].sum())
    assert storage["bus_soc_MWh"].iloc[-1] == pytest.approx(result["bus_soc_MWh"].iloc[-1])
    assert result["bus_trip_energy_MWh"].sum() == pytest.approx(1.5)
    assert result.loc[result["bus_availability_fraction"] == 0.0, "bus_charge_MWh"].sum() == 0
    assert result["grid_export_MWh"].sum() > 0.0
    assert result["bus_soc_fraction"].iloc[-1] == pytest.approx(0.8)
    assert summary["total_variable_cost_EUR"].iloc[0] == pytest.approx(
        result["variable_cost_EUR"].sum()
    )

    charging_price = (
        result["bus_charge_MWh"] * result["electricity_import_price_EUR_per_MWh"]
    ).sum() / result["bus_charge_MWh"].sum()
    export_price = (
        result["grid_export_MWh"] * result["electricity_export_price_EUR_per_MWh"]
    ).sum() / result["grid_export_MWh"].sum()
    assert export_price > charging_price

    # SOC remains continuous when the implemented state is passed to the next window.
    previous_soc = building.electric_vehicle.initial_soc_mwh
    for _, row in result.iterrows():
        expected_soc = (
            previous_soc
            + row["bus_charge_MWh"] * building.electric_vehicle.efficiency_charge
            - row["bus_discharge_MWh"] / building.electric_vehicle.efficiency_discharge
            - row["bus_trip_energy_MWh"]
        )
        assert row["bus_soc_MWh"] == pytest.approx(expected_soc)
        previous_soc = row["bus_soc_MWh"]
