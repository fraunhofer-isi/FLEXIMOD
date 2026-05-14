# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo

from flexi_mod.plants.technologies import GasBoiler, ThermalStorage


def test_thermal_storage_adds_expected_pyomo_block() -> None:
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=[0, 1], ordered=True)
    model.electricity_price = pyo.Param(model.T, initialize={0: 40.0, 1: 50.0})
    model.charge_allowed = pyo.Param(model.T, within=pyo.Binary, initialize={0: 1, 1: 0})
    model.storage = pyo.Block()

    storage = ThermalStorage(
        max_power_charge_mw=4.0,
        max_power_discharge_mw=3.0,
        max_capacity_mwh=10.0,
        min_capacity_mwh=0.0,
        initial_soc_mwh=1.0,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        storage_loss_rate=0.001,
    )
    storage.add_to_model(model, model.storage, model.T, {"dt_hours": 0.25})

    assert hasattr(model.storage, "electric_charge_to_storage")
    assert hasattr(model.storage, "discharge_heat")
    assert hasattr(model.storage, "soc")
    assert hasattr(model.storage, "storage_balance")
    assert hasattr(model.storage, "charge_allowed_limit")


def test_gas_boiler_adds_expected_pyomo_block() -> None:
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=[0, 1], ordered=True)
    model.gas_price = pyo.Param(model.T, initialize={0: 50.0, 1: 60.0})
    model.co2_price = pyo.Param(model.T, initialize={0: 80.0, 1: 80.0})
    model.co2_emission_factor = pyo.Param(initialize=0.2)
    model.boiler = pyo.Block()

    boiler = GasBoiler(
        max_heat_output_mw=5.0,
        min_heat_output_mw=0.0,
        efficiency=0.9,
    )
    boiler.add_to_model(model, model.boiler, model.T, {"dt_hours": 0.25})

    assert hasattr(model.boiler, "heat_out")
    assert hasattr(model.boiler, "fuel_input")
    assert hasattr(model.boiler, "efficiency_constraint")
    assert hasattr(model.boiler, "operating_cost_definition")
    assert hasattr(model.boiler, "co2_cost_definition")


def test_technologies_can_be_built_from_csv_rows() -> None:
    storage = ThermalStorage.from_row(
        pd.Series(
            {
                "max_power_charge": 7,
                "max_power_discharge": 6,
                "max_capacity": 12,
                "min_capacity": 0,
                "initial_soc": 1,
                "efficiency_charge": 0.92,
                "efficiency_discharge": 0.9,
                "storage_loss_rate": 0.001,
                "storage_type": "short-term_with_generator",
            }
        )
    )
    boiler = GasBoiler.from_row(
        pd.Series(
            {
                "max_power": 5,
                "min_power": 0,
                "efficiency": 0.9,
                "fuel_type": "natural_gas",
            }
        )
    )

    assert storage.max_capacity_mwh == 12
    assert boiler.fuel_type == "natural_gas"
