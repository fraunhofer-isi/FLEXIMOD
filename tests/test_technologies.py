# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from flexi_mod.plants.technologies import (
    AmineCCS,
    CryogenicCCS,
    Electrolyser,
    GasBoiler,
    HydrogenBufferStorage,
    OxyfuelCCS,
    ThermalStorage,
)


def test_amine_ccs_uses_plant_computed_capture_capacity() -> None:
    ccs = AmineCCS.from_row(
        pd.Series(
            {
                "capture_efficiency": 0.9,
                "specific_capture_electricity": 0.2,
                "specific_capture_heat": 1.0,
                "minimum_capture_fraction": 0.5,
                "variable_capture_cost": 4.0,
                "heat_cost": 10.0,
            }
        )
    )
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=[0], ordered=True)
    model.electricity_price = pyo.Param(model.T, initialize={0: 50.0})
    model.co2_price = pyo.Param(model.T, initialize={0: 80.0})
    model.ccs = pyo.Block()

    ccs.add_to_model(
        model,
        model.ccs,
        model.T,
        {"dt_hours": 0.25, "max_capture_rate_t_per_h": 2.0},
    )

    assert pyo.value(model.ccs.max_capture_per_step) == pytest.approx(0.5)
    assert pyo.value(model.ccs.specific_electricity_consumption) == pytest.approx(0.2)
    assert hasattr(model.ccs, "co2_balance")
    assert hasattr(model.ccs, "operating_cost_definition")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("capture_efficiency", 1.01, "must be between 0 and 1"),
        ("minimum_capture_fraction", 0.91, "must satisfy"),
        ("specific_electricity_consumption_mwh_per_t", -0.01, "must be non-negative"),
        ("specific_heat_consumption_mwh_per_t", -0.01, "must be non-negative"),
        ("specific_variable_cost_eur_per_t", -0.01, "must be non-negative"),
        ("heat_cost_eur_per_mwh", -0.01, "must be non-negative"),
    ],
)
def test_amine_ccs_rejects_invalid_parameters(field: str, value: float, message: str) -> None:
    parameters = {
        "capture_efficiency": 0.9,
        "specific_electricity_consumption_mwh_per_t": 0.2,
        "specific_heat_consumption_mwh_per_t": 1.0,
        "minimum_capture_fraction": 0.5,
        "specific_variable_cost_eur_per_t": 4.0,
        "heat_cost_eur_per_mwh": 10.0,
    }
    parameters[field] = value

    with pytest.raises(ValueError, match=message):
        AmineCCS(**parameters)


def test_cryogenic_ccs_builds_without_a_heat_requirement() -> None:
    ccs = CryogenicCCS.from_row(
        pd.Series(
            {
                "capture_efficiency": 0.9,
                "specific_capture_electricity": 0.35,
                "minimum_capture_fraction": 0.5,
                "variable_capture_cost": 4.0,
            }
        )
    )
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=[0], ordered=True)
    model.electricity_price = pyo.Param(model.T, initialize={0: 50.0})
    model.co2_price = pyo.Param(model.T, initialize={0: 80.0})
    model.ccs = pyo.Block()

    ccs.add_to_model(
        model,
        model.ccs,
        model.T,
        {"dt_hours": 0.25, "max_capture_rate_t_per_h": 2.0},
    )

    assert pyo.value(model.ccs.max_capture_per_step) == pytest.approx(0.5)
    assert pyo.value(model.ccs.specific_electricity_consumption) == pytest.approx(0.35)
    assert hasattr(model.ccs, "electricity_consumption")
    assert not hasattr(model.ccs, "heat_consumption")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("capture_efficiency", -0.01, "must be between 0 and 1"),
        ("minimum_capture_fraction", 0.91, "must satisfy"),
        ("specific_electricity_consumption_mwh_per_t", -0.01, "must be non-negative"),
        ("specific_variable_cost_eur_per_t", -0.01, "must be non-negative"),
    ],
)
def test_cryogenic_ccs_rejects_invalid_parameters(field: str, value: float, message: str) -> None:
    parameters = {
        "capture_efficiency": 0.9,
        "specific_electricity_consumption_mwh_per_t": 0.35,
        "minimum_capture_fraction": 0.5,
        "specific_variable_cost_eur_per_t": 4.0,
    }
    parameters[field] = value

    with pytest.raises(ValueError, match=message):
        CryogenicCCS(**parameters)


def test_oxyfuel_ccs_builds_without_oxygen_or_heat_requirements() -> None:
    ccs = OxyfuelCCS.from_row(
        pd.Series(
            {
                "recovery_efficiency": 0.95,
                "specific_capture_electricity": 0.3,
                "minimum_recovery_fraction": 0.8,
                "variable_capture_cost": 3.0,
            }
        )
    )
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=[0], ordered=True)
    model.electricity_price = pyo.Param(model.T, initialize={0: 50.0})
    model.co2_price = pyo.Param(model.T, initialize={0: 80.0})
    model.ccs = pyo.Block()

    ccs.add_to_model(
        model,
        model.ccs,
        model.T,
        {"dt_hours": 0.25, "max_capture_rate_t_per_h": 2.0},
    )

    assert pyo.value(model.ccs.max_capture_per_step) == pytest.approx(0.5)
    assert pyo.value(model.ccs.recovery_efficiency) == pytest.approx(0.95)
    assert hasattr(model.ccs, "electricity_consumption")
    assert not hasattr(model.ccs, "heat_consumption")
    assert not hasattr(model.ccs, "oxygen_demand")
    assert not hasattr(model.ccs, "oxygen_generated")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("recovery_efficiency", 1.01, "must be between 0 and 1"),
        ("minimum_recovery_fraction", 0.96, "must satisfy"),
        ("specific_electricity_consumption_mwh_per_t", -0.01, "must be non-negative"),
        ("specific_variable_cost_eur_per_t", -0.01, "must be non-negative"),
    ],
)
def test_oxyfuel_ccs_rejects_invalid_parameters(field: str, value: float, message: str) -> None:
    parameters = {
        "recovery_efficiency": 0.95,
        "specific_electricity_consumption_mwh_per_t": 0.3,
        "minimum_recovery_fraction": 0.8,
        "specific_variable_cost_eur_per_t": 3.0,
    }
    parameters[field] = value

    with pytest.raises(ValueError, match=message):
        OxyfuelCCS(**parameters)


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


def test_power_component_uses_carried_ramp_and_minimum_uptime_state() -> None:
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=[0, 1, 2, 3], ordered=True)
    model.electricity_price = pyo.Param(model.T, initialize={t: 1.0 for t in model.T})
    model.electrolyser = pyo.Block()
    electrolyser = Electrolyser(
        max_power_mw=10.0,
        min_power_mw=1.0,
        efficiency=0.8,
        ramp_up_mw_per_step=0.5,
        ramp_down_mw_per_step=0.5,
        min_operating_steps=4,
        min_down_steps=1,
        initial_operational_status=1,
    )
    electrolyser.add_to_model(
        model,
        model.electrolyser,
        model.T,
        {
            "dt_hours": 0.25,
            "initial_power_in": 1.0,
            "initial_operational_status": 1,
            "initial_consecutive_status_steps": 2,
        },
    )
    model.objective = pyo.Objective(expr=sum(model.electrolyser.power_in[t] for t in model.T))

    pyo.SolverFactory("highs").solve(model)

    assert pyo.value(model.electrolyser.power_in[0]) == pytest.approx(0.875)
    assert pyo.value(model.electrolyser.operational_status[0]) == pytest.approx(1.0)
    assert pyo.value(model.electrolyser.operational_status[1]) == pytest.approx(1.0)


def test_inventory_storage_uses_carried_flow_for_boundary_ramp() -> None:
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=[0, 1], ordered=True)
    model.storage = pyo.Block()
    storage = HydrogenBufferStorage(
        capacity=10.0,
        initial_soc=0.5,
        max_power_charge=4.0,
        max_power_discharge=4.0,
        ramp_up=1.0,
        ramp_down=1.0,
    )
    storage.add_to_model(
        model,
        model.storage,
        model.T,
        {
            "dt_hours": 0.25,
            "initial_soc": 0.5,
            "initial_charge": 1.0,
            "initial_discharge": 0.0,
        },
    )
    model.objective = pyo.Objective(
        expr=sum(model.storage.charge[t] + model.storage.discharge[t] for t in model.T)
    )

    pyo.SolverFactory("highs").solve(model)

    assert pyo.value(model.storage.charge[0]) == pytest.approx(0.75)
