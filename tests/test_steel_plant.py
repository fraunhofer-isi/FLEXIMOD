# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.plants.factory import build_plants
from flexi_mod.plants.steel_plant import SteelDispatchSignals, SteelPlant
from flexi_mod.plants.technologies import (
    BasicOxygenFurnace,
    BlastFurnaceBasicOxygenFurnace,
    DRIPlant,
    DRIStorage,
    ElectricArcFurnace,
    Electrolyser,
    GenericStorage,
    HydrogenBufferStorage,
    ThermalStorage,
)

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DA"


def test_steel_plant_builds_required_and_optional_technologies() -> None:
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=True))

    assert plant.steel_route == "dri_eaf"
    assert isinstance(plant.dri_plant, DRIPlant)
    assert isinstance(plant.eaf, ElectricArcFurnace)
    assert isinstance(plant.components["electrolyser"], Electrolyser)
    assert isinstance(plant.components["hydrogen_buffer_storage"], HydrogenBufferStorage)
    assert isinstance(plant.components["dri_storage"], DRIStorage)
    assert plant.steel_demand_tonnes == 4.0
    assert issubclass(ThermalStorage, GenericStorage)
    assert issubclass(HydrogenBufferStorage, GenericStorage)
    assert issubclass(DRIStorage, GenericStorage)


def test_steel_plant_builds_dri_bof_route() -> None:
    plant = SteelPlant.from_rows("steel_1", _dri_bof_rows())

    assert plant.steel_route == "dri_bof"
    assert isinstance(plant.dri_plant, DRIPlant)
    assert isinstance(plant.bof, BasicOxygenFurnace)


def test_steel_plant_builds_standalone_bf_bof_route() -> None:
    plant = SteelPlant.from_rows("steel_1", _bf_bof_rows("coal"))

    assert plant.steel_route == "bf_bof"
    assert isinstance(plant.bf_bof, BlastFurnaceBasicOxygenFurnace)


def test_shared_plant_factory_selects_steel_plant() -> None:
    plants = build_plants(_steel_rows(include_optional=False))

    assert len(plants) == 1
    assert isinstance(plants[0], SteelPlant)


def test_steel_plant_rejects_missing_required_technology() -> None:
    rows = _steel_rows(include_optional=False)
    rows = rows.loc[rows["technology"] != "eaf"]

    with pytest.raises(ValueError, match="missing required.*eaf"):
        SteelPlant.from_rows("steel_1", rows)


def test_steel_plant_hydrogen_route_and_material_balances_solve() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=True))
    forecasts = _steel_forecasts()

    result = plant.solve_horizon(config, forecasts, _signals())

    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert result["natural_gas_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["electrolyser_hydrogen_output_MWh"].sum() == pytest.approx(
        result["hydrogen_consumption_MWh"].sum()
        + result["hydrogen_storage_charge_MWh"].sum()
        - result["hydrogen_storage_discharge_MWh"].sum()
    )
    assert result["dri_output_t"].sum() + result["dri_storage_discharge_t"].sum() == (
        pytest.approx(result["dri_input_t"].sum() + result["dri_storage_charge_t"].sum())
    )
    assert result["total_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        (
            result["dri_electricity_consumption_MWh"]
            + result["eaf_electricity_consumption_MWh"]
            + result["electrolyser_electricity_consumption_MWh"]
        ).to_numpy()
    )
    assert result["coal_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["co2_emissions_t"].sum() == pytest.approx(
        result["dri_co2_emissions_t"].sum() + result["eaf_co2_emissions_t"].sum()
    )


def test_dri_bof_route_solves_and_satisfies_material_balance() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteelPlant.from_rows("steel_1", _dri_bof_rows())

    result = plant.solve_horizon(config, _steel_forecasts(), _signals())

    assert result["steel_route"].unique().tolist() == ["dri_bof"]
    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert result["bof_steel_output_t"].sum() == pytest.approx(4.0)
    assert result["dri_output_t"].sum() == pytest.approx(result["bof_dri_input_t"].sum())
    assert result["eaf_electricity_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["lime_consumption_t"].sum() == pytest.approx(0.2)


def test_bf_bof_coal_route_solves_and_requires_coal_price() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteelPlant.from_rows("steel_1", _bf_bof_rows("coal"))
    forecasts = _steel_forecasts()

    result = plant.solve_horizon(config, forecasts, _signals())

    assert result["steel_route"].unique().tolist() == ["bf_bof"]
    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert result["coal_consumption_MWh"].sum() == pytest.approx(16.0)
    assert result["hydrogen_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["natural_gas_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["iron_ore_consumption_t"].sum() == pytest.approx(6.0)
    assert result["lime_consumption_t"].sum() == pytest.approx(0.2)

    with pytest.raises(ValueError, match="coal_price"):
        plant.solve_horizon(config, forecasts.drop(columns=["coal_price"]), _signals())


@pytest.mark.parametrize("fuel_type", ["natural_gas", "hydrogen"])
def test_bf_bof_single_fuel_routes_leave_unused_fuels_zero(fuel_type: str) -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteelPlant.from_rows("steel_1", _bf_bof_rows(fuel_type))

    result = plant.solve_horizon(config, _steel_forecasts(), _signals())

    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    if fuel_type == "natural_gas":
        assert result["natural_gas_consumption_MWh"].sum() == pytest.approx(12.0)
        assert result["hydrogen_consumption_MWh"].sum() == pytest.approx(0.0)
    else:
        assert result["hydrogen_consumption_MWh"].sum() == pytest.approx(8.0)
        assert result["natural_gas_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["coal_consumption_MWh"].sum() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("fuel_type", "factor_column"),
    [
        ("coal", "coal_co2_factor"),
        ("natural_gas", "natural_gas_co2_factor"),
        ("hybrid_hydrogen_natural_gas", "natural_gas_co2_factor"),
    ],
)
def test_bf_bof_requires_explicit_applicable_fuel_co2_factor(
    fuel_type: str,
    factor_column: str,
) -> None:
    rows = _bf_bof_rows(fuel_type)
    rows.loc[rows["technology"] == "bf_bof", factor_column] = float("nan")

    with pytest.raises(ValueError, match=factor_column):
        SteelPlant.from_rows("steel_1", rows)


@pytest.mark.parametrize(
    ("fuel_type", "factor_column"),
    [
        ("coal", "coal_co2_factor"),
        ("natural_gas", "natural_gas_co2_factor"),
        ("hybrid_hydrogen_natural_gas", "natural_gas_co2_factor"),
    ],
)
def test_dri_requires_explicit_applicable_fuel_co2_factor(
    fuel_type: str,
    factor_column: str,
) -> None:
    rows = _steel_rows(include_optional=False)
    dri = rows["technology"] == "dri_plant"
    rows.loc[dri, "fuel_type"] = fuel_type
    rows.loc[dri, "specific_coal_consumption"] = 4.0
    rows.loc[dri, factor_column] = float("nan")

    with pytest.raises(ValueError, match=factor_column):
        SteelPlant.from_rows("steel_1", rows)


def test_hydrogen_dri_does_not_require_fossil_fuel_co2_factors() -> None:
    rows = _steel_rows(include_optional=False).drop(
        columns=["natural_gas_co2_factor"], errors="ignore"
    )

    plant = SteelPlant.from_rows("steel_1", rows)

    assert plant.dri_plant is not None
    assert plant.dri_plant.coal_co2_factor_t_per_mwh == pytest.approx(0.0)
    assert plant.dri_plant.natural_gas_co2_factor_t_per_mwh == pytest.approx(0.0)


def test_bf_bof_hybrid_fuel_selects_cheaper_feasible_mix() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteelPlant.from_rows("steel_1", _bf_bof_rows("hybrid_hydrogen_natural_gas"))
    forecasts = _steel_forecasts()
    forecasts["natural_gas_price"] = 10.0
    forecasts["hydrogen_price"] = 200.0

    gas_result = plant.solve_horizon(config, forecasts, _signals())

    assert gas_result["natural_gas_consumption_MWh"].sum() == pytest.approx(12.0)
    assert gas_result["hydrogen_consumption_MWh"].sum() == pytest.approx(0.0)

    forecasts["natural_gas_price"] = 200.0
    forecasts["hydrogen_price"] = 1.0
    hydrogen_result = plant.solve_horizon(config, forecasts, _signals())

    assert hydrogen_result["hydrogen_consumption_MWh"].sum() == pytest.approx(8.0)
    assert hydrogen_result["natural_gas_consumption_MWh"].sum() == pytest.approx(0.0)


def test_bf_bof_hydrogen_is_external_without_electrolyser_and_constrained_with_one() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    external_plant = SteelPlant.from_rows("steel_1", _bf_bof_rows("hydrogen"))
    external_result = external_plant.solve_horizon(config, _steel_forecasts(), _signals())

    assert external_result["hydrogen_consumption_MWh"].sum() == pytest.approx(8.0)
    assert "electrolyser_hydrogen_output_MWh" not in external_result

    electrolyser_plant = SteelPlant.from_rows(
        "steel_1",
        _bf_bof_rows("hydrogen", include_electrolyser=True),
    )
    electrolyser_result = electrolyser_plant.solve_horizon(config, _steel_forecasts(), _signals())

    assert electrolyser_result["hydrogen_consumption_MWh"].sum() == pytest.approx(8.0)
    assert electrolyser_result["electrolyser_hydrogen_output_MWh"].sum() == pytest.approx(8.0)


def test_invalid_steel_route_combinations_fail_clearly() -> None:
    rows = pd.concat([_steel_rows(include_optional=False), _dri_bof_rows().iloc[[1]]])

    with pytest.raises(ValueError, match="ambiguous route"):
        SteelPlant.from_rows("steel_1", rows)

    rows = pd.concat([_bf_bof_rows("coal"), _steel_rows(include_optional=False).iloc[[0]]])

    with pytest.raises(ValueError, match="bf_bof cannot be combined"):
        SteelPlant.from_rows("steel_1", rows)


def test_ambiguous_both_fuel_type_is_rejected() -> None:
    rows = _steel_rows(include_optional=False)
    rows.loc[rows["technology"] == "dri_plant", "fuel_type"] = "both"

    with pytest.raises(ValueError, match="hybrid_hydrogen_natural_gas"):
        SteelPlant.from_rows("steel_1", rows)


def test_steel_plant_without_electrolyser_buys_hydrogen() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=False))

    result = plant.solve_horizon(config, _steel_forecasts(), _signals())

    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert result["hydrogen_consumption_MWh"].sum() > 0
    assert "electrolyser_hydrogen_output_MWh" not in result


def test_forecast_profile_is_summed_as_a_flexible_cumulative_target() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    rows = _steel_rows(include_optional=True)
    rows["steel_demand"] = pd.NA
    rows["demand"] = "steel_production_target"
    plant = SteelPlant.from_rows("steel_1", rows)
    forecasts = _steel_forecasts()
    demand_profile = pd.Series([0.0, 0.0, 0.0, 4.0], index=forecasts.index)
    forecasts["steel_production_target"] = demand_profile

    result = plant.solve_horizon(config, forecasts, _signals())

    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert result["steel_output_t"].to_numpy() != pytest.approx(demand_profile.to_numpy())
    assert set(result["steel_demand_mode"]) == {"forecast_profile"}
    assert set(result["steel_demand_total_t"]) == {4.0}


def test_forecast_profile_uses_default_plant_specific_column() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    rows = _steel_rows(include_optional=False)
    rows["steel_demand"] = pd.NA
    plant = SteelPlant.from_rows("steel_1", rows)
    forecasts = _steel_forecasts()
    forecasts["steel_1_steel_demand"] = [1.0, 0.5, 1.5, 1.0]

    result = plant.solve_horizon(config, forecasts, _signals())

    assert plant.steel_demand_tonnes is None
    assert plant.steel_demand_column == "steel_1_steel_demand"
    assert result["steel_output_t"].sum() == pytest.approx(4.0)


def test_total_target_takes_precedence_over_forecast_profile() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    rows = _steel_rows(include_optional=False)
    rows["demand"] = "unused_steel_profile"
    plant = SteelPlant.from_rows("steel_1", rows)

    result = plant.solve_horizon(config, _steel_forecasts(), _signals())

    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert set(result["steel_demand_mode"]) == {"total_target"}
    assert set(result["steel_demand_total_t"]) == {4.0}


def test_missing_total_and_forecast_profile_fails_clearly() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    rows = _steel_rows(include_optional=False)
    rows["steel_demand"] = pd.NA
    plant = SteelPlant.from_rows("steel_1", rows)

    with pytest.raises(ValueError, match="missing demand column 'steel_1_steel_demand'"):
        plant.solve_horizon(config, _steel_forecasts(), _signals())


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    [
        (-1.0, "negative"),
        (None, "missing or non-numeric"),
        ("not-a-number", "missing or non-numeric"),
        (float("inf"), "non-finite"),
    ],
)
def test_invalid_forecast_profile_values_fail(invalid_value: object, message: str) -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    rows = _steel_rows(include_optional=False)
    rows["steel_demand"] = pd.NA
    rows["demand"] = "steel_profile"
    plant = SteelPlant.from_rows("steel_1", rows)
    forecasts = _steel_forecasts()
    forecasts["steel_profile"] = [1.0, 1.0, 1.0, invalid_value]

    with pytest.raises(ValueError, match=message):
        plant.solve_horizon(config, forecasts, _signals())


@pytest.mark.parametrize(
    ("column", "first", "second"),
    [
        ("steel_demand", 4.0, 5.0),
        ("demand", "steel_profile_a", "steel_profile_b"),
    ],
)
def test_inconsistent_demand_configuration_fails(
    column: str, first: object, second: object
) -> None:
    rows = _steel_rows(include_optional=False)
    rows[column] = [first, second]

    with pytest.raises(ValueError, match=f"inconsistent {column}"):
        SteelPlant.from_rows("steel_1", rows)


@pytest.mark.parametrize("invalid_total", [-1.0, "bad", float("inf")])
def test_invalid_total_target_fails(invalid_total: object) -> None:
    rows = _steel_rows(include_optional=False)
    rows["steel_demand"] = invalid_total

    with pytest.raises(ValueError, match="steel_demand"):
        SteelPlant.from_rows("steel_1", rows)


def test_forecast_discovery_requires_profile_only_without_total_target() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    loader = DataLoader(config, input_dir=CASE_DIR)
    total_rows = _steel_rows(include_optional=False)

    total_required = loader.required_forecast_columns(total_rows)

    assert "steel_1_steel_demand" not in total_required

    profile_rows = total_rows.copy()
    profile_rows["steel_demand"] = pd.NA
    profile_rows["demand"] = "custom_steel_demand"
    profile_required = loader.required_forecast_columns(profile_rows)
    assert "custom_steel_demand" in profile_required

    profile_rows["demand"] = pd.NA
    default_required = loader.required_forecast_columns(profile_rows)
    assert "steel_1_steel_demand" in default_required

    coal_required = loader.required_forecast_columns(_bf_bof_rows("coal"))
    assert "coal_price" in coal_required

    gas_required = loader.required_forecast_columns(_bf_bof_rows("natural_gas"))
    assert "coal_price" not in gas_required


def test_rolling_scalar_target_is_committed_once_and_completed_exactly() -> None:
    config = _rolling_config(horizon_hours=1.0, step_hours=0.5)
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=True))
    forecasts = _extended_steel_forecasts()

    result = plant.solve_rolling(config, forecasts, _signals())

    assert result.index.equals(forecasts.index)
    assert result.index.is_unique
    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert result["cumulative_steel_output_t"].iloc[-1] == pytest.approx(4.0)
    assert result["remaining_steel_demand_t"].iloc[-1] == pytest.approx(0.0)
    assert result["steel_demand_balance_t"].iloc[-1] == pytest.approx(0.0)
    assert result["rolling_window"].nunique() == 4
    assert set(result["steel_demand_total_t"]) == {4.0}


def test_final_rolling_window_reconciles_only_numerical_infeasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _rolling_config(horizon_hours=2.0, step_hours=2.0)
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=False))
    forecasts = _extended_steel_forecasts()
    solve_model = plant._solve_model
    attempted_targets: list[float] = []

    def fail_exact_target_once(config, forecasts, model):
        attempted_targets.append(float(model.steel_demand.value))
        if len(attempted_targets) == 1:
            raise RuntimeError("synthetic numerical boundary")
        return solve_model(config, forecasts, model)

    monkeypatch.setattr(plant, "_solve_model", fail_exact_target_once)

    result = plant.solve_rolling(config, forecasts, _signals())

    assert attempted_targets == pytest.approx([4.0, 3.9999999])
    assert result["steel_output_t"].sum() == pytest.approx(4.0, abs=1e-6)
    assert result["steel_demand_balance_t"].iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_rolling_profile_carries_backlog_and_credit_between_windows() -> None:
    config = _rolling_config(horizon_hours=1.0, step_hours=0.5)
    rows = _steel_rows(include_optional=False)
    rows["steel_demand"] = pd.NA
    rows["demand"] = "steel_profile"
    plant = SteelPlant.from_rows("steel_1", rows)
    forecasts = _extended_steel_forecasts()
    forecasts["steel_profile"] = [0.5] * len(forecasts)

    result = plant.solve_rolling(config, forecasts, _signals())

    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert result["steel_demand_balance_t"].iloc[-1] == pytest.approx(0.0)
    assert (
        result["steel_demand_balance_t"].abs().max() > 1e-8
        or result["steel_output_t"].nunique() > 1
    )


def test_rolling_bf_bof_preserves_route_state_and_final_demand() -> None:
    config = _rolling_config(horizon_hours=1.0, step_hours=0.5)
    plant = SteelPlant.from_rows("steel_1", _bf_bof_rows("natural_gas"))

    result = plant.solve_rolling(config, _extended_steel_forecasts(), _signals())

    assert result["steel_output_t"].sum() == pytest.approx(4.0)
    assert result["cumulative_steel_output_t"].iloc[-1] == pytest.approx(4.0)
    assert result["steel_demand_balance_t"].iloc[-1] == pytest.approx(0.0)
    assert result["bf_bof_operational_status"].isin([0, 1]).all()


def test_rolling_inventory_state_is_continuous_across_commit_boundary() -> None:
    config = _rolling_config(horizon_hours=1.0, step_hours=0.5)
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=True))
    result = plant.solve_rolling(config, _extended_steel_forecasts(), _signals())
    boundary = result.index[result["rolling_window"].diff().fillna(0).ne(0)][0]
    position = result.index.get_loc(boundary)
    previous = result.iloc[position - 1]
    current = result.iloc[position]
    storage = plant.components["hydrogen_buffer_storage"]
    expected_soc = (
        previous["hydrogen_storage_soc"]
        + (
            storage.efficiency_charge * current["hydrogen_storage_charge_MWh"]
            - current["hydrogen_storage_discharge_MWh"] / storage.efficiency_discharge
            - storage.storage_loss_rate * previous["hydrogen_storage_soc"] * storage.capacity
        )
        / storage.capacity
    )

    assert current["hydrogen_storage_soc"] == pytest.approx(expected_soc)


def test_rolling_minimum_uptime_is_carried_across_commit_boundary() -> None:
    config = _rolling_config(horizon_hours=1.0, step_hours=0.5)
    rows = _steel_rows(include_optional=False)
    rows["min_power"] = 0.1
    rows["min_operating_steps"] = 4
    rows["min_down_steps"] = 1
    rows["initial_operational_status"] = 0
    plant = SteelPlant.from_rows("steel_1", rows)

    result = plant.solve_rolling(config, _extended_steel_forecasts(), _signals())

    assert result["eaf_operational_status"].iloc[:4].tolist() == [1, 1, 1, 1]
    assert result["dri_operational_status"].iloc[:4].tolist() == [1, 1, 1, 1]


def test_rolling_window_configuration_is_validated() -> None:
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=False))
    config = _rolling_config(horizon_hours=0.5, step_hours=1.0)

    with pytest.raises(ValueError, match="must not exceed"):
        plant.solve_rolling(config, _extended_steel_forecasts(), _signals())


def test_dri_storage_can_shift_dri_production_to_later_eaf_operation() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    rows = _steel_rows(include_optional=True)
    rows = rows.loc[rows["technology"].isin({"dri_plant", "eaf", "dri_storage"})].copy()
    rows.loc[rows["technology"] == "dri_plant", "fuel_type"] = "natural_gas"
    plant = SteelPlant.from_rows("steel_1", rows)
    forecasts = _steel_forecasts()
    forecasts["electricity_price"] = [1000.0, 1000.0, 0.0, 0.0]
    forecasts["natural_gas_price"] = [0.0, 0.0, 1000.0, 1000.0]

    result = plant.solve_horizon(config, forecasts, _signals())

    assert result["dri_storage_charge_t"].iloc[:2].sum() > 0
    assert result["dri_storage_discharge_t"].iloc[2:].sum() > 0
    assert result["dri_output_t"].iloc[:2].sum() > result["dri_input_t"].iloc[:2].sum()
    assert result["dri_input_t"].iloc[2:].sum() > result["dri_output_t"].iloc[2:].sum()


def test_quarter_hour_power_limits_are_converted_to_interval_energy() -> None:
    config = CaseConfig.from_case_dir(CASE_DIR)
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=True))
    model = plant._build_model(config, _steel_forecasts(), _signals())

    assert float(model.technology_blocks["electrolyser"].max_power) == pytest.approx(5.0)
    assert float(model.technology_blocks["dri_plant"].max_power) == pytest.approx(2.5)
    assert float(model.technology_blocks["eaf"].max_power) == pytest.approx(2.5)


def test_electrolyser_gate_forces_zero_power_when_closed(tmp_path: Path) -> None:
    """Regression guard for the electrolyser_allowed_col addition to _build_model.

    Self-contained (does not depend on CASE_DIR) so it stays runnable even when
    that fixture directory is unavailable.
    """
    config = _minimal_cost_min_config(tmp_path)
    plant = SteelPlant.from_rows("steel_1", _steel_rows(include_optional=True))
    forecasts = _steel_forecasts().copy()
    forecasts["__electrolyser_allowed"] = [True, False, True, False]
    signals = SteelDispatchSignals(
        electricity_price_col="electricity_price",
        natural_gas_price_col="natural_gas_price",
        hydrogen_price_col="hydrogen_price",
        iron_ore_price_col="iron_ore_price",
        lime_price_col="lime_price",
        co2_price_col="co2_price",
        steel_price_col="steel_price",
        electrolyser_allowed_col="__electrolyser_allowed",
    )

    result = plant.solve_horizon(config, forecasts, signals)

    gated_off = result.iloc[[1, 3]]
    assert gated_off["electrolyser_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        [0.0, 0.0]
    )
    assert result["steel_output_t"].sum() == pytest.approx(4.0)


def _minimal_cost_min_config(tmp_path: Path) -> CaseConfig:
    case_dir = tmp_path / "steel_cost_min_case"
    case_dir.mkdir(exist_ok=True)
    (case_dir / "config.yaml").write_text(
        """
cases:
  steel_cost_min_case:
    name: steel_cost_min_case
    country: DE
    timestep_minutes: 15
    simulation_start: "2025-01-01 00:00"
    simulation_end: "2025-01-01 01:00"
    additional_charges: false
    strategy:
      name: steel_cost_minimization
      dispatch:
        dispatch_method: pyomo
        rolling_horizon_enabled: true
        dispatch_horizon_hours: 1
        rolling_step_hours: 1
    solver:
      name: highs
      fallback_solvers: []
      tee: false
    market_sequence:
      - day_ahead
    markets:
      day_ahead:
        enabled: true
        product_resolution: 15min
        gate_close:
          day_relation: D-1
          time: "12:00"
        signals:
          price: electricity_price
""".strip(),
        encoding="utf-8",
    )
    return CaseConfig.from_case_dir(case_dir, study_case="steel_cost_min_case")


def _steel_rows(include_optional: bool) -> pd.DataFrame:
    shared = {
        "name": "steel_1",
        "unit_type": "steel_plant",
        "node": "north",
        "objective": "min_variable_cost",
        "steel_demand": 4.0,
    }
    rows = [
        {
            **shared,
            "technology": "dri_plant",
            "fuel_type": "hydrogen",
            "specific_hydrogen_consumption": 2.0,
            "specific_natural_gas_consumption": 3.0,
            "specific_electricity_consumption": 0.2,
            "specific_iron_ore_consumption": 1.5,
            "natural_gas_co2_factor": 0.2,
            "max_power": 10.0,
            "min_power": 0.0,
        },
        {
            **shared,
            "technology": "eaf",
            "specific_electricity_consumption": 0.5,
            "specific_dri_demand": 1.0,
            "specific_lime_demand": 0.05,
            "lime_co2_factor": 0.1,
            "max_power": 10.0,
            "min_power": 0.0,
        },
    ]
    if include_optional:
        rows.extend(
            [
                {
                    **shared,
                    "technology": "electrolyser",
                    "max_power": 20.0,
                    "min_power": 0.0,
                    "efficiency": 0.8,
                },
                {
                    **shared,
                    "technology": "hydrogen_buffer_storage",
                    "capacity": 20.0,
                    "min_soc": 0.0,
                    "max_soc": 1.0,
                    "initial_soc": 0.0,
                    "max_power_charge": 20.0,
                    "max_power_discharge": 20.0,
                    "efficiency_charge": 1.0,
                    "efficiency_discharge": 1.0,
                    "storage_loss_rate": 0.0,
                },
                {
                    **shared,
                    "technology": "dri_storage",
                    "capacity": 10.0,
                    "min_soc": 0.0,
                    "max_soc": 1.0,
                    "initial_soc": 0.0,
                    "max_power_charge": 10.0,
                    "max_power_discharge": 10.0,
                    "efficiency_charge": 1.0,
                    "efficiency_discharge": 1.0,
                    "storage_loss_rate": 0.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def _dri_bof_rows() -> pd.DataFrame:
    rows = _steel_rows(include_optional=False)
    rows.loc[rows["technology"] == "eaf", "technology"] = "bof"
    rows.loc[rows["technology"] == "bof", "specific_electricity_consumption"] = 0.2
    return rows


def _bf_bof_rows(fuel_type: str, include_electrolyser: bool = False) -> pd.DataFrame:
    shared = {
        "name": "steel_1",
        "unit_type": "steel_plant",
        "node": "north",
        "objective": "min_variable_cost",
        "steel_demand": 4.0,
    }
    rows = [
        {
            **shared,
            "technology": "bf_bof",
            "fuel_type": fuel_type,
            "specific_coal_consumption": 4.0,
            "specific_hydrogen_consumption": 2.0,
            "specific_natural_gas_consumption": 3.0,
            "specific_electricity_consumption": 0.2,
            "specific_iron_ore_consumption": 1.5,
            "specific_lime_demand": 0.05,
            "coal_co2_factor": 0.34,
            "natural_gas_co2_factor": 0.2,
            "lime_co2_factor": 0.1,
            "max_power": 10.0,
            "min_power": 0.0,
        }
    ]
    if include_electrolyser:
        rows.append(
            {
                **shared,
                "technology": "electrolyser",
                "max_power": 20.0,
                "min_power": 0.0,
                "efficiency": 0.8,
            }
        )
    return pd.DataFrame(rows)


def _steel_forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "electricity_price": [30.0, 40.0, 80.0, 100.0],
            "coal_price": [15.0] * 4,
            "natural_gas_price": [50.0] * 4,
            "hydrogen_price": [70.0] * 4,
            "iron_ore_price": [100.0] * 4,
            "lime_price": [20.0] * 4,
            "co2_price": [80.0] * 4,
            "steel_price": [600.0] * 4,
        },
        index=pd.date_range("2025-01-01", periods=4, freq="15min"),
    )


def _extended_steel_forecasts() -> pd.DataFrame:
    periods = 8
    return pd.DataFrame(
        {
            "electricity_price": [20.0, 25.0, 100.0, 110.0, 30.0, 35.0, 90.0, 95.0],
            "coal_price": [15.0] * periods,
            "natural_gas_price": [50.0] * periods,
            "hydrogen_price": [70.0] * periods,
            "iron_ore_price": [100.0] * periods,
            "lime_price": [20.0] * periods,
            "co2_price": [80.0] * periods,
            "steel_price": [600.0] * periods,
        },
        index=pd.date_range("2025-01-01", periods=periods, freq="15min"),
    )


def _rolling_config(horizon_hours: float, step_hours: float) -> CaseConfig:
    config = CaseConfig.from_case_dir(CASE_DIR)
    dispatch = config.case["strategy"]["dispatch"]
    dispatch["rolling_horizon_enabled"] = True
    dispatch["dispatch_horizon_hours"] = horizon_hours
    dispatch["rolling_step_hours"] = step_hours
    return config


def _signals() -> SteelDispatchSignals:
    return SteelDispatchSignals(
        electricity_price_col="electricity_price",
        natural_gas_price_col="natural_gas_price",
        hydrogen_price_col="hydrogen_price",
        iron_ore_price_col="iron_ore_price",
        lime_price_col="lime_price",
        co2_price_col="co2_price",
        steel_price_col="steel_price",
    )
