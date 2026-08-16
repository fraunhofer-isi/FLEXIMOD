# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pytest

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.plants.cement_plant import (
    CementDispatchSignals,
    CementPlant,
    CementRollingState,
    CementStageState,
)
from flexi_mod.plants.factory import build_plants
from flexi_mod.plants.technologies import (
    CementKiln,
    CementPreheater,
    LEILACCementCalciner,
    OxyfuelCementCalciner,
    SimpleCementCalciner,
)
from flexi_mod.simulation.simulation_runner import SimulationRunner


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    """A minimal cement case folder holding just ``config.yaml``.

    Built here rather than pointed at ``data/input/``, which is gitignored and holds no
    cement case, so these tests do not depend on local-only scenario folders.
    """
    case_dir = tmp_path / "cement_case"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_cement_config(), encoding="utf-8")
    return case_dir


def test_cement_plant_builds_preheater_calciner_kiln_route() -> None:
    plant = CementPlant.from_rows("cement_1", _cement_rows())

    assert isinstance(plant.components["preheater"], CementPreheater)
    assert isinstance(plant.components["simple_calciner"], SimpleCementCalciner)
    assert isinstance(plant.components["kiln"], CementKiln)
    assert plant.cement_route == "preheater_simple_calciner_kiln"


def test_shared_plant_factory_selects_cement_plant() -> None:
    plants = build_plants(_cement_rows())

    assert len(plants) == 1
    assert isinstance(plants[0], CementPlant)


def test_cement_plant_rejects_grinding_mill_for_now() -> None:
    rows = pd.concat(
        [
            _cement_rows(),
            pd.DataFrame(
                [
                    {
                        "name": "cement_1",
                        "unit_type": "cement_plant",
                        "technology": "cement_mill",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="grinding mill integration is intentionally excluded"):
        CementPlant.from_rows("cement_1", rows)


def test_cement_preheater_calciner_kiln_solve_and_balances(case_dir: Path) -> None:
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _cement_rows())

    result = plant.solve_horizon(config, _cement_forecasts(include_coal=False), _signals())

    assert result["clinker_output_t"].sum() == pytest.approx(4.0)
    assert result["preheater_raw_meal_output_t"].sum() == pytest.approx(6.0)
    assert result["simple_calciner_clinker_output_t"].sum() == pytest.approx(4.0)
    assert result["kiln_clinker_output_t"].sum() == pytest.approx(4.0)
    assert result["natural_gas_consumption_MWh"].sum() > 0
    assert result["coal_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["hydrogen_consumption_MWh"].sum() == pytest.approx(0.0)
    assert result["co2_emissions_t"].sum() > 0


def test_cement_hydrogen_with_electrolyser_is_constrained_by_electrolyser_output(
    case_dir: Path,
) -> None:
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _hydrogen_cement_rows(include_electrolyser=True))

    result = plant.solve_horizon(config, _cement_forecasts(include_coal=False), _signals())

    assert result["clinker_output_t"].sum() == pytest.approx(4.0)
    assert result["hydrogen_consumption_MWh"].sum() > 0
    assert result["electrolyser_hydrogen_output_MWh"].sum() == pytest.approx(
        result["hydrogen_consumption_MWh"].sum()
    )


def test_cement_hydrogen_without_electrolyser_uses_external_hydrogen(case_dir: Path) -> None:
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _hydrogen_cement_rows(include_electrolyser=False))

    result = plant.solve_horizon(config, _cement_forecasts(include_coal=False), _signals())

    assert result["hydrogen_consumption_MWh"].sum() > 0
    assert "electrolyser_hydrogen_output_MWh" not in result


def test_cement_forecast_discovery_and_conditional_coal_price(case_dir: Path) -> None:
    config = CaseConfig.from_case_dir(case_dir)
    loader = DataLoader(config, input_dir=case_dir)
    total_rows = _cement_rows()

    total_required = loader.required_forecast_columns(total_rows)

    # The per-timestep demand column is always required; there is no scalar total that
    # could stand in for it.
    assert "cement_1_clinker_demand" in total_required
    assert "coal_price" not in total_required

    profile_rows = total_rows.copy()
    profile_rows["demand"] = "custom_clinker_demand"
    profile_required = loader.required_forecast_columns(profile_rows)
    assert "custom_clinker_demand" in profile_required

    coal_rows = total_rows.copy()
    coal_rows.loc[coal_rows["technology"] == "kiln", "fossil_ng_share"] = 0.0
    coal_required = loader.required_forecast_columns(coal_rows)
    assert "coal_price" in coal_required


def test_cement_runner_uses_day_ahead_price_and_writes_outputs(tmp_path: Path) -> None:
    case_dir = tmp_path / "cement_runner"
    output_dir = tmp_path / "output"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_cement_config(), encoding="utf-8")
    _cement_rows().to_csv(case_dir / "plants.csv", index=False)
    forecasts = _cement_forecasts(include_coal=False).copy()
    forecasts = forecasts.rename(columns={"electricity_price": "connected_DA_price"})
    forecasts.reset_index(names="datetime").to_csv(case_dir / "forecasts_df.csv", index=False)

    outputs = SimulationRunner(case_dir, output_dir=output_dir).run()

    assert set(outputs) == {"dispatch_results", "summary_indicators"}
    dispatch = pd.read_csv(outputs["dispatch_results"])
    summary = pd.read_csv(outputs["summary_indicators"])
    assert dispatch["clinker_output_t"].sum() == pytest.approx(4.0)
    assert summary.loc[0, "total_clinker_production_t"] == pytest.approx(4.0)
    assert summary.loc[0, "final_clinker_demand_balance_t"] == pytest.approx(0.0)
    assert not (output_dir / "market_ledger.csv").exists()
    assert not (output_dir / "storage_cost_ledger.csv").exists()


def test_cement_runner_applies_german_regulatory_charges(tmp_path: Path) -> None:
    case_dir = tmp_path / "cement_grid_fees"
    output_dir = tmp_path / "output"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(_cement_config(additional_charges=True), encoding="utf-8")
    _cement_rows().to_csv(case_dir / "plants.csv", index=False)
    forecasts = _cement_forecasts(include_coal=False).copy()
    forecasts = forecasts.rename(columns={"electricity_price": "connected_DA_price"})
    forecasts.reset_index(names="datetime").to_csv(case_dir / "forecasts_df.csv", index=False)
    _additional_charges().to_csv(case_dir / "additional_charges.csv", index=False)

    outputs = SimulationRunner(
        case_dir,
        output_dir=output_dir,
        assumed_grid_tier="low",
    ).run()

    dispatch = pd.read_csv(outputs["dispatch_results"])
    summary = pd.read_csv(outputs["summary_indicators"])
    grid_fees = pd.read_csv(outputs["grid_fee_summary"])

    assert dispatch["actual_electricity_consumption_MWh"].sum() == pytest.approx(
        dispatch["total_electricity_consumption_MWh"].sum()
    )
    assert dispatch["additional_electricity_charge_EUR_per_MWh_el"].gt(0.0).all()
    assert dispatch["additional_electricity_charges_cost_EUR"].sum() > 0.0
    assert grid_fees.loc[0, "grid_fee_total_EUR"] > 0.0
    assert summary.loc[0, "grid_fee_total_EUR"] == pytest.approx(
        grid_fees.loc[0, "grid_fee_total_EUR"]
    )
    assert summary.loc[0, "net_operating_cost_incl_grid_fees_EUR"] == pytest.approx(
        summary.loc[0, "net_operating_cost_EUR"]
        - summary.loc[0, "total_additional_electricity_charges_cost_EUR"]
        + summary.loc[0, "grid_fee_total_EUR"]
    )


@pytest.mark.parametrize(
    ("stages", "route", "terminal"),
    [
        (
            ["preheater", "simple_calciner", "kiln"],
            "preheater_simple_calciner_kiln",
            "kiln",
        ),
        (["preheater", "simple_calciner"], "preheater_simple_calciner", "simple_calciner"),
        (["preheater", "kiln"], "preheater_kiln", "kiln"),
        (["simple_calciner", "kiln"], "simple_calciner_kiln", "kiln"),
        (["kiln"], "kiln", "kiln"),
        (["simple_calciner"], "simple_calciner", "simple_calciner"),
    ],
)
def test_cement_routes_are_named_and_pick_their_terminal_stage(
    stages: list[str], route: str, terminal: str
) -> None:
    plant = CementPlant.from_rows("cement_1", _rows_for_stages(stages))

    assert plant.cement_route == route
    assert plant._terminal_technology_name() == terminal


@pytest.mark.parametrize(
    ("stages", "extra", "message"),
    [
        (["preheater"], None, "at least one terminal technology"),
        (["kiln"], "thermal_storage", "requires a calciner"),
        (["simple_calciner", "kiln"], "hydrogen_buffer_storage", "without an electrolyser"),
    ],
)
def test_cement_route_guards_reject_unbuildable_component_sets(
    stages: list[str], extra: str | None, message: str
) -> None:
    rows = _rows_for_stages(stages)
    if extra is not None:
        rows = pd.concat([rows, _storage_rows(extra)], ignore_index=True)

    with pytest.raises(ValueError, match=message):
        CementPlant.from_rows("cement_1", rows)


def test_preheater_kiln_route_feeds_raw_meal_straight_to_the_kiln(case_dir: Path) -> None:
    """Without a calciner the kiln performs the calcination reaction itself.

    The preheater's waste-heat inlet is fed from the kiln's own clinker output while that
    output is set by the preheater's raw meal, so this route closes a loop that has to
    stay solvable rather than collapse to zero production.
    """
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _rows_for_stages(["preheater", "kiln"]))

    result = plant.solve_horizon(config, _cement_forecasts(include_coal=False), _signals())

    assert plant.cement_route == "preheater_kiln"
    assert result["clinker_output_t"].sum() == pytest.approx(4.0)
    assert result["kiln_clinker_output_t"].sum() == pytest.approx(4.0)
    # raw_meal_out == receiving stage clinker_out * raw_meal_to_clinker_ratio (1.5)
    assert result["preheater_raw_meal_output_t"].sum() == pytest.approx(6.0)


def test_natural_gas_co2_factor_accepts_the_ng_co2_factor_spelling() -> None:
    """DSM input files carry the emission factor as ``ng_co2_factor``.

    Reading only ``natural_gas_co2_factor`` silently fell back to the default, so a
    non-default value in that column was discarded without warning.
    """
    rows = _cement_rows().drop(columns=["natural_gas_co2_factor"])
    rows["ng_co2_factor"] = 0.25

    plant = CementPlant.from_rows("cement_1", rows)

    for stage in ("preheater", "simple_calciner", "kiln"):
        assert plant.components[stage].natural_gas_co2_factor_t_per_mwh == 0.25


def test_physical_system_can_be_attached_twice_to_one_model(case_dir: Path) -> None:
    """The container form must support two trajectories side by side in one model.

    A market model needs the committed plan and a hypothetical "what if the whole aFRR
    bid is called" twin. Pyomo gives every component exactly one parent, so attaching
    twice only works if the destination is a parameter rather than hard-coded - and the
    rules must reach shared values (demand, ratios, waste heat) through the model rather
    than through their own container.
    """
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _cement_rows())
    forecasts = _cement_forecasts(include_coal=False)

    model = plant._build_model(config, forecasts, _signals())
    model.twin = pyo.Block()
    plant._add_physical_system(
        model,
        model.twin,
        model.T,
        config.timestep_minutes / 60.0,
        plant._initial_rolling_state(),
    )

    # Both trajectories exist with their own variables, and the shared parameters stayed
    # on the model rather than being duplicated into either one.
    for trajectory in (model, model.twin):
        assert trajectory.technology_blocks["kiln"].clinker_out is not None
        assert trajectory.total_power_input is not None
        assert trajectory.variable_cost is not None
    assert model.twin.total_power_input is not model.total_power_input
    assert not hasattr(model.twin, "clinker_demand_per_timestep")

    # The twin carries the same physics: solving still meets demand on both.
    plant._solve_model(config, forecasts, model)
    for trajectory in (model, model.twin):
        produced = [pyo.value(trajectory.technology_blocks["kiln"].clinker_out[t]) for t in model.T]
        assert produced == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_scalar_clinker_demand_is_rejected() -> None:
    rows = _cement_rows()
    rows["clinker_demand"] = 4.0

    with pytest.raises(ValueError, match="sets a scalar 'clinker_demand'"):
        CementPlant.from_rows("cement_1", rows)


def test_missing_per_timestep_demand_column_fails_clearly(case_dir: Path) -> None:
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _cement_rows())
    forecasts = _cement_forecasts(include_coal=False).drop(columns=["cement_1_clinker_demand"])

    with pytest.raises(ValueError, match="missing its per-timestep clinker demand column"):
        plant.solve_horizon(config, forecasts, _signals())


def test_hourly_demand_is_met_step_by_step_not_smeared(case_dir: Path) -> None:
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _cement_rows())
    forecasts = _cement_forecasts(include_coal=False)
    # A spike in the most expensive hour: a horizon-total target would shift this
    # production into the cheap hours instead of producing it when it is demanded.
    forecasts["cement_1_clinker_demand"] = [0.5, 0.5, 3.0, 0.5]

    result = plant.solve_horizon(config, forecasts, _signals())

    assert result["clinker_output_t"].to_numpy() == pytest.approx([0.5, 0.5, 3.0, 0.5])


def test_demand_above_stage_capacity_is_infeasible_rather_than_shifted(case_dir: Path) -> None:
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _cement_rows())
    forecasts = _cement_forecasts(include_coal=False)
    # The kiln can make at most max_heat_out / specific_heat_demand = 10 / 0.8 t per step.
    forecasts["cement_1_clinker_demand"] = [1.0, 1.0, 1.0, 40.0]

    with pytest.raises(RuntimeError, match="could not be solved"):
        plant.solve_horizon(config, forecasts, _signals())


def test_rolling_reproduces_single_shot_when_the_window_is_the_whole_period(
    case_dir: Path,
) -> None:
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _cement_rows_with(_storage_rows("thermal_storage")))
    forecasts = _long_cement_forecasts()

    single = plant.solve_horizon(config, forecasts, _signals())
    rolling = plant.solve_rolling(
        _rolling_config(config, horizon_hours=2.0, step_hours=2.0), forecasts, _signals()
    )

    assert rolling["clinker_output_t"].to_numpy() == pytest.approx(
        single["clinker_output_t"].to_numpy()
    )
    assert rolling["variable_cost_EUR"].sum() == pytest.approx(single["variable_cost_EUR"].sum())
    assert rolling["rolling_window"].nunique() == 1


def test_rolling_carries_thermal_storage_fill_across_commit_boundaries(case_dir: Path) -> None:
    """The thermal storage reports absolute MWh, unlike the fractional hydrogen buffer.

    A window that seeded the wrong context key would restart every window from the
    component's initial fill instead of continuing from the last committed one.
    """
    config = _rolling_config(CaseConfig.from_case_dir(case_dir), horizon_hours=1.0, step_hours=0.5)
    plant = CementPlant.from_rows("cement_1", _cement_rows_with(_storage_rows("thermal_storage")))

    result = plant.solve_rolling(config, _long_cement_forecasts(), _signals())

    assert result["rolling_window"].nunique() > 1
    _assert_storage_balance(
        soc=result["thermal_storage_soc_MWh"],
        charge=result["thermal_storage_charge_MWh"],
        discharge=result["thermal_storage_discharge_MWh"],
        capacity=1.0,
        initial_soc=0.0,
    )


def test_rolling_carries_hydrogen_storage_fill_across_commit_boundaries(case_dir: Path) -> None:
    """The hydrogen buffer reports a fraction of capacity, unlike the thermal storage."""
    config = _rolling_config(CaseConfig.from_case_dir(case_dir), horizon_hours=1.0, step_hours=0.5)
    rows = _hydrogen_cement_rows(include_electrolyser=True)
    rows = pd.concat([rows, _storage_rows("hydrogen_buffer_storage")], ignore_index=True)
    plant = CementPlant.from_rows("cement_1", rows)

    result = plant.solve_rolling(config, _long_cement_forecasts(), _signals())

    assert result["rolling_window"].nunique() > 1
    _assert_storage_balance(
        soc=result["hydrogen_storage_soc"],
        charge=result["hydrogen_storage_charge_MWh"],
        discharge=result["hydrogen_storage_discharge_MWh"],
        capacity=10.0,
        initial_soc=0.5,
    )


def test_rolling_carries_oxyfuel_calciner_state_across_commit_boundaries(case_dir: Path) -> None:
    """Regression guard: state carry-over must resolve the calciner's actual key.

    ``_state_after_commit`` rebuilds each stage's result-column name from the technology
    key it stored the stage under. For an oxyfuel calciner that key is
    ``oxyfuel_calciner``, but its columns are still reported as ``simple_calciner_*`` -
    reading the wrong column name raised a ``KeyError`` on the second rolling window,
    since the first window never exercises carry-over at all.
    """
    config = _rolling_config(CaseConfig.from_case_dir(case_dir), horizon_hours=1.0, step_hours=0.5)
    plant = CementPlant.from_rows("cement_1", _oxyfuel_cement_rows())
    forecasts = _long_cement_forecasts()
    forecasts["oxygen_price"] = 20.0

    result = plant.solve_rolling(config, forecasts, _signals())

    assert result["rolling_window"].nunique() > 1
    assert result["clinker_output_t"].sum() > 0.0
    assert (result["simple_calciner_oxygen_consumption_t"] >= 0.0).all()


def test_rolling_serves_out_minimum_downtime_inherited_from_the_previous_window(
    case_dir: Path,
) -> None:
    """A stage shut down late in one window stays down at the start of the next.

    The per-window min up/down constraints only see their own window, so without the
    residual-status constraint the kiln could restart immediately at every boundary.
    """
    config = CaseConfig.from_case_dir(case_dir)
    rows = _cement_rows()
    rows.loc[rows["technology"] == "kiln", "min_down_time"] = 4
    plant = CementPlant.from_rows("cement_1", rows)
    forecasts = _cement_forecasts(include_coal=False)

    def kiln_state(consecutive_status_steps: int) -> CementRollingState:
        return CementRollingState(
            stages={
                "kiln": CementStageState(
                    heat_out_mwh=0.0,
                    operational_status=0,
                    consecutive_status_steps=consecutive_status_steps,
                )
            }
        )

    # Entering the window having served only 1 of the 4 required down steps, the kiln is
    # held off for 3 more while every step demands clinker, so there is no feasible plan.
    with pytest.raises(RuntimeError, match="could not be solved"):
        plant.solve_horizon(config, forecasts, _signals(), initial_state=kiln_state(1))

    # Once the downtime has been served the same window solves normally.
    result = plant.solve_horizon(config, forecasts, _signals(), initial_state=kiln_state(4))
    assert result["clinker_output_t"].to_numpy() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_thermal_storage_charging_is_priced_into_the_objective(case_dir: Path) -> None:
    """Regression guard: the storage block must expose ``operating_cost``.

    It reports its electricity under ``electricity_cost``; when only that existed the
    plant's cost sum skipped the block entirely and the optimiser charged the store for
    free, displacing calciner fuel at no cost.
    """
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _cement_rows_with(_storage_rows("thermal_storage")))
    forecasts = _cement_forecasts(include_coal=False)
    # Electricity far dearer than the gas the discharged heat would displace.
    forecasts["electricity_price"] = [500.0] * 4
    forecasts["natural_gas_price"] = [10.0] * 4

    result = plant.solve_horizon(config, forecasts, _signals())

    assert result["thermal_storage_charge_MWh"].sum() == pytest.approx(0.0, abs=1e-6)


def test_oxyfuel_calciner_is_registered_and_builds() -> None:
    plant = CementPlant.from_rows("cement_1", _oxyfuel_cement_rows())

    assert isinstance(plant.components["oxyfuel_calciner"], OxyfuelCementCalciner)
    assert plant.cement_route == "preheater_oxyfuel_calciner_kiln"


def test_oxyfuel_calciner_rejects_electric_fuel_type() -> None:
    rows = _oxyfuel_cement_rows()
    rows.loc[rows["technology"] == "oxyfuel_calciner", "fuel_type"] = "electricity"

    with pytest.raises(ValueError, match="cannot use fuel_type='electricity'"):
        CementPlant.from_rows("cement_1", rows)


def test_oxyfuel_calciner_csv_row_requires_an_oxygen_coefficient_for_its_fuel() -> None:
    """A fossil calciner row with no natural-gas oxygen coefficient is a config error.

    Silently defaulting it to zero would let the LP burn gas and capture no oxygen for
    it, which is physically wrong rather than merely imprecise.
    """
    rows = _cement_rows()
    is_calciner = rows["technology"] == "simple_calciner"
    rows.loc[is_calciner, "technology"] = "oxyfuel_calciner"

    with pytest.raises(ValueError, match="Missing required numeric plant parameter"):
        CementPlant.from_rows("cement_1", rows)


def test_oxyfuel_calciner_rejects_an_explicit_zero_oxygen_coefficient() -> None:
    """Direct construction (bypassing ``from_row``) must also be validated.

    A field explicitly set to zero is not "missing", so only ``__post_init__`` - not
    ``from_row``'s CSV-completeness check - can catch this one.
    """
    with pytest.raises(ValueError, match="natural_gas_oxygen_demand must be positive"):
        OxyfuelCementCalciner(
            max_heat_out_mw=20.0,
            specific_heat_demand_mwh_per_t=0.7,
            fuel_type="fossil",
            fossil_ng_share=1.0,
            natural_gas_oxygen_demand_t_per_mwh=0.0,
        )


def test_oxyfuel_calciner_oxygen_tracks_fuel_switching(case_dir: Path) -> None:
    """Oxygen demand reads off actual fuel burned, not off clinker output.

    The fixture calciner is 100% fossil with ``eta_fossil=1.0`` and no coal
    (``fossil_ng_share=1.0``), so ``heat_out == natural_gas_in`` exactly; oxygen
    consumption must then equal ``heat_out`` times the coefficient - proving the
    constraint is wired to actual fuel input, not to a fixed per-tonne clinker rate.
    """
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _oxyfuel_cement_rows(natural_gas_oxygen_demand=0.2))

    result = plant.solve_horizon(
        config, _cement_forecasts(include_coal=False, include_oxygen=True), _signals()
    )

    assert result["simple_calciner_heat_output_MWh"].sum() > 0.0, "fixture must burn fuel"
    assert result["simple_calciner_oxygen_consumption_t"].to_numpy() == pytest.approx(
        result["simple_calciner_heat_output_MWh"].to_numpy() * 0.2
    )


def test_oxyfuel_calciner_oxygen_cost_enters_the_objective(case_dir: Path) -> None:
    """A nonzero oxygen price must raise cost by exactly ``oxygen_in * oxygen_price``."""
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows("cement_1", _oxyfuel_cement_rows())
    free_oxygen = _cement_forecasts(include_coal=False, include_oxygen=True)
    free_oxygen["oxygen_price"] = 0.0
    priced_oxygen = _cement_forecasts(include_coal=False, include_oxygen=True)
    priced_oxygen["oxygen_price"] = 20.0

    free_result = plant.solve_horizon(config, free_oxygen, _signals())
    priced_result = plant.solve_horizon(config, priced_oxygen, _signals())

    expected_extra_cost = (priced_result["simple_calciner_oxygen_consumption_t"] * 20.0).sum()
    actual_extra_cost = (
        priced_result["variable_cost_EUR"].sum() - free_result["variable_cost_EUR"].sum()
    )
    assert actual_extra_cost == pytest.approx(expected_extra_cost, rel=1e-6)
    assert expected_extra_cost > 0.0


def test_leilac_calciner_is_registered_and_builds() -> None:
    plant = CementPlant.from_rows("cement_1", _leilac_cement_rows())

    assert isinstance(plant.components["leilac_calciner"], LEILACCementCalciner)
    assert plant.cement_route == "preheater_leilac_calciner_kiln"


@pytest.mark.parametrize("efficiency", [-0.01, 1.01])
def test_leilac_calciner_rejects_invalid_separation_efficiency(efficiency: float) -> None:
    rows = _leilac_cement_rows(direct_separation_efficiency=efficiency)

    with pytest.raises(ValueError, match="must be between 0 and 1"):
        CementPlant.from_rows("cement_1", rows)


def test_electric_leilac_calciner_separates_only_process_co2(case_dir: Path) -> None:
    config = CaseConfig.from_case_dir(case_dir)
    rows = _leilac_cement_rows(
        direct_separation_efficiency=0.95,
        stages=["simple_calciner"],
    )
    rows.loc[rows["technology"] == "leilac_calciner", "fuel_type"] = "electricity"
    rows.loc[rows["technology"] == "leilac_calciner", "eta_electric"] = 1.0
    plant = CementPlant.from_rows("cement_1", rows)

    result = plant.solve_horizon(config, _cement_forecasts(include_coal=False), _signals())

    # Four tonnes of clinker generate 2 t process CO2. At 95% direct separation,
    # 1.9 t is separated and the remaining 0.1 t is emitted.
    assert result["co2_separated_t"].sum() == pytest.approx(1.9)
    assert result["simple_calciner_process_co2_emissions_t"].sum() == pytest.approx(0.1)
    assert result["co2_emissions_t"].sum() == pytest.approx(0.1)


def test_fossil_leilac_calciner_keeps_combustion_co2_in_emissions(case_dir: Path) -> None:
    config = CaseConfig.from_case_dir(case_dir)
    plant = CementPlant.from_rows(
        "cement_1",
        _leilac_cement_rows(
            direct_separation_efficiency=0.95,
            stages=["simple_calciner"],
        ),
    )

    result = plant.solve_horizon(config, _cement_forecasts(include_coal=False), _signals())

    # The process balance is the same 1.9 t separated and 0.1 t residual. The
    # calciner also burns 2 MWh gas, whose 0.4 t CO2 remains fully emitted.
    assert result["co2_separated_t"].sum() == pytest.approx(1.9)
    assert result["simple_calciner_process_co2_emissions_t"].sum() == pytest.approx(0.1)
    assert result["natural_gas_consumption_MWh"].sum() == pytest.approx(2.0)
    assert result["co2_emissions_t"].sum() == pytest.approx(0.5)


def _assert_storage_balance(
    soc: pd.Series,
    charge: pd.Series,
    discharge: pd.Series,
    capacity: float,
    initial_soc: float,
) -> None:
    """Every fill level follows from the previous one, commit boundaries included."""
    previous = initial_soc
    for step in range(len(soc)):
        expected = previous + (charge.iloc[step] - discharge.iloc[step]) / capacity
        assert soc.iloc[step] == pytest.approx(expected, abs=1e-6)
        previous = soc.iloc[step]


def _cement_rows_with(extra: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([_cement_rows(), extra], ignore_index=True)


def _rolling_config(config: CaseConfig, horizon_hours: float, step_hours: float) -> CaseConfig:
    dispatch = config.case["strategy"]["dispatch"]
    dispatch["rolling_horizon_enabled"] = True
    dispatch["dispatch_horizon_hours"] = horizon_hours
    dispatch["rolling_step_hours"] = step_hours
    return config


def _long_cement_forecasts() -> pd.DataFrame:
    """Eight quarter-hours with a price swing the storage can arbitrage."""
    periods = 8
    return pd.DataFrame(
        {
            "electricity_price": [30.0, 40.0, 80.0, 100.0, 20.0, 25.0, 90.0, 95.0],
            "natural_gas_price": [50.0] * periods,
            "hydrogen_price": [70.0] * periods,
            "co2_price": [80.0] * periods,
            "cement_1_clinker_demand": [1.0] * periods,
        },
        index=pd.date_range("2025-01-01", periods=periods, freq="15min"),
    )


def _cement_rows() -> pd.DataFrame:
    shared = {
        "name": "cement_1",
        "unit_type": "cement_plant",
        "node": "north",
        "objective": "min_variable_cost",
        "raw_meal_to_clinker_ratio": 1.5,
        "waste_heat_per_t_clinker": 0.0,
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "technology": "preheater",
                "fuel_type": "electricity",
                "max_heat_out": 10.0,
                "specific_heat_demand": 0.1,
                "eta_electric": 1.0,
            },
            {
                **shared,
                "technology": "simple_calciner",
                "fuel_type": "fossil",
                "fossil_ng_share": 1.0,
                "max_heat_out": 10.0,
                "specific_heat_demand": 0.5,
                "eta_fossil": 1.0,
                "calcination_emission_factor": 0.5,
                "natural_gas_co2_factor": 0.2,
            },
            {
                **shared,
                "technology": "kiln",
                "fuel_type": "fossil",
                "fossil_ng_share": 1.0,
                "max_heat_out": 10.0,
                "specific_heat_demand": 0.8,
                "eta_fossil": 1.0,
                "natural_gas_co2_factor": 0.2,
            },
        ]
    )


def _rows_for_stages(stages: list[str]) -> pd.DataFrame:
    """The standard fixture restricted to *stages*, for exercising each cement route."""
    rows = _cement_rows()
    return rows.loc[rows["technology"].isin(stages)].reset_index(drop=True)


def _storage_rows(technology: str) -> pd.DataFrame:
    shared = {
        "name": "cement_1",
        "unit_type": "cement_plant",
        "node": "north",
        "objective": "min_variable_cost",
        "technology": technology,
        "max_power_charge": 5.0,
        "max_power_discharge": 5.0,
        "efficiency_charge": 1.0,
        "efficiency_discharge": 1.0,
    }
    if technology == "thermal_storage":
        shared |= {"max_capacity": 10.0, "min_capacity": 0.0, "initial_soc": 0.0}
    else:
        shared |= {"capacity": 10.0, "min_soc": 0.0, "max_soc": 1.0, "initial_soc": 0.5}
    return pd.DataFrame([shared])


def _hydrogen_cement_rows(include_electrolyser: bool) -> pd.DataFrame:
    rows = _cement_rows()
    rows.loc[rows["technology"].isin(["simple_calciner", "kiln"]), "fuel_type"] = "hydrogen"
    if include_electrolyser:
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "name": "cement_1",
                            "unit_type": "cement_plant",
                            "technology": "electrolyser",
                            "node": "north",
                            "objective": "min_variable_cost",
                            "max_power": 20.0,
                            "min_power": 0.0,
                            "efficiency": 1.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return rows


def _cement_forecasts(include_coal: bool, include_oxygen: bool = False) -> pd.DataFrame:
    data = {
        "electricity_price": [30.0, 40.0, 80.0, 100.0],
        "natural_gas_price": [50.0] * 4,
        "hydrogen_price": [70.0] * 4,
        "co2_price": [80.0] * 4,
        "cement_1_clinker_demand": [1.0] * 4,
    }
    if include_coal:
        data["coal_price"] = [15.0] * 4
    if include_oxygen:
        data["oxygen_price"] = [20.0] * 4
    return pd.DataFrame(data, index=pd.date_range("2025-01-01", periods=4, freq="15min"))


def _oxyfuel_cement_rows(*, natural_gas_oxygen_demand: float = 0.2) -> pd.DataFrame:
    """The standard fixture with an oxyfuel calciner in place of the simple one."""
    rows = _cement_rows()
    is_calciner = rows["technology"] == "simple_calciner"
    rows.loc[is_calciner, "technology"] = "oxyfuel_calciner"
    rows.loc[is_calciner, "natural_gas_oxygen_demand"] = natural_gas_oxygen_demand
    return rows


def _leilac_cement_rows(
    *,
    direct_separation_efficiency: float = 0.95,
    stages: list[str] | None = None,
) -> pd.DataFrame:
    """The selected route with a LEILAC calciner in place of the simple one."""
    rows = _cement_rows() if stages is None else _rows_for_stages(stages)
    is_calciner = rows["technology"] == "simple_calciner"
    rows.loc[is_calciner, "technology"] = "leilac_calciner"
    rows.loc[is_calciner, "direct_separation_efficiency"] = direct_separation_efficiency
    return rows


def _signals() -> CementDispatchSignals:
    return CementDispatchSignals(electricity_price_col="electricity_price")


def _cement_config(*, additional_charges: bool = False) -> str:
    return f"""
cases:
  cement_runner:
    name: cement_runner
    country: DE
    timestep_minutes: 15
    simulation_start: "2025-01-01 00:00"
    simulation_end: "2025-01-01 00:45"
    additional_charges: {str(additional_charges).lower()}
    strategy:
      name: cement_cost_minimization
      dispatch:
        dispatch_method: pyomo
    solver:
      name: highs
      fallback_solvers: []
      tee: false
    market_sequence:
      - day_ahead
    markets:
      day_ahead:
        enabled: true
        signals:
          price: connected_DA_price
""".strip()


def _additional_charges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component": [
                "grid_energy_charge_high",
                "grid_energy_charge_low",
                "grid_capacity_charge_high",
                "grid_capacity_charge_low",
                "special_network_use_a",
                "special_network_use_b",
                "chp_surcharge",
                "offshore_grid_levy",
                "concession_fee",
                "electricity_tax",
            ],
            "unit": [
                "EUR/MWh",
                "EUR/MWh",
                "EUR/MW.a",
                "EUR/MW.a",
                "EUR/MWh",
                "EUR/MWh",
                "EUR/MWh",
                "EUR/MWh",
                "EUR/MWh",
                "EUR/MWh",
            ],
            "cement_1": [8.0, 9.0, 100.0, 120.0, 2.0, 0.5, 0.1, 0.2, 0.3, 0.4],
        }
    )
