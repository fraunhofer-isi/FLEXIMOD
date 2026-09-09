# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.ledgers.market_ledger import MarketLedger
from flexi_mod.ledgers.storage_cost_ledger import StorageCostLedger
from flexi_mod.markets import BaseMarket, build_markets
from flexi_mod.markets.afrr_energy import AFRRDownEnergyMarket
from flexi_mod.plants.building import Building
from flexi_mod.plants.factory import build_plants
from flexi_mod.plants.steam_generation_plant import DispatchSignals, SteamGenerationPlant
from flexi_mod.regulations import GridFeeResult, build_grid_fee_regulation
from flexi_mod.strategies import build_strategy
from flexi_mod.strategies.building_strategy import BuildingStrategy
from flexi_mod.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy
from flexi_mod.visualisation.analytics import calculate_summary_indicators
from flexi_mod.visualisation.plots import create_case_plots

DAY_AHEAD = "day_ahead"
INTRADAY_CONTINUOUS = "intraday_continuous"
AFRR_ENERGY = "afrr_energy"
AFRR_CAPACITY = "afrr_capacity"


@dataclass(frozen=True)
class OutputOptions:
    save_dispatch_results: bool = True
    save_market_ledger: bool = True
    save_storage_cost_ledger: bool = True
    save_summary_indicators: bool = True
    create_plots: bool = True


@dataclass(frozen=True)
class DecisionWindow:
    """One market-calendar decision window and its committed output slice."""

    number: int
    forecasts: pd.DataFrame
    commit_index: pd.DatetimeIndex


class SimulationRunner:
    """Coordinate data loading, sequential market stages, dispatch and outputs."""

    def __init__(
        self,
        case_dir: str | Path,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        plants_file: str = "plants.csv",
        forecasts_file: str = "forecasts_df.csv",
        study_case: str | None = None,
        output_options: OutputOptions | None = None,
        progress_callback: Callable[[str], None] | None = None,
        assumed_grid_tier: str = "high",
    ):
        self.config = CaseConfig.from_case_dir(case_dir, study_case=study_case)
        self.assumed_grid_tier = assumed_grid_tier
        self.input_dir = Path(input_dir).resolve() if input_dir else Path(case_dir).resolve()
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else self.config.project_root / "data" / "output" / self.config.output_folder_name
        )
        self.output_options = output_options or OutputOptions()
        self._progress_callback = progress_callback
        self.markets = build_markets(self.config)
        self.loader = DataLoader(
            self.config,
            input_dir=self.input_dir,
            plants_file=plants_file,
            forecasts_file=forecasts_file,
        )

    def run(self) -> dict[str, Path | list[Path]]:
        self._progress("Loading input data")
        plants_df = self.loader.load_plants()
        plants = build_plants(plants_df)
        building_plants = [plant for plant in plants if isinstance(plant, Building)]
        steam_plants = [plant for plant in plants if isinstance(plant, SteamGenerationPlant)]
        if building_plants and steam_plants:
            raise ValueError("One simulation case cannot mix building and steam plant types")

        strategy = build_strategy(self.config.strategy_name, self.config)
        extra_required_columns = set(strategy.required_forecast_columns())

        if building_plants:
            if not isinstance(strategy, BuildingStrategy):
                raise ValueError("A building case requires strategy.name='building_v2g'")
            required_columns = self.loader.required_forecast_columns(
                plants_df,
                extra_required_columns=extra_required_columns,
            )
            forecasts = self.loader.load_forecasts(required_columns=required_columns)
            self._progress("Input data loaded")
            return self._run_building_case(building_plants, forecasts, strategy)

        if isinstance(strategy, BuildingStrategy):
            raise ValueError("Strategy 'building_v2g' requires unit_type='building'")

        plants = steam_plants
        additional_charges = self.loader.load_additional_charges(plants_df)
        for plant in plants:
            regulation = build_grid_fee_regulation(
                self.config.country,
                additional_charges.get(plant.name),
                assumed_tier=self.assumed_grid_tier,
            )
            plant.grid_fee_regulation = regulation
            plant.additional_electricity_charge_eur_per_mwh = (
                regulation.marginal_charge_eur_per_mwh()
            )
        # A regulation that declares a dynamic charge column (ES peajes, FR
        # TURPE+accise) requires that column in the forecasts; add it so a
        # missing/misspelled column fails fast at load instead of silently
        # falling back to a zero scalar during dispatch.
        for plant in plants:
            regulation = getattr(plant, "grid_fee_regulation", None)
            dynamic_column = getattr(regulation, "dynamic_charge_column", None)
            if dynamic_column:
                extra_required_columns.add(dynamic_column)
        required_columns = self.loader.required_forecast_columns(
            plants_df,
            extra_required_columns=extra_required_columns,
        )
        forecasts = self.loader.load_forecasts(required_columns=required_columns)
        self._progress("Input data loaded")
        dispatch_results = self._run_market_sequence(plants, forecasts, strategy)

        # --- Emissionsberechnung ---
        # Strom-Emissionsintensität aus forecasts (Spalte: co2_intensity_kgco2_MWh)
        emission_col = "co2_intensity_kgco2_MWh"
        # Grid electricity drawn per timestep. The ETES route charges the store
        # (etes_charge_MWh); the storage-less direct route draws straight into the
        # electric boiler (actual_electricity_consumption_MWh). Both equal the grid
        # draw for their route, so pick whichever column the dispatch produced.
        electricity_col = (
            "etes_charge_MWh"
            if "etes_charge_MWh" in dispatch_results.columns
            else "actual_electricity_consumption_MWh"
        )
        if emission_col in forecasts.columns:
            emission_factors = forecasts[emission_col].reindex(dispatch_results.index).fillna(0.0)
            dispatch_results["electricity_emissions_kg"] = (
                dispatch_results[electricity_col] * emission_factors
            )
        else:
            dispatch_results["electricity_emissions_kg"] = 0.0

        # Gas emission factor per plant (kg/MWh fuel), mapped by plant name so a
        # multi-plant case uses each plant's own factor rather than the first plant's.
        gas_factor_by_plant = {
            plant.name: plant.gas_emissions_factor_kg_per_mwh for plant in plants
        }
        gas_emission_factors = dispatch_results["plant_name"].map(gas_factor_by_plant).fillna(0.0)
        dispatch_results["gas_emissions_kg"] = (
            dispatch_results["gas_input_MWh"] * gas_emission_factors
        )

        # Gesamtemissionen
        dispatch_results["total_emissions_kg"] = (
            dispatch_results["electricity_emissions_kg"] + dispatch_results["gas_emissions_kg"]
        )
        # ----------------------------

        if AFRR_ENERGY in self.config.enabled_markets:
            strategy.afrr_energy_data_quality_summary = _full_period_afrr_quality_summary(
                self.config,
                forecasts,
            )

        grid_fee_results = self._settle_grid_fees(plants, dispatch_results)

        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path | list[Path]] = {}

        if self.output_options.save_dispatch_results:
            path = output_dir / "dispatch_results.csv"
            dispatch_results.reset_index().to_csv(path, index=False)
            output_paths["dispatch_results"] = path

        market_ledger = MarketLedger()
        market_ledger.update_from_dispatch_results(dispatch_results)
        if self.output_options.save_market_ledger:
            output_paths["market_ledger"] = market_ledger.save(output_dir / "market_ledger.csv")

        storage_ledger = StorageCostLedger()
        storage_ledger.build_from_dispatch_results(dispatch_results, plants)
        if self.output_options.save_storage_cost_ledger:
            output_paths["storage_cost_ledger"] = storage_ledger.save(
                output_dir / "storage_cost_ledger.csv"
            )

        summary = calculate_summary_indicators(
            dispatch_results,
            market_ledger=market_ledger.to_dataframe(),
            storage_cost_ledger=storage_ledger.to_dataframe(),
            afrr_energy_data_quality_summary=strategy.afrr_energy_data_quality_summary,
        )
        summary = _attach_grid_fee_summary(summary, grid_fee_results)
        if grid_fee_results and self.output_options.save_summary_indicators:
            path = output_dir / "grid_fee_summary.csv"
            _grid_fee_summary_frame(grid_fee_results).to_csv(path, index=False)
            output_paths["grid_fee_summary"] = path
        if not strategy.afrr_capacity_block_summary.empty:
            strategy.afrr_capacity_block_summary = _update_capacity_block_summary(
                strategy.afrr_capacity_block_summary,
                dispatch_results,
                timestep_hours=self.config.timestep_minutes / 60.0,
            )
        if self.output_options.save_summary_indicators:
            path = output_dir / "summary_indicators.csv"
            summary.to_csv(path, index=False)
            output_paths["summary_indicators"] = path

        if (
            AFRR_ENERGY in self.config.enabled_markets
            and not strategy.afrr_energy_data_quality_summary.empty
        ):
            path = output_dir / "afrr_energy_data_quality_summary.csv"
            strategy.afrr_energy_data_quality_summary.to_csv(path, index=False)
            output_paths["afrr_energy_data_quality_summary"] = path

        if (
            AFRR_CAPACITY in self.config.enabled_markets
            and not strategy.afrr_capacity_block_summary.empty
        ):
            path = output_dir / "afrr_capacity_block_summary.csv"
            strategy.afrr_capacity_block_summary.to_csv(path, index=False)
            output_paths["afrr_capacity_block_summary"] = path

        if self.output_options.create_plots:
            self._progress("Plot creation started")
            output_paths["plots"] = create_case_plots(
                dispatch_results,
                summary,
                output_dir,
                market_ledger=market_ledger.to_dataframe(),
                storage_cost_ledger=storage_ledger.to_dataframe(),
            )
            self._progress("Plots created")

        self._progress("Outputs saved")

        return output_paths

    def _run_building_case(
        self,
        buildings: list[Building],
        forecasts: pd.DataFrame,
        strategy: BuildingStrategy,
    ) -> dict[str, Path | list[Path]]:
        """Run building rolling windows and save building-compatible outputs."""

        dispatch_results = self._run_building_windows(buildings, forecasts, strategy)
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path | list[Path]] = {}

        if self.output_options.save_dispatch_results:
            path = output_dir / "dispatch_results.csv"
            dispatch_results.reset_index().to_csv(path, index=False)
            output_paths["dispatch_results"] = path

        if self.output_options.save_market_ledger:
            path = output_dir / "market_ledger.csv"
            _building_market_ledger(dispatch_results).to_csv(path, index=False)
            output_paths["market_ledger"] = path

        if self.output_options.save_storage_cost_ledger:
            path = output_dir / "storage_cost_ledger.csv"
            _building_storage_ledger(dispatch_results).to_csv(path, index=False)
            output_paths["storage_cost_ledger"] = path

        if self.output_options.save_summary_indicators:
            path = output_dir / "summary_indicators.csv"
            _building_summary(dispatch_results).to_csv(path, index=False)
            output_paths["summary_indicators"] = path

        if self.output_options.create_plots:
            self._progress("Building plots are not implemented; plot creation skipped")
        self._progress("Outputs saved")
        return output_paths

    def _run_building_windows(
        self,
        buildings: list[Building],
        forecasts: pd.DataFrame,
        strategy: BuildingStrategy,
    ) -> pd.DataFrame:
        """Optimize each look-ahead horizon and carry implemented vehicle SOC."""

        windows = _decision_windows(self.config, forecasts)
        self._report_market_calendar_notices()
        dispatch_parts: list[pd.DataFrame] = []
        completed_windows = 0
        total_windows = len(buildings) * len(windows)

        for building in buildings:
            current_soc = {
                bus_name: vehicle.initial_soc_mwh
                for bus_name, vehicle in building.electric_vehicles.items()
            }
            current_renewable_soc = {
                bus_name: 0.0 for bus_name in building.electric_vehicles
            }
            current_monthly_peaks: dict[str, float] = {}
            for window in windows:
                self._progress(
                    _window_progress_message(
                        current=completed_windows + 1,
                        total=total_windows,
                        plant_name=building.name,
                        window_start=pd.Timestamp(window.commit_index[0]),
                        window_end=pd.Timestamp(window.commit_index[-1]),
                    )
                )
                optimized = strategy.decide_day_ahead(
                    building,
                    window.forecasts,
                    initial_soc_mwh=current_soc,
                    initial_renewable_soc_mwh=current_renewable_soc,
                    initial_monthly_peak_mw=current_monthly_peaks,
                )
                committed = optimized.reindex(window.commit_index).copy()
                committed["rolling_window"] = window.number
                dispatch_parts.append(committed)
                current_soc = {
                    bus_name: float(committed[f"{bus_name}_soc_MWh"].iloc[-1])
                    for bus_name in building.electric_vehicles
                }
                if all(
                    f"{bus_name}_renewable_soc_MWh" in committed
                    for bus_name in building.electric_vehicles
                ):
                    current_renewable_soc = {
                        bus_name: float(
                            committed[f"{bus_name}_renewable_soc_MWh"].iloc[-1]
                        )
                        for bus_name in building.electric_vehicles
                    }
                for timestamp, grid_import_mw in committed["grid_import_MW"].items():
                    month = pd.Timestamp(timestamp).strftime("%Y-%m")
                    current_monthly_peaks[month] = max(
                        current_monthly_peaks.get(month, 0.0),
                        float(grid_import_mw),
                    )
                completed_windows += 1
                total_soc = sum(current_soc.values())
                self._progress(
                    f"Window {window.number} completed for {building.name}; "
                    f"total bus SOC = {total_soc:.3f} MWh"
                )

        return (
            pd.concat(dispatch_parts)
            .reset_index()
            .sort_values(["plant_name", "datetime"])
            .set_index("datetime")
        )

    def _settle_grid_fees(
        self,
        plants: list[SteamGenerationPlant],
        dispatch_results: pd.DataFrame,
    ) -> dict[str, GridFeeResult]:
        """Compute the authoritative ex-post grid-fee bill per plant."""

        if not self.config.additional_charges_enabled:
            return {}
        results: dict[str, GridFeeResult] = {}
        for plant in plants:
            regulation = getattr(plant, "grid_fee_regulation", None)
            if regulation is None:
                continue
            plant_rows = dispatch_results[dispatch_results["plant_name"] == plant.name]
            if plant_rows.empty:
                continue
            result = regulation.settle(plant_rows, self.config.timestep_minutes)
            results[plant.name] = result
            self._progress(
                f"Grid fees for {plant.name}: {result.grid_energy_MWh:,.0f} MWh_el, "
                f"full-load hours {result.full_load_hours:,.0f} h/a -> tier "
                f"'{result.realized_tier}' (assumed '{result.assumed_tier}'); billed peak "
                f"{result.billed_peak_MW:,.2f} MW; total grid fee "
                f"{result.grid_fee_total_EUR:,.0f} EUR"
            )
            for message in result.warnings:
                warnings.warn(message, stacklevel=2)
        return results

    def _run_market_sequence(
        self,
        plants: list[SteamGenerationPlant],
        forecasts: pd.DataFrame,
        strategy: HybridETESGasStrategy,
    ) -> pd.DataFrame:
        dispatch_parts: list[pd.DataFrame] = []
        capacity_summary_parts: list[pd.DataFrame] = []
        self._report_market_calendar_notices()
        windows = list(_decision_windows(self.config, forecasts))
        if not windows:
            raise ValueError("No decision windows could be created from forecasts_df.csv")
        total_windows = len(plants) * len(windows)
        progress_counter = 0

        for plant in plants:
            has_storage = plant.has_thermal_storage
            current_soc = plant.etes.initial_soc_mwh if has_storage else 0.0
            for window in windows:
                progress_counter += 1
                window_forecasts = window.forecasts
                commit_index = window.commit_index
                window_start = pd.Timestamp(commit_index[0])
                window_end = pd.Timestamp(commit_index[-1])
                self._progress(
                    _window_progress_message(
                        current=progress_counter,
                        total=total_windows,
                        plant_name=plant.name,
                        window_start=window_start,
                        window_end=window_end,
                    )
                )
                soc_note = f"; initial ETES SoC = {current_soc:.3f} MWh_th" if has_storage else ""
                self._progress(
                    f"Delivery window {window.number} for {plant.name}: "
                    f"{window_start:%Y-%m-%d %H:%M} to {window_end:%Y-%m-%d %H:%M}{soc_note}"
                )

                fixed_positions = _zero_market_positions(window_forecasts.index)
                capacity_reservation = pd.DataFrame(index=window_forecasts.index)
                stage_outputs: dict[str, pd.DataFrame] = {}
                dispatch_stage_ran = False

                for market_name in self.config.market_sequence:
                    market = self.markets[market_name]
                    if not market.enabled:
                        self._progress(f"{_stage_label(market_name)} stage skipped (disabled)")
                        continue

                    timing = _market_timing_message(market, window_start)
                    if timing:
                        self._progress(timing)

                    stage_result = self._run_configured_market(
                        market,
                        plant,
                        window_forecasts,
                        fixed_positions,
                        strategy,
                        capacity_reservation,
                        initial_soc_mwh=current_soc,
                    )
                    if market.name == AFRR_CAPACITY:
                        capacity_reservation = stage_result
                        stage_outputs[AFRR_CAPACITY] = capacity_reservation.copy()
                        capacity_summary_parts.extend(
                            _capacity_summaries_for_commit(
                                strategy.afrr_capacity_block_summary,
                                capacity_reservation,
                                commit_index,
                            )
                        )
                    else:
                        fixed_positions = stage_result
                        stage_outputs[market.name] = fixed_positions.copy()
                        dispatch_stage_ran = True
                    self._progress(f"{_stage_label(market.name)} stage solved for {plant.name}")

                if not dispatch_stage_ran:
                    fixed_positions = _run_zero_electricity_dispatch(
                        plant=plant,
                        config=self.config,
                        forecasts=window_forecasts,
                        capacity_reservation=capacity_reservation,
                        initial_soc_mwh=current_soc,
                    )

                committed = fixed_positions.reindex(commit_index).copy()
                committed = _add_stage_dispatch_columns(committed, stage_outputs)
                dispatch_parts.append(committed)
                if has_storage:
                    current_soc = float(committed["etes_soc_MWh"].iloc[-1])
                    self._progress(
                        f"Delivery window {window.number} completed for {plant.name}; "
                        f"final ETES SoC = {current_soc:.3f} MWh_th"
                    )
                else:
                    self._progress(f"Delivery window {window.number} completed for {plant.name}")

        if capacity_summary_parts:
            strategy.afrr_capacity_block_summary = _combine_capacity_summaries(
                capacity_summary_parts
            )

        combined = (
            pd.concat(dispatch_parts)
            .reset_index()
            .sort_values(["plant_name", "datetime"])
            .set_index("datetime")
        )
        return combined

    def _progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)

    def _report_market_calendar_notices(self) -> None:
        horizon = float(self.config.dispatch_setting("dispatch_horizon_hours", 24))
        step = float(self.config.dispatch_setting("rolling_step_hours", horizon))
        rolling = bool(self.config.dispatch_setting("rolling_horizon_enabled", True))
        if rolling:
            self._progress(
                f"Market calendar: {step:g} h commit window, {horizon:g} h optimisation horizon"
            )
        else:
            self._progress("Market calendar: single full-period decision window")
        capacity_enabled = AFRR_CAPACITY in self.config.enabled_markets
        afrr_energy_enabled = AFRR_ENERGY in self.config.enabled_markets
        if capacity_enabled and not afrr_energy_enabled:
            self._progress(
                "Notice: aFRR capacity is enabled but aFRR energy is disabled; reserved "
                "capacity can earn capacity revenue, but no activation energy is modelled."
            )

    @staticmethod
    def _run_configured_market(
        market: BaseMarket,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        strategy: HybridETESGasStrategy,
        capacity_reservation: pd.DataFrame | None = None,
        initial_soc_mwh: float | None = None,
    ) -> pd.DataFrame:
        if market.name == DAY_AHEAD:
            return strategy.decide_day_ahead(
                plant,
                forecasts,
                capacity_reservation,
                initial_soc_mwh=initial_soc_mwh,
                rolling=False,
            )
        if market.name == INTRADAY_CONTINUOUS:
            return strategy.decide_intraday_continuous(
                plant,
                forecasts,
                fixed_positions,
                capacity_reservation,
                initial_soc_mwh=initial_soc_mwh,
                rolling=False,
            )
        if market.name == AFRR_ENERGY:
            return strategy.decide_afrr_energy(
                plant,
                forecasts,
                fixed_positions,
                capacity_reservation,
                initial_soc_mwh=initial_soc_mwh,
                rolling=False,
            )
        if market.name == AFRR_CAPACITY:
            return strategy.decide_afrr_capacity(
                plant,
                forecasts,
                initial_soc_mwh=initial_soc_mwh,
            )
        raise NotImplementedError(f"Market '{market.name}' is not implemented")

    @staticmethod
    def _build_summary(dispatch_results: pd.DataFrame) -> pd.DataFrame:
        records = []
        for plant_name, group in dispatch_results.groupby("plant_name"):
            records.append(
                {
                    "plant_name": plant_name,
                    "total_electricity_cost_EUR": group["electricity_cost_EUR"].sum(),
                    "total_gas_cost_EUR": group["gas_cost_EUR"].sum(),
                    "total_co2_cost_EUR": group["co2_cost_EUR"].sum(),
                    "total_operating_cost_EUR": group["operating_cost_EUR"].sum(),
                    "total_heat_demand_MWh": group["heat_demand_MWh"].sum(),
                    "total_gas_heat_MWh": group["gas_heat_MWh"].sum(),
                    "total_electric_heat_MWh": group["etes_discharge_MWh"].sum(),
                    "total_etes_charged_MWh": group["etes_charge_MWh"].sum(),
                    "total_etes_discharged_MWh": group["etes_discharge_MWh"].sum(),
                    "final_etes_soc_MWh": group["etes_soc_MWh"].iloc[-1],
                }
            )
        return pd.DataFrame(records)


def _building_summary(dispatch_results: pd.DataFrame) -> pd.DataFrame:
    """Create the compact operational and economic summary for buildings."""

    rows: list[dict[str, object]] = []
    for plant_name, group in dispatch_results.groupby("plant_name"):
        pv_available = float(group["pv_available_generation_MWh"].sum())
        pv_used = float(group["pv_generation_MWh"].sum())
        grid_export = float(group["grid_export_MWh"].sum())
        row: dict[str, object] = {
            "plant_name": plant_name,
            "total_building_demand_MWh": group["building_demand_MWh"].sum(),
            "total_bus_trip_distance_km": group["bus_trip_distance_km"].sum(),
            "total_bus_trip_energy_MWh": group["bus_trip_energy_MWh"].sum(),
            "total_bus_charge_MWh": group["bus_charge_MWh"].sum(),
            "total_bus_discharge_MWh": group["bus_discharge_MWh"].sum(),
            "total_unmet_trip_energy_MWh": group["unmet_trip_energy_MWh"].sum(),
            "total_pv_available_generation_MWh": pv_available,
            "total_pv_generation_MWh": pv_used,
            "total_pv_curtailment_MWh": group["pv_curtailment_MWh"].sum(),
            "pv_utilisation_fraction": (
                pv_used / pv_available if pv_available > 0.0 else 0.0
            ),
            "total_grid_import_MWh": group["grid_import_MWh"].sum(),
            "total_grid_export_MWh": grid_export,
            "peak_grid_import_MW": group["grid_import_MW"].max(),
            "peak_grid_export_MW": group["grid_export_MW"].max(),
            "total_energy_cost": group["energy_cost"].sum(),
            "total_demand_charge_cost": group["demand_charge_cost"].sum(),
            "total_cost": group["total_cost"].sum(),
            "currency": group["currency"].iloc[0],
            "final_bus_soc_MWh": group["bus_soc_MWh"].iloc[-1],
            "final_bus_soc_fraction": group["bus_soc_fraction"].iloc[-1],
        }
        if abs(grid_export) <= 1e-9:
            row["pv_self_consumption_MWh"] = pv_used
            row["pv_self_consumption_fraction"] = (
                pv_used / pv_available if pv_available > 0.0 else 0.0
            )
        if "regional_grid_load_mw" in group:
            row["peak_regional_grid_load_MW"] = group["regional_grid_load_mw"].max()
        if "grid_congestion_weight" in group:
            stress_threshold = (
                float(group["grid_stress_threshold"].iloc[0])
                if "grid_stress_threshold" in group
                else 0.8
            )
            high_load = group["grid_congestion_weight"] >= stress_threshold
            row["congestion_weighted_grid_import_MWh"] = (
                group["grid_import_MWh"] * group["grid_congestion_weight"]
            ).sum()
            row["congestion_weighted_net_grid_import_MWh"] = (
                group["net_grid_import_MWh"] * group["grid_congestion_weight"]
            ).sum()
            row["grid_stress_threshold"] = stress_threshold
            row["grid_import_during_regional_stress_MWh"] = group.loc[
                high_load, "grid_import_MWh"
            ].sum()
            row["bus_discharge_during_regional_stress_MWh"] = group.loc[
                high_load, "bus_discharge_MWh"
            ].sum()
        if "regional_grid_load_state" in group:
            for state in ("normal", "elevated", "stressed"):
                selected = group["regional_grid_load_state"] == state
                row[f"grid_import_during_{state}_grid_load_MWh"] = group.loc[
                    selected, "grid_import_MWh"
                ].sum()
                row[f"bus_discharge_during_{state}_grid_load_MWh"] = group.loc[
                    selected, "bus_discharge_MWh"
                ].sum()
        if "renewable_availability_weight" in group:
            availability = group["renewable_availability_weight"]
            total_charge = float(group["bus_charge_MWh"].sum())
            weighted_charge = float((group["bus_charge_MWh"] * availability).sum())
            row["mean_renewable_availability_weight"] = availability.mean()
            row["renewable_weighted_bus_charge_MWh"] = weighted_charge
            row["renewable_alignment_score"] = (
                weighted_charge / total_charge if total_charge > 0.0 else 0.0
            )
            row["renewable_misalignment_MWh"] = (
                group["bus_charge_MWh"] * (1.0 - availability)
                + group["bus_discharge_MWh"] * availability
            ).sum()
            rich_threshold = (
                float(group["renewable_rich_threshold"].iloc[0])
                if "renewable_rich_threshold" in group
                else float(availability.quantile(0.80))
            )
            renewable_rich = availability >= rich_threshold
            row["renewable_rich_threshold"] = rich_threshold
            row["bus_charge_during_renewable_rich_MWh"] = group.loc[
                renewable_rich, "bus_charge_MWh"
            ].sum()
            row["bus_discharge_during_renewable_rich_MWh"] = group.loc[
                renewable_rich, "bus_discharge_MWh"
            ].sum()
            if "modelled_vre_potential_share_of_demand" in group:
                row["mean_modelled_vre_potential_share_of_demand"] = group[
                    "modelled_vre_potential_share_of_demand"
                ].mean()
        if "renewable_equivalent_charge_MWh" in group:
            row["renewable_equivalent_absorbed_MWh"] = group[
                "renewable_equivalent_charge_MWh"
            ].sum()
            row["renewable_equivalent_delivered_MWh"] = group[
                "renewable_equivalent_discharge_MWh"
            ].sum()
            row["renewable_equivalent_transport_MWh"] = group[
                "renewable_equivalent_trip_energy_MWh"
            ].sum()
            row["renewable_equivalent_discharge_to_site_MWh"] = group[
                "renewable_equivalent_discharge_to_site_MWh"
            ].sum()
            row["renewable_equivalent_grid_export_MWh"] = group[
                "renewable_equivalent_grid_export_MWh"
            ].sum()
            row["renewable_equivalent_conversion_loss_MWh"] = group[
                "renewable_equivalent_conversion_loss_MWh"
            ].sum()
            row["renewable_deficit_weighted_discharge_MWh"] = group[
                "renewable_deficit_weighted_discharge_MWh"
            ].sum()
            row["final_renewable_equivalent_soc_MWh"] = group[
                "renewable_equivalent_soc_MWh"
            ].iloc[-1]
        individual_soc_columns = [
            column
            for column in group.columns
            if column.endswith("_soc_MWh")
            and column != "bus_soc_MWh"
            and not column.endswith("_renewable_soc_MWh")
            and column != "renewable_equivalent_soc_MWh"
        ]
        for column in individual_soc_columns:
            row[f"final_{column}"] = group[column].iloc[-1]
            fraction_column = column.removesuffix("_soc_MWh") + "_soc_fraction"
            row[f"final_{fraction_column}"] = group[fraction_column].iloc[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def _building_market_ledger(dispatch_results: pd.DataFrame) -> pd.DataFrame:
    """Select the building's grid-market positions and settlement values."""

    columns = [
        "plant_name",
        "rolling_window",
        "building_demand_MWh",
        "pv_available_generation_MWh",
        "pv_generation_MWh",
        "pv_curtailment_MWh",
        "grid_import_MWh",
        "grid_export_MWh",
        "net_grid_import_MWh",
        "billing_month",
        "billing_peak_MW",
        "electricity_import_price_per_MWh",
        "electricity_export_price_per_MWh",
        "energy_cost",
        "demand_charge_cost",
        "total_cost",
        "currency",
    ]
    columns.extend(
        column
        for column in (
            "regional_grid_load_mw",
            "regional_grid_load_fraction",
            "grid_congestion_weight",
            "regional_grid_load_source_year",
            "regional_grid_load_profile",
            "regional_grid_load_interpolated",
        )
        if column in dispatch_results
    )
    return dispatch_results.reset_index()[["datetime", *columns]]


def _building_storage_ledger(dispatch_results: pd.DataFrame) -> pd.DataFrame:
    """Select bus-battery operation and state for the storage ledger."""

    columns = [
        "plant_name",
        "rolling_window",
        "bus_availability_fraction",
        "bus_trip_distance_km",
        "bus_trip_energy_MWh",
        "bus_charge_MWh",
        "bus_discharge_MWh",
        "bus_soc_MWh",
        "bus_soc_fraction",
    ]
    individual_columns = [
        column
        for column in dispatch_results.columns
        if column not in columns
        and column != "plant_name"
        and any(
            column.endswith(suffix)
            for suffix in (
                "_availability_fraction",
                "_trip_distance_km",
                "_trip_energy_MWh",
                "_charge_MWh",
                "_discharge_MWh",
                "_soc_MWh",
                "_soc_fraction",
            )
        )
    ]
    return dispatch_results.reset_index()[["datetime", *columns, *individual_columns]]


def _grid_fee_summary_frame(grid_fee_results: dict[str, GridFeeResult]) -> pd.DataFrame:
    rows = []
    for plant_name, result in grid_fee_results.items():
        row: dict[str, object] = {"plant_name": plant_name, **result.as_summary_dict()}
        row["warnings"] = " | ".join(result.warnings)
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_grid_fee_summary(
    summary: pd.DataFrame,
    grid_fee_results: dict[str, GridFeeResult],
) -> pd.DataFrame:
    """Merge per-plant grid-fee scalars into the summary and add the net incl. grid fees."""

    if not grid_fee_results:
        return summary
    grid_df = pd.DataFrame(
        {"plant_name": name, **result.as_summary_dict()}
        for name, result in grid_fee_results.items()
    )
    merged = summary.merge(grid_df, on="plant_name", how="left")
    if {"net_operating_cost_EUR", "grid_fee_total_EUR"}.issubset(merged.columns):
        in_dispatch = (
            merged["total_additional_electricity_charges_cost_EUR"]
            if "total_additional_electricity_charges_cost_EUR" in merged.columns
            else 0.0
        ) + (merged["total_tax_cost_EUR"] if "total_tax_cost_EUR" in merged.columns else 0.0)
        merged["net_operating_cost_incl_grid_fees_EUR"] = (
            merged["net_operating_cost_EUR"] - in_dispatch + merged["grid_fee_total_EUR"]
        )
    return merged


def _add_stage_dispatch_columns(
    final_dispatch: pd.DataFrame,
    stage_outputs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Attach stage-level heat dispatch columns to the final sequential result."""

    dispatch = final_dispatch.copy()
    stage_specs = {
        DAY_AHEAD: "day_ahead",
        INTRADAY_CONTINUOUS: "intraday",
        AFRR_ENERGY: "afrr_energy",
    }
    for market_name, label in stage_specs.items():
        stage = stage_outputs.get(market_name)
        if stage is None:
            continue
        stage = stage.reindex(dispatch.index)
        if "gas_heat_MWh" in stage.columns:
            dispatch[f"gas_heat_after_{label}_MWh"] = stage["gas_heat_MWh"]
        if "etes_discharge_MWh" in stage.columns:
            dispatch[f"etes_discharge_after_{label}_MWh"] = stage["etes_discharge_MWh"]
    return dispatch


def _decision_windows(config: CaseConfig, forecasts: pd.DataFrame) -> list[DecisionWindow]:
    dt_hours = config.timestep_minutes / 60.0
    rolling_enabled = bool(config.dispatch_setting("rolling_horizon_enabled", True))
    if not rolling_enabled:
        index = pd.DatetimeIndex(forecasts.index)
        return [DecisionWindow(number=1, forecasts=forecasts.copy(), commit_index=index)]

    horizon_hours = float(config.dispatch_setting("dispatch_horizon_hours", 24))
    step_hours = float(config.dispatch_setting("rolling_step_hours", horizon_hours))
    horizon_steps = max(1, int(round(horizon_hours / dt_hours)))
    step_steps = max(1, int(round(step_hours / dt_hours)))

    windows: list[DecisionWindow] = []
    position = 0
    number = 1
    while position < len(forecasts):
        horizon = forecasts.iloc[position : position + horizon_steps].copy()
        commit_count = min(step_steps, len(forecasts) - position, len(horizon))
        commit_index = pd.DatetimeIndex(horizon.iloc[:commit_count].index)
        windows.append(
            DecisionWindow(
                number=number,
                forecasts=horizon,
                commit_index=commit_index,
            )
        )
        position += commit_count
        number += 1
    return windows


def _window_progress_message(
    current: int,
    total: int,
    plant_name: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> str:
    if window_start.date() == window_end.date():
        window_label = f"{window_start:%Y-%m-%d}"
    else:
        window_label = f"{window_start:%Y-%m-%d} to {window_end:%Y-%m-%d}"
    remaining = max(total - current, 0)
    return (
        f"Simulating {window_label} for {plant_name} "
        f"({current}/{total} windows, {remaining} remaining)"
    )


def _zero_market_positions(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DA_position_MWh": 0.0,
            "IDC_buy_MWh": 0.0,
            "IDC_sell_MWh": 0.0,
            "final_planned_electricity_MWh": 0.0,
            "actual_electricity_consumption_MWh": 0.0,
        },
        index=index,
    )


def _run_zero_electricity_dispatch(
    plant: SteamGenerationPlant,
    config: CaseConfig,
    forecasts: pd.DataFrame,
    capacity_reservation: pd.DataFrame,
    initial_soc_mwh: float,
) -> pd.DataFrame:
    """Dispatch useful heat with no electricity market procurement."""

    zero_price_col = "__zero_electricity_price_EUR_per_MWh"
    dispatch_forecasts = forecasts.copy()
    dispatch_forecasts[zero_price_col] = 0.0
    gas_price_col = "natural_gas_price"
    if plant.gas_boiler is None:
        raise ValueError(f"Plant '{plant.name}' needs a gas boiler for zero-electricity dispatch")
    gas_benchmark = dispatch_forecasts[gas_price_col].astype(float) / plant.gas_boiler.efficiency
    signals = DispatchSignals(
        electricity_price_col=zero_price_col,
        gas_price_col=gas_price_col,
        gas_benchmark_eur_per_mwh_th=gas_benchmark,
        charge_allowed=pd.Series(False, index=dispatch_forecasts.index),
        additional_electricity_charge_eur_per_mwh=pd.Series(
            float(getattr(plant, "additional_electricity_charge_eur_per_mwh", 0.0)),
            index=dispatch_forecasts.index,
        ),
        tax_rate=getattr(plant.grid_fee_regulation, "electricity_tax_rate", 0.0),
        **_capacity_signal_kwargs(capacity_reservation, dispatch_forecasts.index),
    )
    result = plant.solve_horizon(
        config=config,
        forecasts=dispatch_forecasts,
        signals=signals,
        initial_soc_mwh=initial_soc_mwh,
    )
    result["DA_position_MWh"] = 0.0
    result["final_planned_electricity_MWh"] = 0.0
    result["actual_electricity_consumption_MWh"] = 0.0
    result["electricity_consumption_MWh"] = 0.0
    return result


def _capacity_signal_kwargs(
    capacity_reservation: pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    if capacity_reservation is None or capacity_reservation.empty:
        return {}

    frame = capacity_reservation.reindex(index)
    return {
        "reserved_capacity_mwh": _capacity_float_column(
            frame,
            index,
            "afrr_capacity_reserved_MWh",
        ),
        "afrr_capacity_block_id": _capacity_object_column(
            frame,
            index,
            "afrr_capacity_block_id",
            "",
        ),
        "afrr_capacity_block_duration_h": _capacity_float_column(
            frame,
            index,
            "block_duration_h",
        ),
        "afrr_capacity_price_eur_per_mw_h": _capacity_float_column(
            frame,
            index,
            "capacity_clearing_price_EUR_per_MW_h",
        ),
        "afrr_capacity_pricing_rule": _capacity_object_column(
            frame,
            index,
            "capacity_pricing_rule",
            "",
        ),
        "afrr_capacity_bid_price_eur_per_mw_h": _capacity_float_column(
            frame,
            index,
            "capacity_bid_price_EUR_per_MW_h",
        ),
        "afrr_capacity_settlement_price_eur_per_mw_h": _capacity_float_column(
            frame,
            index,
            "capacity_settlement_price_EUR_per_MW_h",
        ),
        "afrr_capacity_reserved_mw": _capacity_float_column(
            frame,
            index,
            "afrr_capacity_reserved_MW",
        ),
        "afrr_capacity_revenue_eur": _capacity_float_column(
            frame,
            index,
            "afrr_capacity_revenue_EUR",
        ),
        "afrr_capacity_opportunity_cost_eur": _capacity_float_column(
            frame,
            index,
            "afrr_capacity_opportunity_cost_EUR",
        ),
        "afrr_capacity_market_surplus_eur": _capacity_float_column(
            frame,
            index,
            "afrr_capacity_market_surplus_EUR",
        ),
        "afrr_capacity_net_value_eur": _capacity_float_column(
            frame,
            index,
            "afrr_capacity_net_value_EUR",
        ),
    }


def _capacity_float_column(
    capacity_reservation: pd.DataFrame,
    index: pd.DatetimeIndex,
    column: str,
) -> pd.Series:
    if column not in capacity_reservation:
        return pd.Series(0.0, index=index)
    return capacity_reservation[column].astype(float).reindex(index).fillna(0.0)


def _capacity_object_column(
    capacity_reservation: pd.DataFrame,
    index: pd.DatetimeIndex,
    column: str,
    default: object,
) -> pd.Series:
    if column not in capacity_reservation:
        return pd.Series(default, index=index)
    return capacity_reservation[column].reindex(index).fillna(default)


def _capacity_summaries_for_commit(
    block_summary: pd.DataFrame,
    capacity_reservation: pd.DataFrame,
    commit_index: pd.DatetimeIndex,
) -> list[pd.DataFrame]:
    if block_summary.empty or capacity_reservation.empty:
        return []
    if "afrr_capacity_block_id" not in capacity_reservation.columns:
        return []
    block_ids = (
        capacity_reservation.reindex(commit_index)["afrr_capacity_block_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if not block_ids:
        return []
    selected = block_summary[block_summary["block_id"].astype(str).isin(block_ids)].copy()
    return [selected] if not selected.empty else []


def _combine_capacity_summaries(parts: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(parts, ignore_index=True)
    if "block_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["block_id"], keep="first")
    return combined.sort_values("block_start").reset_index(drop=True)


def _full_period_afrr_quality_summary(config: CaseConfig, forecasts: pd.DataFrame) -> pd.DataFrame:
    market = AFRRDownEnergyMarket("afrr_energy", config.market("afrr_energy"))
    timestep_hours = config.timestep_minutes / 60.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = market.prepare_market_data(
            forecasts,
            timestep_hours=timestep_hours,
        )
    return prepared.quality_summary


def _market_timing_message(market: BaseMarket, delivery_start: pd.Timestamp) -> str:
    events = []
    open_text = _gate_text("opens", market.gate_open, delivery_start)
    close_text = _gate_text("closes", market.gate_close, delivery_start)
    if open_text:
        events.append(open_text)
    if close_text:
        events.append(close_text)
    if not events:
        return ""
    return f"{_stage_label(market.name)} gate: " + ", ".join(events)


def _gate_text(label: str, gate: dict[str, object], delivery_start: pd.Timestamp) -> str:
    if not gate:
        return ""
    if "day_relation" in gate and "time" in gate:
        relation = str(gate["day_relation"])
        event_time = _day_relation_timestamp(delivery_start, relation, str(gate["time"]))
        return f"{label} {relation} {gate['time']} ({event_time:%Y-%m-%d %H:%M})"
    if "relative_to_delivery_start_minutes" in gate:
        minutes = int(gate["relative_to_delivery_start_minutes"])
        event_time = delivery_start + pd.Timedelta(minutes=minutes)
        if minutes < 0:
            relation = f"{abs(minutes)} min before delivery start"
        elif minutes > 0:
            relation = f"{minutes} min after delivery start"
        else:
            relation = "at delivery start"
        return f"{label} {relation} ({event_time:%Y-%m-%d %H:%M} for first timestep)"
    return ""


def _day_relation_timestamp(
    delivery_start: pd.Timestamp,
    relation: str,
    time_text: str,
) -> pd.Timestamp:
    text = relation.strip().upper()
    if not text.startswith("D"):
        raise ValueError(f"Unsupported market day_relation '{relation}'")
    offset_text = text[1:] or "+0"
    offset_days = int(offset_text)
    hour, minute = [int(part) for part in time_text.split(":", maxsplit=1)]
    return delivery_start.normalize() + pd.Timedelta(days=offset_days, hours=hour, minutes=minute)


def _update_capacity_block_summary(
    block_summary: pd.DataFrame,
    dispatch_results: pd.DataFrame,
    timestep_hours: float,
) -> pd.DataFrame:
    if "afrr_capacity_block_id" not in dispatch_results.columns:
        return block_summary
    summary = block_summary.copy()
    grouped = dispatch_results.groupby("afrr_capacity_block_id", dropna=False)
    activated = grouped["afrr_energy_activated_MWh"].sum()
    energy_cost = grouped["afrr_energy_cost_EUR"].sum()
    min_charge_headroom = grouped["available_charge_headroom_after_schedule_MWh"].min()
    min_storage_headroom = grouped["available_storage_headroom_after_schedule_MWh"].min()
    summary = summary.set_index("block_id")
    summary["activated_energy_MWh"] = activated.reindex(summary.index).fillna(0.0)
    summary["activation_cost_EUR"] = energy_cost.reindex(summary.index).fillna(0.0)
    summary["charge_headroom_MW"] = (
        min_charge_headroom.reindex(summary.index).fillna(0.0) / timestep_hours
    )
    summary["storage_headroom_MW"] = (
        min_storage_headroom.reindex(summary.index).fillna(0.0) / timestep_hours
    )
    return summary.reset_index()


def _stage_label(market_name: str) -> str:
    labels = {
        AFRR_CAPACITY: "aFRR capacity",
        DAY_AHEAD: "Day-ahead",
        INTRADAY_CONTINUOUS: "Intraday continuous",
        AFRR_ENERGY: "aFRR energy",
    }
    return labels.get(market_name, market_name)
