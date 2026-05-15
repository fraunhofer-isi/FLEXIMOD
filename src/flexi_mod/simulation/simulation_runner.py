# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader
from flexi_mod.ledgers.market_ledger import MarketLedger
from flexi_mod.ledgers.storage_cost_ledger import StorageCostLedger
from flexi_mod.markets import BaseMarket, build_markets
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant
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


class SimulationRunner:
    """Coordinate data loading, sequential market stages, dispatch and outputs."""

    def __init__(
        self,
        case_dir: str | Path,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        plants_file: str = "plants.csv",
        forecasts_file: str = "forecasts_df.csv",
        output_options: OutputOptions | None = None,
    ):
        self.config = CaseConfig.from_case_dir(case_dir)
        self.input_dir = Path(input_dir).resolve() if input_dir else Path(case_dir).resolve()
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else self.config.project_root / "data" / "output" / self.config.case_name
        )
        self.output_options = output_options or OutputOptions()
        self.markets = build_markets(self.config)
        self.loader = DataLoader(
            self.config,
            input_dir=self.input_dir,
            plants_file=plants_file,
            forecasts_file=forecasts_file,
        )

    def run(self) -> dict[str, Path | list[Path]]:
        plants_df = self.loader.load_plants()
        plants = SteamGenerationPlant.from_plants_dataframe(plants_df)
        strategy = HybridETESGasStrategy(self.config)
        required_columns = self.loader.required_forecast_columns(
            plants_df,
            extra_required_columns=strategy.required_forecast_columns(),
        )
        forecasts = self.loader.load_forecasts(required_columns=required_columns)
        dispatch_results = self._run_market_sequence(plants, forecasts, strategy)

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

        if self.output_options.create_plots:
            output_paths["plots"] = create_case_plots(
                dispatch_results,
                summary,
                output_dir,
                market_ledger=market_ledger.to_dataframe(),
                storage_cost_ledger=storage_ledger.to_dataframe(),
            )

        return output_paths

    def _run_market_sequence(
        self,
        plants: list[SteamGenerationPlant],
        forecasts: pd.DataFrame,
        strategy: HybridETESGasStrategy,
    ) -> pd.DataFrame:
        dispatch_parts: list[pd.DataFrame] = []
        enabled_markets = [
            market for market in self.markets.values() if market.name in self.config.enabled_markets
        ]

        for plant in plants:
            fixed_positions = pd.DataFrame(index=forecasts.index)
            stage_outputs: dict[str, pd.DataFrame] = {}
            for market in enabled_markets:
                fixed_positions = self._run_configured_market(
                    market,
                    plant,
                    forecasts,
                    fixed_positions,
                    strategy,
                )
                stage_outputs[market.name] = fixed_positions.copy()

            if fixed_positions.empty:
                raise NotImplementedError("The MVP requires the day_ahead market to be enabled")

            dispatch_parts.append(_add_stage_dispatch_columns(fixed_positions, stage_outputs))

        combined = (
            pd.concat(dispatch_parts)
            .reset_index()
            .sort_values(["plant_name", "datetime"])
            .set_index("datetime")
        )
        return combined

    @staticmethod
    def _run_configured_market(
        market: BaseMarket,
        plant: SteamGenerationPlant,
        forecasts: pd.DataFrame,
        fixed_positions: pd.DataFrame,
        strategy: HybridETESGasStrategy,
    ) -> pd.DataFrame:
        if market.name == DAY_AHEAD:
            return strategy.decide_day_ahead(plant, forecasts)
        if market.name == INTRADAY_CONTINUOUS:
            if fixed_positions.empty:
                raise NotImplementedError("IDC requires fixed day-ahead positions")
            return strategy.decide_intraday_continuous(plant, forecasts, fixed_positions)
        if market.name == AFRR_ENERGY:
            if fixed_positions.empty:
                raise NotImplementedError("aFRR energy requires fixed DA or IDC positions")
            return strategy.decide_afrr_energy(plant, forecasts, fixed_positions)
        if market.name == AFRR_CAPACITY:
            raise NotImplementedError(
                "aFRR capacity is present as a market placeholder but is disabled in the MVP"
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
