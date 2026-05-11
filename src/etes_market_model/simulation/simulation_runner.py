# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from etes_market_model.config.case_config import CaseConfig
from etes_market_model.data.data_loader import DataLoader
from etes_market_model.ledgers.market_ledger import MarketLedger
from etes_market_model.ledgers.storage_cost_ledger import StorageCostLedger
from etes_market_model.markets.market_stage import MarketStage
from etes_market_model.plants.steam_generation_plant import SteamGenerationPlant
from etes_market_model.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy
from etes_market_model.visualisation.plots import create_case_plots


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

        summary = self._build_summary(dispatch_results)
        if self.output_options.save_summary_indicators:
            path = output_dir / "summary_indicators.csv"
            summary.to_csv(path, index=False)
            output_paths["summary_indicators"] = path

        if self.output_options.create_plots:
            output_paths["plots"] = create_case_plots(dispatch_results, summary, output_dir)

        return output_paths

    def _run_market_sequence(
        self,
        plants: list[SteamGenerationPlant],
        forecasts: pd.DataFrame,
        strategy: HybridETESGasStrategy,
    ) -> pd.DataFrame:
        dispatch_parts: list[pd.DataFrame] = []
        enabled = set(self.config.enabled_markets)

        for plant in plants:
            fixed_positions = pd.DataFrame(index=forecasts.index)
            if MarketStage.AFRR_CAPACITY.value in enabled:
                raise NotImplementedError(
                    "aFRR capacity is present as a module placeholder but is disabled in the MVP"
                )

            if MarketStage.DAY_AHEAD.value in enabled:
                dispatch = strategy.decide_day_ahead(plant, forecasts)
                fixed_positions = dispatch
            else:
                raise NotImplementedError("The MVP requires the day_ahead market to be enabled")

            if MarketStage.INTRADAY_CONTINUOUS.value in enabled:
                raise NotImplementedError(
                    "intraday_continuous is present as a module placeholder but is "
                    "disabled in the MVP"
                )
            if MarketStage.AFRR_ENERGY.value in enabled:
                raise NotImplementedError(
                    "aFRR energy is present as a module placeholder but is disabled in the MVP"
                )

            dispatch_parts.append(fixed_positions)

        combined = (
            pd.concat(dispatch_parts)
            .reset_index()
            .sort_values(["plant_name", "datetime"])
            .set_index("datetime")
        )
        return combined

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
                    "total_unmet_heat_penalty_EUR": group["unmet_heat_penalty_EUR"].sum(),
                    "total_operating_cost_EUR": group["operating_cost_EUR"].sum(),
                    "total_heat_demand_MWh": group["heat_demand_MWh"].sum(),
                    "total_gas_heat_MWh": group["gas_heat_MWh"].sum(),
                    "total_electric_heat_MWh": group["etes_discharge_MWh"].sum(),
                    "total_etes_charged_MWh": group["etes_charge_MWh"].sum(),
                    "total_etes_discharged_MWh": group["etes_discharge_MWh"].sum(),
                    "final_etes_soc_MWh": group["etes_soc_MWh"].iloc[-1],
                    "total_unmet_heat_MWh": group["unmet_heat_MWh"].sum(),
                }
            )
        return pd.DataFrame(records)
