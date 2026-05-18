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
from flexi_mod.plants.steam_generation_plant import DispatchSignals, SteamGenerationPlant
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
        output_options: OutputOptions | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.config = CaseConfig.from_case_dir(case_dir)
        self.input_dir = Path(input_dir).resolve() if input_dir else Path(case_dir).resolve()
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else self.config.project_root / "data" / "output" / self.config.case_name
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
        plants = SteamGenerationPlant.from_plants_dataframe(plants_df)
        additional_charges = self.loader.load_additional_charges(plants_df)
        for plant in plants:
            plant.additional_electricity_charge_eur_per_mwh = additional_charges.get(
                plant.name,
                0.0,
            )
        strategy = HybridETESGasStrategy(self.config)
        required_columns = self.loader.required_forecast_columns(
            plants_df,
            extra_required_columns=strategy.required_forecast_columns(),
        )
        forecasts = self.loader.load_forecasts(required_columns=required_columns)
        self._progress("Input data loaded")
        dispatch_results = self._run_market_sequence(plants, forecasts, strategy)
        if AFRR_ENERGY in self.config.enabled_markets:
            strategy.afrr_energy_data_quality_summary = _full_period_afrr_quality_summary(
                self.config,
                forecasts,
            )

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
        if not strategy.afrr_capacity_block_summary.empty:
            strategy.afrr_capacity_block_summary = _update_capacity_block_summary(
                strategy.afrr_capacity_block_summary,
                dispatch_results,
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

        for plant in plants:
            current_soc = plant.etes.initial_soc_mwh
            for window in windows:
                window_forecasts = window.forecasts
                commit_index = window.commit_index
                window_start = pd.Timestamp(commit_index[0])
                window_end = pd.Timestamp(commit_index[-1])
                self._progress(
                    f"Delivery window {window.number} for {plant.name}: "
                    f"{window_start:%Y-%m-%d %H:%M} to {window_end:%Y-%m-%d %H:%M}; "
                    f"initial ETES SoC = {current_soc:.3f} MWh_th"
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
                current_soc = float(committed["etes_soc_MWh"].iloc[-1])
                self._progress(
                    f"Delivery window {window.number} completed for {plant.name}; "
                    f"final ETES SoC = {current_soc:.3f} MWh_th"
                )

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
        additional_electricity_charge_eur_per_mwh=(plant.additional_electricity_charge_eur_per_mwh),
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
            "capacity_price_EUR_per_MW_h",
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
    summary["total_afrr_energy_activated_MWh_in_block"] = activated.reindex(summary.index).fillna(
        0.0
    )
    summary["total_afrr_energy_cost_EUR_in_block"] = energy_cost.reindex(summary.index).fillna(0.0)
    summary["min_final_planned_headroom_MW"] = min_charge_headroom.reindex(summary.index).fillna(
        0.0
    )
    summary["min_storage_headroom_MW"] = min_storage_headroom.reindex(summary.index).fillna(0.0)
    return summary.reset_index()


def _stage_label(market_name: str) -> str:
    labels = {
        AFRR_CAPACITY: "aFRR capacity",
        DAY_AHEAD: "Day-ahead",
        INTRADAY_CONTINUOUS: "Intraday continuous",
        AFRR_ENERGY: "aFRR energy",
    }
    return labels.get(market_name, market_name)
