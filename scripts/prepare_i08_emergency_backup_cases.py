# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate targeted regional-high-load-correlated I08 emergency V2G case inputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

THRESHOLD = 0.9
RECOVERY_HOURS = 48
TIMESTEP_HOURS = 0.25
EVENT_DURATIONS_HOURS = (8, 4)


@dataclass(frozen=True)
class OutageEvent:
    """One selected regional-high-load-correlated outage sensitivity."""

    event_id: str
    duration_hours: int
    start: pd.Timestamp
    end: pd.Timestamp
    cumulative_weight: float


def select_events(forecast: pd.DataFrame, threshold: float = THRESHOLD) -> list[OutageEvent]:
    """Select strongest non-overlapping 8 h and 4 h windows above ``threshold``."""

    weights = forecast["grid_congestion_weight"].astype(float)
    if not weights.between(0.0, 1.0).all():
        raise ValueError("grid_congestion_weight must remain between zero and one")

    selected: list[OutageEvent] = []
    for duration_hours in EVENT_DURATIONS_HOURS:
        intervals = int(round(duration_hours / TIMESTEP_HOURS))
        candidates: list[tuple[float, pd.Timestamp, pd.Timestamp]] = []
        for end_position in range(intervals - 1, len(forecast)):
            window = weights.iloc[end_position - intervals + 1 : end_position + 1]
            if (window >= threshold).all():
                candidates.append(
                    (
                        float(window.sum()),
                        pd.Timestamp(window.index[0]),
                        pd.Timestamp(window.index[-1]),
                    )
                )
        if not candidates:
            raise ValueError(f"No {duration_hours} h interval meets the stress threshold")

        for score, start, end in sorted(candidates, key=lambda item: (-item[0], item[1])):
            overlaps = any(not (end < item.start or start > item.end) for item in selected)
            if not overlaps:
                selected.append(
                    OutageEvent(
                        event_id=f"I08{chr(ord('a') + len(selected))}",
                        duration_hours=duration_hours,
                        start=start,
                        end=end,
                        cumulative_weight=score,
                    )
                )
                break
        else:  # pragma: no cover - depends on unusual input profiles
            raise ValueError(f"No non-overlapping {duration_hours} h interval meets the threshold")
    return selected


def prepare_episode_forecast(
    forecast: pd.DataFrame,
    event: OutageEvent,
    recovery_hours: int = RECOVERY_HOURS,
) -> pd.DataFrame:
    """Return the outage and recovery slice with explicit islanding signals."""

    recovery_end = event.end + pd.Timedelta(hours=recovery_hours)
    episode = forecast.loc[event.start:recovery_end].copy()
    if episode.index[-1] != recovery_end:
        raise ValueError("Forecast does not cover the requested recovery horizon")
    outage = (episode.index >= event.start) & (episode.index <= event.end)
    episode["synthetic_outage_event"] = outage.astype(int)
    episode["grid_connection_available"] = (~outage).astype(int)
    episode["synthetic_outage_event_id"] = event.event_id
    episode["synthetic_outage_duration_hours"] = event.duration_hours
    episode["synthetic_outage_trigger_weight"] = THRESHOLD
    return episode


def inherited_soc(
    v1g_dispatch_file: Path,
    event_start: pd.Timestamp,
    bus_names: list[str],
) -> dict[str, float]:
    """Read each bus SOC at the last interval before the outage starts."""

    dispatch = pd.read_csv(v1g_dispatch_file, parse_dates=["datetime"])
    timestamps = pd.DatetimeIndex(dispatch["datetime"])
    if timestamps.tz is not None:
        timestamps = timestamps.tz_localize(None)
    dispatch.index = timestamps
    prior = event_start - pd.Timedelta(minutes=15)
    if prior not in dispatch.index:
        raise ValueError(f"V1G dispatch does not contain the required pre-outage timestamp {prior}")
    row = dispatch.loc[prior]
    return {name: float(row[f"{name}_soc_MWh"]) for name in bus_names}


def _case_config(case_name: str, description: str, forecast: pd.DataFrame) -> dict:
    return {
        "cases": {
            case_name: {
                "name": case_name,
                "description": description,
                "country": "TH",
                "timestep_minutes": 15,
                "simulation_start": str(forecast.index[0]),
                "simulation_end": str(forecast.index[-1]),
                "timezone": "Asia/Bangkok",
                "additional_charges": False,
                "solver": {"name": "highs", "fallback_solvers": [], "tee": False},
                "market_sequence": ["day_ahead"],
                "markets": {
                    "day_ahead": {
                        "enabled": True,
                        "product_resolution": "15min",
                        "signals": {
                            "price": "electricity_import_price_thb_per_mwh",
                            "export_price": "v2g_export_price_current_thb_per_mwh",
                        },
                    }
                },
                "strategy": {
                    "name": "building_v2g",
                    "dispatch": {
                        "dispatch_method": "pyomo",
                        "rolling_horizon_enabled": False,
                        "currency": "THB",
                        "dispatch_objective": "max_emergency_v2g_transfer",
                        "emergency_event_column": "synthetic_outage_event",
                        "grid_connection_available_column": "grid_connection_available",
                        "vehicle_discharge_enabled": True,
                        "grid_export_limit_mw": 0.0,
                        "demand_charge_per_kw_month": 210,
                    },
                },
            }
        }
    }


def _plants_with_initial_soc(base_plants: pd.DataFrame, soc_mwh: dict[str, float]) -> pd.DataFrame:
    plants = base_plants.copy()
    buses = plants["technology"].eq("electric_vehicle")
    for index, row in plants.loc[buses].iterrows():
        bus_name = str(row["component_name"])
        capacity = float(row["battery_capacity_mwh"])
        plants.loc[index, "initial_soc"] = soc_mwh[bus_name] / capacity
    return plants


def generate_cases(
    annual_forecast_file: Path,
    v1g_dispatch_file: Path,
    base_plants_file: Path,
    output_root: Path,
) -> list[Path]:
    """Create four I08 case directories and return them."""

    forecast = pd.read_csv(annual_forecast_file, index_col="datetime", parse_dates=True)
    base_plants = pd.read_csv(base_plants_file)
    bus_names = base_plants.loc[
        base_plants["technology"].eq("electric_vehicle"), "component_name"
    ].astype(str).tolist()
    selected = select_events(forecast)
    prepared_soc = {
        str(row["component_name"]): 0.8 * float(row["battery_capacity_mwh"])
        for _, row in base_plants.loc[base_plants["technology"].eq("electric_vehicle")].iterrows()
    }
    created: list[Path] = []

    for event in selected:
        episode = prepare_episode_forecast(forecast, event)
        soc_options = {
            "prepared": prepared_soc,
            "baseline": inherited_soc(v1g_dispatch_file, event.start, bus_names),
        }
        for soc_basis, initial_soc in soc_options.items():
            case_name = f"building_v2g_emergency_backup_{event.duration_hours}h_{soc_basis}"
            case_dir = output_root / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            description = (
                f"{event.duration_hours} h regional-high-load-correlated islanded-depot V2G "
                f"technical potential with {soc_basis} pre-outage SOC"
            )
            config = _case_config(case_name, description, episode)
            (case_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            _plants_with_initial_soc(base_plants, initial_soc).to_csv(
                case_dir / "plants.csv", index=False
            )
            episode.rename_axis("datetime").to_csv(case_dir / "forecasts_df.csv")
            created.append(case_dir)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annual_forecast_file", type=Path)
    parser.add_argument("v1g_dispatch_file", type=Path)
    parser.add_argument("base_plants_file", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    for case_dir in generate_cases(
        args.annual_forecast_file,
        args.v1g_dispatch_file,
        args.base_plants_file,
        args.output_root,
    ):
        print(f"Created {case_dir}")


if __name__ == "__main__":
    main()
