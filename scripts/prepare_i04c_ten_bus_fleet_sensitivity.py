#!/usr/bin/env python3
"""Prepare reproducible ten-bus peak-demand-response sensitivity inputs.

This is an illustrative future-fleet case, not a statement of the current
BMTA route allocation.  It represents three route-7ก buses, four route-79
buses and three route-101 buses.  Within each route, duty profiles are
staggered by 15-minute increments to avoid assuming that every bus begins
and ends each trip at precisely the same time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "input" / "building_v2b_grid_support_annual"
TARGET = ROOT / "data" / "input" / "building_v2b_ten_bus_peak_demand_sensitivity"

# The study fleet is illustrative.  Counts follow the relative representative
# round-trip frequency in the existing route templates (6 : 8 : 6).
FLEET = {
    "bus_route_7k": {"count": 3, "offset_steps": (0, 4, 8)},
    "bus_route_79": {"count": 4, "offset_steps": (0, 2, 4, 6)},
    "bus_route_101": {"count": 3, "offset_steps": (0, 3, 6)},
}
TIMESTEPS_PER_DAY = 96
CHARGER_POWER_MW = 0.15


def stagger_daily_profile(values: pd.Series, offset_steps: int) -> np.ndarray:
    """Cyclically shift each daily profile while retaining daily route energy."""

    array = values.to_numpy()
    if len(array) % TIMESTEPS_PER_DAY:
        raise ValueError("The forecast must contain full 15-minute days")
    daily = array.reshape(-1, TIMESTEPS_PER_DAY)
    return np.roll(daily, offset_steps, axis=1).reshape(-1)


def vehicle_row(name: str, route: str) -> dict[str, object]:
    return {
        "name": "building_1",
        "unit_type": "building",
        "technology": "electric_vehicle",
        "component_name": name,
        "node": "grid_node",
        "demand": "building_1_electricity_demand",
        "battery_capacity_mwh": 0.3,
        "mileage_mwh_per_km": 0.00125,
        "max_power_charge": CHARGER_POWER_MW,
        "max_power_discharge": CHARGER_POWER_MW,
        "min_soc": 0.0,
        "max_soc": 1.0,
        "initial_soc": 0.8,
        "terminal_soc": 0.8,
        "efficiency_charge": 0.95,
        "efficiency_discharge": 0.95,
        "power_flow_directionality": "bidirectional",
        "availability_column": f"{name}_availability",
        "trip_distance_column": f"{name}_trip_distance_km",
        "max_power": "",
        "generation_column": "",
        "route_template": route,
    }


def charger_row(number: int) -> dict[str, object]:
    return {
        "name": "building_1",
        "unit_type": "building",
        "technology": "charging_station",
        "component_name": f"charger_{number:02d}",
        "node": "grid_node",
        "demand": "building_1_electricity_demand",
        "battery_capacity_mwh": "",
        "mileage_mwh_per_km": "",
        "max_power_charge": CHARGER_POWER_MW,
        "max_power_discharge": CHARGER_POWER_MW,
        "min_soc": "",
        "max_soc": "",
        "initial_soc": "",
        "terminal_soc": "",
        "efficiency_charge": "",
        "efficiency_discharge": "",
        "power_flow_directionality": "bidirectional",
        "availability_column": "",
        "trip_distance_column": "",
        "max_power": "",
        "generation_column": "",
        "route_template": "",
    }


def prepare_forecast(source: pd.DataFrame) -> pd.DataFrame:
    forecast = source.copy()
    for route, settings in FLEET.items():
        if len(settings["offset_steps"]) != settings["count"]:
            raise ValueError(f"Missing stagger offsets for {route}")
        for number, offset in enumerate(settings["offset_steps"], start=1):
            name = f"{route}_{number:02d}"
            forecast[f"{name}_availability"] = stagger_daily_profile(
                source[f"{route}_availability"], offset
            )
            forecast[f"{name}_trip_distance_km"] = stagger_daily_profile(
                source[f"{route}_trip_distance_km"], offset
            )
    return forecast


def prepare_config() -> dict[str, object]:
    shared_case = {
        "country": "TH",
        "timestep_minutes": 15,
        "simulation_start": "2024-01-01 00:00",
        "simulation_end": "2024-12-31 23:45",
        "timezone": "Asia/Bangkok",
        "additional_charges": False,
        # Match the established annual-sensitivity configuration.  This affects
        # solution time only, not the case assumptions or model formulation.
        "solver": {
            "name": "highs",
            "fallback_solvers": [],
            "tee": False,
            "options": {"threads": 24},
        },
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
    }
    common_dispatch = {
        "dispatch_method": "pyomo",
        "rolling_horizon_enabled": True,
        "dispatch_horizon_hours": 48,
        "rolling_step_hours": 24,
        "currency": "THB",
        "grid_congestion_weight_column": "grid_congestion_weight",
        "demand_charge_per_kw_month": 210,
    }
    peak_event_window = {
        # The 6 May event is the representative peak-demand day used in the
        # I04 profile.  The following day is retained so the 48-hour rolling
        # horizon can preserve the next-duty and terminal-SOC constraints.
        "simulation_start": "2024-05-06 00:00",
        "simulation_end": "2024-05-07 23:45",
    }
    return {
        "cases": {
            "building_v1g_ten_bus_reference_annual": {
                **shared_case,
                "name": "building_v1g_ten_bus_reference_annual",
                "description": (
                    "Illustrative ten-bus future-fleet V1G reference: three Route-7ก, "
                    "four Route-79 and three Route-101 buses. This is not a verified "
                    "current BMTA fleet allocation."
                ),
                "strategy": {
                    "name": "building_v2g",
                    "dispatch": {
                        **common_dispatch,
                        "dispatch_objective": "min_cost",
                        "vehicle_discharge_enabled": False,
                        "grid_export_limit_mw": 0.0,
                    },
                },
            },
            "building_v2g_ten_bus_peak_demand_technical_annual": {
                **shared_case,
                "name": "building_v2g_ten_bus_peak_demand_technical_annual",
                "description": (
                    "Illustrative ten-bus annual maximum power-system peak-demand "
                    "technical potential with ten 150 kW bidirectional chargers and "
                    "a 1.5 MW bidirectional connection. It retains route and SOC "
                    "constraints but uses a single maximum-service solve per horizon; "
                    "it is not an approved capacity or an economic dispatch case."
                ),
                "strategy": {
                    "name": "building_v2g",
                    "dispatch": {
                        **common_dispatch,
                        "dispatch_objective": "max_grid_support_technical",
                        "vehicle_discharge_enabled": True,
                        "grid_export_limit_mw": 1.5,
                    },
                },
            },
            "building_v1g_ten_bus_peak_event_reference": {
                **shared_case,
                **peak_event_window,
                "name": "building_v1g_ten_bus_peak_event_reference",
                "description": (
                    "Illustrative ten-bus V1G reference for the representative "
                    "6 May 2024 power-system peak-demand event."
                ),
                "strategy": {
                    "name": "building_v2g",
                    "dispatch": {
                        **common_dispatch,
                        "dispatch_objective": "min_cost",
                        "vehicle_discharge_enabled": False,
                        "grid_export_limit_mw": 0.0,
                    },
                },
            },
            "building_v1g_ten_bus_peak_event_response": {
                **shared_case,
                **peak_event_window,
                "name": "building_v1g_ten_bus_peak_event_response",
                "description": (
                    "Illustrative ten-bus 6 May 2024 peak-demand response using "
                    "managed charging only; it is not electricity supplied by buses."
                ),
                "strategy": {
                    "name": "building_v2g",
                    "dispatch": {
                        **common_dispatch,
                        "dispatch_objective": "max_grid_support",
                        "vehicle_discharge_enabled": False,
                        "grid_export_limit_mw": 0.0,
                    },
                },
            },
            "building_v2g_ten_bus_peak_event_technical": {
                **shared_case,
                **peak_event_window,
                "name": "building_v2g_ten_bus_peak_event_technical",
                "description": (
                    "Illustrative ten-bus V2G technical potential during the "
                    "6 May 2024 peak-demand event, with ten 150 kW bidirectional "
                    "chargers and a 1.5 MW bidirectional connection. This is a "
                    "future-infrastructure sensitivity, not an approved capacity."
                ),
                "strategy": {
                    "name": "building_v2g",
                    "dispatch": {
                        **common_dispatch,
                        "dispatch_objective": "max_grid_support",
                        "vehicle_discharge_enabled": True,
                        "grid_export_limit_mw": 1.5,
                    },
                },
            },
        }
    }


def main() -> None:
    source_forecast = pd.read_csv(SOURCE / "forecasts_df.csv")
    if len(source_forecast) != 366 * TIMESTEPS_PER_DAY:
        raise ValueError("Expected the complete 2024 leap-year forecast")
    TARGET.mkdir(parents=True, exist_ok=True)
    prepare_forecast(source_forecast).to_csv(TARGET / "forecasts_df.csv", index=False)

    rows: list[dict[str, object]] = []
    for route, settings in FLEET.items():
        for number in range(1, int(settings["count"]) + 1):
            rows.append(vehicle_row(f"{route}_{number:02d}", route))
    rows.extend(charger_row(number) for number in range(1, sum(item["count"] for item in FLEET.values()) + 1))
    pd.DataFrame(rows).to_csv(TARGET / "plants.csv", index=False)
    with (TARGET / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(prepare_config(), stream, allow_unicode=True, sort_keys=False)
    print(f"Prepared {TARGET} with {sum(item['count'] for item in FLEET.values())} buses and chargers")


if __name__ == "__main__":
    main()
