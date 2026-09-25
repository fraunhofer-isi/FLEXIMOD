# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark price helpers shared by steam-plant market strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant

GAS_PRICE_SIGNAL = "natural_gas_price"


def gas_based_heat_cost(
    plant: SteamGenerationPlant,
    forecasts: pd.DataFrame,
) -> pd.Series:
    """Gas heat benchmark: fuel cost per MWh of thermal output."""

    if plant.gas_boiler is None:
        raise ValueError(f"Plant '{plant.name}' has no gas boiler")

    gas_input_per_mwh_heat = 1.0 / plant.gas_boiler.efficiency
    benchmark = forecasts[GAS_PRICE_SIGNAL].astype(float) * gas_input_per_mwh_heat
    benchmark.name = "gas_based_heat_benchmark_EUR_per_MWh_th"
    # TODO: Add CO2 cost to this benchmark when CO2 is enabled in gas cost accounting.
    return benchmark


def electricity_trading_benchmark(
    plant: SteamGenerationPlant,
    gas_heat_benchmark: pd.Series,
) -> pd.Series:
    """Electricity trading benchmark: delivered-heat value per MWh of electricity."""

    if plant.etes is None:
        raise ValueError(f"Plant '{plant.name}' has no ETES component")

    delivered_heat_per_mwh_electric = (
        plant.etes.efficiency_charge * plant.etes.efficiency_discharge
    )
    if delivered_heat_per_mwh_electric <= 0:
        raise ValueError("ETES charge/discharge efficiencies must be positive")

    benchmark = gas_heat_benchmark.astype(float) * delivered_heat_per_mwh_electric
    benchmark.name = "electricity_trading_benchmark_EUR_per_MWh_el"
    return benchmark


def delivered_electricity_price(
    market_price: pd.Series,
    tax_rate: float,
    additional_charges: pd.Series,
) -> pd.Series:
    """Return total delivered electricity price including charges and tax.

    Formula: (market_price + additional_charges) * (1 + tax_rate)
    - DE: tax_rate=0.0, charges=scalar -> price + scalar
    - ES: tax_rate>0, charges=time-series -> (price + charges) * (1 + tax)
    """

    base = market_price.astype(float) + additional_charges.astype(float)
    return base * (1.0 + tax_rate)
