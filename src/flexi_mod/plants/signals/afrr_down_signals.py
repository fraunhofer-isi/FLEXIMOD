# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""AFRR down energy market signals for the steam-generation plant."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from flexi_mod.plants.constants import DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL


@dataclass
class AFRRDownSignals:
    da_price_col: str
    idc_price_col: str
    gas_price_col: str
    da_position_mwh: pd.Series
    idc_buy_mwh: pd.Series
    idc_sell_mwh: pd.Series
    final_planned_electricity_mwh: pd.Series
    afrr_energy_price: pd.Series
    afrr_system_activation_mwh: pd.Series
    afrr_energy_bid_mwh: pd.Series
    afrr_energy_activated_mwh: pd.Series
    gas_benchmark_eur_per_mwh_th: pd.Series
    electricity_trading_benchmark_eur_per_mwh_el: pd.Series
    afrr_energy_bid_price: pd.Series | None = None
    afrr_energy_delivered_bid_price: pd.Series | None = None
    afrr_energy_capacity_backed_bid_mwh: pd.Series | None = None
    afrr_energy_free_bid_mwh: pd.Series | None = None
    afrr_energy_capacity_backed_activated_mwh: pd.Series | None = None
    afrr_energy_free_activated_mwh: pd.Series | None = None
    additional_electricity_charge_eur_per_mwh: pd.Series | None = None
    tax_rate: float = 0.0
    afrr_headroom_binding: pd.Series | None = None
    afrr_curtailment_mwh: pd.Series | None = None
    reserved_capacity_mwh: pd.Series | None = None
    afrr_capacity_block_id: pd.Series | None = None
    afrr_capacity_block_duration_h: pd.Series | None = None
    afrr_capacity_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_pricing_rule: pd.Series | None = None
    afrr_capacity_bid_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_settlement_price_eur_per_mw_h: pd.Series | None = None
    afrr_capacity_reserved_mw: pd.Series | None = None
    afrr_capacity_revenue_eur: pd.Series | None = None
    afrr_capacity_opportunity_cost_eur: pd.Series | None = None
    afrr_capacity_market_surplus_eur: pd.Series | None = None
    afrr_capacity_net_value_eur: pd.Series | None = None
    co2_price_col: str | None = None
    co2_emission_factor_t_per_mwh_fuel: float = DEFAULT_CO2_EMISSION_FACTOR_T_PER_MWH_FUEL
