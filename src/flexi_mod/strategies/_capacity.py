# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""aFRR-capacity signal and storage-headroom helpers for steam-plant strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant


def capacity_signal_kwargs(
    capacity_reservation: pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    """Project a capacity-reservation frame onto the forecast index as signals."""

    if capacity_reservation is None or capacity_reservation.empty:
        return {}

    frame = capacity_reservation.reindex(index)
    return {
        "reserved_capacity_mwh": _capacity_column(frame, index, "afrr_capacity_reserved_MWh"),
        "afrr_capacity_block_id": _capacity_object_column(
            frame,
            index,
            "afrr_capacity_block_id",
            "",
        ),
        "afrr_capacity_block_duration_h": _capacity_column(frame, index, "block_duration_h"),
        "afrr_capacity_price_eur_per_mw_h": _capacity_column(
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
        "afrr_capacity_bid_price_eur_per_mw_h": _capacity_column(
            frame,
            index,
            "capacity_bid_price_EUR_per_MW_h",
        ),
        "afrr_capacity_settlement_price_eur_per_mw_h": _capacity_column(
            frame,
            index,
            "capacity_settlement_price_EUR_per_MW_h",
        ),
        "afrr_capacity_reserved_mw": _capacity_column(frame, index, "afrr_capacity_reserved_MW"),
        "afrr_capacity_revenue_eur": _capacity_column(frame, index, "afrr_capacity_revenue_EUR"),
        "afrr_capacity_opportunity_cost_eur": _capacity_column(
            frame,
            index,
            "afrr_capacity_opportunity_cost_EUR",
        ),
        "afrr_capacity_market_surplus_eur": _capacity_column(
            frame,
            index,
            "afrr_capacity_market_surplus_EUR",
        ),
        "afrr_capacity_net_value_eur": _capacity_column(
            frame,
            index,
            "afrr_capacity_net_value_EUR",
        ),
    }


def future_storage_input_headroom_mwh(
    plant: SteamGenerationPlant,
    baseline_soc: pd.Series,
    replaceable_gas_heat: pd.Series,
) -> pd.Series:
    """Return feasible extra thermal inventory before each timestep's heat outlet.

    A volume can fit at delivery time but still overfill ETES later when fixed
    day-ahead or intraday electricity raises the baseline storage trajectory.
    Working backward makes each value reserve enough room for every later fixed
    charge while crediting only replaceable gas heat as a valid storage outlet.
    """

    baseline_soc_values = baseline_soc.to_numpy(dtype=float)
    replaceable_heat_values = replaceable_gas_heat.to_numpy(dtype=float)
    retention = 1.0 - plant.etes.storage_loss_rate
    discharge_efficiency = plant.etes.efficiency_discharge
    max_capacity = plant.etes.max_capacity_mwh
    allowed_before = [0.0] * len(baseline_soc_values)
    next_allowed_before = float("inf")

    for position in range(len(baseline_soc_values) - 1, -1, -1):
        capacity_headroom = max(0.0, max_capacity - baseline_soc_values[position])
        if retention > 0.0:
            allowed_after_outlet = min(capacity_headroom, next_allowed_before / retention)
        else:
            allowed_after_outlet = capacity_headroom
        allowed_before[position] = (
            allowed_after_outlet + replaceable_heat_values[position] / discharge_efficiency
        )
        next_allowed_before = allowed_before[position]

    return pd.Series(allowed_before, index=baseline_soc.index)


def project_reserved_capacity_claim(
    plant: SteamGenerationPlant,
    capacity_backed_bid: pd.Series,
    baseline_soc: pd.Series,
    replaceable_gas_heat: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Project the baseline SOC as if every reserved interval were fully activated.

    Returns the claimed SOC trajectory and the replaceable gas heat left over
    for free bids. Sizing free bids against these keeps them out of the storage
    room a later capacity-backed activation is entitled to.
    """

    retention = 1.0 - plant.etes.storage_loss_rate
    charge_efficiency = plant.etes.efficiency_charge
    discharge_efficiency = plant.etes.efficiency_discharge
    max_capacity = plant.etes.max_capacity_mwh
    claim = 0.0
    claimed_soc = []
    remaining_gas_heat = []

    for position in range(len(baseline_soc)):
        claim *= retention
        claim += max(0.0, float(capacity_backed_bid.iloc[position])) * charge_efficiency
        outlet = min(float(replaceable_gas_heat.iloc[position]), claim * discharge_efficiency)
        claim -= outlet / discharge_efficiency
        claimed_soc.append(min(max_capacity, float(baseline_soc.iloc[position]) + claim))
        remaining_gas_heat.append(float(replaceable_gas_heat.iloc[position]) - outlet)

    index = baseline_soc.index
    return pd.Series(claimed_soc, index=index), pd.Series(remaining_gas_heat, index=index)


def _capacity_column(
    capacity_reservation: pd.DataFrame | None,
    index: pd.DatetimeIndex,
    column: str,
) -> pd.Series:
    """Numeric capacity column reindexed onto the forecast index, zero-filled."""

    if (
        capacity_reservation is None
        or capacity_reservation.empty
        or column not in capacity_reservation
    ):
        return pd.Series(0.0, index=index)
    return capacity_reservation[column].astype(float).reindex(index).fillna(0.0)


def _capacity_object_column(
    capacity_reservation: pd.DataFrame,
    index: pd.DatetimeIndex,
    column: str,
    default: object,
) -> pd.Series:
    """Object-typed capacity column reindexed onto the forecast index, default-filled."""

    if column not in capacity_reservation:
        return pd.Series(default, index=index)
    return capacity_reservation[column].reindex(index).fillna(default)
