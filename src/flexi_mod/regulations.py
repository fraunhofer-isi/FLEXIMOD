# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Network-tariff (grid-fee) regulations — one module, country-selected.

This single file holds everything the engine needs for electricity network
charges:

* :class:`GridFeeResult`        - the ex-post settlement returned per plant.
* :class:`GridFeeRegulation`    - the small interface the strategy + runner call.
* :class:`NullGridFeeRegulation`- no-op (used when additional charges are disabled).
* :class:`GermanGridFeeRegulation` - Germany (Netzentgelte, §19(2) StromNEV).
* :func:`build_grid_fee_regulation` - factory selecting the regulation by country.

To add another country, write one more ``GridFeeRegulation`` subclass below and
register it in ``_REGISTRY``. The case ``config.country`` then selects it.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class GridFeeConfigError(ValueError):
    """Raised when the per-plant charge table or options cannot be interpreted."""


# --------------------------------------------------------------------- result
@dataclass
class GridFeeResult:
    """Authoritative ex-post grid-fee settlement for one plant over the run."""

    grid_energy_MWh: float = 0.0
    annual_peak_MW: float = 0.0
    window_peak_MW: float = 0.0
    billed_peak_MW: float = 0.0
    full_load_hours: float = 0.0
    assumed_tier: str = "n/a"
    realized_tier: str = "n/a"
    tier_assumption_held: bool = True
    energy_charge_EUR: float = 0.0
    capacity_charge_EUR: float = 0.0
    special_network_use_EUR: float = 0.0
    levies_EUR: float = 0.0
    grid_fee_total_EUR: float = 0.0
    # Extra cost to add on top of the in-dispatch marginal charge already booked,
    # so the reported economics are correct without re-running the dispatch:
    #   ex_post_addition = capacity charge + group-A premium + energy tier true-up
    ex_post_addition_EUR: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_summary_dict(self) -> dict[str, float | str | bool]:
        """Flatten to scalar columns for ``summary_indicators`` / ``grid_fee_summary``."""

        return {
            "grid_annual_energy_MWh": self.grid_energy_MWh,
            "grid_annual_peak_MW": self.annual_peak_MW,
            "grid_high_load_window_peak_MW": self.window_peak_MW,
            "grid_billed_peak_MW": self.billed_peak_MW,
            "grid_full_load_hours": self.full_load_hours,
            "grid_assumed_tier": self.assumed_tier,
            "grid_realized_tier": self.realized_tier,
            "grid_tier_assumption_held": self.tier_assumption_held,
            "grid_fee_energy_charge_EUR": self.energy_charge_EUR,
            "grid_fee_capacity_charge_EUR": self.capacity_charge_EUR,
            "grid_fee_special_network_use_EUR": self.special_network_use_EUR,
            "grid_fee_levies_EUR": self.levies_EUR,
            "grid_fee_total_EUR": self.grid_fee_total_EUR,
            "grid_fee_ex_post_addition_EUR": self.ex_post_addition_EUR,
        }


# ------------------------------------------------------------------ interface
class GridFeeRegulation(ABC):
    """National network-tariff rule set used by the strategy and the runner."""

    @abstractmethod
    def marginal_charge_eur_per_mwh(self) -> float:
        """Per-MWh adder that enters the effective strike price and Pyomo objective."""

    @abstractmethod
    def charging_block_mask(self, forecasts: pd.DataFrame) -> pd.Series:
        """Boolean per-timestep mask; ``True`` where grid-charging must be blocked."""

    @abstractmethod
    def settle(self, dispatch_results: pd.DataFrame, timestep_minutes: int) -> GridFeeResult:
        """Compute the authoritative annual grid-fee bill from the realized dispatch."""

    def tier_prompt_options(self) -> list[dict[str, str]]:
        """Return interactive-prompt choices for the assumed full-load-hour tier.

        Each entry has keys ``key`` (the value to pass as ``assumed_tier``),
        ``label`` (human-readable threshold), and ``rate`` (rate description).
        Return an empty list when the regulation has no tiered energy charge.
        """
        return []


class NullGridFeeRegulation(GridFeeRegulation):
    """No-op regulation used when ``additional_charges`` is disabled for the case."""

    def marginal_charge_eur_per_mwh(self) -> float:
        return 0.0

    def charging_block_mask(self, forecasts: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=forecasts.index)

    def settle(self, dispatch_results: pd.DataFrame, timestep_minutes: int) -> GridFeeResult:
        return GridFeeResult()

    # tier_prompt_options() inherited → returns [] (no prompt shown)


# --------------------------------------------------------------------- Germany
class GermanGridFeeRegulation(GridFeeRegulation):
    """Germany: full-load-hour tiers, special network use A/B, atypical grid use."""

    # Statutory constants (German regulation — intentionally not data/config driven).
    SPECIAL_USE_THRESHOLD_MWH = 1000.0
    FULL_LOAD_HOURS_THRESHOLD_H = 2500.0
    DEFAULT_ASSUMED_TIER = "high"

    # High-load-window (local wall-clock) definition, in minutes-of-day.
    _WINTER_MONTHS = (1, 2, 12)
    _AUTUMN_MONTHS = (9, 10, 11)
    _WINTER_WINDOW = (9 * 60 + 45, 19 * 60 + 30)  # 09:45–19:30
    _AUTUMN_WINDOWS = (
        (11 * 60 + 15, 13 * 60 + 45),
        (15 * 60, 19 * 60 + 15),
    )  # 11:15–13:45, 15:00–19:15

    def __init__(
        self,
        plant_charges: pd.DataFrame,
        assumed_tier: str = DEFAULT_ASSUMED_TIER,
        avoid_high_load_window: bool = True,
        high_load_window_column: str = "high_load_window",
        capacity_peak_basis: str = "high_load_window",
    ):
        if assumed_tier not in {"high", "low"}:
            raise GridFeeConfigError("assumed_tier must be 'high' or 'low'")
        if capacity_peak_basis not in {"high_load_window", "annual"}:
            raise GridFeeConfigError("capacity_peak_basis must be 'high_load_window' or 'annual'")
        self._assumed_tier = assumed_tier
        self._avoid_window = bool(avoid_high_load_window)
        self._window_col = high_load_window_column
        self._capacity_peak_basis = capacity_peak_basis
        self._missing_col_warned = False
        self._parse_charges(plant_charges)

    # ------------------------------------------------------------------ parsing
    def _parse_charges(self, plant_charges: pd.DataFrame) -> None:
        energy: dict[str, float] = {}
        capacity: dict[str, float] = {}
        special: dict[str, float] = {}
        levies = 0.0

        for _, row in plant_charges.iterrows():
            component = str(row["component"]).strip().lower()
            value = float(row["value"])
            tier = "low" if "<2500" in component else "high"
            if "grid energy" in component:
                energy[tier] = value
            elif "grid capacity" in component:
                capacity[tier] = value
            elif "group a" in component or "group_a" in component:
                special["a"] = value
            elif "group b" in component or "group_b" in component:
                special["b"] = value
            else:
                levies += value  # CHP, offshore, concession, electricity tax, ...

        # A category that is present must be complete (both tiers / both groups);
        # a fully absent category defaults to zero so partial tariffs still load.
        incomplete = []
        for label, store, keys in [
            ("grid energy", energy, ("high", "low")),
            ("grid capacity", capacity, ("high", "low")),
        ]:
            if store and any(key not in store for key in keys):
                incomplete += [f"{label} ({key} tier)" for key in keys if key not in store]
        if special and any(key not in special for key in ("a", "b")):
            incomplete += [
                f"special network use (group {key.upper()})"
                for key in ("a", "b")
                if key not in special
            ]
        if incomplete:
            raise GridFeeConfigError(
                "additional_charges.csv has an incomplete tiered component; missing: "
                + ", ".join(incomplete)
            )

        self._energy = {"high": energy.get("high", 0.0), "low": energy.get("low", 0.0)}
        self._capacity = {"high": capacity.get("high", 0.0), "low": capacity.get("low", 0.0)}
        self._special_a = special.get("a", 0.0)
        self._special_b = special.get("b", 0.0)
        self._levies = levies

    def tier_prompt_options(self) -> list[dict[str, str]]:
        if self._energy["high"] == self._energy["low"] == 0.0:
            return []
        return [
            {
                "key": "high",
                "label": f">= {self.FULL_LOAD_HOURS_THRESHOLD_H:.0f} h/a",
                "rate": f"{self._energy['high']:.2f} EUR/MWh",
            },
            {
                "key": "low",
                "label": f"< {self.FULL_LOAD_HOURS_THRESHOLD_H:.0f} h/a",
                "rate": f"{self._energy['low']:.2f} EUR/MWh",
            },
        ]

    # ------------------------------------------------------------ interface (A)
    def marginal_charge_eur_per_mwh(self) -> float:
        # Marginal MWh sits beyond the 1 GWh special-use threshold -> group B rate.
        return self._levies + self._energy[self._assumed_tier] + self._special_b

    def charging_block_mask(self, forecasts: pd.DataFrame) -> pd.Series:
        if not self._avoid_window:
            return pd.Series(False, index=forecasts.index)
        if self._window_col in forecasts.columns:
            return pd.to_numeric(forecasts[self._window_col], errors="coerce").fillna(0.0) >= 0.5
        if not self._missing_col_warned:
            warnings.warn(
                f"Column '{self._window_col}' not found in forecasts; high-load-window "
                "avoidance is disabled. Add it with "
                "GermanGridFeeRegulation.compute_high_load_window(forecasts.index).",
                stacklevel=2,
            )
            self._missing_col_warned = True
        return pd.Series(False, index=forecasts.index)

    # ------------------------------------------------------------ interface (B)
    def settle(self, dispatch_results: pd.DataFrame, timestep_minutes: int) -> GridFeeResult:
        dt_h = timestep_minutes / 60.0
        consumption = pd.to_numeric(
            dispatch_results.get("actual_electricity_consumption_MWh"), errors="coerce"
        ).fillna(0.0)
        grid_energy = float(consumption.sum())
        power = consumption / dt_h if dt_h > 0 else consumption * 0.0
        annual_peak = float(power.max()) if len(power) else 0.0

        # Window flag follows the German rules (identical to the precomputed
        # forecast column the dispatch gate used, which is generated from them).
        window_flag = self.compute_high_load_window(dispatch_results.index) == 1
        window_power = power[window_flag.to_numpy(dtype=bool)] if len(power) else power
        window_peak = float(window_power.max()) if len(window_power) else 0.0

        if annual_peak > 0:
            full_load_hours = grid_energy / annual_peak
            realized_tier = "high" if full_load_hours >= self.FULL_LOAD_HOURS_THRESHOLD_H else "low"
        else:
            # No grid withdrawal -> no tier to assess; treat the assumption as held.
            full_load_hours = 0.0
            realized_tier = self._assumed_tier
        tier_held = realized_tier == self._assumed_tier
        billed_peak = (
            window_peak if self._capacity_peak_basis == "high_load_window" else annual_peak
        )

        thr = self.SPECIAL_USE_THRESHOLD_MWH
        energy_charge = grid_energy * self._energy[realized_tier]
        capacity_charge = self._capacity[realized_tier] * billed_peak
        special = (
            min(grid_energy, thr) * self._special_a + max(0.0, grid_energy - thr) * self._special_b
        )
        levies = self._levies * grid_energy
        total = energy_charge + capacity_charge + special + levies

        group_a_premium = (self._special_a - self._special_b) * min(grid_energy, thr)
        tier_trueup = (self._energy[realized_tier] - self._energy[self._assumed_tier]) * grid_energy
        ex_post_addition = capacity_charge + group_a_premium + tier_trueup

        result_warnings: list[str] = []
        if not tier_held:
            result_warnings.append(
                f"Assumed {self._assumed_tier} full-load-hour tier did not hold "
                f"(realized {full_load_hours:.0f} h/a -> '{realized_tier}'). The dispatch used "
                f"the '{self._assumed_tier}'-tier energy rate; the bill is corrected ex-post, "
                f"but re-run with --assumed-grid-tier {realized_tier} for a fully "
                "self-consistent dispatch."
            )

        return GridFeeResult(
            grid_energy_MWh=grid_energy,
            annual_peak_MW=annual_peak,
            window_peak_MW=window_peak,
            billed_peak_MW=billed_peak,
            full_load_hours=full_load_hours,
            assumed_tier=self._assumed_tier,
            realized_tier=realized_tier,
            tier_assumption_held=tier_held,
            energy_charge_EUR=energy_charge,
            capacity_charge_EUR=capacity_charge,
            special_network_use_EUR=special,
            levies_EUR=levies,
            grid_fee_total_EUR=total,
            ex_post_addition_EUR=ex_post_addition,
            warnings=result_warnings,
        )

    # --------------------------------------------------- high-load-window rules
    @classmethod
    def compute_high_load_window(cls, index: pd.DatetimeIndex) -> pd.Series:
        """Return the 0/1 high-load-window flag for each (local wall-clock) timestamp.

        A 15-min step starting at ``t`` is in-window when ``window_start <= t <
        window_end``. Working days Monday–Friday only; weekends are never in-window.
        """

        idx = pd.DatetimeIndex(index)
        minutes = idx.hour.to_numpy() * 60 + idx.minute.to_numpy()
        month = idx.month.to_numpy()
        is_weekday = idx.weekday.to_numpy() < 5

        winter = np.isin(month, cls._WINTER_MONTHS)
        autumn = np.isin(month, cls._AUTUMN_MONTHS)

        w_start, w_end = cls._WINTER_WINDOW
        winter_win = (minutes >= w_start) & (minutes < w_end)

        autumn_win = np.zeros_like(minutes, dtype=bool)
        for a_start, a_end in cls._AUTUMN_WINDOWS:
            autumn_win |= (minutes >= a_start) & (minutes < a_end)

        flag = is_weekday & ((winter & winter_win) | (autumn & autumn_win))
        return pd.Series(flag.astype(int), index=idx)


# --------------------------------------------------------------------- factory
# Register one regulation per ISO country code. Adding a country is a one-line change.
_REGISTRY: dict[str, type[GridFeeRegulation]] = {
    "DE": GermanGridFeeRegulation,
}


def build_grid_fee_regulation(
    country: str,
    plant_charges: pd.DataFrame | None,
    *,
    assumed_tier: str = "high",
    avoid_high_load_window: bool = True,
    high_load_window_column: str = "high_load_window",
    capacity_peak_basis: str = "high_load_window",
) -> GridFeeRegulation:
    """Build the regulation for ``country``; ``None`` charges -> :class:`NullGridFeeRegulation`."""

    if plant_charges is None:
        return NullGridFeeRegulation()
    key = str(country).strip().upper()
    regulation_cls = _REGISTRY.get(key)
    if regulation_cls is None:
        raise GridFeeConfigError(
            f"No grid-fee regulation registered for country '{country}'. "
            f"Available: {', '.join(sorted(_REGISTRY))}"
        )
    return regulation_cls(
        plant_charges,
        assumed_tier=assumed_tier,
        avoid_high_load_window=avoid_high_load_window,
        high_load_window_column=high_load_window_column,
        capacity_peak_basis=capacity_peak_basis,
    )
