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
* :class:`SpanishGridFeeRegulation` - Spain (peajes 6.xTD + IEE + capacity charge).
* :class:`FrenchGridFeeRegulation` - France (TURPE + accise + fixed annual charges).
* :func:`build_grid_fee_regulation` - factory selecting the regulation by country.

To add another country, write one more ``GridFeeRegulation`` subclass below and
register it in ``_REGISTRY``. The case ``config.country`` then selects it.
"""

from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Standard forecasts_df.csv column holding the time-varying per-MWh grid energy
# charge (Arbeitspreis). Shared, country-neutral name for every regulation that
# uses a dynamic charge (ES peajes, FR TURPE+accise, …).
GRID_ENERGY_CHARGE_COLUMN = "grid_energy_charge"


class GridFeeConfigError(ValueError):
    """Raised when the per-plant charge table or options cannot be interpreted."""


# --------------------------------------------------------------------- result
@dataclass
class GridFeeResult:
    """Authoritative ex-post grid-fee settlement for one plant over the run."""

    grid_energy_MWh: float = 0.0
    annual_peak_MW: float = 0.0
    billed_peak_MW: float = 0.0
    energy_charge_EUR: float = 0.0
    capacity_charge_EUR: float = 0.0
    # Flat per-MWh levies (e.g. German CHP/offshore/concession/Stromsteuer, or a
    # country's static EUR/MWh levies). Note: a *multiplicative* electricity tax
    # such as the Spanish IEE goes in ``electricity_tax_EUR``, not here.
    levies_EUR: float = 0.0
    # Multiplicative electricity tax settled on the electricity bill (Spanish IEE).
    # Pairs with ``GridFeeRegulation.electricity_tax_rate``; 0 for DE/FR.
    electricity_tax_EUR: float = 0.0
    grid_fee_total_EUR: float = 0.0
    # Extra cost to add on top of the in-dispatch marginal charge already booked,
    # so the reported economics are correct without re-running the dispatch:
    #   ex_post_addition = capacity charge + group-A premium + energy tier true-up
    ex_post_addition_EUR: float = 0.0

    # --- Germany-specific detail (default/neutral for other countries) ----------
    # These describe the German full-load-hour tiers and §19(2) high-load window;
    # ES/FR leave them at their defaults. A future country-neutral redesign would
    # move them into an optional per-country detail object.
    full_load_hours: float = 0.0
    window_peak_MW: float = 0.0
    assumed_tier: str = "n/a"
    realized_tier: str = "n/a"
    tier_assumption_held: bool = True
    special_network_use_EUR: float = 0.0
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
            "grid_fee_electricity_tax_EUR": self.electricity_tax_EUR,
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

    @property
    def electricity_tax_rate(self) -> float:
        """Multiplicative tax applied on (market_price + charges). Default 0."""
        return 0.0

    @property
    def dynamic_charge_column(self) -> str | None:
        """Forecasts column with time-varying charges, or None for scalar-only."""
        return None

    # Constructor parameters that are behavioural options rather than named
    # charge values. Regulations that mix charges and options (Germany) override
    # this so from_charges_frame can tell the two apart.
    _BEHAVIOURAL_PARAMS: frozenset[str] = frozenset()

    @classmethod
    def _charge_param_names(cls) -> set[str]:
        """The constructor's named charge parameters (excluding behavioural options)."""
        return {
            name
            for name in inspect.signature(cls).parameters
            if name not in cls._BEHAVIOURAL_PARAMS
        }

    @classmethod
    def _charges_from_frame(cls, plant_charges: pd.DataFrame) -> dict[str, float]:
        """Map ``additional_charges.csv`` component → value, rejecting unknown names.

        An unknown/misspelled component raises instead of being silently dropped
        into a levy bucket — the failure mode behind the France dropped-charges bug.
        """
        valid = cls._charge_param_names()
        charges: dict[str, float] = {}
        for _, row in plant_charges.iterrows():
            name = str(row["component"]).strip()
            if name not in valid:
                raise GridFeeConfigError(
                    f"Unknown grid-fee charge '{name}' for {cls.__name__}. "
                    f"Supported charges: {', '.join(sorted(valid))}."
                )
            charges[name] = float(row["value"])
        return charges

    @classmethod
    def from_charges_frame(cls, plant_charges: pd.DataFrame, **options) -> GridFeeRegulation:
        """Build the regulation from the ``additional_charges.csv`` frame.

        Default: map each component name to the constructor's named charge
        parameter of the same name (rejecting unknowns) and ignore behavioural
        ``options``. Regulations that also take behavioural options or need extra
        validation (Germany) override this.
        """
        del options  # charge-only regulations take no behavioural options
        return cls(**cls._charges_from_frame(plant_charges))


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

    # Constructor parameters that are behavioural options rather than charge
    # values. Everything else in __init__ is a named static charge, which is how
    # from_charges_frame tells the two apart when mapping the CSV.
    _BEHAVIOURAL_PARAMS = frozenset(
        {"assumed_tier", "avoid_high_load_window", "high_load_window_column", "capacity_peak_basis"}
    )

    def __init__(
        self,
        *,
        # --- Static charges (from additional_charges.csv) -----------------------
        grid_energy_charge_high: float = 0.0,  # EUR/MWh  — full-load-hours >= 2500 h/a
        grid_energy_charge_low: float = 0.0,  # EUR/MWh  — full-load-hours < 2500 h/a
        grid_capacity_charge_high: float = 0.0,  # EUR/MW.a — >= 2500 h/a
        grid_capacity_charge_low: float = 0.0,  # EUR/MW.a — < 2500 h/a
        special_network_use_a: float = 0.0,  # EUR/MWh  — group A (first 1 GWh)
        special_network_use_b: float = 0.0,  # EUR/MWh  — group B (remainder)
        chp_surcharge: float = 0.0,  # EUR/MWh
        offshore_grid_levy: float = 0.0,  # EUR/MWh
        concession_fee: float = 0.0,  # EUR/MWh
        electricity_tax: float = 0.0,  # EUR/MWh  — flat Stromsteuer levy
        # --- Behavioural options (from config, not charges) ---------------------
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

        # Group the named charges into the internal structures the rest of the
        # class uses (tier dicts, special A/B, summed levies).
        self._energy = {"high": grid_energy_charge_high, "low": grid_energy_charge_low}
        self._capacity = {"high": grid_capacity_charge_high, "low": grid_capacity_charge_low}
        self._special_a = special_network_use_a
        self._special_b = special_network_use_b
        self._levies = chp_surcharge + offshore_grid_levy + concession_fee + electricity_tax

    # -------------------------------------------------------------- CSV adapter
    @classmethod
    def from_charges_frame(cls, plant_charges: pd.DataFrame, **options) -> GermanGridFeeRegulation:
        """Build from additional_charges.csv.

        Each ``component`` must match one of the constructor's named charge
        parameters (e.g. ``grid_energy_charge_high``, ``chp_surcharge``). Unknown
        component names raise, so a typo can't silently disappear into a levy
        bucket. The ``unit`` column is descriptive; the expected unit for each
        charge is documented on the constructor. Behavioural ``options``
        (assumed_tier, capacity_peak_basis, …) are forwarded to the constructor.
        """
        charges = cls._charges_from_frame(plant_charges)
        cls._validate_tier_completeness(charges)
        return cls(**charges, **options)

    @staticmethod
    def _validate_tier_completeness(charges: dict[str, float]) -> None:
        """Both halves of a tiered/paired charge must be given if either is."""
        pairs = (
            ("grid_energy_charge_high", "grid_energy_charge_low"),
            ("grid_capacity_charge_high", "grid_capacity_charge_low"),
            ("special_network_use_a", "special_network_use_b"),
        )
        missing = [
            other
            for a, b in pairs
            for one, other in ((a, b), (b, a))
            if one in charges and other not in charges
        ]
        if missing:
            raise GridFeeConfigError(
                "additional_charges.csv has an incomplete tiered component; missing: "
                + ", ".join(sorted(set(missing)))
            )

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


# --------------------------------------------------------------------- Spain
class SpanishGridFeeRegulation(GridFeeRegulation):
    """Spain: 6.xTD access tariff (peajes) + IEE + capacity charge.

    Spanish industrial electricity cost:
    1. Market price (OMIE day-ahead / intraday)
    2. Access tariff (peajes de acceso) — time-varying by period (punta, llano, valle)
       → provided as a column in forecasts_df.csv
    3. Impuesto Eléctrico (IEE) — multiplicative tax on (market + peajes)
       → rate = 5.11269632%
    4. Grid capacity charge (potencia contratada) — EUR/MW.a based on annual peak

    Delivered price for strategy decisions:
        (market_price + peajes) × (1 + IEE_rate)

    The capacity charge is not marginal (depends on peak, not energy) and is
    settled ex-post only.
    """

    # Impuesto Especial sobre la Electricidad (IEE)
    ELECTRICITY_TAX_RATE = 0.007669044  # = 0.05112696 × 0.15

    # Column in forecasts_df.csv containing the time-varying access tariff
    DYNAMIC_CHARGE_COLUMN = GRID_ENERGY_CHARGE_COLUMN

    def __init__(self, *, grid_capacity_charge: float = 0.0):
        # EUR/MW.a — potencia contratada, settled ex-post on the annual peak.
        self._capacity_eur_per_mw_a = grid_capacity_charge
        # Spain's per-MWh charges (peajes) live entirely in the dynamic
        # grid_energy_charge column, so there is no static EUR/MWh levy here.
        self._levies_eur_per_mwh = 0.0

    # ─── Properties for strategy access ───────────────────────────

    @property
    def electricity_tax_rate(self) -> float:
        return self.ELECTRICITY_TAX_RATE

    @property
    def dynamic_charge_column(self) -> str | None:
        return self.DYNAMIC_CHARGE_COLUMN

    # ─── Interface (A): dispatch-time ─────────────────────────────

    def marginal_charge_eur_per_mwh(self) -> float:
        """Per-MWh adder for the Pyomo objective.

        For Spain, the dynamic peajes are passed as a time-series directly into
        the Pyomo model via DispatchSignals. Only static EUR/MWh levies (if any)
        enter here. The capacity charge is non-marginal (peak-based) and settled
        ex-post only.
        """
        return self._levies_eur_per_mwh

    def charging_block_mask(self, forecasts: pd.DataFrame) -> pd.Series:
        """Spain has no high-load-window avoidance mechanism."""
        return pd.Series(False, index=forecasts.index)

    # ─── Interface (B): ex-post settlement ────────────────────────

    def settle(self, dispatch_results: pd.DataFrame, timestep_minutes: int) -> GridFeeResult:
        """Compute the authoritative annual Spanish grid-fee bill.

        Components:
        - Energy charges (peajes): already in dispatch as additional_electricity_charges_cost
        - Capacity charge: annual peak MW × EUR/MW.a (annual, not prorated)
        - IEE tax: applied on total electricity cost (market + peajes + levies)
        - Static levies: EUR/MWh × total energy
        """
        dt_h = timestep_minutes / 60.0
        consumption = pd.to_numeric(
            dispatch_results.get("actual_electricity_consumption_MWh"), errors="coerce"
        ).fillna(0.0)
        grid_energy = float(consumption.sum())
        power = consumption / dt_h if dt_h > 0 else consumption * 0.0
        annual_peak = float(power.max()) if len(power) else 0.0
        full_load_hours = grid_energy / annual_peak if annual_peak > 0 else 0.0

        # Energy charges (dynamic peajes already computed in dispatch)
        charges_col = "additional_electricity_charges_cost_EUR"
        if charges_col in dispatch_results.columns:
            energy_charge = float(
                pd.to_numeric(dispatch_results[charges_col], errors="coerce").fillna(0.0).sum()
            )
        else:
            energy_charge = 0.0

        # Capacity charge: annual rate (EUR/MW.a) × annual peak. Not prorated by
        # simulation length — the rate is an annual value driven only by the peak,
        # consistent with the German and French capacity charge.
        capacity_charge = self._capacity_eur_per_mw_a * annual_peak

        # Static levies
        levies_energy = self._levies_eur_per_mwh * grid_energy

        # IEE tax on total electricity cost (market + peajes + levies)
        market_cost_col = "electricity_market_cost_EUR"
        if market_cost_col in dispatch_results.columns:
            market_cost = float(
                pd.to_numeric(dispatch_results[market_cost_col], errors="coerce").fillna(0.0).sum()
            )
        else:
            warnings.warn(
                f"Column '{market_cost_col}' not found in dispatch_results; "
                "IEE taxable base excludes market cost — Spanish grid-fee bill will be "
                "understated. Ensure dispatch_results includes electricity_market_cost_EUR.",
                stacklevel=2,
            )
            market_cost = 0.0
        taxable_base = market_cost + energy_charge + levies_energy
        iee_tax = taxable_base * self.ELECTRICITY_TAX_RATE

        total = energy_charge + capacity_charge + levies_energy + iee_tax

        # ex_post_addition: costs not captured in the per-timestep marginal dispatch.
        # IEE (iee_tax) is already in the Pyomo objective via tax_cost and therefore
        # already in operating_cost_EUR — only the capacity charge is ex-post.
        ex_post_addition = capacity_charge

        return GridFeeResult(
            grid_energy_MWh=grid_energy,
            annual_peak_MW=annual_peak,
            window_peak_MW=0.0,
            billed_peak_MW=annual_peak,
            full_load_hours=full_load_hours,
            assumed_tier="n/a",
            realized_tier="n/a",
            tier_assumption_held=True,
            energy_charge_EUR=energy_charge,
            capacity_charge_EUR=capacity_charge,
            special_network_use_EUR=0.0,
            levies_EUR=levies_energy,
            electricity_tax_EUR=iee_tax,
            grid_fee_total_EUR=total,
            ex_post_addition_EUR=ex_post_addition,
        )

    def tier_prompt_options(self) -> list[dict[str, str]]:
        """Spain has no tiered energy charges — no prompt needed."""
        return []


# --------------------------------------------------------------------- France
class FrenchGridFeeRegulation(GridFeeRegulation):
    """France: TURPE + accise (dynamic, from forecasts) + fixed annual charges (ex-post).

    French industrial electricity cost structure:
    1. Market price (EPEX SPOT day-ahead / intraday)
    2. TURPE (Tarif d'Utilisation des Réseaux Publics d'Électricité)
       — time-varying by period, provided as column in forecasts_df.csv
    3. Accise sur l'électricité (formerly CSPE/TICFE)
       — time-varying, provided as column in forecasts_df.csv
    4. Fixed annual charges (EUR/MW.a), settled ex-post on the realized annual
       peak with no dispatch feedback. All EUR/MW.a rows are summed, covering:
       - Capacity Obligation (Obligation de Capacité)
       - fixed TURPE components (management, metering, fixed withdrawal)

    TURPE and accise are pre-summed into the standard forecasts column
    ``grid_energy_charge``.

    Delivered price for strategy decisions:
        market_price + grid_energy_charge
    (all additive, no multiplicative tax for industrial consumers)
    """

    DYNAMIC_CHARGE_COLUMN = GRID_ENERGY_CHARGE_COLUMN

    def __init__(
        self,
        *,
        capacity_obligation: float = 0.0,  # EUR/MW.a — Obligation de Capacité
        turpe_management: float = 0.0,  # EUR/MW.a — TURPE composante de gestion
        turpe_metering: float = 0.0,  # EUR/MW.a — TURPE composante de comptage
        turpe_fix: float = 0.0,  # EUR/MW.a — TURPE fixed withdrawal (soutirage)
    ):
        # All fixed annual EUR/MW.a charges, summed; settled ex-post on the peak.
        self._fixed_annual_eur_per_mw_a = (
            capacity_obligation + turpe_management + turpe_metering + turpe_fix
        )
        # France's per-MWh charges (TURPE energy + accise) live entirely in the
        # dynamic grid_energy_charge column, so there is no static EUR/MWh levy here.
        self._levies_eur_per_mwh = 0.0

    # ─── Properties for strategy access ───────────────────────────

    @property
    def electricity_tax_rate(self) -> float:
        return 0.0

    @property
    def dynamic_charge_column(self) -> str | None:
        return self.DYNAMIC_CHARGE_COLUMN

    # ─── Interface (A): dispatch-time ─────────────────────────────

    def marginal_charge_eur_per_mwh(self) -> float:
        """Per-MWh adder for the Pyomo objective.

        Dynamic charges (TURPE + accise) are passed as a time-series via
        DispatchSignals. Static EUR/MWh levies (if any) enter here as a scalar.
        The capacity obligation is non-marginal and settled ex-post only.
        """
        return self._levies_eur_per_mwh

    def charging_block_mask(self, forecasts: pd.DataFrame) -> pd.Series:
        """France has no high-load-window avoidance mechanism."""
        return pd.Series(False, index=forecasts.index)

    # ─── Interface (B): ex-post settlement ────────────────────────

    def settle(self, dispatch_results: pd.DataFrame, timestep_minutes: int) -> GridFeeResult:
        """Compute the authoritative annual French grid-fee bill.

        Components:
        - Energy charges (TURPE + accise): already in dispatch as
          additional_electricity_charges_cost_EUR
        - Fixed annual charges (Capacity Obligation + fixed TURPE components):
          summed EUR/MW.a × realized annual peak (prorated)
        """
        dt_h = timestep_minutes / 60.0
        consumption = pd.to_numeric(
            dispatch_results.get("actual_electricity_consumption_MWh"), errors="coerce"
        ).fillna(0.0)
        grid_energy = float(consumption.sum())
        power = consumption / dt_h if dt_h > 0 else consumption * 0.0
        annual_peak = float(power.max()) if len(power) else 0.0
        full_load_hours = grid_energy / annual_peak if annual_peak > 0 else 0.0

        # Energy charges (dynamic TURPE + accise already computed in dispatch)
        charges_col = "additional_electricity_charges_cost_EUR"
        if charges_col in dispatch_results.columns:
            energy_charge = float(
                pd.to_numeric(dispatch_results[charges_col], errors="coerce").fillna(0.0).sum()
            )
        else:
            energy_charge = 0.0

        # Fixed annual charges (Capacity Obligation + fixed TURPE components):
        # summed EUR/MW.a × realized annual peak. Not prorated by simulation length
        # — the rate is an annual value driven only by the peak, consistent with
        # the German and Spanish capacity charge.
        capacity_charge = self._fixed_annual_eur_per_mw_a * annual_peak

        # Static levies (already in dispatch as additional_electricity_charges_cost_EUR)
        levies_energy = self._levies_eur_per_mwh * grid_energy

        total = energy_charge + capacity_charge + levies_energy
        ex_post_addition = capacity_charge

        return GridFeeResult(
            grid_energy_MWh=grid_energy,
            annual_peak_MW=annual_peak,
            window_peak_MW=0.0,
            billed_peak_MW=annual_peak,
            full_load_hours=full_load_hours,
            assumed_tier="n/a",
            realized_tier="n/a",
            tier_assumption_held=True,
            energy_charge_EUR=energy_charge,
            capacity_charge_EUR=capacity_charge,
            special_network_use_EUR=0.0,
            levies_EUR=levies_energy,
            grid_fee_total_EUR=total,
            ex_post_addition_EUR=ex_post_addition,
        )

    def tier_prompt_options(self) -> list[dict[str, str]]:
        """France has no tiered energy charges — no prompt needed."""
        return []


# --------------------------------------------------------------------- factory
# Register one regulation per ISO country code. Adding a country is a one-line change.
_REGISTRY: dict[str, type[GridFeeRegulation]] = {
    "DE": GermanGridFeeRegulation,
    "ES": SpanishGridFeeRegulation,
    "FR": FrenchGridFeeRegulation,
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
    # These options are German-specific (full-load-hour tier + §19(2) high-load
    # window). Non-German regulations accept and ignore them via **kwargs.
    # from_charges_frame maps the CSV to each regulation's constructor.
    return regulation_cls.from_charges_frame(
        plant_charges,
        assumed_tier=assumed_tier,
        avoid_high_load_window=avoid_high_load_window,
        high_load_window_column=high_load_window_column,
        capacity_peak_basis=capacity_peak_basis,
    )
