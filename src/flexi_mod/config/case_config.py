# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a case configuration is incomplete or inconsistent."""


@dataclass
class CaseConfig:
    """Typed access wrapper around a case-study ``config.yaml`` file."""

    raw: dict[str, Any]
    config_path: Path
    project_root: Path
    study_case: str
    case: dict[str, Any]

    @classmethod
    def from_case_dir(
        cls,
        case_dir: str | Path,
        study_case: str | None = None,
        case_name: str | None = None,
    ) -> CaseConfig:
        case_dir = Path(case_dir).resolve()
        config_path = case_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find config.yaml in {case_dir}")
        return cls.from_file(config_path, study_case=study_case, case_name=case_name)

    @classmethod
    def from_file(
        cls,
        config_path: str | Path,
        study_case: str | None = None,
        case_name: str | None = None,
    ) -> CaseConfig:
        config_path = Path(config_path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        selected_case = _select_case(raw, study_case=study_case, case_name=case_name)
        config = cls(
            raw=raw,
            config_path=config_path,
            project_root=_find_project_root(config_path),
            study_case=selected_case,
            case=dict(raw["cases"][selected_case]),
        )
        config.validate()
        return config

    @property
    def case_name(self) -> str:
        return str(self.case["name"])

    @property
    def country(self) -> str:
        return str(self.case["country"])

    @property
    def timestep_minutes(self) -> int:
        return int(self.case["timestep_minutes"])

    @property
    def simulation_start(self) -> str:
        return str(self.case["simulation_start"])

    @property
    def simulation_end(self) -> str:
        return str(self.case["simulation_end"])

    @property
    def timezone(self) -> str | None:
        value = self.case.get("timezone")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @property
    def additional_charges_enabled(self) -> bool:
        return bool(self.case.get("additional_charges", False))

    @property
    def strategy_name(self) -> str:
        return str(self.case.get("strategy", {}).get("name", ""))

    @property
    def output_folder_name(self) -> str:
        return f"{self.case_name}_{self.strategy_name}"

    @property
    def solver_name(self) -> str:
        return str(self.case.get("solver", {}).get("name", "highs"))

    @property
    def solver_fallbacks(self) -> list[str]:
        return list(self.case.get("solver", {}).get("fallback_solvers", []))

    @property
    def solver_tee(self) -> bool:
        return bool(self.case.get("solver", {}).get("tee", False))

    @property
    def market_sequence(self) -> list[str]:
        return list(self.case.get("market_sequence", []))

    @property
    def enabled_markets(self) -> list[str]:
        markets = self.case.get("markets", {})
        return [
            market_name
            for market_name in self.market_sequence
            if bool(markets.get(market_name, {}).get("enabled", False))
        ]

    def market(self, market_name: str) -> dict[str, Any]:
        try:
            return self.case["markets"][market_name]
        except KeyError as exc:
            raise ConfigError(f"Market '{market_name}' is not defined in config.yaml") from exc

    def market_signal(self, market_name: str, signal_name: str) -> str:
        market = self.market(market_name)
        try:
            return str(market["signals"][signal_name])
        except KeyError as exc:
            raise ConfigError(
                f"Market '{market_name}' is missing required signal '{signal_name}'"
            ) from exc

    def dispatch_setting(self, name: str, default: Any = None) -> Any:
        return self.case.get("strategy", {}).get("dispatch", {}).get(name, default)

    def validate(self) -> None:
        if "cases" not in self.raw:
            old_sections = {"case", "strategy", "market_sequence", "markets"}.intersection(self.raw)
            if old_sections:
                raise ConfigError(
                    "config.yaml uses the old top-level case format. FLEXIMOD now "
                    "requires a top-level 'cases:' mapping, with strategy, solver, "
                    "market_sequence and markets nested under each case."
                )
            raise ConfigError("config.yaml must define a top-level 'cases:' mapping")

        required_sections = ["strategy", "solver", "market_sequence", "markets"]
        missing = [section for section in required_sections if section not in self.case]
        if missing:
            raise ConfigError(
                f"Case '{self.study_case}' is missing required section(s): " + ", ".join(missing)
            )

        for field in ["name", "country", "timestep_minutes", "simulation_start", "simulation_end"]:
            if field not in self.case:
                raise ConfigError(f"cases.{self.study_case}.{field} is required")
        if str(self.case["name"]) != self.study_case:
            raise ConfigError(
                f"cases.{self.study_case}.name must match the selected study case "
                f"'{self.study_case}'"
            )
        if "additional_charges" in self.case and not isinstance(
            self.case["additional_charges"], bool
        ):
            raise ConfigError("cases.<case_name>.additional_charges must be true or false")

        supported_strategies = {
            "hybrid_etes_gas",
            "hybrid_etes_gas_pay_as_cleared_capacity",
            "steel_cost_minimization",
            "electrified_steel",
            "electrified_steel_rule_based",
            "cement_cost_minimization",
        }
        strategy_name = str(self.case["strategy"].get("name", ""))
        if strategy_name not in supported_strategies:
            options = ", ".join(sorted(supported_strategies))
            raise ConfigError(f"strategy.name must be one of: {options}")

        dispatch = self.case["strategy"].get("dispatch", {})
        if dispatch.get("dispatch_method") != "pyomo":
            raise ConfigError("Only strategy.dispatch.dispatch_method='pyomo' is implemented")

        if strategy_name in {
            "steel_cost_minimization",
            "electrified_steel",
            "electrified_steel_rule_based",
            "cement_cost_minimization",
        }:
            if not bool(self.case["markets"].get("day_ahead", {}).get("enabled", False)):
                raise ConfigError(f"{strategy_name} requires an enabled day_ahead market")
        if strategy_name in {"steel_cost_minimization", "cement_cost_minimization"}:
            unsupported_enabled_markets = [
                name
                for name in self.market_sequence
                if name != "day_ahead"
                and bool(self.case["markets"].get(name, {}).get("enabled", False))
            ]
            if unsupported_enabled_markets:
                raise ConfigError(
                    f"{strategy_name} supports only day_ahead price-taking dispatch; "
                    "disable other markets until a market bidding strategy is configured"
                )
        if strategy_name in {
            "steel_cost_minimization",
            "electrified_steel",
            "electrified_steel_rule_based",
        }:
            horizon_hours = float(dispatch.get("dispatch_horizon_hours", 48))
            step_hours = float(dispatch.get("rolling_step_hours", 24))
            if horizon_hours <= 0 or step_hours <= 0:
                raise ConfigError("Steel rolling horizon and step hours must be positive")
            if step_hours > horizon_hours:
                raise ConfigError(
                    "strategy.dispatch.rolling_step_hours must not exceed dispatch_horizon_hours"
                )
            timestep_hours = self.timestep_minutes / 60.0
            for field, hours in {
                "dispatch_horizon_hours": horizon_hours,
                "rolling_step_hours": step_hours,
            }.items():
                steps = hours / timestep_hours
                if abs(steps - round(steps)) > 1e-9:
                    raise ConfigError(
                        f"strategy.dispatch.{field} must align with case.timestep_minutes"
                    )

        if strategy_name in {"electrified_steel", "electrified_steel_rule_based"}:
            if str(self.case["country"]).upper() != "DE":
                raise ConfigError(f"{strategy_name} currently requires country='DE'")
            expected_sequence = ["afrr_capacity", "day_ahead", "afrr_energy"]
            enabled_sequence = [
                name
                for name in self.market_sequence
                if bool(self.case["markets"].get(name, {}).get("enabled", False))
            ]
            if enabled_sequence != expected_sequence:
                raise ConfigError(
                    f"{strategy_name} market_sequence must contain enabled markets in this "
                    "order: afrr_capacity, day_ahead, afrr_energy"
                )
            product_length = str(
                self.case["markets"].get("afrr_capacity", {}).get("product_length", "4h")
            )
            product_hours = _duration_hours(product_length)
            if horizon_hours + 1e-9 < step_hours + product_hours:
                raise ConfigError(
                    f"{strategy_name} dispatch_horizon_hours must cover rolling_step_hours "
                    "plus one complete aFRR capacity product"
                )

        markets = self.case["markets"]
        for market_name in self.market_sequence:
            if market_name not in markets:
                raise ConfigError(f"market_sequence references undefined market '{market_name}'")
            market = markets[market_name]
            if not bool(market.get("enabled", False)):
                continue
            if "signals" not in market:
                raise ConfigError(f"Enabled market '{market_name}' must define signals")
            if market_name == "day_ahead" and "price" not in market["signals"]:
                raise ConfigError("Enabled day_ahead market must define signals.price")
            if market_name == "intraday_continuous" and "price" not in market["signals"]:
                raise ConfigError("Enabled intraday_continuous market must define signals.price")
            if market_name == "afrr_energy":
                signals = market["signals"]
                if "price" not in signals:
                    raise ConfigError("Enabled afrr_energy market must define signals.price")
                if "system_activation" not in signals:
                    raise ConfigError(
                        "Enabled afrr_energy market must define signals.system_activation"
                    )
                rules = market.get("product_rules", {})
                validity_period = int(rules.get("validity_period_minutes", 0))
                if validity_period != self.timestep_minutes:
                    raise ConfigError(
                        "afrr_energy.product_rules.validity_period_minutes must match "
                        "case.timestep_minutes"
                    )
            if market_name == "afrr_capacity":
                signals = market["signals"]
                if "price" not in signals:
                    raise ConfigError("Enabled afrr_capacity market must define signals.price")
                supported_capacity_price_units = {
                    "EUR_per_MW_per_h",
                    "EUR_per_MW_per_product",
                }
                capacity_price_unit = str(market.get("price_unit", "EUR_per_MW_per_h"))
                if capacity_price_unit not in supported_capacity_price_units:
                    options = ", ".join(sorted(supported_capacity_price_units))
                    raise ConfigError(f"afrr_capacity.price_unit must be one of: {options}")
        self._validate_market_order()

    def _validate_market_order(self) -> None:
        markets = self.case["markets"]
        if not bool(markets.get("afrr_capacity", {}).get("enabled", False)):
            return
        if not bool(markets.get("day_ahead", {}).get("enabled", False)):
            return
        try:
            capacity_index = self.market_sequence.index("afrr_capacity")
            day_ahead_index = self.market_sequence.index("day_ahead")
        except ValueError:
            return
        if capacity_index > day_ahead_index:
            raise ConfigError(
                "Enabled afrr_capacity must appear before day_ahead in market_sequence"
            )


def available_study_cases(config_path: str | Path) -> list[str]:
    """Return study-case keys declared in a config file."""

    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cases = raw.get("cases")
    if not isinstance(cases, dict):
        return []
    return sorted(str(name) for name in cases)


def _duration_hours(value: str) -> float:
    text = str(value).strip().lower()
    if text.endswith("min"):
        return float(text.removesuffix("min")) / 60.0
    if text.endswith("h"):
        return float(text.removesuffix("h"))
    raise ConfigError(f"Unsupported duration '{value}'")


def _select_case(
    raw: dict[str, Any],
    study_case: str | None = None,
    case_name: str | None = None,
) -> str:
    if study_case and case_name and study_case != case_name:
        raise ConfigError("--study-case and --case-name must refer to the same case")
    selected = study_case or case_name
    cases = raw.get("cases")
    if not isinstance(cases, dict) or not cases:
        old_sections = {"case", "strategy", "market_sequence", "markets"}.intersection(raw)
        if old_sections:
            raise ConfigError(
                "config.yaml uses the old top-level case format. FLEXIMOD now requires "
                "a top-level 'cases:' mapping."
            )
        raise ConfigError("config.yaml must define a non-empty top-level 'cases:' mapping")
    if selected is None:
        if len(cases) == 1:
            return str(next(iter(cases)))
        options = ", ".join(sorted(str(name) for name in cases))
        raise ConfigError(
            "config.yaml defines multiple study cases. Select one with --study-case "
            f"or --case-name. Available study cases: {options}"
        )
    if selected not in cases:
        options = ", ".join(sorted(str(name) for name in cases))
        raise ConfigError(f"Unknown study case '{selected}'. Available study cases: {options}")
    return selected


def _find_project_root(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()
