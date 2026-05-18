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

    @classmethod
    def from_case_dir(cls, case_dir: str | Path) -> CaseConfig:
        case_dir = Path(case_dir).resolve()
        config_path = case_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find config.yaml in {case_dir}")
        return cls.from_file(config_path)

    @classmethod
    def from_file(cls, config_path: str | Path) -> CaseConfig:
        config_path = Path(config_path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        config = cls(
            raw=raw,
            config_path=config_path,
            project_root=_find_project_root(config_path),
        )
        config.validate()
        return config

    @property
    def case_name(self) -> str:
        return str(self.raw["case"]["name"])

    @property
    def country(self) -> str:
        return str(self.raw["case"]["country"])

    @property
    def timestep_minutes(self) -> int:
        return int(self.raw["case"]["timestep_minutes"])

    @property
    def simulation_start(self) -> str:
        return str(self.raw["case"]["simulation_start"])

    @property
    def simulation_end(self) -> str:
        return str(self.raw["case"]["simulation_end"])

    @property
    def additional_charges_enabled(self) -> bool:
        return bool(self.raw["case"].get("additional_charges", False))

    @property
    def solver_name(self) -> str:
        return str(self.raw.get("solver", {}).get("name", "highs"))

    @property
    def solver_fallbacks(self) -> list[str]:
        return list(self.raw.get("solver", {}).get("fallback_solvers", []))

    @property
    def solver_tee(self) -> bool:
        return bool(self.raw.get("solver", {}).get("tee", False))

    @property
    def market_sequence(self) -> list[str]:
        return list(self.raw.get("market_sequence", []))

    @property
    def enabled_markets(self) -> list[str]:
        markets = self.raw.get("markets", {})
        return [
            market_name
            for market_name in self.market_sequence
            if bool(markets.get(market_name, {}).get("enabled", False))
        ]

    def market(self, market_name: str) -> dict[str, Any]:
        try:
            return self.raw["markets"][market_name]
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
        return self.raw.get("strategy", {}).get("dispatch", {}).get(name, default)

    def validate(self) -> None:
        required_sections = [
            "case",
            "strategy",
            "solver",
            "market_sequence",
            "markets",
        ]
        missing = [section for section in required_sections if section not in self.raw]
        if missing:
            raise ConfigError(f"Missing required config section(s): {', '.join(missing)}")

        for field in ["name", "country", "timestep_minutes", "simulation_start", "simulation_end"]:
            if field not in self.raw["case"]:
                raise ConfigError(f"case.{field} is required")
        if "additional_charges" in self.raw["case"] and not isinstance(
            self.raw["case"]["additional_charges"], bool
        ):
            raise ConfigError("case.additional_charges must be true or false")

        if self.raw["strategy"].get("name") != "hybrid_etes_gas":
            raise ConfigError("Only strategy.name='hybrid_etes_gas' is implemented in the MVP")

        dispatch = self.raw["strategy"].get("dispatch", {})
        if dispatch.get("dispatch_method") != "pyomo":
            raise ConfigError("Only strategy.dispatch.dispatch_method='pyomo' is implemented")

        markets = self.raw["markets"]
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
                if str(market.get("price_unit", "EUR_per_MW_per_h")) != "EUR_per_MW_per_h":
                    raise ConfigError("afrr_capacity.price_unit must be 'EUR_per_MW_per_h'")
                afrr_energy = markets.get("afrr_energy", {})
                if "afrr_energy" not in self.market_sequence:
                    raise ConfigError(
                        "Enabled afrr_capacity requires afrr_energy in market_sequence"
                    )
                if not bool(afrr_energy.get("enabled", False)):
                    raise ConfigError(
                        "Enabled afrr_capacity requires afrr_energy.enabled=true so "
                        "capacity-backed activation can use cleaned aFRR energy data"
                    )
                afrr_energy_signals = afrr_energy.get("signals", {})
                if "price" not in afrr_energy_signals:
                    raise ConfigError("Enabled afrr_capacity requires afrr_energy.signals.price")
                if "system_activation" not in afrr_energy_signals:
                    raise ConfigError(
                        "Enabled afrr_capacity requires afrr_energy.signals.system_activation"
                    )
        self._validate_market_order()

    def _validate_market_order(self) -> None:
        markets = self.raw["markets"]
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


def _find_project_root(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()
