# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Collect run provenance and serialize it to a JSON metadata file."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from flexi_mod.config.case_config import CaseConfig
from flexi_mod.data.data_loader import DataLoader

PLANT_TECHNOLOGY_COLUMN = "technology"
PLANT_PARAMETER_COLUMNS = (
    "max_power",
    "efficiency",
    "max_capacity",
    "min_capacity",
    "max_power_charge",
    "max_power_discharge",
)
FORECAST_PARAMETER_COLUMNS = ("natural_gas_price", "co2_price")


class ProvenanceCollector:
    """Gather run, code, input, parameter and environment metadata for one run."""

    # Bump SCHEMA_VERSION on every update to the emitted JSON structure.
    SCHEMA_VERSION = "1.0"

    def __init__(self, config: CaseConfig, loader: DataLoader, project_root: Path) -> None:
        self.config = config
        self.loader = loader
        self.project_root = project_root

    def build(
        self,
        plants: pd.DataFrame,
        forecasts: pd.DataFrame,
        started_at: datetime,
        duration_seconds: float,
    ) -> dict:
        """Assemble the full provenance document as a JSON-serialisable dict."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run": self._run_section(started_at, duration_seconds),
            "code": self._code_section(),
            "inputs": self._inputs_section(plants, forecasts),
            "parameters": self._parameters_section(plants, forecasts),
            "environment": self._environment_section(),
        }

    @staticmethod
    def to_json(metadata: dict) -> str:
        """Serialise a provenance document to indented JSON."""
        return json.dumps(metadata, indent=2)

    def _run_section(self, started_at: datetime, duration_seconds: float) -> dict:
        # The run is uniquely identified by its UTC start timestamp, so id and
        # started_at are intentionally the same value.
        run_id = _iso_utc(started_at)
        return {
            "id": run_id,
            "started_at": run_id,
            "duration_seconds": int(duration_seconds),
        }

    def _code_section(self) -> dict:
        # Git may be unavailable (e.g. a released package without the repo), so
        # each field degrades to "unknown" rather than failing the whole run.
        return {
            "repository": _repository_name(self.project_root),
            "commit": _git_output(self.project_root, "rev-parse", "--short", "HEAD") or "unknown",
            "branch": _git_output(self.project_root, "branch", "--show-current") or "unknown",
            "dirty": bool(_git_output(self.project_root, "status", "--porcelain")),
        }

    def _inputs_section(
        self,
        plants: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> list[dict]:
        inputs = [
            self._input_record("forecasts", self.loader.forecasts_path, forecasts),
            self._input_record("plants", self.loader.plants_path, plants),
        ]
        # additional_charges.csv is optional, so it is listed only when in use.
        if self.config.additional_charges_enabled:
            charges = self._load_charges_for_count(self.loader.additional_charges_path)
            inputs.append(
                self._input_record("additional_charges", self.loader.additional_charges_path, charges)
            )
        inputs.append(self._input_record("config", self.config.config_path))
        return inputs

    def _input_record(self, name: str, path: Path, frame: pd.DataFrame | None = None) -> dict:
        record: dict = {"name": name, "path": self._relative_path(path)}
        if frame is not None and not frame.empty:
            record["rows"] = len(frame)
        return record

    def _load_charges_for_count(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, skipinitialspace=True)
        except OSError:
            return None

    def _parameters_section(self, plants: pd.DataFrame, forecasts: pd.DataFrame) -> dict:
        return {
            "simulation": self._simulation_parameters(),
            "markets": self._market_parameters(),
            "forecasts": self._forecast_parameters(forecasts),
            "plants": self._plant_parameters(plants),
            "strategy": self._strategy_parameters(),
            "solver": self._solver_parameters(),
        }

    def _simulation_parameters(self) -> dict:
        return {
            "case_name": self.config.case_name,
            "country": self.config.country,
            "timestep_minutes": self.config.timestep_minutes,
            "simulation_start": self.config.simulation_start,
            "simulation_end": self.config.simulation_end,
            "timezone": self.config.timezone,
        }

    def _market_parameters(self) -> dict:
        return {
            "enabled": list(self.config.enabled_markets),
            "sequence": list(self.config.market_sequence),
        }

    def _forecast_parameters(self, forecasts: pd.DataFrame) -> dict:
        # Only a single value per price column is recorded (here the first
        # numeric sample); a missing or entirely blank column is omitted.
        params: dict = {}
        for column in FORECAST_PARAMETER_COLUMNS:
            values = _numeric_column(forecasts, column)
            if values is not None:
                params[column] = values
        return params

    def _plant_parameters(self, plants: pd.DataFrame) -> list[dict]:
        # Keep each plant record lean: only spec columns that actually hold a
        # value are emitted, so blank optional specs do not appear as nulls.
        records = []
        for _, row in plants.iterrows():
            record = {
                "name": str(row.get("name", "")),
                "technology": str(row.get(PLANT_TECHNOLOGY_COLUMN, "")),
            }
            for column in PLANT_PARAMETER_COLUMNS:
                value = _numeric_param(row, column)
                if value is not None:
                    record[column] = value
            records.append(record)
        return records

    def _strategy_parameters(self) -> dict:
        return {
            "name": self.config.strategy_name,
            "dispatch_method": self.config.dispatch_setting("dispatch_method"),
            "rolling_horizon_enabled": self.config.dispatch_setting("rolling_horizon_enabled"),
            "dispatch_horizon_hours": self.config.dispatch_setting("dispatch_horizon_hours"),
            "rolling_step_hours": self.config.dispatch_setting("rolling_step_hours"),
        }

    def _solver_parameters(self) -> dict:
        return {
            "name": self.config.solver_name,
            "fallback_solvers": list(self.config.solver_fallbacks),
        }

    def _environment_section(self) -> dict:
        return {
            "python_version": platform.python_version(),
            "package_lock": "requirements.txt",
            "platform": sys.platform,
        }

    def _relative_path(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(path)


def _git_output(project_root: Path, *args: str) -> str | None:
    # Any git failure (missing binary, non-repo, or command error) returns None;
    # callers fall back to "unknown" instead of crashing the run.
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _repository_name(project_root: Path) -> str:
    url = _git_output(project_root, "config", "--get", "remote.origin.url")
    if not url:
        return project_root.name
    name = url.rstrip("/").split("/")[-1]
    return name[:-4] if name.endswith(".git") else name


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _numeric_column(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(numeric.iloc[0]) if not numeric.empty else None


def _numeric_param(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)
