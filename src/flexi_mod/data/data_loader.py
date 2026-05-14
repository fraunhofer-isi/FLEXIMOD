# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flexi_mod.config.case_config import CaseConfig


class DataValidationError(ValueError):
    """Raised when input data does not satisfy the configured case requirements."""


class DataLoader:
    """Load and validate the three input files used by a case study."""

    def __init__(
        self,
        config: CaseConfig,
        input_dir: str | Path | None = None,
        plants_file: str = "plants.csv",
        forecasts_file: str = "forecasts_df.csv",
    ):
        self.config = config
        self.input_dir = Path(input_dir).resolve() if input_dir else config.config_path.parent
        self.plants_file = plants_file
        self.forecasts_file = forecasts_file

    @property
    def plants_path(self) -> Path:
        return self.input_dir / self.plants_file

    @property
    def forecasts_path(self) -> Path:
        return self.input_dir / self.forecasts_file

    def load_plants(self) -> pd.DataFrame:
        path = self.plants_path
        if not path.exists():
            raise FileNotFoundError(f"plants.csv not found at {path}")

        plants = pd.read_csv(path, skipinitialspace=True)
        required = {"name", "unit_type", "technology"}
        missing = required - set(plants.columns)
        if missing:
            raise DataValidationError(
                f"plants.csv is missing required column(s): {', '.join(sorted(missing))}"
            )
        if plants.empty:
            raise DataValidationError("plants.csv contains no plant rows")

        plants["name"] = plants["name"].astype(str).str.strip()
        plants["technology"] = plants["technology"].astype(str).str.strip()
        return plants

    def load_forecasts(self, required_columns: set[str] | None = None) -> pd.DataFrame:
        path = self.forecasts_path
        if not path.exists():
            raise FileNotFoundError(f"forecasts_df.csv not found at {path}")

        forecasts = pd.read_csv(path, skipinitialspace=True)
        datetime_col = _find_datetime_column(forecasts)
        forecasts[datetime_col] = _parse_datetime_column(forecasts[datetime_col], datetime_col)
        forecasts = (
            forecasts.rename(columns={datetime_col: "datetime"}).set_index("datetime").sort_index()
        )
        forecasts = forecasts[~forecasts.index.duplicated(keep="first")]

        if self._requires_native_resolution():
            forecasts = self._slice_time_range(forecasts)
            forecasts = self._ensure_resolution(forecasts)
            self._check_expected_period_count(forecasts)
        else:
            forecasts = self._ensure_resolution(forecasts)
            forecasts = self._filter_time_range(forecasts)
        self._check_required_columns(forecasts, required_columns or set())
        return forecasts

    def required_forecast_columns(
        self,
        plants: pd.DataFrame,
        extra_required_columns: set[str] | None = None,
    ) -> set[str]:
        required = set(extra_required_columns or set())

        for market_name in self.config.enabled_markets:
            market = self.config.market(market_name)
            signals = market.get("signals", {})
            if market_name in {"day_ahead", "intraday_continuous"}:
                if "price" in signals:
                    required.add(str(signals["price"]))
                continue
            required.update(str(column) for column in signals.values())

        for plant_name, plant_rows in plants.groupby("name"):
            demand_column = _demand_column_for_plant(str(plant_name), plant_rows)
            required.add(demand_column)

        return required

    def _filter_time_range(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        filtered = self._slice_time_range(forecasts)
        self._check_expected_period_count(filtered)
        return filtered

    def _slice_time_range(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        start = pd.Timestamp(self.config.simulation_start)
        end = pd.Timestamp(self.config.simulation_end)
        filtered = forecasts.loc[(forecasts.index >= start) & (forecasts.index <= end)].copy()
        if filtered.empty:
            raise DataValidationError(
                f"forecasts_df.csv has no rows in configured simulation range {start} to {end}"
            )
        return filtered

    def _check_expected_period_count(self, forecasts: pd.DataFrame) -> None:
        start = pd.Timestamp(self.config.simulation_start)
        end = pd.Timestamp(self.config.simulation_end)
        expected_periods = (
            int(((end - start).total_seconds() / 60) / self.config.timestep_minutes) + 1
        )
        if len(forecasts) != expected_periods:
            raise DataValidationError(
                f"Filtered forecast range has {len(forecasts)} rows, expected {expected_periods} "
                f"for {self.config.timestep_minutes}-minute resolution"
            )

    def _ensure_resolution(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        if len(forecasts.index) < 2:
            raise DataValidationError("forecasts_df.csv needs at least two timestamps")

        target_minutes = self.config.timestep_minutes
        observed_minutes = _infer_step_minutes(forecasts.index)
        if observed_minutes == target_minutes:
            return forecasts

        no_resample_markets = {"intraday_continuous", "afrr_energy"}
        active_no_resample = no_resample_markets.intersection(self.config.enabled_markets)
        if active_no_resample:
            raise DataValidationError(
                "Intraday continuous or aFRR energy is enabled, so forecasts_df.csv "
                "must already use "
                f"the configured {target_minutes}-minute timestep. Observed "
                f"{observed_minutes}-minute data. IDC and aFRR energy signals are not "
                "resampled or forward-filled in this implementation."
            )

        if observed_minutes > target_minutes and observed_minutes % target_minutes == 0:
            # Hourly market data is commonly supplied for DA. Forward-fill to the model
            # step only when the source grid is a clean multiple of the configured step.
            full_index = pd.date_range(
                start=forecasts.index.min(),
                end=forecasts.index.max(),
                freq=f"{target_minutes}min",
            )
            return forecasts.reindex(full_index).ffill()

        raise DataValidationError(
            f"Forecast time resolution is {observed_minutes} minutes, "
            f"but config requires {target_minutes} minutes"
        )

    def _requires_native_resolution(self) -> bool:
        native_resolution_markets = {"intraday_continuous", "afrr_energy"}
        return bool(native_resolution_markets.intersection(self.config.enabled_markets))

    @staticmethod
    def _check_required_columns(forecasts: pd.DataFrame, required_columns: set[str]) -> None:
        missing = sorted(required_columns - set(forecasts.columns))
        if missing:
            raise DataValidationError(
                "forecasts_df.csv is missing required column(s): " + ", ".join(missing)
            )


def _find_datetime_column(frame: pd.DataFrame) -> str:
    for candidate in ["datetime", "timestamp", "time"]:
        if candidate in frame.columns:
            return candidate
    raise DataValidationError("forecasts_df.csv must contain a datetime column")


def _parse_datetime_column(series: pd.Series, column_name: str) -> pd.Series:
    if series.astype(str).str.match(r"\s*\d{1,2}\.\d{1,2}\.\d{4}").any():
        try:
            return pd.to_datetime(series, errors="raise", format="mixed", dayfirst=True)
        except (TypeError, ValueError) as dotted_error:
            raise DataValidationError(
                f"Could not parse datetime column '{column_name}' in forecasts_df.csv. "
                "Use a consistent datetime format such as YYYY-MM-DD HH:MM or DD.MM.YYYY HH:MM."
            ) from dotted_error

    try:
        return pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError):
        try:
            return pd.to_datetime(series, errors="raise", format="mixed", dayfirst=True)
        except (TypeError, ValueError) as second_error:
            raise DataValidationError(
                f"Could not parse datetime column '{column_name}' in forecasts_df.csv. "
                "Use a consistent datetime format such as YYYY-MM-DD HH:MM or DD.MM.YYYY HH:MM."
            ) from second_error


def _infer_step_minutes(index: pd.DatetimeIndex) -> int:
    diffs = index.to_series().diff().dropna().dt.total_seconds().div(60)
    unique = sorted(set(int(value) for value in diffs))
    if len(unique) != 1:
        raise DataValidationError(
            "forecasts_df.csv must use a regular time grid; observed steps are "
            + ", ".join(str(value) for value in unique)
            + " minutes"
        )
    return unique[0]


def _demand_column_for_plant(plant_name: str, plant_rows: pd.DataFrame) -> str:
    if "demand" in plant_rows.columns:
        values = [
            str(value).strip()
            for value in plant_rows["demand"].dropna().tolist()
            if str(value).strip()
        ]
        if values:
            return values[0]
    return f"{plant_name}_heat_demand"
