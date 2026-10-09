# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from flexi_mod.config.case_config import CaseConfig


class DataValidationError(ValueError):
    """Raised when input data does not satisfy the configured case requirements."""


@dataclass(frozen=True)
class PlantDefinition:
    """One multi-component plant read from ``plants.csv``.

    ``plants.csv`` is intentionally denormalised for modellers: each row
    describes one technology and rows with the same ``name`` belong to one
    plant.  This definition restores a model-friendly shape without imposing a
    technology-specific schema.  Plant-level values live in ``metadata`` and
    each technology keeps its own parameter dictionary in ``components``.
    """

    name: str
    unit_type: str
    metadata: dict[str, Any]
    components: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CaseInputs:
    """Structured plant definitions and their common, validated forecasts."""

    plants_by_type: dict[str, list[PlantDefinition]]
    forecasts: pd.DataFrame

    @property
    def plants(self) -> list[PlantDefinition]:
        """Return all plants in their order of appearance in ``plants.csv``."""

        return [plant for plants in self.plants_by_type.values() for plant in plants]

    def forecast_for(self, plant: PlantDefinition | str, signal_name: str) -> pd.Series | None:
        """Return a plant-specific signal, falling back to a global signal.

        A request for ``steel_demand`` from plant ``steel_1`` resolves
        ``steel_1_steel_demand`` first and then the global ``steel_demand``
        column.  This mirrors ASSUME's input convention while also allowing
        shared price and fuel columns in one wide ``forecasts_df.csv``.
        """

        plant_name = plant.name if isinstance(plant, PlantDefinition) else str(plant)
        return get_plant_forecast_column(self.forecasts, plant_name, signal_name)

    def require_forecast_for(self, plant: PlantDefinition | str, signal_name: str) -> pd.Series:
        """Return a required plant/global forecast or raise a clear input error."""

        series = self.forecast_for(plant, signal_name)
        if series is not None:
            return series
        plant_name = plant.name if isinstance(plant, PlantDefinition) else str(plant)
        raise DataValidationError(
            f"forecasts_df.csv is missing forecast '{signal_name}' for plant '{plant_name}'. "
            f"Expected '{plant_name}_{signal_name}' or '{signal_name}'."
        )


def get_plant_forecast_column(
    forecasts: pd.DataFrame | None,
    plant_name: str,
    signal_name: str,
) -> pd.Series | None:
    """Return a plant-specific forecast column, falling back to a global one.

    Plant-specific columns take precedence over shared columns.  Passing an
    already-prefixed signal name is supported and does not duplicate the plant
    prefix.  This helper is deliberately technology-agnostic so new plant
    models can request their own profiles without changes to :class:`DataLoader`.
    """

    if forecasts is None:
        return None
    prefix = f"{plant_name}_"
    candidates = (
        (signal_name,)
        if signal_name.startswith(prefix)
        else (f"{prefix}{signal_name}", signal_name)
    )
    for column in candidates:
        if column in forecasts.columns:
            return forecasts[column]
    return None


class DataLoader:
    """Load flat case files and expose structured, multi-component plant inputs."""

    def __init__(
        self,
        config: CaseConfig,
        input_dir: str | Path | None = None,
        plants_file: str = "plants.csv",
        forecasts_file: str = "forecasts_df.csv",
        additional_charges_file: str = "additional_charges.csv",
    ):
        self.config = config
        self.input_dir = Path(input_dir).resolve() if input_dir else config.config_path.parent
        self.plants_file = plants_file
        self.forecasts_file = forecasts_file
        self.additional_charges_file = additional_charges_file

    @property
    def plants_path(self) -> Path:
        return self.input_dir / self.plants_file

    @property
    def forecasts_path(self) -> Path:
        return self.input_dir / self.forecasts_file

    @property
    def additional_charges_path(self) -> Path:
        return self.input_dir / self.additional_charges_file

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

    def load_plant_definitions(
        self,
        plants: pd.DataFrame | None = None,
    ) -> dict[str, list[PlantDefinition]]:
        """Group technology rows into generic plant/component definitions.

        The return value follows the same shape as ASSUME's DSM input loader:
        plants are grouped by ``unit_type`` and retain all technology-specific
        values in a ``components`` mapping.  This method only understands the
        input grammar; it deliberately does not validate a plant's physical
        topology.  That validation belongs to the registered plant model.

        ``component_id`` may be supplied to distinguish multiple components of
        the same technology.  If omitted, the technology name is used as the
        component key and therefore must be unique within a plant.
        """

        frame = self.load_plants() if plants is None else plants.copy()
        result: dict[str, list[PlantDefinition]] = {}
        component_id_column = "component_id" if "component_id" in frame.columns else None

        for plant_name, rows in frame.groupby("name", sort=False):
            unit_types = {
                str(value).strip().lower()
                for value in rows["unit_type"].dropna().tolist()
                if str(value).strip()
            }
            if len(unit_types) != 1:
                raise DataValidationError(
                    f"Plant '{plant_name}' must use exactly one non-empty unit_type"
                )
            unit_type = unit_types.pop()
            metadata = _plant_metadata(rows, str(plant_name), unit_type)
            components: dict[str, dict[str, Any]] = {}

            for _, row in rows.iterrows():
                technology = str(row["technology"]).strip().lower()
                if not technology:
                    raise DataValidationError(f"Plant '{plant_name}' has an empty technology name")
                component_key = technology
                if component_id_column and _has_value(row[component_id_column]):
                    component_key = str(row[component_id_column]).strip()
                if component_key in components:
                    raise DataValidationError(
                        f"Plant '{plant_name}' defines duplicate component '{component_key}'. "
                        "Use a unique component_id when a technology occurs more than once."
                    )
                components[component_key] = _component_parameters(
                    row=row,
                    component_id_column=component_id_column,
                )

            definition = PlantDefinition(
                name=str(plant_name),
                unit_type=unit_type,
                metadata=metadata,
                components=components,
            )
            result.setdefault(unit_type, []).append(definition)
        return result

    def load_case_inputs(
        self,
        required_columns: set[str] | None = None,
    ) -> CaseInputs:
        """Load grouped plant definitions and the validated forecast frame together.

        Existing callers may continue using :meth:`load_plants` and
        :meth:`load_forecasts`.  New, generic plant models should use this
        method and resolve profiles via :meth:`CaseInputs.forecast_for`.
        """

        plants = self.load_plants()
        forecasts = self.load_forecasts(required_columns=required_columns)
        return CaseInputs(
            plants_by_type=self.load_plant_definitions(plants),
            forecasts=forecasts,
        )

    def load_additional_charges(self, plants: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Load plant-specific network-tariff components.

        Returns one tidy frame per plant with columns ``component``, ``unit`` and
        ``value``. The loader stays country-agnostic: tier/levy/threshold
        interpretation is performed later by the selected grid-fee regulation
        (see :mod:`flexi_mod.grid_fees`). Returns an empty mapping when the case
        has ``additional_charges`` disabled.
        """

        plant_names = sorted(str(name) for name in plants["name"].dropna().unique())
        if not self.config.additional_charges_enabled:
            return {}

        path = self.additional_charges_path
        if not path.exists():
            raise FileNotFoundError(
                "cases.<case_name>.additional_charges=true but additional_charges.csv "
                f"was not found at {path}"
            )

        charges = pd.read_csv(path, skipinitialspace=True)
        required = {"component", "unit"}
        missing = required - set(charges.columns)
        if missing:
            raise DataValidationError(
                "additional_charges.csv is missing required column(s): "
                + ", ".join(sorted(missing))
            )
        if charges.empty:
            raise DataValidationError("additional_charges.csv contains no charge rows")

        units = charges["unit"].astype(str).str.strip()
        allowed_units = {"EUR/MWh", "EUR/MW.a"}
        invalid_units = sorted(set(units) - allowed_units)
        if invalid_units:
            raise DataValidationError(
                "additional_charges.csv supports units 'EUR/MWh' and 'EUR/MW.a'; found: "
                + ", ".join(invalid_units)
            )

        missing_plants = [name for name in plant_names if name not in charges.columns]
        if missing_plants:
            raise DataValidationError(
                "additional_charges.csv is missing plant column(s): " + ", ".join(missing_plants)
            )

        components = charges["component"].astype(str).str.strip()
        result: dict[str, pd.DataFrame] = {}
        for plant_name in plant_names:
            values = pd.to_numeric(charges[plant_name], errors="coerce")
            if values.isna().any():
                bad_components = charges.loc[values.isna(), "component"].astype(str).tolist()
                raise DataValidationError(
                    f"additional_charges.csv has non-numeric value(s) for plant "
                    f"'{plant_name}' in component(s): {', '.join(bad_components)}"
                )
            result[plant_name] = pd.DataFrame(
                {
                    "component": components.to_numpy(),
                    "unit": units.to_numpy(),
                    "value": values.to_numpy(dtype=float),
                }
            )
        return result

    def load_forecasts(self, required_columns: set[str] | None = None) -> pd.DataFrame:
        path = self.forecasts_path
        if not path.exists():
            raise FileNotFoundError(f"forecasts_df.csv not found at {path}")

        forecasts = pd.read_csv(path, skipinitialspace=True)
        datetime_col = _find_datetime_column(forecasts)
        parsed_datetime = _parse_datetime_column(forecasts[datetime_col], datetime_col)
        forecasts[datetime_col] = _case_datetime_index(
            parsed_datetime,
            timezone=self.config.timezone,
        )
        forecasts = (
            forecasts.rename(columns={datetime_col: "datetime"}).set_index("datetime").sort_index()
        )
        forecasts = forecasts[~forecasts.index.duplicated(keep="first")]

        if self._requires_native_resolution():
            forecasts = self._slice_time_range(forecasts)
            forecasts = self._ensure_resolution(forecasts)
            self._check_expected_period_count(forecasts)
        else:
            forecasts = self._slice_time_range(forecasts)
            forecasts = self._ensure_resolution(forecasts)
            self._check_expected_period_count(forecasts)
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
            unit_types = {
                str(value).strip().lower() for value in plant_rows["unit_type"].dropna().tolist()
            }
            if unit_types & {"building", "bus_depot", "electric_bus_depot"}:
                from flexi_mod.plants.building import building_profile_columns

                required.update(building_profile_columns(str(plant_name), plant_rows))
            else:
                demand_column = _demand_column_for_plant(str(plant_name), plant_rows)
                required.add(demand_column)

        return required

    def _filter_time_range(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        filtered = self._slice_time_range(forecasts)
        self._check_expected_period_count(filtered)
        return filtered

    def _slice_time_range(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        start = _timestamp_for_index(
            pd.Timestamp(self.config.simulation_start),
            forecasts.index,
            "simulation_start",
        )
        end = _timestamp_for_index(
            pd.Timestamp(self.config.simulation_end),
            forecasts.index,
            "simulation_end",
        )
        filtered = forecasts.loc[(forecasts.index >= start) & (forecasts.index <= end)].copy()
        if filtered.empty:
            raise DataValidationError(
                f"forecasts_df.csv has no rows in configured simulation range {start} to {end}"
            )
        return filtered

    def _check_expected_period_count(self, forecasts: pd.DataFrame) -> None:
        start = pd.Timestamp(self.config.simulation_start)
        end = pd.Timestamp(self.config.simulation_end)
        expected_periods = _expected_period_count(
            start,
            end,
            self.config.timestep_minutes,
            self.config.timezone,
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
        observed_minutes = _infer_step_minutes(forecasts.index, self.config.timezone)
        if observed_minutes == target_minutes:
            return forecasts

        no_resample_markets = {"intraday_continuous", "afrr_energy", "afrr_capacity"}
        active_no_resample = no_resample_markets.intersection(self.config.enabled_markets)
        if active_no_resample:
            raise DataValidationError(
                "Intraday continuous, aFRR energy, or aFRR capacity is enabled, so "
                "forecasts_df.csv must already use "
                f"the configured {target_minutes}-minute timestep. Observed "
                f"{observed_minutes}-minute data. IDC and aFRR signals are not "
                "resampled or forward-filled in this implementation."
            )

        if observed_minutes > target_minutes and observed_minutes % target_minutes == 0:
            # Hourly market data is commonly supplied for DA. Forward-fill to the model
            # step only when the source grid is a clean multiple of the configured step.
            start = pd.Timestamp(self.config.simulation_start)
            end = pd.Timestamp(self.config.simulation_end)
            indexed_start = _timestamp_for_index(start, forecasts.index, "simulation_start")
            if forecasts.index.min() > indexed_start:
                raise DataValidationError(
                    "Forecast data start after the configured simulation_start, so DA-only "
                    "resampling would create leading missing values"
                )
            full_index = _case_date_range(
                start=start,
                end=end,
                timestep_minutes=target_minutes,
                timezone=self.config.timezone,
            )
            return forecasts.reindex(full_index).ffill()

        raise DataValidationError(
            f"Forecast time resolution is {observed_minutes} minutes, "
            f"but config requires {target_minutes} minutes"
        )

    def _requires_native_resolution(self) -> bool:
        native_resolution_markets = {"intraday_continuous", "afrr_energy", "afrr_capacity"}
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


def _case_datetime_index(series: pd.Series, timezone: str | None = None) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(series)
    if timezone is None or index.tz is not None:
        return index
    try:
        return index.tz_localize(timezone, ambiguous="infer", nonexistent="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Could not interpret forecast timestamps in timezone '{timezone}'. "
            "For spring DST changes, remove nonexistent local timestamps. For autumn "
            "DST changes, include both duplicated local timestamps in chronological order "
            "or provide timezone-aware timestamps."
        ) from exc


def _infer_step_minutes(index: pd.DatetimeIndex, timezone: str | None = None) -> int:
    validation_index = _elapsed_time_index(index, timezone)
    diffs = validation_index.to_series().diff().dropna().dt.total_seconds().div(60)
    unique = sorted(set(int(value) for value in diffs))
    if len(unique) != 1:
        raise DataValidationError(
            "forecasts_df.csv must use a regular time grid; observed steps are "
            + ", ".join(str(value) for value in unique)
            + " minutes"
        )
    return unique[0]


def _expected_period_count(
    start: pd.Timestamp,
    end: pd.Timestamp,
    timestep_minutes: int,
    timezone: str | None = None,
) -> int:
    if timezone is None:
        elapsed_minutes = (end - start).total_seconds() / 60
    else:
        start_utc = _local_timestamp_to_utc(start, timezone, label="simulation_start")
        end_utc = _local_timestamp_to_utc(end, timezone, label="simulation_end")
        elapsed_minutes = (end_utc - start_utc).total_seconds() / 60
    return int(elapsed_minutes / timestep_minutes) + 1


def _case_date_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    timestep_minutes: int,
    timezone: str | None = None,
) -> pd.DatetimeIndex:
    if timezone is None:
        return pd.date_range(start=start, end=end, freq=f"{timestep_minutes}min")
    try:
        return pd.date_range(
            start=start,
            end=end,
            freq=f"{timestep_minutes}min",
            tz=timezone,
        )
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Could not create a {timestep_minutes}-minute datetime grid in timezone "
            f"'{timezone}'. Check case.timezone, simulation_start and simulation_end."
        ) from exc


def _elapsed_time_index(index: pd.DatetimeIndex, timezone: str | None) -> pd.DatetimeIndex:
    """Return timestamps on an elapsed-time axis for robust DST-aware grid checks."""

    if timezone is None:
        return index
    if index.tz is not None:
        return index.tz_convert("UTC")
    try:
        return index.tz_localize(timezone, ambiguous="infer", nonexistent="raise").tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Could not validate forecast timestamps in timezone '{timezone}'. "
            "If the data cover the autumn DST clock change, use timezone-aware "
            "timestamps or make duplicated local times distinguishable."
        ) from exc


def _timestamp_for_index(
    timestamp: pd.Timestamp,
    index: pd.DatetimeIndex,
    label: str,
) -> pd.Timestamp:
    if index.tz is None:
        return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    try:
        if timestamp.tzinfo is not None:
            return timestamp.tz_convert(index.tz)
        return timestamp.tz_localize(index.tz, ambiguous="raise", nonexistent="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Configured {label}={timestamp} cannot be interpreted unambiguously in "
            f"timezone '{index.tz}'. Choose an existing, unambiguous local timestamp "
            "or provide timezone-aware timestamps."
        ) from exc


def _local_timestamp_to_utc(
    timestamp: pd.Timestamp,
    timezone: str,
    label: str,
) -> pd.Timestamp:
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert("UTC")
    try:
        return timestamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise").tz_convert(
            "UTC"
        )
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Configured {label}={timestamp} cannot be interpreted unambiguously in "
            f"timezone '{timezone}'. Choose an existing, unambiguous local timestamp "
            "or provide timezone-aware timestamps."
        ) from exc


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


_PLANT_METADATA_COLUMNS = ("node", "objective", "demand")


def _plant_metadata(
    rows: pd.DataFrame,
    plant_name: str,
    unit_type: str,
) -> dict[str, Any]:
    """Extract values shared by a plant's technology rows.

    Only universally understood plant fields are lifted out of component
    dictionaries.  Everything else remains component-specific, even when it
    happens to be repeated in a CSV, so the loader never guesses a physical
    parameter's ownership.
    """

    metadata: dict[str, Any] = {"name": plant_name, "unit_type": unit_type}
    for column in _PLANT_METADATA_COLUMNS:
        if column not in rows.columns:
            continue
        values = [value for value in rows[column].tolist() if _has_value(value)]
        if not values:
            continue
        unique_values = {_comparison_value(value) for value in values}
        if len(unique_values) != 1:
            raise DataValidationError(
                f"Plant '{plant_name}' has inconsistent plant-level '{column}' values"
            )
        metadata[column] = values[0]
    return metadata


def _component_parameters(
    row: pd.Series,
    component_id_column: str | None,
) -> dict[str, Any]:
    """Return populated technology-specific fields from one plant CSV row."""

    excluded_columns = {
        "name",
        "unit_type",
        "technology",
        *_PLANT_METADATA_COLUMNS,
    }
    if component_id_column:
        excluded_columns.add(component_id_column)
    return {
        column: value
        for column, value in row.items()
        if column not in excluded_columns and _has_value(value)
    }


def _has_value(value: Any) -> bool:
    """Return whether a CSV cell carries a meaningful value."""

    return not pd.isna(value) and str(value).strip() != ""


def _comparison_value(value: Any) -> str:
    """Normalise a scalar only for cross-row consistency checks."""

    return str(value).strip()
