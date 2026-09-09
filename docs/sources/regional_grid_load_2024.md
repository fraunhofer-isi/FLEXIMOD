<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Regional grid-load input

## Source

- Zenodo record: <https://doi.org/10.5281/zenodo.17109911>
- File used: `system_2024.csv`
- Dataset version: 1.0
- Author: Phumthep Bunnak
- License: CC BY 4.0
- MD5: `5c1f09d6641ad6addb6f282f981e808a`
- Original resolution and unit: hourly MW
- Original field: `metropolitan_demand`

Only the 2024 file is used. The 2023 file is not part of the model inputs.

## Cleaning

The preparation script restricts the data to the 2024 calendar year, prefers
the exact `00:00` value over the duplicate `00:05` value, and creates a complete
hourly index. It linearly interpolates the three absent timestamps:

- 2024-12-07 23:00;
- 2024-12-08 09:00;
- 2024-12-19 17:00.

The cleaned 2024 thresholds are:

- annual maximum: 11,408.2 MW;
- P90: 9,512.28 MW;
- P99: 10,476.459 MW.

## One-day example profile

The building examples run on a Monday in September. Their regional load is
the hour-by-hour mean of the five Mondays in September 2024: September 2, 9,
16, 23, and 30. Hourly means are linearly interpolated to 15-minute values.
The final three quarter-hours interpolate between the Monday 23:00 mean and
the mean at 00:00 on the following Tuesdays.

The following columns are appended to every building example forecast:

- `regional_grid_load_mw`;
- `regional_grid_load_fraction`, divided by the 2024 annual maximum;
- `grid_congestion_weight`, scaled from zero at P90 to one at P99;
- `regional_grid_load_source_year`;
- `regional_grid_load_profile`;
- `regional_grid_load_interpolated`.

Regional load is an external grid-condition signal. It is not added to the
depot electricity balance and does not represent the loading of a particular
feeder or transformer.

## Reproduction

Download `system_2024.csv` from the Zenodo record, then run:

```powershell
python scripts/prepare_regional_grid_load.py `
  path\to\system_2024.csv `
  regional_grid_load_15min.csv
```

The full-year building forecast uses the chronological series instead:

```powershell
python scripts/prepare_annual_building_forecast.py `
  path\to\system_2024.csv `
  data\input\building_v1g_baseline\forecasts_df.csv `
  annual_forecasts_df.csv
```

Three missing hourly observations are linearly interpolated. Quarter-hours
between source hours are also linearly interpolated, with the source file's
1 January 2025 midnight observation used as the final endpoint. P90 and P99
are calculated from the cleaned hourly 2024 series. The annual output retains
the actual 2024 order. The continuous congestion weight remains zero at or
below P90, increases linearly between P90 and P99, and reaches one at P99.
For reporting, weights from 0.8 upward are labelled `stressed`, positive
weights below 0.8 are `elevated`, and zero is `normal`.
