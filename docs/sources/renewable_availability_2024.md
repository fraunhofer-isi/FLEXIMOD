<!--
SPDX-FileCopyrightText: FLEXIMOD Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Renewable-availability proxy

## Meaning

`renewable_availability_weight` is a modelled variable-renewable availability
proxy, not the observed renewable share of grid electricity. It combines local
2024 PV and wind weather conditions with national installed-capacity weights.
Hydropower, biomass, curtailment, outages, imports, and transmission limits are
not represented.

## Weather source

The hourly weather input comes from the NASA POWER Hourly API:

- location: latitude 13.7563, longitude 100.5018;
- period requested: 31 December 2023 through 31 December 2024;
- time standard: UTC, converted to `Asia/Bangkok`;
- parameters: `ALLSKY_SFC_SW_DWN`, `T2M`, and `WS50M`;
- API version returned: v2.10.0;
- source JSON SHA-256:
  `ace0bb8c66a1e22698e2b77585ec78553f5dd66c6c0e18ec9e652c5d8cd712df`.

Exact request endpoint:

<https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN,T2M,WS50M&community=RE&longitude=100.5018&latitude=13.7563&start=20231231&end=20241231&format=JSON&time-standard=UTC>

NASA POWER documents that its meteorological series are based on MERRA-2 and
that the hourly service returns hourly average values:

- <https://power.larc.nasa.gov/docs/methodology/meteorology/>;
- <https://power.larc.nasa.gov/docs/services/api/temporal/hourly/>;
- <https://power.larc.nasa.gov/docs/referencing/>.

## Conversion to capacity factors

The transparent PV proxy uses horizontal all-sky irradiance, ambient
temperature, a 45 degrees C nominal operating cell temperature, a -0.4%/K
temperature coefficient, and 10% system losses. Values are clipped to 0--1.

The wind proxy extrapolates 50 m wind speed to 100 m with a 1/7 power law. A
generic turbine curve uses 3 m/s cut-in, 12 m/s rated, and 25 m/s cut-out wind
speeds. Output rises cubically between cut-in and rated speed.

These assumptions provide a reproducible availability indicator. They do not
represent a particular PV module, array orientation, wind farm, or turbine.

## Capacity weighting

The primary signal uses the year-consistent end-2024 IRENA capacities:

- solar PV: 3,383 MW;
- wind: 1,544 MW.

The forecast also includes
`renewable_availability_weight_2025_capacity`, using 6,837 MW PV and 1,544 MW
wind as an expansion sensitivity. IRENA reports 6,842 MW total solar in 2025;
6,837 MW is the PV component compatible with the PV weather model.

Source: <https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2026/Mar/IRENA_DAT_RE_capacity_statistics_2026.pdf>.

For capacity year `y`, the signal is:

```text
availability[t] = (PV_capacity[y] * PV_CF[t]
                   + wind_capacity[y] * wind_CF[t])
                  / (PV_capacity[y] + wind_capacity[y])
```

No second min--max normalization is applied.

## Connection to power-system demand

National demand is reconstructed for every hour by summing north, south,
metropolitan, central, and northeast demand in `system_2024.csv`. The derived
potential indicators are:

```text
modelled_vre_potential_mw[t]
    = PV_capacity_2024 * PV_CF[t] + wind_capacity_2024 * wind_CF[t]

modelled_vre_potential_share_of_demand[t]
    = modelled_vre_potential_mw[t] / national_system_demand_mw[t]
```

The second quantity is a potential-share proxy, not actual renewable
generation. Power-system source and cleaning are documented in
`docs/sources/regional_grid_load_2024.md`.

Hourly values are repeated across the four corresponding 15-minute intervals.
This preserves hourly average power and energy instead of inventing
sub-hourly weather changes.

## Reproduction

Download the NASA POWER JSON with the parameters above and the documented 2024
power-system CSV, then run:

```powershell
python scripts/prepare_renewable_signal.py `
  nasa_power_weather_2024.json `
  system_2024.csv `
  data\input\building_v1g_annual\forecasts_df.csv `
  renewable_forecasts_df.csv
```
