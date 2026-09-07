# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

# ============================================================
# KONFIGURATION – hier je Land anpassen
# ============================================================
FORECASTS_FILE   = "forecasts_df.csv"
CO2_FILE         = "fr_co2_intensity.csv"   # fr_co2_intensity.csv / es_co2_intensity.csv
TIMEZONE         = "Europe/Paris"           # Europe/Paris / Europe/Madrid
CO2_SOURCE_COL   = "co2_intensity_gco2_kwh"
CO2_TARGET_COL   = "co2_intensity_kgco2_MWh"
# FR:
DATETIME_FORMAT  = "%Y-%m-%d %H:%M:%S"
# DE/ES (wie bisher):
# DATETIME_FORMAT  = "%d.%m.%Y %H:%M"
# ============================================================

# Einlesen
forecasts_df = pd.read_csv(FORECASTS_FILE)
co2_df = pd.read_csv(CO2_FILE)

# forecasts_df datetime: Lokalzeit → UTC
forecasts_df["datetime_parsed"] = (
    pd.to_datetime(forecasts_df["datetime"], format=DATETIME_FORMAT)
    .dt.tz_localize(TIMEZONE, ambiguous="infer", nonexistent="shift_forward")
    .dt.tz_convert("UTC")
)

# co2_df datetime: bereits UTC
co2_df["datetime_parsed"] = pd.to_datetime(co2_df["datetime_utc"], utc=True)

# Matching per Left-Join
co2_slim = co2_df[["datetime_parsed", CO2_SOURCE_COL]].copy()

# Falls Spalte schon existiert (Wiederholung für DE), erst entfernen
if CO2_TARGET_COL in forecasts_df.columns:
    forecasts_df.drop(columns=[CO2_TARGET_COL], inplace=True)
    print(f"Alte Spalte '{CO2_TARGET_COL}' wurde ersetzt.")

forecasts_df = forecasts_df.merge(co2_slim, on="datetime_parsed", how="left")

# Umbenennen
forecasts_df.rename(columns={CO2_SOURCE_COL: CO2_TARGET_COL}, inplace=True)

# Fehlende Werte: Forward Fill
forecasts_df[CO2_TARGET_COL] = forecasts_df[CO2_TARGET_COL].ffill()

# Hilfsspalte entfernen & speichern
forecasts_df.drop(columns=["datetime_parsed"], inplace=True)
forecasts_df.to_csv(FORECASTS_FILE, index=False)

# Zusammenfassung
n_missing = forecasts_df[CO2_TARGET_COL].isna().sum()
n_total   = len(forecasts_df)
print(f"Fertig! Zeilen gesamt: {n_total} | Noch fehlend nach ffill: {n_missing}")
print(forecasts_df[["datetime", CO2_TARGET_COL]].head(10))