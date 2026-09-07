# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

# --- 1. heat_profile.csv einlesen ---
heat = pd.read_csv(
    "heat_profile.csv",
    sep=";",
    decimal=",",
    usecols=["Hour", "T2"]
)

# --- 2. Auf 15-Minuten-Schritte erweitern (forward fill / repeat) ---
# Jede Stunde 4x wiederholen → 8760 * 4 = 35040 Zeilen
heat_15min = heat["T2"].repeat(4).reset_index(drop=True)

# --- 3. forecasts_df.csv einlesen ---
forecasts = pd.read_csv("forecasts_df.csv")

# --- 4. Längencheck ---
if len(heat_15min) != len(forecasts):
    raise ValueError(
        f"Zeilenanzahl stimmt nicht überein: "
        f"heat_profile (expanded) = {len(heat_15min)}, "
        f"forecasts_df = {len(forecasts)}"
    )

# --- 5. Spalte ersetzen ---
forecasts["plant_1_heat_demand"] = heat_15min.values

# --- 6. Ergebnis speichern ---
forecasts.to_csv("forecasts_df.csv", index=False)

print("Fertig. plant_1_heat_demand wurde erfolgreich ersetzt.")
print(f"Erste Werte:\n{forecasts[['datetime','plant_1_heat_demand']].head(8)}")