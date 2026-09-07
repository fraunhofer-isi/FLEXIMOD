# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

# ============================================================================
# 1. Neue Lastkurve laden
# ============================================================================
df_new = pd.read_csv("heat_profile.csv", sep=";", dayfirst=True, decimal=",",
                     parse_dates=["datetime"])
df_new = df_new[["datetime", "10"]].copy()
df_new.rename(columns={"10": "plant_1_heat_demand"}, inplace=True)

# Duplikate entfernen
df_new = df_new.drop_duplicates(subset=["datetime"], keep="first")

# Zu 15-min mit Forward Fill aufblasen
df_new = df_new.set_index("datetime")
new_index = pd.date_range(start=df_new.index.min(), end=df_new.index.max(), freq="15min")
df_new = df_new.reindex(new_index).ffill().reset_index()
df_new.rename(columns={"index": "datetime"}, inplace=True)

print("Neue Lastkurve (15-min) Beispiel:")
print(df_new.head(6))

# ============================================================================
# 2. Alte Datei laden
# ============================================================================
df_old = pd.read_csv("forecasts_df.csv", dayfirst=True, parse_dates=["datetime"])

# Sicherstellen, dass beide datetime-Spalten naive Timestamps sind
df_new["datetime"] = pd.to_datetime(df_new["datetime"])
df_old["datetime"] = pd.to_datetime(df_old["datetime"], dayfirst=True)

# ============================================================================
# 3. Merge
# ============================================================================
df_result = df_old.drop(columns=["plant_1_heat_demand"]).merge(
    df_new[["datetime", "plant_1_heat_demand"]],
    on="datetime",
    how="left"
)

# ============================================================================
# 4. Spaltenreihenfolge (plant_1_heat_demand an Position 2)
# ============================================================================
cols = df_result.columns.tolist()
cols.remove("plant_1_heat_demand")
cols.insert(1, "plant_1_heat_demand")
df_result = df_result[cols]

# ============================================================================
# 5. Speichern
# ============================================================================
df_result["datetime"] = df_result["datetime"].dt.strftime("%d.%m.%Y %H:%M")
df_result.to_csv("forecasts_df.csv", index=False)

print("\nforecasts_df.csv aktualisiert.")
print(f"Min: {df_result['plant_1_heat_demand'].min():.4f}, Max: {df_result['plant_1_heat_demand'].max():.4f}")
print(f"NaN-Werte: {df_result['plant_1_heat_demand'].isna().sum()}")
print(f"Anzahl Reihen: {len(df_result)}")
print("\nErste Reihen:")
print(df_result[["datetime", "plant_1_heat_demand"]].head(6))