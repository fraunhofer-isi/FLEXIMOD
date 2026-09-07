# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

# Laden
df = pd.read_csv("forecasts_df.csv", parse_dates=["datetime"])

# Skalierungsparameter
old_min = df["plant_1_heat_demand"].min()
old_max = df["plant_1_heat_demand"].max()
new_min = 0.1
new_max = 0.9

# Spalte direkt überschreiben
df["plant_1_heat_demand"] = (
    new_min
    + (df["plant_1_heat_demand"] - old_min)
    / (old_max - old_min)
    * (new_max - new_min)
).round(2)

# Speichern
df.to_csv("forecasts_df.csv", index=False)

print("forecasts_df.csv aktualisiert.")
print(f"Min: {df['plant_1_heat_demand'].min()}, Max: {df['plant_1_heat_demand'].max()}")