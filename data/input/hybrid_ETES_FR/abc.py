# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

df = pd.read_csv("forecasts_df.csv")

# Statistiken für die Spalte 'plant_1_heat_demand' berechnen
mean_val = df["plant_1_heat_demand"].mean()
median_val = df["plant_1_heat_demand"].median()
min_val = df["plant_1_heat_demand"].min()
max_val = df["plant_1_heat_demand"].max()

# Werte mit print ausgeben
print("Mean:", mean_val)
print("Median:", median_val)
print("Min:", min_val)
print("Max:", max_val)