# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

# --- Einlesen ---
forecasts = pd.read_csv("forecasts_df.csv")
afrr_cleaned = pd.read_csv("ES_aFRR_capacity_cleaned.csv")

# --- Sicherheitscheck ---
assert len(forecasts) == len(afrr_cleaned), \
    f"Zeilenlänge stimmt nicht überein: {len(forecasts)} vs {len(afrr_cleaned)}"

# --- Werte direkt überschreiben ---
forecasts["aFRR_capacity_down_price"] = afrr_cleaned["price_EUR_per_MW_per_ISP"].values

# --- Spalte entfernen ---
forecasts.drop(columns=["ES_aFRR_capacity_down_marginal_price"], inplace=True)

# --- Speichern ---
forecasts.to_csv("forecasts_df.csv", index=False)
print("✅ Fertig. Gespeichert: forecasts_df_updated.csv")