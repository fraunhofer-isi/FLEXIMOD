# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

# CSV-Datei einlesen
df = pd.read_csv('forecasts_df.csv')

# gas price
gas_price = 74.7
# CO2 price
# co2_price = 20.1

# set prices in forecasts
df['natural_gas_price'] = gas_price
# df['co2_price'] = co2_price

# Datei speichern
df.to_csv('forecasts_df.csv', index=False)

print(f"natural_gas_price auf {gas_price} gesetzt und Datei gespeichert!")