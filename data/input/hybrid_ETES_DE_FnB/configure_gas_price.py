# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

# CSV-Datei einlesen
df = pd.read_csv('forecasts_df.csv')

# gas price, incl. CO2
gas_price = 106.1 # 86.0 + 20.1 = 106.1

# natural_gas_price auf 60.15 setzen
df['natural_gas_price'] = gas_price

# Datei speichern
df.to_csv('forecasts_df.csv', index=False)

print(f"natural_gas_price auf {gas_price} gesetzt und Datei gespeichert!")