# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

# 1. CSV-Datei einlesen
df = pd.read_csv('forecasts_df.csv')

# 2. Spalte durch 4 teilen und direkt zuweisen
df['aFRR_capacity_down_price'] = df['aFRR_capacity_down_price'] * 4

# 3. Datei überschreiben (index=False sorgt dafür, dass keine neue Index-Spalte hinzugefügt wird)
df.to_csv('forecasts_df.csv', index=False)