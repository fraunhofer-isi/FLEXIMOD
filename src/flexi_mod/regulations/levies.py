# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Levies, surcharges and taxes on electricity, kept apart from the grid fees.

The electricity bill of an industrial consumer has two kinds of per-MWh charges:

* Grid fees (network tariff): what the grid operator charges for using the grid.
  They depend on the tariff design (tiers, ordered capacity, ...) and are modelled in
  ``grid_fees.py``, one class per design.
* Levies, surcharges and taxes: fixed amounts per MWh set by law. They do not depend
  on how the grid fee is designed and are the same under every German design.
  They are modelled here, once, and every German regulation uses the same values.

Where the levies act in the model
---------------------------------
Every MWh drawn from the grid pays the sum of all levies, whatever the hour and
whatever the load. The sum is therefore added to the electricity price the plant sees
(together with the energy part of the grid fee) and it is part of the settlement
after the run.

The German levies are rows of ``additional_charges.csv``, all in EUR/MWh:

    chp_surcharge        CHP surcharge (KWKG-Umlage)
    offshore_grid_levy   Offshore grid levy (Offshore-Netzumlage)
    concession_fee       Concession fee (Konzessionsabgabe)
    electricity_tax      Electricity tax as a flat amount (Stromsteuer)

A row with any other name is not a levy. It must be a grid-fee charge of the selected
regulation, otherwise the model stops with an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The names of the levy rows in additional_charges.csv.
LEVY_NAMES = ("chp_surcharge", "offshore_grid_levy", "concession_fee", "electricity_tax")


@dataclass(frozen=True)
class ElectricityLevies:
    """The levies and taxes that one plant pays on every MWh it draws from the grid."""

    chp_surcharge: float = 0.0  # EUR/MWh
    offshore_grid_levy: float = 0.0  # EUR/MWh
    concession_fee: float = 0.0  # EUR/MWh
    electricity_tax: float = 0.0  # EUR/MWh

    @property
    def eur_per_mwh(self) -> float:
        """All levies added up: what one MWh drawn from the grid costs in levies (EUR/MWh)."""
        return (
            self.chp_surcharge
            + self.offshore_grid_levy
            + self.concession_fee
            + self.electricity_tax
        )


def split_levies(charges: pd.DataFrame) -> tuple[ElectricityLevies, pd.DataFrame]:
    """Split the rows of ``additional_charges.csv`` into levies and grid-fee charges.

    ``charges`` has one row per charge, with the columns ``component`` and ``value``.
    Returns two things:

    1. the levies of the plant (a levy that is not in the CSV counts as zero),
    2. the remaining rows, which belong to the grid fee.
    """

    levy_values = {}  # levy name -> value in EUR/MWh
    is_grid_fee_row = []  # one True/False per row: True if the row is not a levy

    for _, row in charges.iterrows():
        name = str(row["component"]).strip()
        if name in LEVY_NAMES:
            levy_values[name] = float(row["value"])
            is_grid_fee_row.append(False)
        else:
            is_grid_fee_row.append(True)

    levies = ElectricityLevies(**levy_values)
    grid_fee_rows = charges[np.array(is_grid_fee_row, dtype=bool)]
    return levies, grid_fee_rows
