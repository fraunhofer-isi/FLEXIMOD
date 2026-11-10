# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from flexi_mod.regulations import (
    ElectricityLevies,
    GermanGridFeeRegulation,
    GridFeeConfigError,
    split_levies,
)


def charges(rows: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component": list(rows),
            "unit": ["EUR/MWh"] * len(rows),
            "value": list(rows.values()),
        }
    )


def test_levies_add_up_to_the_cost_of_one_mwh():
    levies = ElectricityLevies(
        chp_surcharge=2.77, offshore_grid_levy=8.16, concession_fee=1.1, electricity_tax=0.5
    )
    assert levies.eur_per_mwh == pytest.approx(12.53)


def test_levies_not_given_are_zero():
    assert ElectricityLevies().eur_per_mwh == 0.0


def test_split_levies_separates_levy_rows_from_grid_fee_rows():
    frame = charges(
        {"grid_energy_charge_high": 19.9, "chp_surcharge": 2.77, "electricity_tax": 0.5}
    )

    levies, grid_fee_rows = split_levies(frame)

    assert levies.chp_surcharge == 2.77
    assert levies.electricity_tax == 0.5
    assert levies.concession_fee == 0.0
    assert grid_fee_rows["component"].tolist() == ["grid_energy_charge_high"]


def test_german_regulation_takes_its_levies_from_the_csv_rows():
    frame = charges(
        {
            "grid_energy_charge_high": 19.9,
            "grid_energy_charge_low": 43.6,
            "chp_surcharge": 2.77,
            "concession_fee": 1.1,
        }
    )

    regulation = GermanGridFeeRegulation.from_charges_frame(frame)

    assert regulation.marginal_charge_eur_per_mwh() == pytest.approx(19.9 + 2.77 + 1.1)


def test_german_error_for_an_unknown_row_still_lists_the_levy_names():
    frame = charges(
        {"grid_energy_charge_high": 19.9, "grid_energy_charge_low": 43.6, "chp_surcharg": 2.77}
    )

    with pytest.raises(GridFeeConfigError, match="'chp_surcharg'.*chp_surcharge"):
        GermanGridFeeRegulation.from_charges_frame(frame)
