# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd
import pytest
from conftest import case_config_text

from flexi_mod.config.case_config import CaseConfig, ConfigError
from flexi_mod.plants.steam_generation_plant import SteamGenerationPlant
from flexi_mod.regulations import GermanGridFeeRegulation, SpanishGridFeeRegulation
from flexi_mod.strategies import build_strategy
from flexi_mod.strategies.hybrid_electric_gas_boiler_strategy import (
    HybridElectricGasBoilerStrategy,
)
from flexi_mod.strategies.hybrid_etes_gas_strategy import HybridETESGasStrategy

ETES_CASE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "input" / "hybrid_ETES_DA_ID_aFRR_energy"
)


def test_direct_strategy_uses_full_gas_and_co2_benchmark(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    plant = _direct_plant()
    strategy = HybridElectricGasBoilerStrategy(config)
    forecasts = _forecasts()

    gas_heat_benchmark = strategy.calculate_gas_based_heat_cost(plant, forecasts)
    electricity_benchmark = strategy.calculate_electricity_trading_benchmark(
        plant,
        gas_heat_benchmark,
    )

    assert gas_heat_benchmark.to_numpy() == pytest.approx([125.0] * len(forecasts))
    assert electricity_benchmark.to_numpy() == pytest.approx([118.75] * len(forecasts))


def test_direct_strategy_accepts_explicit_zero_co2_price(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    plant = _direct_plant()
    strategy = HybridElectricGasBoilerStrategy(config)
    forecasts = _forecasts()
    forecasts["co2_price"] = 0.0

    gas_heat_benchmark = strategy.calculate_gas_based_heat_cost(plant, forecasts)

    assert gas_heat_benchmark.to_numpy() == pytest.approx([100.0] * len(forecasts))


def test_direct_strategy_rejects_blank_co2_price(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    plant = _direct_plant()
    strategy = HybridElectricGasBoilerStrategy(config)
    forecasts = _forecasts()
    forecasts["co2_price"] = float("nan")

    with pytest.raises(ValueError, match="blank cells are not zero"):
        strategy.calculate_gas_based_heat_cost(plant, forecasts)


def test_direct_strategy_runs_da_idc_and_free_afrr_energy(tmp_path: Path) -> None:
    config = _write_config(tmp_path, margin=1.0)
    plant = _direct_plant()
    plant.additional_electricity_charge_eur_per_mwh = 10.0
    strategy = HybridElectricGasBoilerStrategy(config)
    forecasts = _forecasts()

    day_ahead = strategy.decide_day_ahead(
        plant,
        forecasts,
        rolling=False,
    )
    intraday = strategy.decide_intraday_continuous(
        plant,
        forecasts,
        day_ahead,
        rolling=False,
    )
    result = strategy.decide_afrr_energy(
        plant,
        forecasts,
        intraday,
        rolling=False,
    )

    assert day_ahead["DA_position_MWh"].sum() == pytest.approx(0.0)
    assert intraday["final_planned_electricity_MWh"].sum() == pytest.approx(0.0)
    assert intraday["gas_heat_MWh"].to_numpy() == pytest.approx([0.5] * len(intraday))

    # 0.5 MWh gas heat / 0.95 electric-boiler efficiency = 2.105 MW_el
    # physical headroom, rounded down to the configured 1 MW increment.
    assert result["afrr_energy_bid_MW"].to_numpy() == pytest.approx([2.0] * len(result))
    assert result["afrr_energy_free_bid_MWh"].to_numpy() == pytest.approx(
        result["afrr_energy_bid_MWh"].to_numpy()
    )
    assert result["afrr_energy_capacity_backed_bid_MWh"].sum() == pytest.approx(0.0)

    # The 1 MW activation signal is 0.25 MWh at a 15-minute resolution and
    # is not rounded to the 1 MW bid increment.
    assert result["afrr_energy_activated_MWh"].to_numpy() == pytest.approx([0.25] * len(result))
    assert result["afrr_energy_free_activated_MWh"].to_numpy() == pytest.approx(
        result["afrr_energy_activated_MWh"].to_numpy()
    )
    assert result["afrr_energy_capacity_backed_activated_MWh"].sum() == pytest.approx(0.0)
    assert result["electric_boiler_heat_MWh"].to_numpy() == pytest.approx(
        [0.25 * 0.95] * len(result)
    )
    assert result["gas_heat_MWh"].to_numpy() == pytest.approx([0.5 - 0.25 * 0.95] * len(result))
    assert result["actual_electricity_consumption_MWh"].to_numpy() == pytest.approx(
        (result["final_planned_electricity_MWh"] + result["afrr_energy_activated_MWh"]).to_numpy()
    )

    # Delivered strike = 118.75 gas-derived benchmark - 1.00 margin.
    # The submitted raw bid removes the 10 EUR/MWh electricity charge.
    assert result["afrr_energy_delivered_bid_price_EUR_per_MWh"].to_numpy() == pytest.approx(
        [117.75] * len(result)
    )
    assert result["afrr_energy_bid_price_EUR_per_MWh"].to_numpy() == pytest.approx(
        [107.75] * len(result)
    )


def test_direct_day_ahead_uses_co2_cost_when_selecting_boiler(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    plant = _direct_plant()
    strategy = HybridElectricGasBoilerStrategy(config)
    forecasts = _forecasts(da_price=105.0, idc_price=105.0)

    result = strategy.decide_day_ahead(plant, forecasts, rolling=False)

    # Electricity is more expensive than gas alone, but cheaper than gas plus
    # its explicit CO2 cost.
    assert result["DA_position_MWh"].sum() > 0.0
    assert result["gas_heat_MWh"].sum() == pytest.approx(0.0)


def test_direct_strategy_does_not_bid_without_post_idc_gas_heat(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    plant = _direct_plant()
    strategy = HybridElectricGasBoilerStrategy(config)
    forecasts = _forecasts(da_price=10.0, idc_price=10.0)

    day_ahead = strategy.decide_day_ahead(plant, forecasts, rolling=False)
    intraday = strategy.decide_intraday_continuous(
        plant,
        forecasts,
        day_ahead,
        rolling=False,
    )
    result = strategy.decide_afrr_energy(
        plant,
        forecasts,
        intraday,
        rolling=False,
    )

    assert intraday["gas_heat_MWh"].sum() == pytest.approx(0.0)
    assert result["afrr_energy_bid_MWh"].sum() == pytest.approx(0.0)
    assert result["afrr_energy_activated_MWh"].sum() == pytest.approx(0.0)


def test_direct_strategy_blocks_expensive_or_subminimum_afrr_bid(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    plant = _direct_plant()
    strategy = HybridElectricGasBoilerStrategy(config)

    expensive_forecasts = _forecasts(afrr_price=120.0)
    expensive_day_ahead = strategy.decide_day_ahead(
        plant,
        expensive_forecasts,
        rolling=False,
    )
    expensive_idc = strategy.decide_intraday_continuous(
        plant,
        expensive_forecasts,
        expensive_day_ahead,
        rolling=False,
    )
    expensive = strategy.decide_afrr_energy(
        plant,
        expensive_forecasts,
        expensive_idc,
        rolling=False,
    )
    assert expensive["afrr_energy_bid_MWh"].sum() == pytest.approx(0.0)

    subminimum_forecasts = _forecasts(heat_demand_mw=0.5)
    subminimum_day_ahead = strategy.decide_day_ahead(
        plant,
        subminimum_forecasts,
        rolling=False,
    )
    subminimum_idc = strategy.decide_intraday_continuous(
        plant,
        subminimum_forecasts,
        subminimum_day_ahead,
        rolling=False,
    )
    subminimum = strategy.decide_afrr_energy(
        plant,
        subminimum_forecasts,
        subminimum_idc,
        rolling=False,
    )
    assert subminimum["available_load_headroom_after_schedule_MWh"].min() > 0.0
    assert subminimum["afrr_energy_bid_MWh"].sum() == pytest.approx(0.0)


def test_direct_strategy_reads_afrr_bid_increment_from_config(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path, bid_increment_mw=0.5)
    plant = _direct_plant()
    strategy = HybridElectricGasBoilerStrategy(config)
    forecasts = _forecasts(heat_demand_mw=1.7)
    day_ahead = strategy.decide_day_ahead(plant, forecasts, rolling=False)
    intraday = strategy.decide_intraday_continuous(
        plant,
        forecasts,
        day_ahead,
        rolling=False,
    )

    result = strategy.decide_afrr_energy(
        plant,
        forecasts,
        intraday,
        rolling=False,
    )

    assert result["afrr_energy_bid_MW"].to_numpy() == pytest.approx([1.5] * len(result))


def test_direct_strategy_accepts_break_even_afrr_clearing_price(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    plant = _direct_plant()
    strategy = HybridElectricGasBoilerStrategy(config)
    forecasts = _forecasts(afrr_price=118.75)
    day_ahead = strategy.decide_day_ahead(plant, forecasts, rolling=False)
    intraday = strategy.decide_intraday_continuous(
        plant,
        forecasts,
        day_ahead,
        rolling=False,
    )

    result = strategy.decide_afrr_energy(
        plant,
        forecasts,
        intraday,
        rolling=False,
    )

    assert result["afrr_energy_bid_MW"].min() > 0.0
    assert result["afrr_energy_activated_MWh"].min() > 0.0


def test_direct_strategy_converts_delivered_bid_with_tax_and_charge(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    plant = _direct_plant()
    plant.grid_fee_regulation = SpanishGridFeeRegulation()
    strategy = HybridElectricGasBoilerStrategy(config)
    forecasts = _forecasts()
    forecasts["grid_energy_charge"] = 10.0
    tax_rate = plant.grid_fee_regulation.electricity_tax_rate
    expected_raw_bid = 118.75 / (1.0 + tax_rate) - 10.0
    forecasts["aFRR_energy_down_price"] = expected_raw_bid
    day_ahead = strategy.decide_day_ahead(plant, forecasts, rolling=False)
    intraday = strategy.decide_intraday_continuous(
        plant,
        forecasts,
        day_ahead,
        rolling=False,
    )

    result = strategy.decide_afrr_energy(
        plant,
        forecasts,
        intraday,
        rolling=False,
    )

    assert result["afrr_energy_bid_price_EUR_per_MWh"].to_numpy() == pytest.approx(
        [expected_raw_bid] * len(result)
    )
    assert result["afrr_energy_delivered_bid_price_EUR_per_MWh"].to_numpy() == pytest.approx(
        [118.75] * len(result)
    )
    assert result["afrr_energy_activated_MWh"].min() > 0.0


def test_direct_strategy_blocks_afrr_for_missing_price_or_high_load_window(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    plant = _direct_plant()
    strategy = HybridElectricGasBoilerStrategy(config)

    missing = _forecasts()
    missing["aFRR_energy_down_price"] = pd.NA
    missing_da = strategy.decide_day_ahead(plant, missing, rolling=False)
    missing_idc = strategy.decide_intraday_continuous(
        plant,
        missing,
        missing_da,
        rolling=False,
    )
    with pytest.warns(UserWarning, match="price contains missing"):
        missing_result = strategy.decide_afrr_energy(
            plant,
            missing,
            missing_idc,
            rolling=False,
        )
    assert missing_result["afrr_energy_bid_MWh"].sum() == pytest.approx(0.0)
    assert missing_result["afrr_energy_activated_MWh"].sum() == pytest.approx(0.0)

    blocked_plant = _direct_plant()
    blocked_plant.grid_fee_regulation = GermanGridFeeRegulation(
        avoid_high_load_window=True,
    )
    blocked_strategy = HybridElectricGasBoilerStrategy(config)
    blocked = _forecasts()
    blocked["high_load_window"] = 1
    blocked_da = blocked_strategy.decide_day_ahead(
        blocked_plant,
        blocked,
        rolling=False,
    )
    blocked_idc = blocked_strategy.decide_intraday_continuous(
        blocked_plant,
        blocked,
        blocked_da,
        rolling=False,
    )
    blocked_result = blocked_strategy.decide_afrr_energy(
        blocked_plant,
        blocked,
        blocked_idc,
        rolling=False,
    )
    assert blocked_result["afrr_energy_bid_MWh"].sum() == pytest.approx(0.0)
    assert blocked_result["afrr_energy_activated_MWh"].sum() == pytest.approx(0.0)


def test_direct_intraday_can_buy_or_sell_against_gas_benchmark(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    plant = _direct_plant()
    strategy = HybridElectricGasBoilerStrategy(config)

    buy_forecasts = _forecasts(da_price=200.0, idc_price=10.0)
    buy_da = strategy.decide_day_ahead(plant, buy_forecasts, rolling=False)
    buy_result = strategy.decide_intraday_continuous(
        plant,
        buy_forecasts,
        buy_da,
        rolling=False,
    )
    assert buy_da["DA_position_MWh"].sum() == pytest.approx(0.0)
    assert buy_result["IDC_buy_MWh"].sum() > 0.0
    assert buy_result["gas_heat_MWh"].sum() == pytest.approx(0.0)

    sell_forecasts = _forecasts(da_price=10.0, idc_price=200.0)
    sell_da = strategy.decide_day_ahead(plant, sell_forecasts, rolling=False)
    sell_result = strategy.decide_intraday_continuous(
        plant,
        sell_forecasts,
        sell_da,
        rolling=False,
    )
    assert sell_da["DA_position_MWh"].sum() > 0.0
    assert sell_result["IDC_sell_MWh"].to_numpy() == pytest.approx(
        sell_da["DA_position_MWh"].to_numpy()
    )
    assert sell_result["gas_heat_MWh"].sum() > 0.0


def test_direct_strategy_rejects_etes_route() -> None:
    config = CaseConfig.from_case_dir(ETES_CASE_DIR)
    rows = pd.read_csv(ETES_CASE_DIR / "plants.csv")
    plant = SteamGenerationPlant.from_plants_dataframe(rows)[0]
    strategy = HybridElectricGasBoilerStrategy(config)
    forecasts = pd.DataFrame(
        {
            "natural_gas_price": [80.0],
            "co2_price": [100.0],
        },
        index=pd.date_range("2025-01-01", periods=1, freq="15min"),
    )

    with pytest.raises(ValueError, match="direct electric-boiler"):
        strategy.calculate_gas_based_heat_cost(plant, forecasts)


def test_direct_strategy_rejects_afrr_capacity_call(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    strategy = HybridElectricGasBoilerStrategy(config)

    with pytest.raises(ValueError, match="does not support aFRR capacity"):
        strategy.decide_afrr_capacity(_direct_plant(), _forecasts())


def test_strategy_registry_builds_direct_strategy(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    strategy = build_strategy(config.strategy_name, config)

    assert isinstance(strategy, HybridElectricGasBoilerStrategy)


def test_both_hybrid_strategies_default_afrr_margin_to_zero(tmp_path: Path) -> None:
    direct_config = _write_config(tmp_path)
    etes_config = CaseConfig.from_case_dir(ETES_CASE_DIR)

    assert HybridElectricGasBoilerStrategy(direct_config).afrr_energy_bid_margin_eur_per_mwh == 0.0
    assert HybridETESGasStrategy(etes_config).afrr_energy_bid_margin_eur_per_mwh == 0.0


@pytest.mark.parametrize("margin", ["-1", ".nan", ".inf", "not-a-number", "true"])
def test_config_rejects_invalid_afrr_energy_margin(
    tmp_path: Path,
    margin: str,
) -> None:
    with pytest.raises(ConfigError, match="finite non-negative"):
        _write_config(tmp_path, margin=margin)


def test_config_rejects_capacity_for_direct_strategy(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not support enabled aFRR capacity"):
        _write_config(tmp_path, capacity_enabled=True)


def test_config_requires_day_ahead_for_direct_strategy(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="requires enabled day_ahead"):
        _write_config(tmp_path, day_ahead_enabled=False)


def test_config_requires_intraday_before_direct_afrr(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="requires enabled intraday_continuous"):
        _write_config(tmp_path, intraday_enabled=False)


@pytest.mark.parametrize("direction", ["up", "positive"])
def test_config_rejects_afrr_up_for_direct_strategy(
    tmp_path: Path,
    direction: str,
) -> None:
    with pytest.raises(ConfigError, match="supports only.*direction"):
        _write_config(tmp_path, afrr_direction=direction)


@pytest.mark.parametrize(
    ("rule_name", "rule_value"),
    [
        ("min_bid_mw", "-1"),
        ("min_bid_mw", ".nan"),
        ("min_bid_mw", ".inf"),
        ("min_bid_mw", "not-a-number"),
        ("min_bid_mw", "true"),
        ("bid_increment_mw", "0"),
        ("bid_increment_mw", ".nan"),
        ("bid_increment_mw", ".inf"),
        ("bid_increment_mw", "not-a-number"),
        ("bid_increment_mw", "true"),
    ],
)
def test_config_rejects_invalid_afrr_bid_rules(
    tmp_path: Path,
    rule_name: str,
    rule_value: str,
) -> None:
    kwargs = {rule_name: rule_value}
    with pytest.raises(ConfigError, match=rule_name):
        _write_config(tmp_path, **kwargs)


@pytest.mark.parametrize(
    "market_sequence",
    [
        ("intraday_continuous", "day_ahead", "afrr_energy"),
        ("day_ahead", "afrr_energy", "intraday_continuous"),
    ],
)
def test_config_rejects_wrong_direct_market_order(
    tmp_path: Path,
    market_sequence: tuple[str, ...],
) -> None:
    with pytest.raises(ConfigError, match="requires enabled markets in this order"):
        _write_config(tmp_path, market_sequence=market_sequence)


def _write_config(
    tmp_path: Path,
    *,
    margin: float | str | None = None,
    capacity_enabled: bool = False,
    day_ahead_enabled: bool = True,
    intraday_enabled: bool = True,
    afrr_direction: str = "down",
    market_sequence: tuple[str, ...] = (
        "afrr_capacity",
        "day_ahead",
        "intraday_continuous",
        "afrr_energy",
    ),
    min_bid_mw: float | str = 1.0,
    bid_increment_mw: float | str = 1.0,
) -> CaseConfig:
    case_dir = tmp_path / f"direct_strategy_{len(list(tmp_path.iterdir()))}"
    case_dir.mkdir()
    margin_line = (
        f"    afrr_energy_bid_margin_eur_per_mwh: {margin}\n" if margin is not None else ""
    )
    market_sequence_lines = "\n".join(f"  - {market}" for market in market_sequence)
    (case_dir / "config.yaml").write_text(
        case_config_text(
            f"""
case:
  name: direct_strategy
  country: DE
  timestep_minutes: 15
  simulation_start: "2025-01-01 00:00"
  simulation_end: "2025-01-01 00:45"
strategy:
  name: hybrid_electric_gas_boiler
  dispatch:
    dispatch_method: pyomo
    rolling_horizon_enabled: false
{margin_line.rstrip()}
solver:
  name: highs
  fallback_solvers: []
  tee: false
market_sequence:
{market_sequence_lines}
markets:
  afrr_capacity:
    enabled: {str(capacity_enabled).lower()}
    direction: down
    product_length: "4h"
    price_unit: "EUR_per_MW_per_h"
    signals:
      price: aFRR_capacity_down_price
  day_ahead:
    enabled: {str(day_ahead_enabled).lower()}
    signals:
      price: DE_DA_price
  intraday_continuous:
    enabled: {str(intraday_enabled).lower()}
    allowed_actions:
      buy: true
      sell: true
    signals:
      price: DE_ID3_price
  afrr_energy:
    enabled: true
    direction: {afrr_direction}
    clearing_mechanism: pay_as_cleared
    product_resolution: "15min"
    product_rules:
      min_bid_mw: {min_bid_mw}
      bid_increment_mw: {bid_increment_mw}
      validity_period_minutes: 15
    signals:
      price: aFRR_energy_down_price
      system_activation: aFRR_energy_down_quantity
    interpretation:
      activation_unit: MW
""".strip()
        ),
        encoding="utf-8",
    )
    return CaseConfig.from_case_dir(case_dir)


def _direct_plant() -> SteamGenerationPlant:
    rows = pd.DataFrame(
        [
            {
                "name": "direct_1",
                "unit_type": "steam_plant",
                "technology": "boiler",
                "fuel_type": "natural_gas",
                "demand": "direct_1_heat_demand",
                "max_power": 4.0,
                "min_power": 0.0,
                "efficiency": 0.8,
                "gas_emissions_factor_kg_per_mwh": 200.0,
            },
            {
                "name": "direct_1",
                "unit_type": "steam_plant",
                "technology": "electric_boiler",
                "fuel_type": "electricity",
                "demand": "direct_1_heat_demand",
                "max_power": 4.0,
                "min_power": 0.0,
                "efficiency": 0.95,
                "gas_emissions_factor_kg_per_mwh": 200.0,
            },
        ]
    )
    return SteamGenerationPlant.from_rows("direct_1", rows)


def _forecasts(
    *,
    da_price: float = 200.0,
    idc_price: float = 200.0,
    afrr_price: float = 90.0,
    heat_demand_mw: float = 2.0,
) -> pd.DataFrame:
    index = pd.date_range("2025-01-01 00:00", periods=4, freq="15min")
    return pd.DataFrame(
        {
            "direct_1_heat_demand": heat_demand_mw,
            "DE_DA_price": da_price,
            "DE_ID3_price": idc_price,
            "aFRR_energy_down_price": afrr_price,
            "aFRR_energy_down_quantity": 1.0,
            "aFRR_capacity_down_price": 0.0,
            "natural_gas_price": 80.0,
            "co2_price": 100.0,
        },
        index=index,
    )
