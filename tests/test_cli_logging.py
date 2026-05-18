# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

from flexi_mod.simulation import run_case
from flexi_mod.simulation.cli_logging import CliLogger, output_summary, print_verbose_outputs


def test_cli_warning_capture_prints_friendly_notice_without_trace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = CliLogger()

    with logger.capture_warnings():
        warnings.warn(
            "No IDC sell/reduction volumes found. Skipping IDC sell source and compensation plot.",
            stacklevel=1,
        )

    output = capsys.readouterr().out
    assert "Notice: IDC sell source plot skipped" in output
    assert "plot_idc_sell_source_and_compensation" not in output


def test_output_summary_is_compact_and_verbose_lists_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    outputs = {
        "dispatch_results": Path("dispatch_results.csv"),
        "market_ledger": Path("market_ledger.csv"),
        "plots": [Path("plot_1.png"), Path("plot_2.png")],
    }

    assert output_summary(outputs) == "2 table file(s), 2 plot file(s)"

    print_verbose_outputs(CliLogger(verbose=False), outputs)
    assert capsys.readouterr().out == ""

    print_verbose_outputs(CliLogger(verbose=True), outputs)
    verbose_output = capsys.readouterr().out
    assert "dispatch_results.csv" in verbose_output
    assert "plot_1.png" in verbose_output


def test_run_case_concise_output_hides_individual_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_dir = _write_cli_case(tmp_path, additional_charges=False)
    _patch_runner(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_case.py", "--case", str(case_dir)])

    run_case.main()

    output = capsys.readouterr().out
    assert "Case started: cli_case" in output
    assert "Additional charges: disabled; market prices are used directly." in output
    assert "Simulating 2025-01-01 for plant_1 (1/1 windows, 0 remaining)" in output
    assert "Day-ahead stage solved for plant_1" not in output
    assert "Case completed: 1 table file(s), 1 plot file(s) saved." in output
    assert "Created outputs:" not in output
    assert "dispatch_results.csv" not in output


def test_run_case_verbose_output_lists_created_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_dir = _write_cli_case(tmp_path, additional_charges=True)
    _patch_runner(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_case.py", "--case", str(case_dir), "--verbose"])

    run_case.main()

    output = capsys.readouterr().out
    assert "Additional charges: enabled; plant_1 = 12.10 EUR/MWh_el." in output
    assert "Day-ahead stage solved for plant_1" in output
    assert "dispatch_results.csv" in output
    assert "plot_1.png" in output


def test_run_case_missing_additional_charges_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_dir = _write_cli_case(tmp_path, additional_charges=True, write_charges=False)
    _patch_runner(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_case.py", "--case", str(case_dir)])

    with pytest.raises(SystemExit):
        run_case.main()

    output = capsys.readouterr().out
    assert "Additional charges are enabled" in output
    assert "Add the file or set case.additional_charges: false" in output


def _patch_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    import flexi_mod.simulation.simulation_runner as runner_module

    class FakeRunner:
        def __init__(self, *args: Any, progress_callback=None, **kwargs: Any) -> None:
            self.progress_callback = progress_callback

        def run(self) -> dict[str, Path | list[Path]]:
            if self.progress_callback is not None:
                self.progress_callback(
                    "Simulating 2025-01-01 for plant_1 (1/1 windows, 0 remaining)"
                )
                self.progress_callback("Day-ahead stage solved for plant_1")
                self.progress_callback("Outputs saved")
            return {
                "dispatch_results": Path("dispatch_results.csv"),
                "plots": [Path("plot_1.png")],
            }

    monkeypatch.setattr(runner_module, "SimulationRunner", FakeRunner)


def _write_cli_case(
    tmp_path: Path,
    additional_charges: bool,
    write_charges: bool = True,
) -> Path:
    case_dir = tmp_path / "cli_case"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        f"""
case:
  name: cli_case
  country: DE
  timestep_minutes: 15
  simulation_start: "2025-01-01 00:00"
  simulation_end: "2025-01-01 00:45"
  additional_charges: {str(additional_charges).lower()}
strategy:
  name: hybrid_etes_gas
  dispatch:
    dispatch_method: pyomo
solver:
  name: highs
  fallback_solvers: []
  tee: false
market_sequence:
  - day_ahead
markets:
  day_ahead:
    enabled: true
    signals:
      price: DE_DA_price
""".strip(),
        encoding="utf-8",
    )
    (case_dir / "plants.csv").write_text(
        "\n".join(
            [
                "name,unit_type,technology,demand,fuel_type,max_power,efficiency",
                "plant_1,steam_plant,thermal_storage,plant_1_heat_demand,,,",
                "plant_1,steam_plant,boiler,,natural_gas,5,0.9",
            ]
        ),
        encoding="utf-8",
    )
    if additional_charges and write_charges:
        (case_dir / "additional_charges.csv").write_text(
            "\n".join(
                [
                    "component,unit,plant_1",
                    "Network consumption price,EUR/MWh,6.9",
                    "Metering and operation,EUR/MWh,2.0",
                    "Concession fees,EUR/MWh,1.1",
                    "Surcharges and levies,EUR/MWh,1.6",
                    "Electricity tax,EUR/MWh,0.5",
                ]
            ),
            encoding="utf-8",
        )
    return case_dir
