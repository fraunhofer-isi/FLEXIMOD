#!/usr/bin/env python3
"""Configure every generated cement case for DA plus energy-only aFRR participation.

Each case keeps its aFRR capacity market definition and price signal for provenance,
but disables participation and removes it from the active market sequence.  The model
then submits only free aFRR-energy bids and records zero capacity reservation and
capacity revenue.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO_ROOT / "data" / "input" / "cement_inputs"
EXPECTED_SEQUENCE = ["afrr_capacity", "day_ahead", "afrr_energy"]
ENERGY_ONLY_SEQUENCE = ["day_ahead", "afrr_energy"]


def update_config(config_path: Path, *, check: bool) -> bool:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw.get("cases", {})) != 1:
        raise ValueError(f"{config_path}: expected exactly one case")
    case_name, case = next(iter(raw["cases"].items()))
    markets = case["markets"]
    sequence = case["market_sequence"]
    configured = (
        sequence == ENERGY_ONLY_SEQUENCE and markets["afrr_capacity"].get("enabled") is False
    )
    if check:
        if not configured:
            raise ValueError(f"{config_path}: {case_name} is not configured for energy-only aFRR")
        return False
    if sequence != EXPECTED_SEQUENCE:
        raise ValueError(f"{config_path}: unexpected market sequence {sequence!r}")
    if markets["afrr_capacity"].get("enabled") is not True:
        raise ValueError(f"{config_path}: aFRR capacity is not enabled before conversion")
    case["market_sequence"] = ENERGY_ONLY_SEQUENCE
    markets["afrr_capacity"]["enabled"] = False
    temporary_path = config_path.with_suffix(".yaml.tmp")
    temporary_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    temporary_path.replace(config_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without changing files.")
    args = parser.parse_args()
    configs = sorted(INPUT_ROOT.glob("*/config.yaml"))
    if len(configs) != 456:
        raise RuntimeError(f"Expected 456 generated cement configs, found {len(configs)}")
    changed = sum(update_config(path, check=args.check) for path in configs)
    action = "Validated" if args.check else "Updated"
    print(f"{action} {len(configs)} cement configurations ({changed} changed).")


if __name__ == "__main__":
    main()
