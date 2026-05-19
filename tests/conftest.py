# SPDX-FileCopyrightText: FLEXIMOD Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def case_config_text(old_style_yaml: str, case_name: str | None = None) -> str:
    """Convert compact old-style fixture YAML into the required cases: shape."""

    raw = yaml.safe_load(old_style_yaml) or {}
    case = dict(raw.pop("case"))
    selected = case_name or str(case["name"])
    case["name"] = selected
    for section in ["strategy", "solver", "market_sequence", "markets"]:
        if section in raw:
            case[section] = raw[section]
    return yaml.safe_dump({"cases": {selected: case}}, sort_keys=False)
