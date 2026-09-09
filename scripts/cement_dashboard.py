#!/usr/bin/env python3
"""Live, in-place dashboard for a cement_run batch.

Polls the small per-worker status files written by run_case.py's worker
processes (data/output/.worker_status/<pid>.status) and redraws a fixed table
once a second -- no scrolling, only the numbers change, like `top`.

Usage: .venv/bin/python scripts/cement_dashboard.py
"""

from __future__ import annotations

import time
from pathlib import Path

STATUS_DIR = Path(__file__).resolve().parents[1] / "data" / "output" / ".worker_status"
REFRESH_SECONDS = 1.0


def _read_rows() -> list[tuple[str, str, str, int, int, str, str]]:
    rows = []
    if not STATUS_DIR.exists():
        return rows
    for path in sorted(STATUS_DIR.glob("*.status")):
        try:
            case, plant, current, total, date_label, updated = (
                path.read_text().strip().split("\t")
            )
            current_i = int(current)
            total_i = int(total) if total.isdigit() else 0
        except (ValueError, OSError):
            continue
        rows.append((path.stem, case, plant, current_i, total_i, date_label, updated))
    rows.sort(key=lambda r: -(r[3] / r[4]) if r[4] else 0)
    return rows


def _render(rows: list[tuple[str, str, str, int, int, str, str]]) -> str:
    lines = [
        f"cement_run batch — live status ({time.strftime('%H:%M:%S')}, "
        f"{len(rows)} active worker(s))",
        f"{'PID':<9} {'CASE':<34} {'PLANT':<24} {'PROGRESS':<20} {'PCT':<7} "
        f"{'DATE':<12} {'UPDATED':<9}",
        "-" * 118,
    ]
    for pid, case, plant, current, total, date_label, updated in rows:
        progress_str = f"{current}/{total}" if total else f"{current}/?"
        pct_str = f"{current / total * 100:5.1f}%" if total else "  ?  "
        lines.append(
            f"{pid:<9} {case:<34} {plant:<24} {progress_str:<20} {pct_str:<7} "
            f"{date_label:<12} {updated:<9}"
        )
    return "\n".join(lines)


def main() -> None:
    try:
        while True:
            rows = _read_rows()
            # Cursor home + clear-to-end, then redraw: avoids flicker vs a full
            # clear, and keeps the table pinned at the top of the pane.
            print("\033[H\033[J" + _render(rows), flush=True)
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
