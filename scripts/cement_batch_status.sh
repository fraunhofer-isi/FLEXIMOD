#!/usr/bin/env bash
# Live progress snapshot for the cement_run batch, sampled directly from each
# worker's in-memory state via py-spy (works regardless of stdout buffering).
set -uo pipefail

cd /root/FLEXIMOD

echo "=== $(date) ==="
echo

echo "--- authoritative parent state ---"
PARENT_PID=$(pgrep -f "python3.13 .venv/bin/fleximod-run-case" | head -1)
if [ -n "${PARENT_PID:-}" ]; then
    .venv/bin/py-spy dump --pid "$PARENT_PID" --locals 2>&1 | grep -A2 "completed:\|failures:"
else
    echo "parent process not found"
fi
echo

# NOTE: window_number sampled via py-spy --locals was tried here and dropped.
# On this Python 3.13 build it reads wildly inconsistent values for the same
# worker between consecutive samples (physically impossible rates), so it is
# not trustworthy progress signal -- only alive/dead + CPU%, and completions
# on disk, are reported below.
echo "--- worker liveness (CPU%, confirms real work vs hung) ---"
printf "%-10s %-45s %-8s\n" "PID" "CASE" "CPU%"
for pid in $(pgrep -f "multiprocessing.spawn"); do
    case_name=$(grep -oP "(?<=pid $pid\): ).*" data/output/cement_run.log 2>/dev/null | tail -1)
    cpu=$(ps -p "$pid" -o pcpu= 2>/dev/null | tr -d ' ')
    printf "%-10s %-45s %-8s\n" "$pid" "${case_name:-?}" "${cpu:-?}"
done

echo
echo "--- real completions on disk (folders newer than the batch's log file) ---"
if [ -f data/output/cement_run.log ]; then
    found=$(find data/output -maxdepth 1 -type d -newer data/output/cement_run.log 2>/dev/null | grep -v "^data/output$")
    if [ -n "$found" ]; then
        echo "$found"
    else
        echo "(none yet)"
    fi
else
    echo "cement_run.log not found"
fi
