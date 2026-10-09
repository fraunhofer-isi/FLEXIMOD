#!/usr/bin/env bash
# Start the cement batch only after the queued steel rerun succeeds and its
# compressed result catalogue is complete.  Intended to run in its own tmux
# session; all progress is logged to data/output/cement_rerun_24workers.log.

set -euo pipefail

cd /root/FLEXIMOD

steel_log=data/output/steel_rerun_24workers.log
cement_log=data/output/cement_rerun_24workers.log

echo "[$(date -Is)] Waiting for the steel batch to finish successfully." | tee -a "$cement_log"
while ! rg -q '^All 432 steel simulations completed in ' "$steel_log"; do
    if rg -q 'FAILED|Completed with [1-9][0-9]* failed case\(s\)' "$steel_log"; then
        echo "[$(date -Is)] Steel batch failed; cement batch will not start." | tee -a "$cement_log"
        exit 1
    fi
    sleep 60
done

steel_case_count=$(find data/output/steel_results -mindepth 1 -maxdepth 1 -type d | wc -l)
steel_artifact_count=$(find data/output/steel_results -type f -name '*.csv.zst' | wc -l)
if [[ "$steel_case_count" -ne 432 || "$steel_artifact_count" -ne 2592 ]]; then
    echo "[$(date -Is)] Steel verification failed: $steel_case_count result folders, $steel_artifact_count artifacts." | tee -a "$cement_log"
    exit 1
fi

echo "[$(date -Is)] Steel verification passed; starting 456 cement cases with 24 workers." | tee -a "$cement_log"
MPLCONFIGDIR=/tmp/fleximod-matplotlib .venv/bin/python -u scripts/run_cement_cases_to_results.py --workers 24 >> "$cement_log" 2>&1
