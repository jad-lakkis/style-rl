#!/usr/bin/env bash
# Keeps ONE training run going on a pod: runs scripts/train12.py in parts of at most 2 hours (every part resumes from latest.zip and the saved replay buffer)
# and starts it again after a crash, until the time written in END_FILE has passed. END_FILE holds one number, a unix time (seconds), and is read before every part,
# so the run can be lengthened or shortened by changing that number; nothing has to be restarted.
#   bash runpod/run_loop.sh /workspace/end_epoch.txt /workspace/runs12/x/smooth.log --driver smooth --backend ocd --out /workspace/runs12/x/smooth --threads 2 --train-freq 1 ...
# The arguments after the log file go to scripts/train12.py as they are.
END_FILE=$1
LOG=$2
shift 2
cd "$(dirname "$0")/.."
source "${OCD_DIR:-/workspace/Open-Car-Dynamics}/install/local_setup.sh"
fails=0
while [ "$fails" -lt 8 ]; do
  rem=$(python3 -c "import time; print(round((float(open('$END_FILE').read()) - time.time()) / 3600.0, 3))")
  python3 -c "import sys; sys.exit(0 if $rem > 0.1 else 1)" || break
  part=$(python3 -c "print(min($rem, 2.0))")
  echo "=== start $(date -u +%H:%M:%S) UTC, $rem h to the end, this part $part h" >> "$LOG"
  t0=$(date +%s)
  if python3 scripts/train12.py "$@" --max-hours "$part" >> "$LOG" 2>&1 && [ $(( $(date +%s) - t0 )) -gt 120 ]; then
    fails=0
  else
    fails=$((fails + 1))
    echo "=== stopped early or crashed, attempt $fails of 8, again in 30 s" >> "$LOG"
    sleep 30
  fi
done
echo "=== loop ended $(date -u +%H:%M:%S) UTC" >> "$LOG"
