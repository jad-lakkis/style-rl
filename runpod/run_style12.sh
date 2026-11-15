#!/usr/bin/env bash
# Whole-lap 12-number style training: two agents in parallel on one CPU pod (attacker and smooth), logs and results on the volume.
#   HOURS  training time limit per agent (default 3)      TAG  run folder name (default s12a)      EXTRA  more arguments for scripts/train12.py
# Start it in the background:  nohup bash runpod/run_style12.sh > /workspace/run_style12.log 2>&1 &
# When both agents are done the pod is stopped (so that it stops billing); /workspace/watchdog.py does the same after a fixed time.
cd "$(dirname "$0")/.."
source "${OCD_DIR:-/workspace/Open-Car-Dynamics}/install/local_setup.sh"
pip3 install -q -r requirements.txt 2>&1 | tail -1
HOURS="${HOURS:-3}"
TAG="${TAG:-s12a}"
RUNS="/workspace/runs12/$TAG"
mkdir -p "$RUNS"
PIDS=()
for d in attacker smooth; do
  python3 scripts/train12.py --driver "$d" --backend ocd --out "$RUNS/$d" --max-hours "$HOURS" --threads 2 --train-freq 1 ${EXTRA:-} > "$RUNS/$d.log" 2>&1 &
  PIDS+=($!)
  echo "started $d (pid $!), logs in $RUNS/$d.log"
done
wait "${PIDS[@]}"
echo "both agents finished ($(date -u +%H:%M:%S) UTC)"
python3 /workspace/watchdog.py
