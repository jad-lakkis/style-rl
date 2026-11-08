#!/usr/bin/env bash
# Trains the two opposite drivers (attacker, smooth) in parallel and streams both logs to the pod log.
# HOURS = time limit (default 9.5), RES = residual authority of the agent (default 0.35; 1.0 = full authority).
cd "$(dirname "$0")/.."
source "${OCD_DIR:-/workspace/Open-Car-Dynamics}/install/local_setup.sh"
HOURS="${HOURS:-9.5}"
RES="${RES:-0.35}"
mkdir -p /workspace/runs
PIDS=()
for d in attacker smooth; do
  python3 scripts/train.py --driver "$d" --backend ocd --out "/workspace/runs/$d" --max-hours "$HOURS" --threads 2 --res-steer "$RES" --res-pedal "$RES" > "/workspace/runs/$d.log" 2>&1 &
  PIDS+=($!)
  echo "started $d (pid $!)"
done
sleep 20
tail -n +1 -F /workspace/runs/attacker.log /workspace/runs/smooth.log &
TAIL=$!
wait "${PIDS[@]}"
sleep 5
kill "$TAIL" 2>/dev/null
echo "TRAINING FINISHED"
