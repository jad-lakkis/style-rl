#!/usr/bin/env bash
# Trains four agents in parallel and streams their logs to the pod log:
#   attacker and smooth (the two opposite driver models), each in
#     residual mode (a base driver drives, the agent adds a correction) and
#     pure mode     (the agent's three numbers ARE steering, throttle, brake; no help).
# HOURS = time limit (default 9.5), RES = residual authority (default 0.35; 1.0 = full), MODES = which modes (default "residual pure").
cd "$(dirname "$0")/.."
source "${OCD_DIR:-/workspace/Open-Car-Dynamics}/install/local_setup.sh"
HOURS="${HOURS:-9.5}"
RES="${RES:-0.35}"
MODES="${MODES:-residual pure}"
TAG="$(cat runpod/RUN_TAG 2>/dev/null || echo v1)"   # bump the tag whenever the base driver or the environment changes, so old checkpoints are not resumed
RUNS="/workspace/runs/$TAG"
mkdir -p "$RUNS"
PIDS=()
LOGS=()
for d in attacker smooth; do
  for m in $MODES; do
    name="${d}_${m}"
    python3 scripts/train.py --driver "$d" --mode "$m" --backend ocd --out "$RUNS/$name" --max-hours "$HOURS" --threads 2 --res-steer "$RES" --res-pedal "$RES" > "$RUNS/$name.log" 2>&1 &
    PIDS+=($!)
    LOGS+=("$RUNS/$name.log")
    echo "started $name (pid $!)"
  done
done
sleep 20
tail -n +1 -F "${LOGS[@]}" &
TAIL=$!
wait "${PIDS[@]}"
sleep 5
kill "$TAIL" 2>/dev/null
echo "TRAINING FINISHED"
