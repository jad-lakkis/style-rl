#!/usr/bin/env bash
# Trains four agents in parallel and streams their logs to the pod log:
#   attacker and smooth (the two opposite driver models), each in
#     residual mode (a base driver drives, the agent adds a correction) and
#     pure mode     (the agent's three numbers ARE steering, throttle, brake; no help).
# HOURS = time limit (default: runpod/HOURS), RES = residual authority (default 0.35; 1.0 = full), MODES = which modes (default "residual pure").
cd "$(dirname "$0")/.."
source "${OCD_DIR:-/workspace/Open-Car-Dynamics}/install/local_setup.sh"
pip3 install -q -r requirements.txt 2>&1 | tail -1   # picks up packages added after the pod was set up (tensorboard)
HOURS="${HOURS:-$(cat runpod/HOURS 2>/dev/null || echo 6.5)}"   # training length; runpod/HOURS is set so that the pod stays inside the budget
RES="${RES:-0.35}"
TAG="$(cat runpod/RUN_TAG 2>/dev/null || echo v1)"   # bump the tag whenever the base driver or the environment changes, so old checkpoints are not resumed
RUNS="/workspace/runs/$TAG"
mkdir -p "$RUNS"
# watchdog: whatever happens, the pod is stopped 45 minutes after the planned end (the budget), and the DONE marker prevents a restart from training again
( sleep "$(python3 -c "print(int(($HOURS + 0.75) * 3600))")"; echo "WATCHDOG: time is up, stopping the pod"; touch /workspace/DONE; bash runpod/stop_pod.sh ) &
PIDS=()
LOGS=()
BC="--warm-start --learning-starts 0 --ent-init 0.05 --bc-weight ${BC_WEIGHT:-5} --bc-decay ${BC_DECAY:-400000}"
# agents: pure_bc2 = the agent's three numbers are the commands, started as a copy of the base driver and pulled towards it while the pull fades (anchor on the mean action, capped entropy coefficient);
#         pure_bc3 = the same with 5x the reward for progress, so that lap time matters against the style/smoothness terms;
#         residual = base driver + correction (stopped at 23:40 UTC: after 380k steps it was 1-3 s slower than the base driver with a much worse style error; its folders stay)
SPECS="${SPECS:-pure_bc2 pure_bc3}"
for d in attacker smooth; do
  for spec in $SPECS; do
    case "$spec" in
      residual)  mode=residual; extra="" ;;
      pure_bc2)  mode=pure; extra="$BC" ;;
      pure_bc3)  mode=pure; extra="$BC --prog-weight 0.1" ;;
      *) echo "unknown agent $spec"; continue ;;
    esac
    name="${d}_${spec}"
    python3 scripts/train.py --driver "$d" --mode "$mode" --backend ocd --out "$RUNS/$name" --max-hours "$HOURS" --threads 2 --res-steer "$RES" --res-pedal "$RES" $extra > "$RUNS/$name.log" 2>&1 &
    PIDS+=($!)
    LOGS+=("$RUNS/$name.log")
    echo "started $name (pid $!)"
  done
done
sleep 20
tail -n +1 -F "${LOGS[@]}" &
TAIL=$!
( while true; do sleep 1800; echo "=== SUMMARY $(date -u +%H:%M) UTC"; python3 scripts/compare_runs.py --runs "$RUNS" --only "pure_bc[23]" 2>&1 | tail -70; done ) &
SUMM=$!
wait "${PIDS[@]}"
sleep 5
kill "$TAIL" "$SUMM" 2>/dev/null
python3 scripts/compare_runs.py --runs "$RUNS" 2>&1 | tee "$RUNS/summary.txt"
touch /workspace/DONE
bash runpod/stop_pod.sh
echo "TRAINING FINISHED"
