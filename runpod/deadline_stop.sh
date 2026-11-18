#!/usr/bin/env bash
# usage: runpod/deadline_stop.sh OUT DEADLINE_EPOCH
# Independent timer. At the deadline it asks the run to stop cleanly (the file OUT/STOP), waits up to 5 minutes, then makes sure nothing is left running, packs the results and stops the pod.
# It does not depend on run_drive.sh, so it also covers a run that hangs.
OUT=$1; DEAD=$2
while [ "$(date +%s)" -lt "$DEAD" ]; do sleep 30; done
echo "=== deadline reached $(date -u +%T) UTC: asking the run to stop" >> "$OUT.log"
touch "$OUT/STOP"
for i in $(seq 1 20); do pgrep -f "scripts/drive_train.py" > /dev/null || break; sleep 15; done
pkill -f run_drive.sh; pkill -TERM -f "scripts/drive_train.py"; sleep 45; pkill -9 -f "scripts/drive_train.py"
echo "=== deadline stop: packing the results and stopping the pod $(date -u +%T) UTC" >> "$OUT.log"
bash "${REPO:-/workspace/style-rl-drive}/runpod/finish_drive.sh" "$OUT" >> "$OUT.log" 2>&1
