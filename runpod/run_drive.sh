#!/usr/bin/env bash
# usage: runpod/run_drive.sh OUT [scripts/drive_train.py arguments...]
# Trains the driving agent into OUT and starts it again from its last checkpoint if it crashes (at most 8 times), until it ends by itself (exit 0). Log: OUT.log
OUT=$1; shift
cd /workspace/style-rl
source /opt/ros/humble/setup.bash 2>/dev/null
source /workspace/Open-Car-Dynamics/install/local_setup.sh
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$OUT")"
tries=0
while [ $tries -lt 8 ]; do
  echo "=== start $(date -u +%H:%M:%S) UTC (try $tries)" >> "$OUT.log"
  python3 scripts/drive_train.py --out "$OUT" "$@" --resume >> "$OUT.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] && break
  tries=$((tries+1)); echo "=== crashed (exit $rc), restart $tries in 30 s" >> "$OUT.log"; sleep 30
done
echo "=== loop ended $(date -u +%H:%M:%S) UTC" >> "$OUT.log"
