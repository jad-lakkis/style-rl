#!/usr/bin/env bash
# Start command of the pod: gets the code, builds OCD, runs the smoke test, then the overnight training. The pod stops by itself when this script ends.
set -uo pipefail
mkdir -p /workspace && cd /workspace
[ -d style-rl ] || git clone https://github.com/jad-lakkis/style-rl.git
cd style-rl && git pull --ff-only || true
echo "=== SETUP ($(date -u +%H:%M:%S) UTC)"
bash runpod/setup_pod.sh 2>&1 | tee /workspace/setup.log
if ! grep -q "setup done" /workspace/setup.log; then echo "SETUP FAILED: see /workspace/setup.log"; tail -40 /workspace/setup.log; exit 1; fi
source /workspace/Open-Car-Dynamics/install/local_setup.sh
echo "=== SMOKE TEST ($(date -u +%H:%M:%S) UTC)"
python3 scripts/smoke_test.py --backend ocd 2>&1 | tee /workspace/smoke.log
code=${PIPESTATUS[0]}
if [ "$code" = "1" ]; then echo "SMOKE TEST FAILED, not starting the long run"; exit 1; fi
if [ "$code" = "3" ]; then export RES=1.0; echo "base driver weak: residual authority 1.0"; fi
echo "=== TRAINING ($(date -u +%H:%M:%S) UTC)"
bash runpod/run_overnight.sh
