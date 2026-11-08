#!/usr/bin/env bash
# Start command of the pod: gets the code, builds OCD, runs the smoke test, then the overnight training.
# RunPod restarts this command when it ends, so a marker file /workspace/DONE stops a finished run from starting again, and the pod then stops itself.
set -o pipefail   # no -u: the ROS/OCD setup scripts read unset variables
mkdir -p /workspace && cd /workspace

stop_pod() {
  echo "stopping this pod"
  if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then runpodctl stop pod "$RUNPOD_POD_ID" || true; fi
  sleep 600   # fallback: wait here (the pod can also be stopped from outside)
}

if [ -f /workspace/DONE ]; then echo "already finished earlier (see /workspace/runs)"; stop_pod; exit 0; fi
[ -d style-rl ] || git clone https://github.com/jad-lakkis/style-rl.git
cd style-rl && git pull --ff-only || true
echo "=== SETUP ($(date -u +%H:%M:%S) UTC)"
bash runpod/setup_pod.sh 2>&1 | tee /workspace/setup.log
if ! grep -q "setup done" /workspace/setup.log; then echo "SETUP FAILED: see /workspace/setup.log"; tail -40 /workspace/setup.log; sleep 60; exit 1; fi
source /workspace/Open-Car-Dynamics/install/local_setup.sh
echo "=== SMOKE TEST ($(date -u +%H:%M:%S) UTC)"
python3 scripts/smoke_test.py --backend ocd 2>&1 | tee /workspace/smoke.log
code=${PIPESTATUS[0]}
if [ "$code" = "1" ]; then echo "SMOKE TEST FAILED, not starting the long run"; sleep 60; exit 1; fi
if [ "$code" = "3" ]; then export RES=1.0; echo "base driver weak: residual authority 1.0"; fi
echo "=== TRAINING ($(date -u +%H:%M:%S) UTC)"
bash runpod/run_overnight.sh
touch /workspace/DONE
stop_pod
