#!/usr/bin/env bash
# Start command of the pod: gets the code, builds OCD (once, it stays on the volume), measures the car, checks the base driver, then trains.
# - While the file runpod/HOLD exists in the repository the pod does not train: it waits for a new commit (git pull every 30 s) and checks again, so no pod restart is needed.
# - RunPod restarts this command when it ends, so a marker file /workspace/DONE stops a finished run from starting again; the pod then stops itself through the RunPod API.
set -o pipefail   # no -u: the ROS/OCD setup scripts read unset variables
mkdir -p /workspace && cd /workspace

stop_pod() {
  bash /workspace/style-rl/runpod/stop_pod.sh   # a script of the repository, so it can be fixed without restarting the pod
  sleep 600                                      # if the pod is still running after that, wait here (RunPod restarts this command when it ends)
}

if [ -f /workspace/DONE ]; then echo "already finished earlier (see /workspace/runs)"; stop_pod; exit 0; fi
[ -d style-rl ] || git clone https://github.com/jad-lakkis/style-rl.git
cd style-rl && git pull --ff-only || true
echo "=== POD: id ${RUNPOD_POD_ID:-unknown}; RunPod variables present: $(env | grep -o '^RUNPOD[A-Z_]*' | tr '\n' ' ')"
if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
  curl -s -o /dev/null -w "API check (read this pod): HTTP %{http_code}\n" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" || true
fi
echo "=== SETUP ($(date -u +%H:%M:%S) UTC)"
bash runpod/setup_pod.sh 2>&1 | tee /workspace/setup.log
if ! grep -q "setup done" /workspace/setup.log; then echo "SETUP FAILED: see /workspace/setup.log"; tail -40 /workspace/setup.log; sleep 60; exit 1; fi
source /workspace/Open-Car-Dynamics/install/local_setup.sh
QUIET='^(=== OCD|Parameter file|Mass parameter|Wheelbase|Lift coefficient|Drive calibration|Brake calibration|Maximum steering)'
echo "=== MEASURE THE CAR ($(date -u +%H:%M:%S) UTC)"
python3 scripts/measure_ocd_envelope.py --out /workspace/ocd_envelope.json 2>&1 | grep -v -E "$QUIET" | tee /workspace/envelope.log
last=""
while true; do
  git pull -q --ff-only >/dev/null 2>&1 || true
  last=$(git rev-parse --short HEAD)
  echo "=== DIAGNOSE, repository at $last ($(date -u +%H:%M:%S) UTC)"
  python3 scripts/diagnose_ocd.py 2>&1 | grep -v -E "$QUIET" | tee /workspace/diagnose.log
  [ -f runpod/HOLD ] || break
  echo "HOLD: runpod/HOLD exists in the repository, not training. Waiting for a new commit (checked every 30 s)."
  while [ "$(git rev-parse --short HEAD)" = "$last" ]; do sleep 30; git pull -q --ff-only >/dev/null 2>&1 || true; done
done
echo "=== SMOKE TEST ($(date -u +%H:%M:%S) UTC)"
python3 scripts/smoke_test.py --backend ocd 2>&1 | tee /workspace/smoke.log
code=${PIPESTATUS[0]}
if [ "$code" = "1" ]; then echo "SMOKE TEST FAILED, not starting the long run"; sleep 60; exit 1; fi
if [ "$code" = "3" ]; then export RES=1.0; echo "base driver weak: residual authority 1.0"; fi
echo "=== TRAINING ($(date -u +%H:%M:%S) UTC)"
bash runpod/run_overnight.sh
touch /workspace/DONE
stop_pod
