#!/usr/bin/env bash
# Stops the pod this script runs in (so that a finished run stops billing). Uses the RunPod REST API with the key RunPod puts in the container; falls back to runpodctl.
# Docs: https://docs.runpod.io/pods/manage-pods ("Stop a Pod")
echo "stopping this pod ($(date -u +%H:%M:%S) UTC)"
if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "no RUNPOD_POD_ID / RUNPOD_API_KEY in this container: the pod has to be stopped from outside"; exit 1
fi
code=$(curl -s -o /tmp/stop_reply.txt -w "%{http_code}" -X POST "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" --data '{"action":"stop"}')
echo "stop through the REST API: HTTP $code"
if [ "$code" != "200" ] && [ "$code" != "202" ] && [ "$code" != "204" ]; then
  head -c 300 /tmp/stop_reply.txt; echo
  curl -fsSL -o /usr/local/bin/runpodctl "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64" && chmod +x /usr/local/bin/runpodctl
  runpodctl pod stop "$RUNPOD_POD_ID" && echo "stopped through runpodctl" || echo "runpodctl could not stop the pod"
fi
