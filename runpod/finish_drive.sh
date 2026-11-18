#!/usr/bin/env bash
# usage: runpod/finish_drive.sh OUT
# Packs the small result files of a run (not the replay buffer) into OUT_results.tgz, then stops this pod so that it stops billing (NO_STOP=1: only pack).
OUT=$1
B=$(basename "$OUT"); cd "$(dirname "$OUT")" || exit 1
tar czf "${B}_results.tgz" --exclude=replay_buffer.pkl --exclude='*.tmp.zip' "$B" "$B.log" 2>/dev/null
ls -la "${B}_results.tgz"
[ -n "$NO_STOP" ] && exit 0
bash "${REPO:-/workspace/style-rl-drive}/runpod/stop_pod.sh"
