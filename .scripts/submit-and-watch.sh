#!/usr/bin/env bash
# Submit /tmp/spec.json, poll status until terminal or timeout, print events.
set -euo pipefail
KEY="${TRAINPIPE_API_KEY:-local-dev-secret}"
BASE="${TRAINPIPE_BASE:-http://127.0.0.1:8080}"
HDR="X-API-Key: $KEY"
SPEC="${1:-/tmp/spec.json}"

resp=$(curl -s -H "$HDR" -H "Content-Type: application/json" -X POST "$BASE/experiments" -d @"$SPEC")
echo "submit: $resp"
id=$(printf '%s' "$resp" | python3 -c 'import json,sys; print(json.load(sys.stdin)["experiment_id"])')
echo "id=$id"

for i in $(seq 1 30); do
  detail=$(curl -s -H "$HDR" "$BASE/experiments/$id")
  status=$(printf '%s' "$detail" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  printf '[%2ds] status=%s\n' "$((i*2))" "$status"
  case "$status" in
    completed|failed|cancelled) break ;;
  esac
  sleep 2
done

echo "--- final detail ---"
printf '%s' "$detail" | python3 -m json.tool
echo "--- gpus ---"
curl -s -H "$HDR" "$BASE/gpus" | python3 -m json.tool
echo "--- log tail ---"
log=$(printf '%s' "$detail" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("log_path") or "")')
if [ -n "$log" ] && [ -f "$log" ]; then
  echo "log: $log"
  tail -30 "$log"
else
  echo "(no log file)"
fi
