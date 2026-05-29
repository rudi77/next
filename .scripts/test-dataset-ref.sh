#!/usr/bin/env bash
set -euo pipefail
KEY="local-dev-secret"
BASE="http://127.0.0.1:8080"
HDR="X-API-Key: $KEY"

echo "--- upload mini.jsonl ---"
RESP=$(curl -s -H "$HDR" -F "file=@/tmp/mini.jsonl" -F "name=mini-chat-v2" "$BASE/datasets")
echo "$RESP" | python3 -m json.tool
DS_ID=$(printf '%s' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "ds_id=$DS_ID"

cat > /tmp/spec_ref.json <<JSON
{
  "name": "ds-ref-clean",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "sft_type": "lora",
  "dataset": ["ds:__DS__"],
  "hyperparameters": {"num_train_epochs": 1, "per_device_train_batch_size": 1, "logging_steps": 1, "max_length": 256}
}
JSON
sed -i "s|__DS__|$DS_ID|" /tmp/spec_ref.json

echo "--- spec ---"
cat /tmp/spec_ref.json
echo
echo "--- submit ---"
SUBMIT=$(curl -s -H "$HDR" -H "Content-Type: application/json" -X POST "$BASE/experiments" -d @/tmp/spec_ref.json)
echo "$SUBMIT" | python3 -m json.tool
EXP_ID=$(printf '%s' "$SUBMIT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["experiment_id"])')

echo
echo "--- stored spec.dataset (resolved path expected) ---"
curl -s -H "$HDR" "$BASE/experiments/$EXP_ID" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("dataset:", d["spec"]["dataset"]); print("status:", d["status"])'

echo
echo "--- malformed ds: returns 422 ---"
curl -s -H "$HDR" -H "Content-Type: application/json" -X POST "$BASE/experiments" \
  -d '{"model":"m","dataset":["ds:not-hex!"]}' \
  | python3 -m json.tool
