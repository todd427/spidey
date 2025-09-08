#!/usr/bin/env bash
set -euo pipefail

# (Optional) activate your venv if you use one
# source "${HOME}/venvs/trainingEnv/bin/activate"

# Model config: flat layout, your namespace baked in
export TODDRIC_MODEL="${TODDRIC_MODEL:-./toddric-1_5b-merged-v1}"
export TODDRIC_MODEL_ID="${TODDRIC_MODEL_ID:-toddie314/toddric-1_5b-merged-v1}"
export TODDRIC_MODEL_REVISION="${TODDRIC_MODEL_REVISION:-main}"
export HF_TOKEN="${HF_TOKEN:-}"   # set if needed

# Ensure model exists (idempotent)
./scripts/get_model.sh

# Your existing runtime knobs (keep whatever you use)
export RL_MAX="${RL_MAX:-30}"
export RL_WINDOW="${RL_WINDOW:-300}"
export TODDRIC_DEVICE_MAP='{"":0}'
export TODDRIC_ATTN="${TODDRIC_ATTN:-eager}"
export TODDRIC_ALLOW_DOMAINS="${TODDRIC_ALLOW_DOMAINS:-youtube.com,youtu.be}"
export TODDRIC_SMS_MAXNEW="${TODDRIC_SMS_MAXNEW:-60}"
export TODDRIC_BEARER="${TODDRIC_BEARER:-7ff03c53-5529-43a2-b549-695957c8d161}"
export TOKEN_COOKIE_SECURE=1

# Start API (adjust module/app if different)
exec uvicorn app_toddric:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --timeout-keep-alive 30 \
  --loop uvloop \
  --http h11 \
  --log-level info \
  --access-log

