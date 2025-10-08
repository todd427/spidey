#!/usr/bin/env bash
# run_toddric.sh — quiet, idempotent runner with pip sentinel "pipped"
set -euo pipefail

# -------- Defaults (override via env or .env) --------
export TORCH_DEVICE=cuda
export DEVICE_MAP=auto
export TORCH_DTYPE=bfloat16   # or: float16
# env or CLI flags for your server/eval
export MAX_NEW_TOKENS=64
export TEMPERATURE=0.0
export TOP_P=0.9
export TOP_K=40
export DO_SAMPLE=0        # use greedy for crisp answers
export REPETITION_PENALTY=1.15


APP_MODULE="${APP_MODULE:-app_toddric:app}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
REQS_FILE="${REQS_FILE:-requirements.txt}"
SENTINEL="${PIP_SENTINEL:-pipped}"        # sentinel file to mark deps installed
PIP_LOG="${PIP_LOG:-pip_install.log}"     # pip output goes here (quiet by default)
FORCE_PIP="${FORCE_PIP:-0}"               # set to 1 to force reinstall deps

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

MODEL_ID="${MODEL_ID:-toddie314/toddric-1_5b-merged-v1}"
MODEL_DIR="${MODEL_DIR:-./models/toddric-1_5b-merged-v1}"
MODEL_REV="${MODEL_REV:-main}"
NO_DOWNLOAD="${NO_DOWNLOAD:-0}"           # 1 = skip downloading

# -------- enter repo root & load .env if present --------
cd "$(dirname "$0")"
if [[ -f .env ]]; then set -a; source ./.env; set +a; fi

# -------- venv --------
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# -------- compute deps signature (invalidate when reqs/python change) --------
py_sig="$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if [[ -f "$REQS_FILE" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    reqs_sig="$(sha256sum "$REQS_FILE" | awk '{print $1}')"
  else
    reqs_sig="$(python - <<'PY'
import hashlib,sys
with open(sys.argv[1],'rb') as f:
    print(hashlib.sha256(f.read()).hexdigest())
PY
"$REQS_FILE")"
  fi
else
  reqs_sig="no-reqs"
fi
sentinel_contents="py=$py_sig reqs=$reqs_sig"

# -------- quiet deps install (once) --------
need_pip=1
if [[ -f "$SENTINEL" ]]; then
  if [[ "$(cat "$SENTINEL" 2>/dev/null || true)" == "$sentinel_contents" ]] && [[ "$FORCE_PIP" != "1" ]]; then
    need_pip=0
  fi
fi

if [[ "$need_pip" == "1" ]]; then
  echo "[deps] Installing Python deps… (logging to $PIP_LOG)"
  : > "$PIP_LOG"
  # Upgrade core tooling quietly
  python -m pip install -q -U pip wheel >>"$PIP_LOG" 2>&1

  if [[ -f "$REQS_FILE" ]]; then
    python -m pip install -q -r "$REQS_FILE" >>"$PIP_LOG" 2>&1
  else
    python -m pip install -q fastapi "uvicorn[standard]" transformers accelerate safetensors huggingface_hub >>"$PIP_LOG" 2>&1
  fi

  # Ensure the hf CLI exists
  if ! command -v hf >/dev/null 2>&1; then
    python -m pip install -q "huggingface_hub[cli]" >>"$PIP_LOG" 2>&1
  fi

  echo "$sentinel_contents" > "$SENTINEL"
  echo "[deps] Done."
else
  echo "[deps] Skipping (found $SENTINEL). To force, set FORCE_PIP=1 or delete $SENTINEL."
fi

# -------- model --------
if [[ "$NO_DOWNLOAD" != "1" ]]; then
  if [[ -x ./get_model.sh ]]; then
    ./get_model.sh "$MODEL_ID" "$MODEL_DIR" "$MODEL_REV"
  else
    echo "[model] get_model.sh not found or not executable; skipping download."
  fi
else
  echo "[model] NO_DOWNLOAD=1; using existing: $MODEL_DIR"
fi

# -------- environment for app --------
export MODEL_ID MODEL_DIR
export SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-prompts/system_toddric.md}"
export TORCH_DEVICE="${TORCH_DEVICE:-auto}"
# dev convenience
export ALLOW_NO_AUTH="${ALLOW_NO_AUTH:-1}"

echo "[run] ${APP_MODULE} on http://${HOST}:${PORT}  (model: ${MODEL_DIR:-$MODEL_ID})"
exec uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" --reload

