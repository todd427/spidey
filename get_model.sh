#!/usr/bin/env bash
# get_model.sh — download a HF model snapshot with legacy-friendly `hf download`
# Usage: ./get_model.sh <model_id> <local_dir> [revision]
# Notes:
#   - If the repo is private/gated, set HF_TOKEN=hf_xxx first (env is enough).
#   - No `set -x` → your token won't get echoed.
#   - Verifies that config.json or *.safetensors exist before "success".

set -euo pipefail

MODEL_ID="${1:-}"
LOCAL_DIR="${2:-}"
REVISION="${3:-main}"

if [[ -z "$MODEL_ID" || -z "$LOCAL_DIR" ]]; then
  echo "Usage: $0 <model_id> <local_dir> [revision]" >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "[model] ERROR: 'hf' CLI not found. Activate your venv and: pip install 'huggingface_hub[cli]'" >&2
  exit 3
fi

# Optional: repo-local cache
: "${HF_HOME:=$PWD/.hf-cache}"
export HF_HOME

# If token provided, log in non-interactively (idempotent, quiet)
if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
fi

TMP_DIR="${LOCAL_DIR}.partial.$$"
rm -rf "$TMP_DIR" 2>/dev/null || true

echo "[model] repo:    $MODEL_ID"
echo "[model] dest:    $LOCAL_DIR"
echo "[model] rev:     $REVISION"
echo "[model] HF_HOME: $HF_HOME"
echo "[model] token:   $([[ -n ${HF_TOKEN:-} ]] && echo present || echo none)"

# Download (env HF_TOKEN is picked up automatically; no --token so we don't echo it)
# Older CLIs may not support --local-dir-use-symlinks; omit for compatibility.
if ! hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$TMP_DIR"; then
  echo "[model] ERROR: download failed for '$MODEL_ID - $REVISION'. Typo in repo name or no access?" >&2
  rm -rf "$TMP_DIR" || true
  exit 10
fi

# Verify snapshot has real payload
if [[ ! -f "$TMP_DIR/config.json" && -z "$(ls -1 "$TMP_DIR"/*.safetensors 2>/dev/null || true)" ]]; then
  echo "[model] ERROR: No config.json or *.safetensors found in $TMP_DIR (incomplete snapshot)" >&2
  ls -la "$TMP_DIR" || true
  rm -rf "$TMP_DIR" || true
  exit 11
fi

# Atomic move into place
rm -rf "$LOCAL_DIR" 2>/dev/null || true
mv "$TMP_DIR" "$LOCAL_DIR"

echo "[model] OK → $LOCAL_DIR contains:"
ls -1 "$LOCAL_DIR" | sed 's/^/  - /'

