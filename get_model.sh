#!/usr/bin/env bash
set -euo pipefail

# Where the model should live on disk (flat in repo root)
MODEL_DIR="${TODDRIC_MODEL:-./toddric-1_5b-merged-v1}"

# Where to fetch it from (Hugging Face)
MODEL_ID="${TODDRIC_MODEL_ID:-toddie314/toddric-1_5b-merged-v1}"
REVISION="${TODDRIC_MODEL_REVISION:-main}"

# Optional token (needed if private / to avoid rate limits)
HF_TOKEN="${HF_TOKEN:-}"

# If present and non-empty, do nothing
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null || true)" ]; then
  echo "[get_model] Using existing model at: $MODEL_DIR"
  exit 0
fi

echo "[get_model] Need model at $MODEL_DIR (id: $MODEL_ID, rev: $REVISION)"

# Ensure the HF 'hf' CLI is available
if ! command -v hf >/dev/null 2>&1; then
  echo "[get_model] Installing hf CLI..."
  python3 -m pip -q install -U "huggingface_hub[cli]"
fi

# Non-interactive auth if token provided
if [ -n "$HF_TOKEN" ]; then
  echo "[get_model] Logging into HF non-interactively"
  hf auth login --token "$HF_TOKEN" >/dev/null
fi

mkdir -p "$MODEL_DIR"
echo "[get_model] Downloading snapshot to $MODEL_DIR ..."
# --local-dir-use-symlinks False => copy real files (avoids symlink weirdness)
hf download "$MODEL_ID" \
  --revision "$REVISION" \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False

echo "[get_model] Ready → $MODEL_DIR"

