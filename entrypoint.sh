#!/usr/bin/env bash
set -euo pipefail

# Ensure HF cache exists and is writable by appuser, then drop privileges.
mkdir -p "${HF_HOME:-/cache/hf}"
chown -R 1000:1000 "${HF_HOME:-/cache/hf}" 2>/dev/null || true
find "${HF_HOME:-/cache/hf}" -name "*.lock" -delete 2>/dev/null || true

# Also make sure app-owned dirs are sane (optional but handy)
mkdir -p /app/models /app/store
chown -R 1000:1000 /app/models /app/store 2>/dev/null || true

# Drop to non-root and exec
exec gosu 1000:1000 "$@"

