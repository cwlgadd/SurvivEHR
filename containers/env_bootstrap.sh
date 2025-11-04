#!/usr/bin/env bash
# containers/env_bootstrap.sh
# Create/update a venv on /rds using the CONTAINER's Python 3.10.
# No activation: call $VENV_PATH/bin/{python,pip} directly. Non-interactive.

set -euo pipefail
cd "$(dirname "$0")/.."

# Centralised paths (IMG_PATH, VENV_PATH, UV_CACHE_DIR, TMPDIR, etc.)
source containers/env.paths.sh

# Ensure host-side dirs exist
mkdir -p "$IMG_DIR" "$VENV_DIR" "$UV_CACHE_DIR" "$TMPDIR"

if [[ ! -f "$IMG_PATH" ]]; then
  echo "[env_bootstrap] Container image not found at: $IMG_PATH"
  echo "Run: bash containers/container_build.sh"
  exit 1
fi

# All steps occur inside the container; no activation on host
apptainer exec \
  --env VENV_PATH="$VENV_PATH",UV_CACHE_DIR="$UV_CACHE_DIR",TMPDIR="$TMPDIR" \
  "$IMG_PATH" bash -s <<'BASH'
set -euo pipefail
export PATH="/root/.local/bin:$PATH"

PY="/usr/local/bin/python3.10"
echo "[container] Using local Python: $($PY -V)"

# If venv exists but is broken (missing python), wipe and recreate
if [ -d "$VENV_PATH" ] && [ ! -x "$VENV_PATH/bin/python" ]; then
  echo "[container] Existing venv is incomplete; removing..."
  rm -rf "$VENV_PATH"
fi

# Create venv if missing (stdlib venv = deterministic, includes ensurepip)
if [ ! -d "$VENV_PATH" ]; then
  echo "[container] Creating venv at: $VENV_PATH"
  "$PY" -m venv "$VENV_PATH"
fi

# Ensure pip is present (some distros omit seeding by default)
if ! "$VENV_PATH/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "[container] Bootstrapping pip via ensurepip..."
  "$VENV_PATH/bin/python" -m ensurepip --upgrade
fi

# Make sure basic build tooling is modern inside the venv
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel

# Install project requirements (exactly what you listed)
echo "[container] Installing requirements from requirements.txt"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export TMPDIR="$TMPDIR"
# Use pip to avoid activation; uv not required for install here
"$VENV_PATH/bin/pip" install -r requirements.txt

# Report
echo "[container] Python: $("$VENV_PATH/bin/python" -V)"
echo "[container] Pip:    $("$VENV_PATH/bin/pip" --version)"
BASH

# IDE convenience: symlink .venv → external venv (no activation)
ln -sfn "$VENV_PATH" "$LOCAL_DOT_VENV_LINK"
echo "[host] Venv ready at: $VENV_PATH"
echo "[host] Symlinked to:  $LOCAL_DOT_VENV_LINK"
