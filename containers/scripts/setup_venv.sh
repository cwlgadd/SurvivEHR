#!/usr/bin/env bash
set -euo pipefail

# Choose the Python you want embedded into the venv
PY_VER="${PY_VER:-3.11}"
VENV_DIR="${VENV_DIR:-$HOME/venvs}"
VENV_NAME="${VENV_NAME:-yourproj-py${PY_VER}}"
VENV_PATH="${VENV_DIR}/${VENV_NAME}"

mkdir -p "$VENV_DIR"

# 1) Create a completely self-contained venv (uv will fetch Python if missing)
uv venv --python "$PY_VER" "$VENV_PATH"

# 2) Activate
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

# 3) Install your requirements (no --user, inside the venv)
#    (uv pip is drop-in compatible; it’s much faster, but pip works too)
uv pip install --upgrade pip
uv pip install -r "$(git rev-parse --show-toplevel)/requirements.txt"

# Optional: dev tools
if [[ -f "$(git rev-parse --show-toplevel)/requirements-dev.txt" ]]; then
  uv pip install -r "$(git rev-parse --show-toplevel)/requirements-dev.txt"
fi

# 4) Show versions for sanity
python -V
pip list | head -50
