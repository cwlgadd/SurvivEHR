#!/usr/bin/env bash
# Centralised paths (edit once)

# ---- RDS roots ----
export RDS_ROOT="/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford"


# Virtual environments live off-repo
export VENV_DIR="$RDS_ROOT/virtual_envs"
export VENV_PATH="$VENV_DIR/SurvivEHR-3.10.4"

# uv cache + tmp (keep off $HOME where space may be limited)
export UV_CACHE_DIR="$RDS_ROOT/uv-cache"
export TMPDIR="$RDS_ROOT/tmp"

# (Optional) offline wheels location
export WHEELHOUSE="$RDS_ROOT/wheelhouse"

# Built container image path (NOT in repo)
export IMG_DIR="$RDS_ROOT/containers"
export IMG_PATH="$IMG_DIR/uv-python3104.sif"

# Editor convenience: symlink .venv → external venv
export LOCAL_DOT_VENV_LINK=".venv"

