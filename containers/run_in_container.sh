#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source containers/env.paths.sh

[[ -f "$IMG_PATH" ]] || { echo "Missing image: $IMG_PATH"; exit 1; }

# Auto-enable GPU passthrough if GPUs are allocated
NV_FLAG=""
if [[ -n "${SLURM_GPUS:-}" || -n "${SLURM_JOB_GPUS:-}" || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NV_FLAG="--nv"
fi

# If first arg looks like a Python file, auto-prefix with python
if [[ "${1:-}" == *.py ]]; then
  set -- python "$@"
fi

# If user typed: run_in_container.sh python …, keep it; otherwise run exactly what they passed.
apptainer exec $NV_FLAG "$IMG_PATH" /bin/sh -c "
  export UV_CACHE_DIR='$UV_CACHE_DIR'
  export TMPDIR='$TMPDIR'
  export MPLBACKEND=Agg
  export PATH='$VENV_PATH/bin':\"\$PATH\"
  exec \"\$@\"
" sh "$@"
