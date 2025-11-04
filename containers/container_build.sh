#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
source containers/env.paths.sh
mkdir -p "$IMG_DIR"

# Required by BlueBEAR docs before building
unset APPTAINER_BIND || true

apptainer build "$IMG_PATH" containers/uv-python3104.def
echo "Built image: $IMG_PATH"
