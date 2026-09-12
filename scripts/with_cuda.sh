#!/usr/bin/env bash
set -euo pipefail
project_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
if [[ -d "$project_root/.cache/nvidia-580.173.02" ]] && [[ "$(cat /proc/driver/nvidia/version 2>/dev/null)" == *580.173.02* ]]; then
  export LD_LIBRARY_PATH="$project_root/.cache/nvidia-580.173.02${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
