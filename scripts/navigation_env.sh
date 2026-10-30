#!/usr/bin/env bash
# Source from the authorized project root; all caches and runtime data stay here.
NAVIGATION_PROJECT_ROOT="$PWD"
if [ ! -f "$NAVIGATION_PROJECT_ROOT/backend/radar_platform/navigation/io.py" ]; then
  echo 'Source navigation_env.sh from the radar platform project root.' >&2
  return 1
fi
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="$NAVIGATION_PROJECT_ROOT/.cache/navigation/numba"
export MPLCONFIGDIR="$NAVIGATION_PROJECT_ROOT/.cache/navigation/matplotlib"
export UV_CACHE_DIR="$NAVIGATION_PROJECT_ROOT/.cache/navigation/uv"
export TMPDIR="$NAVIGATION_PROJECT_ROOT/.cache/navigation/tmp"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
