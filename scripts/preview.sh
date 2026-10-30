#!/usr/bin/env bash
set -euo pipefail

RADAR_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RADAR_RUNTIME_ROOT="${RADAR_RUNTIME_ROOT:-${RADAR_PROJECT_DIR}/.runtime}"
export OPENBLAS_NUM_THREADS="${RADAR_BLAS_THREADS:-1}"

if [[ ! -x "${RADAR_PROJECT_DIR}/.venv/bin/python" ]]; then
  echo "请先在项目根目录运行 scripts/setup.sh 安装环境。" >&2
  exit 1
fi
cd "${RADAR_PROJECT_DIR}/frontend"
npm run build
cd "${RADAR_PROJECT_DIR}"
echo "仿真工作台：http://${RADAR_PREVIEW_HOST:-127.0.0.1}:${RADAR_PREVIEW_PORT:-8000}/simulate"
exec "${RADAR_PROJECT_DIR}/.venv/bin/python" -m radar_platform serve \
  --host "${RADAR_PREVIEW_HOST:-127.0.0.1}" --port "${RADAR_PREVIEW_PORT:-8000}"
