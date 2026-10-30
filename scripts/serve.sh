#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RADAR_RUNTIME_ROOT="${RADAR_RUNTIME_ROOT:-${PROJECT_DIR}/.runtime}"

if [[ ! -f "${PROJECT_DIR}/frontend/dist/index.html" ]]; then
  echo "Frontend build missing. Run: cd frontend && npm run build" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
exec python -m radar_platform.cli serve --host "${RADAR_API_HOST:-127.0.0.1}" --port "${RADAR_API_PORT:-8000}"
