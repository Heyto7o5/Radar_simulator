#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RADAR_RUNTIME_ROOT="${RADAR_RUNTIME_ROOT:-${PROJECT_DIR}/.runtime}"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then kill "${BACKEND_PID}" 2>/dev/null || true; fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then kill "${FRONTEND_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

cd "${PROJECT_DIR}"
python -m radar_platform.cli serve --host "${RADAR_API_HOST:-127.0.0.1}" --port "${RADAR_API_PORT:-8000}" &
BACKEND_PID=$!

cd "${PROJECT_DIR}/frontend"
npm run dev -- --host "${RADAR_UI_HOST:-127.0.0.1}" &
FRONTEND_PID=$!

wait -n "${BACKEND_PID}" "${FRONTEND_PID}"
