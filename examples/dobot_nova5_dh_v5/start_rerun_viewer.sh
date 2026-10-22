#!/usr/bin/env bash
# Run on the viewing computer, in its graphical desktop session.
set -euo pipefail

viewer_python="${RERUN_PYTHON:-python}"
exec "$viewer_python" -m rerun \
    --bind "${RERUN_BIND:-0.0.0.0}" \
    --port "${RERUN_PORT:-9876}" \
    --memory-limit "${RERUN_MEMORY_LIMIT:-1GB}" \
    --drop-at-latency 500ms \
    --expect-data-soon \
    "$@"
