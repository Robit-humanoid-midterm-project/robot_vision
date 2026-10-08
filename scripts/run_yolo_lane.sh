#!/usr/bin/env bash
set -eo pipefail
experiment_root=/home/doyeon/.codex/.chatgpt-projects/g-p-6aba6659faf48191b8a2d11202c2dc7b
yolo_python=${ROBOT_VISION_YOLO_PYTHON:-$experiment_root/.venv-line/bin/python}
export YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR:-/tmp/robot_vision_ultralytics}
mkdir -p "$YOLO_CONFIG_DIR"
if [[ ! -x "$yolo_python" ]]; then
  echo "YOLO Python not found: $yolo_python; set ROBOT_VISION_YOLO_PYTHON" >&2
  exit 1
fi
exec "$yolo_python" "$(dirname "${BASH_SOURCE[0]}")/yolo_lane_node.py" "$@"
