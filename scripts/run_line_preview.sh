#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
if [[ -f /home/doyeon/colcon_ws/install/setup.bash ]]; then
  source /home/doyeon/colcon_ws/install/setup.bash
fi
set -u
experiment_root=/home/doyeon/.codex/.chatgpt-projects/g-p-6aba6659faf48191b8a2d11202c2dc7b
export YOLO_CONFIG_DIR="$experiment_root/line_preview/.ultralytics"
mkdir -p "$YOLO_CONFIG_DIR"
exec "$experiment_root/.venv-line/bin/python" "$(dirname "${BASH_SOURCE[0]}")/insta360_line_preview.py" "$@"
