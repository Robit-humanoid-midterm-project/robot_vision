#!/usr/bin/env bash
# Run after sourcing the target computer's ROS setup. No Codex runtime required.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_root=${XDG_DATA_HOME:-$HOME/.local/share}/robot_vision
ros_python=${ROBOT_VISION_ROS_PYTHON:-/usr/bin/python3}
if [[ $# -gt 1 ]]; then echo "Usage: bash $0 [path/to/weights.pt]" >&2; exit 2; fi
weights_source=${1:-}
if [[ -z "$weights_source" && -f "$script_dir/../models/weights.pt" ]]; then
  weights_source=$script_dir/../models/weights.pt
fi
if ! "$ros_python" -c 'import rclpy' 2>/dev/null; then
  echo "ROS Python unavailable. Source /opt/ros/<your-distro>/setup.bash first." >&2
  exit 1
fi
if [[ -z "$weights_source" ]]; then
  weights_source=$("$ros_python" -c 'from ament_index_python.packages import get_package_share_directory; from pathlib import Path; print(Path(get_package_share_directory("robot_vision")) / "models/weights.pt")')
fi
if [[ ! -f "$weights_source" ]]; then
  echo "Weights not found: $weights_source; pass a valid checkpoint path." >&2
  exit 1
fi
if [[ -x "$data_root/venv/bin/python" ]] && "$data_root/venv/bin/python" -m pip --version >/dev/null 2>&1; then
  base_version=$("$ros_python" -c 'import sys; print(sys.version_info[:2])')
  venv_version=$("$data_root/venv/bin/python" -c 'import sys; print(sys.version_info[:2])')
  if [[ "$base_version" != "$venv_version" ]]; then
    echo "Existing YOLO venv Python version differs from ROS Python. Use a matching environment via ROBOT_VISION_YOLO_PYTHON." >&2
    exit 1
  fi
elif ! "$ros_python" -m venv --system-site-packages "$data_root/venv"; then
  echo "Install the matching Ubuntu venv package (usually sudo apt install python3-venv), then rerun." >&2
  exit 1
fi
"$data_root/venv/bin/python" -m pip install --upgrade pip
"$data_root/venv/bin/python" -m pip install 'torch>=2.7' torchvision --index-url https://download.pytorch.org/whl/cpu
"$data_root/venv/bin/python" -m pip install -r "$script_dir/requirements-yolo.txt"
mkdir -p "$data_root/models"
if [[ $(realpath "$weights_source") != "$data_root/models/weights.pt" ]]; then
  install -m 644 "$weights_source" "$data_root/models/weights.pt"
fi
ROBOT_VISION_YOLO_PYTHON="$data_root/venv/bin/python" bash "$script_dir/run_yolo_lane.sh" --check
printf 'YOLO environment ready: %s\n' "$data_root/venv"
