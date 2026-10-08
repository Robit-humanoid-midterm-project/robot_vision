#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${ROS_DISTRO:-} ]]; then
  shopt -s nullglob
  ros_setups=(/opt/ros/*/setup.bash)
  if [[ ${#ros_setups[@]} -ne 1 ]]; then
    echo "Source /opt/ros/<your-distro>/setup.bash and your workspace/install/setup.bash first." >&2
    exit 1
  fi
  source "${ros_setups[0]}"
fi
workspace_root=${ROBOT_VISION_WORKSPACE:-$(cd "$script_dir/../../.." && pwd)}
if [[ -f "$workspace_root/install/setup.bash" ]]; then
  source "$workspace_root/install/setup.bash"
fi
exec bash "$script_dir/run_yolo_lane.sh" --preview "$@"
