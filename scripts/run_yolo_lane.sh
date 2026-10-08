#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_root=${XDG_DATA_HOME:-$HOME/.local/share}/robot_vision
if [[ -n ${ROBOT_VISION_YOLO_PYTHON:-} ]]; then
  yolo_python=$ROBOT_VISION_YOLO_PYTHON
elif [[ -x "$data_root/venv/bin/python" ]]; then
  yolo_python=$data_root/venv/bin/python
elif [[ -n ${VIRTUAL_ENV:-} && -x "$VIRTUAL_ENV/bin/python" ]]; then
  yolo_python=$VIRTUAL_ENV/bin/python
else
  yolo_python=$(command -v python3)
fi
if ! command -v "$yolo_python" >/dev/null; then
  echo "YOLO Python not found: $yolo_python" >&2
  echo "Run: bash $script_dir/setup_yolo.sh" >&2
  exit 1
fi
export YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR:-$data_root/ultralytics}
mkdir -p "$YOLO_CONFIG_DIR"
"$yolo_python" - <<'PY'
import importlib.util, sys
required=('rclpy','cv2','numpy','torch','ultralytics')
missing=[name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print('Missing Python modules: '+', '.join(missing), file=sys.stderr)
    print('Source ROS/workspace setup.bash and run scripts/setup_yolo.sh.', file=sys.stderr)
    sys.exit(1)
PY
if [[ ${1:-} == --check ]]; then
  "$yolo_python" - "$script_dir" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
import rclpy, cv2, numpy, torch, ultralytics
from robot_vision.msg import LaneFrame
print('Python:',sys.executable)
print('ROS + LaneFrame: OK')
print('torch:',torch.__version__,'ultralytics:',ultralytics.__version__)
print('numpy:',numpy.__version__,'OpenCV:',cv2.__version__)
PY
  exit 0
fi
if [[ ${1:-} == --preview ]]; then
  shift
  exec "$yolo_python" "$script_dir/insta360_line_preview.py" "$@"
fi
exec "$yolo_python" "$script_dir/yolo_lane_node.py" "$@"
