## 실행 방법 
```
cd ~/colcon_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select insta360_usb_cam robot_vision
source ~/colcon_ws/install/setup.bash
ros2 launch robot_vision obstacle_distance.launch.py
```
```
v4l2-ctl -d /dev/video0 --set-ctrl=pan_absolute=0 
v4l2-ctl -d /dev/video0 --set-ctrl=tilt_absolute=-77000
v4l2-ctl -d /dev/video0 --set-ctrl=zoom_absolute=100
```

한문단씩 두 터미널에 실행하시면 됩니다. 

## YOLO 좌우 기준선

`obstacle_distance.launch.py`는 Insta360 카메라 → YOLO 추론 노드 → 기존 C++ 계산/화면 노드를 함께 시작한다.
좌우 기준선은 `weights.pt`의 `left-sideline`/`right-sideline` 분할로 검출한다.
분할 마스크의 각 행 중심에 직선을 맞추며 좌우 종류는 모델 클래스로 결정한다.
원본 영상과 선 결과를 같은 `LaneFrame`에 담아 다른 프레임의 좌표가 섞이지 않게 한다.

```bash
ros2 launch robot_vision obstacle_distance.launch.py weights:="$HOME/Downloads/weights.pt"
# 카메라가 이미 실행 중이면:
ros2 launch robot_vision obstacle_distance.launch.py start_camera:=false
```

- `/vision/left_lane_line`, `/vision/right_lane_line`: 각각 `LaneLine`, 검출 여부·원본 좌표·신뢰도.
- `/vision/lane_line`: 신뢰도가 더 높은 대표 선 하나(기존 구독자 호환).
- `/vision/lane_mask`: YOLO 분할 마스크.
- `/vision/yolo_lane_frame`: 해당 원본 JPEG, 좌우 선, 마스크, 추론 시간.

설정은 `config/obstacle_distance.yaml`의 `yolo_lane`에 있다. 기본 신뢰도 0.5, 영상 입력 640, CPU 처리 목표 15FPS.
1초 넘게 지연된 추론 결과는 버리고, 입력이 끊기면 좌우 선을 무효로 만든다.
기본 설정은 장애물 행·깊이를 쓰지 않고 YOLO 원본 선 좌표를 왜곡 보정한 다음 실측 지면 좌표로 변환한다. BEV 영상에서 재검출하거나 YOLO를 재학습할 필요가 없다.

시작할 때 로봇 바닥 중심을 왼쪽 기준선에서 **0.7m**에 놓고 카메라 팬·틸트·줌을 저장된 네 점과 같은 상태로 유지한다. 화면에 `locking reference`가 표시되는 동안 로봇을 정지시킨다. 일관된 선 관측 5회로 로봇 기준점을 고정하고 `FIELD X FROM LEFT`가 표시되면 이동한다. 시작부터 선이 없으면 기준점이 잡히지 않으며 결과는 -1000이다.

`left_x_1_dist`는 왼쪽 선에서 로봇까지의 수직 거리, 즉 필드 가로좌표(m)이다. 오른쪽은 `right_x_2_dist`로 전달한다. 한쪽만 보이면 실측 필드 폭 1.4m로 반대쪽을 계산한다. 양쪽이 보일 때 합이 필드 폭과 크게 다르거나 서로 평행하지 않으면 미측정으로 처리한다. 장애물이 없어도 선만 유효하면 좌표가 갱신된다.

양쪽 선 미검출, 해상도 불일치, 보정 영역에 충분한 선 점이 없음, 계산 품질 불량은 미측정(-1000)이다. 카메라 입력이 끊겼다가 돌아와도 이미 고정된 로봇 기준점을 현재 위치로 다시 잡지 않는다. 기준점을 새로 잡으려면 시작 위치로 돌려놓고 노드를 다시 실행한다.

지면 설정은 `obstacle_distance`의 `ground_*`, `field_width_m`, `start_from_left_m`이다. 보정 화면의 네 점과 실측 가로 1.86m, 전방 가까운 경계 1.5m·먼 경계 3.75m를 사용한다. 카메라 또는 장착 위치를 바꾸면 다시 보정해야 한다. 고정 호모그래피는 보행 중 몸체 pitch/roll을 보정하지 않으므로 정지 상태에서 먼저 실측과 대조한 뒤 보행 오차를 확인한다. 초기 버전은 시간 평활화로 지연을 추가하지 않는다.

`ground_field_enabled: false`를 지정하면 비교용 기존 장애물 행 깊이 방식으로 돌아간다. 모델은 직접 미터 거리를 출력하지 않는다.

다른 컴퓨터에서 이 저장소를 내려받은 뒤, 그 컴퓨터에서 아래를 실행한다. 사용자 이름과 Codex 폴더는 필요하지 않다. Python 가상환경은 컴퓨터마다 새로 만든다.

```bash
# 설치된 ROS 배포판으로 변경 (Jazzy 예시)
source /opt/ros/jazzy/setup.bash
sudo apt install python3-venv
cd ~/colcon_ws
colcon build --packages-up-to robot_vision insta360_usb_cam
source install/setup.bash
ros2 run robot_vision setup_yolo.sh
ros2 run robot_vision run_yolo_lane.sh --check
ros2 launch robot_vision obstacle_distance.launch.py
```

설치기는 사용자 홈의 `.local/share/robot_vision/venv`에 CPU 추론 환경을 만들고, 모델을 `.local/share/robot_vision/models/weights.pt`에 복사한다. `XDG_DATA_HOME`이 있으면 그 위치를 사용한다. 런처는 이 환경, 활성화된 가상환경, 기본 Python 순서로 찾는다. 별도 환경은 `ROBOT_VISION_YOLO_PYTHON`으로 지정할 수 있다.
모델 기본 경로는 패키지 models → 사용자 데이터 폴더 → 사용자 Downloads 순서로 찾는다. 명시적인 `weights:=...` 또는 `ROBOT_VISION_WEIGHTS`가 우선한다. 모델이 없으면 경로를 표시하고 실행을 중단한다.
`--check`는 Python 의존성과 빌드된 `LaneFrame` 메시지를 확인한다. 카메라 영상이 있어도 통합 화면이 계속 대기하면 먼저 YOLO 노드의 오류를 확인한다. 통합 화면은 YOLO가 보낸 원본 프레임을 사용한다.
검증 버전은 ultralytics 8.4.174, numpy 1.26.4, opencv-python 4.11.0.86이다. CUDA/Jetson 환경은 해당 장비에 맞는 PyTorch를 별도로 설치하고 Python 경로를 지정한다.

학습한 YOLO26 Nano 좌우 선 모델은 `models/weights.pt`로 저장소에 포함되고 빌드 때 설치된다. 별도 모델을 사용할 때만 `weights:=/절대경로/model.pt`를 지정한다. 가상환경은 Git에 포함하지 않고 설치 스크립트가 해당 컴퓨터에 생성한다.

## 현재 ROS2 msg

토픽: /vision2master
타입: humanoid_interfaces/msg/VisionData
현재 필드: timestamp, left_x_1_dist, right_x_2_dist, obstacle_1~3
단위: m, 장애물 순서 (전방 y, 좌우 x)
좌우 부호: 왼쪽 음수·오른쪽 양수
미측정: -1000

## 코드 전체 흐름

카메라 영상 -> YOLO 좌우 기준선 추론(원본과 결과를 함께 전달) -> 빨강, 파랑 색상 마스크 생성 -> 잡음 제거 -> 윤곽선 추출 -> 사각형 조건 검사 -> 거리 계산 -> 토픽 발행 및 윤곽선·거리 라벨 표시

색상 영역의 윤곽선은 거리 계산 성공 여부와 관계없이 표시한다. 사각형·거리 조건을 통과한 판에는 거리 라벨을 함께 표시한다.

### 1. 색상 영역 찾기 

[color_region_detector.cpp]의 detect_color_regions()가 담당한다. 
영상을 BGR에서 HSV 색상 공간으로 변환한 뒤 기존 설정값으로 빨간색, 파란색 픽셀을 골라 마스크를 만든다. 이후 3x3 커널로 잡음 제거한다. 

### 2. 윤곽선 만들기 

전처리한 마스크에 색상 영역의 바깥 윤곽선을 찾는다. 내부 구멍의 윤곽선은 따로 검출하지 않는다. 윤곽선 면적이 기존 min_area_px보다 작으면 제외하고, 나머지는 다음 정보로 저장한다.

| 저장값 | 의미 |
|---|---|
| `color` | 빨강 / 파랑 |
| `bounds` | 영역을 감싸는 사각형 |
| `center` | 보이는 영역의 면적 중심 |
| `area_px` | 윤곽선 면적 |
| `touches_border` | 화면 가장자리에 닿았는지 |
| `contour` | 실제 윤곽선 점들 |

자료구조는 [distance_estimator.hpp]의 ImageCandidate에 있다.

화면에 잘렸거나 정사각형처럼 보이지 않아도 위치 후보는 남는다. 단, 아직은 장애물로 확정된 물체가 아니라 색상 영역 후보다(거리측정 대상 판별이 아니라는 의미이다.).

### 3. 거리 계산 

[distance_estimator.cpp]는 색상 검출 결과를 받은 뒤 기존 거리 계산을 수행한다.

결과가 두 종류로 나뉘어 있다.
- image_candidates: 색상 영역의 위치 후보
- detections: 사각형·거리 조건까지 통과한 결과

따라서 거리 계산에 실패해도 image_candidates는 사라지지 않는다. 입력 해상도가 보정 기준과 달라도 영상 위치 후보는 남고, 거리 계산만 중단한다.

### 4. 화면에 표시  

[vision_viewer.cpp]에서 image_candidates를 순회하며 윤곽선과 계산된 거리 라벨을 그린다.

-------------------------------------------

## BEV 네 점 선택 구현 

`obstacle_distance.ros__parameters`에는 기존 영상·캘리브레이션·색상 설정이 있고,
`camera_bev_preview.ros__parameters`에는 BEV 설정이 있습니다.
중복된 세부 검출/추적 설정은 코드 기본값을 사용하며 기존 동작값은 유지합니다.

카메라 노드 실행 
```bash
source ~/colcon_ws/install/setup.bash
ros2 launch insta360_usb_cam usb_cam.launch.py
```

BEV와 선 검출 화면 실행 
```
source ~/colcon_ws/install/setup.bash
ros2 launch robot_vision bev.launch.py
config:=$HOME/colcon_ws/src/robot_vision/config/obstacle_distance.yaml
```

- 좌표 수정: `yaml`파일 안에 `camera_bev_preview` 아래 `source_points`.
  `[x1, y1, x2, y2, x3, y3, x4, y4]` 형식. 초기 `[]`는 미선택입니다.
- 한 창의 왼쪽 위 원본 화면에서 바닥 직사각형의 **먼 왼쪽 → 먼 오른쪽 → 가까운 오른쪽 → 가까운 왼쪽**을 클릭합니다.
  클릭 좌표는 왜곡 보정 좌표로 변환해 저장합니다. 첫 클릭에 영상이 정지하고 네 번째 클릭 후 BEV가 표시됩니다.
- S: 네 점만 저장 후 실시간 복귀. R: 재선택. Q/Esc: 종료.
  YAML의 다른 값과 안내 주석은 저장 시 유지됩니다. 수동 수정 후 재실행합니다.
- 화면은 원본 카메라(왼쪽 위), BEV(오른쪽 위), 흰색 전처리 마스크(왼쪽 아래), 선 검출 결과(오른쪽 아래)의 4분할입니다.
  왼쪽 직선은 초록색, 오른쪽 직선은 주황색으로 표시합니다. `HELD`는 직전 검출을 잠깐 유지한 상태,
  `MISSING`은 선을 놓친 상태입니다. BEV 검출은 새 프레임에서 최대 약 15회/초 수행합니다.
  흰색 후보의 밝기·채도와 잔디 인접 비율은 같은 YAML의 `line_white_min_value`,
  `line_white_max_saturation`, `line_min_grass_support`로 조정합니다.
- `bev_size`: 출력 [가로, 세로] 픽셀.
- `bev_width_m`: BEV 영상 좌우 끝에 대응하는 바닥 폭의 근삿값(1.86m). 가로 픽셀 축척은 이 값으로 고정합니다.
  `field_width_m`: 같은 전방 위치의 양쪽 흰 선 사이 실측 거리(1.4m).
  `robot_x_from_left_m`: 로봇 바닥 중심이 왼쪽 흰 선에서 떨어진 거리(0.7m).
  `near_edge_forward_m`, `far_edge_forward_m`: 로봇에서 가까운/먼 변까지의 전방 거리(1.5m, 3.75m).
  전방 거리는 BEV 세로 픽셀로 환산합니다. 전방 2.63m에 해당하는 같은 BEV 행에서
  왼쪽 또는 오른쪽 선 하나가 안정적으로 5프레임 관측되면 0.7m 실측 거리로 로봇 중심 픽셀을 고정합니다.
  `LOCKED` 이후에는 한쪽 선만 보여도 해당 선까지의 가로 거리를 표시합니다.
  관측되지 않은 선은 `--`입니다. 처음 중심을 잡기 전에는 `WAIT LINE`으로 표시됩니다.
  R 또는 S를 누르면 중심 기준을 다시 잡습니다. `BEV WIDTH ~1.86m`의 `~`는 근삿값이라는 뜻입니다.
- K/D, 영상 토픽, 해상도는 같은 YAML의 기존 설정에서 읽습니다.
  보정 후 같은 K를 유지하고 자르지 않습니다. 좌표 원점은 보정 영상 왼쪽 위, x 오른쪽/y 아래입니다.
- 현재는 GUI 미리보기이며 BEV 출력 토픽은 없습니다. 카메라 위치·높이·팬/틸트 변경 시 다시 선택하세요.
- 장애물 launch의 `image_topic`, `viewer` 인자를 생략하면 YAML 값을 사용합니다.
  BEV는 YAML의 영상 토픽을 그대로 사용합니다.
- 저장 원본 영상: `ros2 run robot_vision bev_node --image /절대경로/field.png --config /home/doyeon/colcon_ws/src/robot_vision/config/obstacle_distance.yaml`.
- 현재는 선택한 바닥 사각형 안의 픽셀→미터 환산과 좌우 선까지의 가로 거리 표시까지 구현했습니다.
  각도·장애물 위치 계산은 이후 단계입니다.

## 카메라 펜 틸트 조정 방법 

- 3600 = 1°이며, 카메라의 각도 범위는 다음과 같다.
Pan  : -145° ~ +145°
Tilt :  -90° ~ +100°
- 실행 방법
카메라 실행(이것만으로 카메라화면이 뜨지 않는다. 단순 pan tilt 바꾸는 용도) : 
source /home/robit/colcon_ws/install/setup.bash
ros2 launch insta360_usb_cam usb_cam.launch.py

pan을 돌리고 싶을때 : 
v4l2-ctl -d /dev/video0 --set-ctrl=pan_absolute=0 
tilt를 돌리고 싶을때 : 
v4l2-ctl -d /dev/video0 --set-ctrl=tilt_absolute=-50000
zoom값 확인 : 
v4l2-ctl -d /dev/video0 --list-ctrls | grep zoom
zoom값 설정(예 : 104) :
v4l2-ctl -d /dev/video0 --set-ctrl=zoom_absolute=104


맨 뒤에 숫자를 바꿔 카메라 각도를 조절할 수 있다.

source /home/doyeon/colcon_ws/install/setup.bash
ros2 topic echo /vision2master --field obstacle_1

## 화면에 뜨는 값은 위에서 순서대로 다음과 같다. 

| 화면 항목 | 의미 |
|---|---|
| **Range** | 카메라 → 판 아래쪽 중심의 직선거리 |
| **Ground** | 카메라 바로 아래 바닥 지점 → 판까지의 바닥거리 |
| **Forward** | 바닥거리 중 전방 방향 성분 |
| **Lateral** | 좌우 편차: 왼쪽 `−`, 오른쪽 `+` |

## 선 학습용 원본 영상 녹화

카메라 노드를 먼저 켠 뒤, 다른 터미널에서 JPEG 압축 ROS 영상 토픽을 녹화합니다.
카메라 장치를 중복으로 열지 않으며, JPEG를 풀어 보정 기준과 같은 640×480 컬러 영상을 저장합니다.
카메라 패키지를 새로 빌드한 뒤에는 기존 카메라 프로세스를 종료하고 다시 실행해야 토픽 타입이 바뀝니다.

```bash
source ~/colcon_ws/install/setup.bash
ros2 launch insta360_usb_cam usb_cam.launch.py
```

```bash
source ~/colcon_ws/install/setup.bash
ros2 run robot_vision record_dataset --output "$HOME/colcon_ws/robot_vision_dataset/000_000_000_run01.avi"
```

실행하면 640×480 원본 카메라 미리보기 창이 열리고 빨간 `REC`와 저장 프레임 수가 표시됩니다.
미리보기 표시만 덧그리며 저장되는 영상에는 글씨를 넣지 않습니다.
녹화는 미리보기에서 `Q`/`Esc`를 누르거나 창을 닫거나 터미널에서 `Ctrl+C`로 끝냅니다.
MJPG AVI 영상과 같은 이름의 `.timestamps.csv`가 저장됩니다.
파일이 이미 있으면 덮어쓰지 않고 종료합니다. 첫 영상은 예를 들어
`empty_run01.avi`, 다음 영상은 `110_110_100_run01.avi`로 저장하세요.
입력 영상이 640×480이 아니면 녹화하지 않고 오류를 출력합니다.
화면의 BEV나 전처리 마스크가 아닌 원본 카메라 프레임을 저장합니다.
AVI의 재생 FPS는 기본 30으로 기록하며, 실제 수신 시각은 CSV에 보존합니다.

## 별도 사진 촬영 방법

cd /home/robit/colcon_ws/src/robot_vision
g++ -std=c++17 src/capture_dataset.cpp -o capture_dataset $(pkg-config --cflags --libs opencv4)
./capture_dataset

## 파일별 역할

카메라 영상을 받은 뒤 장애물·차선·행을 분석하고, 계산 결과를 로봇용 메시지와 화면에 사용한다.
장애물 기준과 캘리브레이션을 사용하며, 전체 처리율은 YOLO 처리 속도(기본 목표 15FPS)에 제한된다.
실제 FPS는 카메라 입력과 처리 시간에 따라 달라진다.

| 파일 | 역할 |
|---|---|
| `obstacle_distance_node.cpp` | ROS 설정, 영상 수신, 처리 주기 제한, 결과 발행, 영상 끊김 감시, 처리 FPS 기록 |
| `vision_pipeline.cpp` | 장애물·차선·행 검출 호출, 행 추적과 프레임 결과 관리 |
| `field_geometry.cpp` | 차선과 행의 교차점, 좌우 경계 거리 계산 |
| `vision_message_builder.cpp` | 장애물·차선·로봇 제어용 ROS 메시지 생성 |
| `vision_viewer.cpp` | 윤곽선·거리·행 표시, 전처리 화면 생성, 통합 창 표시와 키 입력 |
| `color_region_detector.cpp` | 빨강·파랑 색상 마스크와 영역 후보 추출 |
| `distance_estimator.cpp` | 판의 위치·거리 추정 및 가려진 판의 검출 복구 |
| `scripts/yolo_lane_node.py`, `scripts/yolo_lane_fit.py` | 원본 프레임의 YOLO 분할과 좌우 중심선 추정 |
| `lane_line_estimator.cpp` | 이전 검출기의 참조·회귀 검사 코드. 통합 실행에서는 호출하지 않음 |
| `row_line_estimator.cpp` | 장애물 아래쪽 경계의 행 추정, 여러 프레임의 추적 결과 평활화 |
| `bev_node.cpp` | 별도 BEV 보정·미리보기 도구 |
| `record_dataset.cpp` | Insta360 ROS 원본 영상을 640×480 AVI와 프레임 시간 CSV로 녹화 |
| `capture_dataset.cpp` | 별도 원본·마스크 사진 저장 도구. 수동 빌드하며 YOLO 추론은 수행하지 않음 |


`test/`의 검사 코드는 검출·거리·메시지와 프레임 제한 동작을 확인한다.
일반 카메라 프로그램 실행 중에는 검사 코드가 동작하지 않는다.

yaml에 해당하는 값이 있으면 우선적으로 yaml에 있는 값이 적용되고 
yaml에 존재하지 않는값은 obstacle_distance_node.cpp에 declare_parameter에 저장된다.

'''
source /opt/ros/jazzy/setup.bash
source ~/colcon_ws/install/setup.bash
ros2 topic echo /vision2master
'''
