## 실행 방법 
```
cd ~/colcon_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select insta360_usb_cam robot_vision
source ~/colcon_ws/install/setup.bash
ros2 launch robot_vision obstacle_distance.launch.py
```
```
v4l2-ctl -d /dev/video4 --set-ctrl=pan_absolute=0 
v4l2-ctl -d /dev/video4 --set-ctrl=tilt_absolute=-77000
v4l2-ctl -d /dev/video4 --set-ctrl=zoom_absolute=100
```

한문단씩 두 터미널에 실행하시면 됩니다. 

## 현재 ROS2 msg

토픽: /vision2master
타입: humanoid_interfaces/msg/VisionData
현재 필드: timestamp, left_x_1_dist, right_x_2_dist, obstacle_1~3
단위: m, 장애물 순서 (전방 y, 좌우 x)
좌우 부호: 왼쪽 음수·오른쪽 양수
미측정: -1000

## 코드 전체 흐름

카메라 영상 -> 빨강, 파랑 색상 마스크 생성 -> 잡음 제거 -> 윤곽선 추출 -> 사각형 조건 검사 -> 거리 계산 -> 토픽 발행 및 윤곽선·거리 라벨 표시

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
v4l2-ctl -d /dev/video2 --list-ctrls | grep zoom
zoom값 설정(예 : 104) :
v4l2-ctl -d /dev/video4 --set-ctrl=zoom_absolute=104


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
ros2 run robot_vision record_dataset --output "$HOME/colcon_ws/robot_vision_dataset/110_110_100_run01.avi"
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
검출 기준과 캘리브레이션은 유지하며, 처리 목표는 `max_processing_fps: 30.0`이다.
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
| `lane_line_estimator.cpp` | 흰색 바닥 경계선 검출 |
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
