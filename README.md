## 실행 방법 

cd ~/colcon_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select insta360_usb_cam robot_vision
source ~/colcon_ws/install/setup.bash
ros2 launch robot_vision obstacle_distance.launch.py

## 코드 전체 흐름

카메라 영상 -> 빨강, 파랑 색상 마스크 생성 -> 잡음 제거 -> 윤곽선 추출 -> 사각형 조건 검사 -> 거리 계산 -> 토픽 발행(현제 화면에 거리계산값은 나오지 않음. 추출한 윤곽선만 표시)

현재 윤곽선만 보이게 하였으며 사각형 조건 검사후 거리를 계산해야 장애물이라 가정하겠다. 추후 변경할 예정.

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

[obstacle_distance_node.cpp]에서 image_candidates를 순회하며 윤곽선을 그린다.

-------------------------------------------

## BEV 네 점 선택 구현 

`obstacle_distance.ros__parameters`에는 기존 영상·캘리브레이션·색상 설정이 있고,
`camera_bev_preview.ros__parameters`에는 BEV 설정이 있습니다.
중복된 세부 검출/추적 설정은 코드 기본값을 사용하며 기존 동작값은 유지합니다.

기존 카메라가 실행된 ROS 환경에서:

```bash
cd /home/robit/colcon_ws
colcon build --packages-select robot_vision --cmake-args -DAMENT_CMAKE_SYMLINK_INSTALL=OFF
source install/setup.bash
ros2 launch robot_vision bev.launch.py config:=/home/doyeon/colcon_ws/src/robot_vision/config/obstacle_distance.yaml
```

- 좌표 수정: `yaml`파일 안에 `camera_bev_preview` 아래 `source_points`.
  `[x1, y1, x2, y2, x3, y3, x4, y4]` 형식. 초기 `[]`는 미선택입니다.
- 왜곡 보정 화면에서 바닥 직사각형의 **먼 왼쪽 → 먼 오른쪽 → 가까운 오른쪽 → 가까운 왼쪽**을 클릭합니다.
  첫 클릭에 영상이 정지하고 네 번째 클릭 후 BEV가 표시됩니다.
- S: 네 점만 저장 후 실시간 복귀. R: 재선택. Q/Esc: 종료.
  YAML의 다른 값과 안내 주석은 저장 시 유지됩니다. 수동 수정 후 재실행합니다.
- `bev_size`: 출력 [가로, 세로] 픽셀.
- `field_width_m`, `field_height_m`: 선택한 바닥 직사각형의 실측 가로·세로(m).
  `0.0`은 미입력이며 **지금은 기록만 합니다**. 미터 환산과 각도 계산은 다음 단계입니다.
- K/D, 영상 토픽, 해상도는 같은 YAML의 기존 설정에서 읽습니다.
  보정 후 같은 K를 유지하고 자르지 않습니다. 좌표 원점은 보정 영상 왼쪽 위, x 오른쪽/y 아래입니다.
- 현재는 GUI 미리보기이며 BEV 출력 토픽은 없습니다. 카메라 위치·높이·팬/틸트 변경 시 다시 선택하세요.
- 장애물 launch의 `image_topic`, `viewer` 인자를 생략하면 YAML 값을 사용합니다.
  BEV는 YAML의 영상 토픽을 그대로 사용합니다.
- 저장 원본 영상: `ros2 run robot_vision bev_node --image /절대경로/field.png --config /home/doyeon/colcon_ws/src/robot_vision/config/obstacle_distance.yaml`.
- 현재 bev자체만 구현 점으로 찍고 확정후 매프레임마다 bev화면도 뽑아내어 각도 및 가로상의 위치 및 장애물 위치 구할 예정 

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
v4l2-ctl -d /dev/video0 --set-ctrl=tilt_absolute=40000
- 

source /home/doyeon/colcon_ws/install/setup.bash
ros2 topic echo /vision2master --field obstacle_1