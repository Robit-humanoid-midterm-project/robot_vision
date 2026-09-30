# robot_vision

Insta360 영상에서 **보이는 면이 40 × 40 cm인 빨강·파랑 장애물**을 검출하고, 면의 **아랫변 중앙**까지의 거리를 추정하는 ROS 2 Jazzy 패키지입니다. 검출 결과는 ROS 토픽과 디버그 영상으로 내보냅니다. 장애물들의 밑변을 같은 줄로 묶어 화면에 연장선을 그리는 기능은 현재 시험 단계입니다.

현재 위치값은 **카메라 기준 추정값**입니다. 필드 절대 좌표, 흰색 양쪽 주행선 검출, 장애물 줄 번호(1·2·3)는 아직 구현하지 않았습니다.

## 실행

```bash
cd ~/colcon_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to robot_vision
source install/setup.bash
ros2 launch robot_vision obstacle_distance.launch.py
```

기본 설정으로 `insta360_usb_cam`과 거리 측정 노드가 함께 실행됩니다. 카메라가 이미 실행 중이면 `start_camera:=false`, 화면을 띄울 수 없는 환경이면 `viewer:=false`를 붙입니다. 두 번째 전처리 창만 끄려면 [설정 파일](config/obstacle_distance.yaml)의 `show_preprocess: false`를 사용합니다. 설정을 바꾼 뒤에는 다시 빌드하고 노드를 재시작하거나, 아래처럼 소스 설정 파일을 직접 지정합니다.

```bash
ros2 launch robot_vision obstacle_distance.launch.py config_file:=$HOME/colcon_ws/src/robot_vision/config/obstacle_distance.yaml
```

카메라 토픽 이름은 `/camera1/camera/compressed_image`지만 실제 메시지 형식은 압축 영상이 아닌 **`sensor_msgs/Image`(BGR8, 640 × 480)**입니다. 이름만 보고 `CompressedImage`로 구독하면 영상이 나오지 않습니다.

## 파일 구성과 처리 순서

| 파일 | 역할 |
| --- | --- |
| [obstacle_distance.launch.py](launch/obstacle_distance.launch.py) | 카메라와 거리 측정 노드 실행 |
| [obstacle_distance.yaml](config/obstacle_distance.yaml) | 카메라 내부 파라미터, 실제 장애물 크기, 렌즈 높이, 색 범위와 검출 기준 |
| [distance_estimator.cpp](src/distance_estimator.cpp), [헤더](include/robot_vision/distance_estimator.hpp) | HSV 마스크 → 사각형 후보 → PnP 자세/거리 → 바닥 거리 계산 |
| [row_line_estimator.cpp](src/row_line_estimator.cpp), [헤더](include/robot_vision/row_line_estimator.hpp) | 영상에서 보이는 장애물 밑변 추출 및 같은 줄의 연장선 추정 |
| [obstacle_distance_node.cpp](src/obstacle_distance_node.cpp) | 영상 구독, 결과 메시지·디버그 영상 발행, 화면 표시, 영상 끊김 처리 |
| [ObstacleDetection.msg](msg/ObstacleDetection.msg), [ObstacleArray.msg](msg/ObstacleArray.msg) | 장애물별 위치·거리와 프레임 상태를 전달하는 ROS 메시지 |
| [test_distance_estimator.cpp](test/test_distance_estimator.cpp) | 거리 수식, 두 색 검출과 붙어 보이는 면 등 C++ 검사 |
| [calibrate.launch.py](launch/calibrate.launch.py) | 체커보드 카메라 보정을 별도로 실행 |

처리 흐름은 다음과 같습니다.

1. 빨강은 OpenCV HSV의 `H=0~12`와 `168~179` 두 구간, 파랑은 `H=105~125`를 마스킹합니다. OpenCV의 8비트 HSV 색상각 H는 **0~179**이고, 빨강이 양 끝에 걸쳐 있어 두 구간이 필요합니다. S/V 범위도 [YAML](config/obstacle_distance.yaml)에 있습니다.
2. 마스크에서 노이즈를 제거한 뒤 색 영역의 외곽과 네 모서리 후보를 찾습니다. 최소 면적, 가장자리 길이, 화면 경계, 사각형과 색 영역의 겹침률(`min_fill_ratio`), 재투영 오차, 거리 범위로 후보를 걸러냅니다.
3. 빨강·파랑 모두 같은 색의 앞뒤 면이 붙어 보이면 채도 차이로 나누고, 필요하면 보이는 네 변으로 면을 근사합니다. 이 결과는 화면의 `~`와 메시지의 `color_split_estimate=true`로 표시합니다. 네 변이 충분하지 않으면 거리를 내지 않습니다.
4. **실제 면 크기 0.40 m × 0.40 m**, 검출된 네 모서리 픽셀, 카메라 행렬 `K`와 왜곡 계수 `D`를 OpenCV PnP에 넣습니다.(기존 Insta_360패키지에 있던 보정값을 사용함) 정사각형용 IPPE와 일반 반복 풀이 후보 중 재투영 오차가 작은 유효 해를 선택합니다. 이때 추정한 3D 점은 면 중심이 아니라 **아랫변 중앙**입니다.
5. 아래 거리값과 같은 줄 추정선을 계산해 화면과 토픽에 발행합니다.

## 거리값과 좌표계

장애물 아랫변 중앙의 카메라 광학 좌표를 `P=(x_c, y_c, z_c)` m라고 둡니다. 광학 좌표는 **오른쪽이 +x, 아래쪽이 +y, 카메라 앞쪽이 +z**입니다. `P`는 [ObstacleDetection.msg](msg/ObstacleDetection.msg)의 `position`이며 `/vision/obstacles`의 `header.frame_id`는 기본적으로 `camera1_optical_frame`입니다. 로봇 몸체나 필드 좌표계로 변환하는 TF는 현재 없습니다.

화면 표시는 다음과 같이 계산합니다. `h`는 [YAML](config/obstacle_distance.yaml)의 `camera_height_m`이며 기본 0.60 m입니다.

| 화면 표시 | ROS 필드 | 뜻과 계산 |
| --- | --- | --- |
| `BOTTOM` | `distance_m` | 렌즈 중심에서 장애물 아랫변 중앙까지의 3D 직선거리 `D = ‖P‖` |
| `L/R` | `position.x` | 렌즈 기준 좌우 성분 `x_c`; 오른쪽이 양수 |
| `Y GROUND` | `ground_distance_m` | 렌즈 바로 아래 바닥점부터 장애물 아랫변 중앙까지의 **바닥 위 직선거리** `G = √(D² − h²)` |
| `X FORWARD` | `forward_distance_m` | 위 바닥 거리에서 카메라가 향한 **앞쪽 성분** `F = √(G² − x_c²)` |

좌우 성분은 영상에서 중심이 얼마나 벗어났는지와 거리 추정값으로 구합니다. 왜곡을 무시한 단순한 관계는 `x_c ≈ (u − c_x) × z_c / f_x`입니다. 현재 코드에서는 이 식만으로 끝내지 않고 네 모서리와 `K/D`를 사용한 **PnP 결과의 x 좌표**를 씁니다. 영상 한 픽셀을 고정 길이로 바꾸는 방식이 아닙니다.

예를 들어 `BOTTOM=1.50 m`, `L/R=+0.30 m`, `h=0.75 m`라면 `Y GROUND≈1.30 m`, `X FORWARD≈1.26 m`입니다. **화면의 X/Y는 필드 지도상의 절대 좌표가 아닙니다.** 특히 `Y GROUND`는 좌우와 앞쪽을 합친 바닥 위 거리이며 ROS `position.y`(광학 좌표의 아래쪽)와 다릅니다. `D<h` 또는 `G<|x_c|`이면 바닥 투영이 성립하지 않아 유효 플래그를 `false`로 두고 화면에 `--`를 표시합니다.

이 계산은 장애물 아랫변 중앙이 바닥에 닿고, 렌즈 아래 바닥과 장애물 바닥이 같은 높이라는 가정에 의존합니다. 앞쪽 성분을 주행 방향 거리로 쓰려면 카메라의 좌우 기울기와 방향도 확인해야 합니다. `camera_height_m=0.75`는 나중에 실제 장착 높이에 맞춰 수정할 수 있습니다.

## ROS 출력과 디버깅

| 토픽 | 내용 |
| --- | --- |
| `/vision/obstacles` | `robot_vision/msg/ObstacleArray`: 영상 시각, 보정 확인 상태, 장애물별 색·위치·거리·모서리·재투영 오차. 검출이 없으면 빈 `detections` 배열 |
| `/vision/obstacle_debug` | 사각형, 거리, 밑변과 같은 줄의 연장선을 그린 BGR 영상 |
| `/vision/obstacle_mask` | 빨강·파랑으로 분리된 색 마스크 |
| `/vision/obstacle_preprocess` | 각 색의 HSV 직후 `RAW`와 노이즈 제거 후 `CLEAN` 마스크 및 검출 통계 |

`Obstacle distance - RED / BLUE - 40cm` 창에는 위 거리와 색을 표시합니다. 초록 실선은 실제 영상에서 추출한 장애물 밑변이고, 노란 점선은 영상 높이와 기울기가 비슷한 밑변을 한 줄로 묶어 화면 폭으로 연장한 **추정선**입니다. 일부만 보이는 밑변도 후보가 되지만 완전히 가려지면 복원할 수 없습니다. 이 기능은 **현재 디버그 영상에만 표시**되며 장애물 줄 번호나 주행용 좌표를 메시지로 발행하지 않습니다.

`Obstacle preprocess - RED / BLUE` 창에서는 다음 순서로 확인합니다.

- `RAW`에서 물체가 검으면 해당 색 HSV 범위를 확인합니다.
- `RAW`에는 있는데 `CLEAN`에서 사라지면 노이즈 제거 또는 색 영역 분리 단계를 확인합니다.
- `CLEAN`에는 있는데 검출 수가 0이면 면적·사각형 적합도(`fit`)·재투영 오차(`reproj`)와 화면 경계 조건을 확인합니다.
- 색 영역은 맞지만 서로 붙은 면이라면 `~` 표시와 `color_split_estimate`를 확인하고 실측 거리와 비교합니다.

영상이 설정된 `image_timeout_s`(기본 1초) 이상 끊기면 `/vision/obstacles`에 `status=image_timeout`과 빈 목록을 발행하고 표시 영상을 지웁니다. `calibration_verified=false`인 동안에는 결과에 미검증 상태가 표시됩니다.

## 정확도와 보정

현재 [설정 파일](config/obstacle_distance.yaml)의 `K/D`는 Insta360 드라이버에 있던 값입니다. **실제 640 × 480 발행 영상에 맞는지는 검증되지 않았으므로** `calibration_verified: false`가 기본값입니다. 거리 노드는 드라이버의 CameraInfo를 그대로 쓰지 않고 YAML의 `K/D`를 읽습니다. 영상 해상도가 설정과 다르면 수치 출력을 중단합니다. `fit`이나 `reproj`가 좋아도 카메라 값이나 실제 면 크기가 틀리면 미터 거리도 틀립니다.

정면 1 m, 1.5 m, 2 m에서 렌즈 부근부터 **장애물 아랫변 중앙**까지 실측한 값과 `BOTTOM`을 비교하고, 좌우 및 비스듬한 위치에서도 확인합니다. `Y GROUND`와 `X FORWARD`는 렌즈 높이와 바닥 조건도 함께 확인해야 합니다. 화면 밖으로 잘린 면, 가려진 모서리, 실제 40 cm와 다른 물체, 같은 색 배경은 오검출 또는 거리 오차가 날 수 있습니다.

체커보드로 내부 파라미터를 다시 구하려면 가로·세로 **내부 교차점 수**와 실제 한 칸 길이를 사용합니다. 사용한 판이 내부 교차점 8 × 6개이고 한 칸이 **3 cm**인 경우의 실행 예시는 다음과 같습니다. 교차점 수는 실제 판에서 세어 확인하세요.

```bash
ros2 launch robot_vision calibrate.launch.py board_size:=8x6 square_size:=0.03
```

카메라가 이미 실행 중이면 `start_camera:=false`를 붙입니다. 체커보드 전체를 중앙과 화면 각 모서리, 여러 거리와 기울기에서 촬영하고 공식 도구의 X/Y/Size/Skew 수집 상태를 확인한 뒤 `CALIBRATE` → `SAVE`합니다. 저장된 결과와 실제 영상 크기를 확인하고 YAML의 `K/D`에 반영해야 합니다. 드라이버는 `set_camera_info` 서비스를 제공하지 않아 보정 도구의 `COMMIT` 자동 적용은 지원하지 않습니다. 초점·줌·크롭·해상도를 바꾸면 다시 검증해야 합니다.

검사 명령:

```bash
cd ~/colcon_ws
source /opt/ros/jazzy/setup.bash
colcon test --packages-select robot_vision
colcon test-result --verbose
```

참고: [OpenCV PnP 문서](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html), [ROS camera_calibration 도구](https://index.ros.org/p/camera_calibration/).
