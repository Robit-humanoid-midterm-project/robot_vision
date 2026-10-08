# Insta360 좌우 선 검출 실험

학습된 `/home/doyeon/Downloads/weights.pt`를 사용한다. 원본 영상과 좌우 세로선 분할 결과를 나란히 표시한다. 파란색은 왼쪽, 노란색은 오른쪽이다. 검출 영역 안에 관측된 선의 중심을 표시하며 거리 계산은 아직 하지 않는다.

터미널 1 — 카메라 연결 후:

```bash
source /home/doyeon/colcon_ws/install/setup.bash
ros2 launch insta360_usb_cam usb_cam.launch.py
```

터미널 2:

```bash
bash /home/doyeon/colcon_ws/src/robot_vision/scripts/run_line_preview.sh
```

`Q`/`Esc`: 종료. `S`: 현재 화면을 실행한 디렉터리에 저장.
기본 신뢰도는 0.50이며 실행 옵션으로 변경한다 (`--conf 0.3`, `--conf 0.61`).
최신 프레임만 처리해 처리 지연에 따른 영상 누적을 방지한다. 카메라가 끊기면 대기 화면으로 바뀐다.
CPU에서 실행하며 화면 상단에 처리 시간이 표시된다.

사진으로 테스트:

```bash
bash /home/doyeon/colcon_ws/src/robot_vision/scripts/run_line_preview.sh --image /path/to/image.jpg --output /tmp/lines.jpg
```


