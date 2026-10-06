// 파일 역할: 영상 처리기의 인터페이스와 행 추적 설정을 선언한다.

#pragma once

#include "robot_vision/vision_frame_result.hpp"

namespace robot_vision {

// 최근 몇 프레임을 평균낼지, 행을 같은 대상으로 볼 위치·기울기 오차와 미검출 허용치를 묶는다.
struct RowTrackerConfig {
    int history_frames{5};
    double match_y_px{35.0};
    double match_slope{0.18};
    int max_missing_frames{3};
};

// Calculates results only; ROS scheduling, publishing and rendering stay outside.
class VisionPipeline {
  public:
    // 보정·색상·차선·행 설정을 받아 계산기와 추적기를 준비한다.
    VisionPipeline(DetectorConfig detector, LaneConfig lane, double camera_height_m,
                   RowTrackerConfig rows = {});
    // BGR 영상의 장애물 결과를 만든다. 이어 complete()로 나머지 계산을 채운다.
    VisionFrameResult detect_obstacles(const cv::Mat &frame) const;
    // 전달된 프레임 결과에 차선·행·경계 거리·바닥 투영을 추가한다.
    void complete(const cv::Mat &frame, VisionFrameResult &result);
    // 영상 끊김 후 기존 행 이력을 버리는 외부 호출용 인터페이스이다.
    void reset_tracking();

  private:
    DetectorConfig detector_config_;
    LaneConfig lane_config_;
    double camera_height_m_;
    DistanceEstimator estimator_;
    RowLineTracker row_tracker_;
};

} // namespace robot_vision
