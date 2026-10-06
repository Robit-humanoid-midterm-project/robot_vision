// 파일 역할: 한 프레임의 검출·추적·거리 계산을 연결한다.
// ROS 토픽이나 창을 직접 다루지 않고 VisionFrameResult에 결과를 모은다.

#include "robot_vision/vision_pipeline.hpp"

#include <stdexcept>
#include <utility>
#include <opencv2/imgproc.hpp>

namespace robot_vision {

// 검출 설정과 카메라 높이를 보관하고 거리 추정기·행 추적기를 초기화한다.
VisionPipeline::VisionPipeline(DetectorConfig detector, LaneConfig lane, double camera_height_m,
                               RowTrackerConfig rows)
    : detector_config_(std::move(detector)), lane_config_(std::move(lane)),
      camera_height_m_(camera_height_m), estimator_(detector_config_),
      row_tracker_(rows.history_frames, rows.match_y_px, rows.match_slope, rows.max_missing_frames)
{
}

// BGR 영상에서 장애물 색상 후보와 위치·거리 결과를 만든다.
// 다음 complete() 단계에 전달할 프레임 결과를 반환한다.
VisionFrameResult VisionPipeline::detect_obstacles(const cv::Mat &frame) const
{
    VisionFrameResult result;
    result.obstacles = estimator_.detect(frame);
    return result;
}

// 장애물 결과가 들어 있는 프레임에 차선, 평활화한 행, 경계 거리와 바닥 투영을 추가한다.
// 차선 계산 실패는 오류 문자열과 빈 마스크로 남겨 나머지 처리를 계속한다.
void VisionPipeline::complete(const cv::Mat &frame, VisionFrameResult &result)
{
    try {
        cv::Mat color_exclusion;
        // 검출한 색상 영역을 조금 넓혀 차선 검출에서 제외한다. 판 위의 밝은 부분을 흰 선으로 고르지 않게 한다.
        cv::cvtColor(result.obstacles.mask_preview, color_exclusion, cv::COLOR_BGR2GRAY);
        cv::dilate(color_exclusion, color_exclusion, cv::getStructuringElement(cv::MORPH_RECT, {7, 7}));
        result.lane = estimate_lane_line(frame, lane_config_, color_exclusion);
    } catch (const std::exception &error) {
        result.lane_error = error.what();
        result.lane.mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    }
    // 현재 영상의 판 아래쪽 선분으로 행을 찾고, 이전 프레임과 연결해 위치·기울기를 평균낸다.
    result.rows = row_tracker_.smooth(
        estimate_row_lines(result.obstacles.colors, result.obstacles.detections, frame.size()), frame.size());
    // 차선과 가장 가까운 행의 교차점을 이용해 좌우 경계 거리를 추정한다.
    result.geometry = estimate_field_geometry(
        result.obstacles, result.lane, result.rows, frame.size(), detector_config_);
    // 장애물 배열과 같은 순서로 바닥 투영 결과를 저장한다. 계산 불가한 항목은 nullopt로 남는다.
    result.ground_projections.clear();
    for (const auto &detection : result.obstacles.detections)
        result.ground_projections.push_back(project_to_ground(detection, camera_height_m_));
}

// 행 추적 이력을 비운다. 영상 끊김 감시 코드가 호출한다.
void VisionPipeline::reset_tracking()
{
    row_tracker_.reset();
}

} // namespace robot_vision
