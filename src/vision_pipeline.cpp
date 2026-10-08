// 파일 역할: 한 프레임의 검출·추적·거리 계산을 연결한다.
// ROS 토픽이나 창을 직접 다루지 않고 VisionFrameResult에 결과를 모은다.

#include "robot_vision/vision_pipeline.hpp"

#include <stdexcept>
#include <utility>
#include <opencv2/imgproc.hpp>

namespace robot_vision {

// 검출 설정과 카메라 높이를 보관하고 거리 추정기·행 추적기를 초기화한다.
VisionPipeline::VisionPipeline(DetectorConfig detector, LaneConfig lane, double camera_height_m,
                               RowTrackerConfig rows, std::optional<GroundFieldConfig> ground)
    : detector_config_(std::move(detector)), lane_config_(std::move(lane)),
      camera_height_m_(camera_height_m), estimator_(detector_config_),
      row_tracker_(rows.history_frames, rows.match_y_px, rows.match_slope, rows.max_missing_frames),
      boundary_selector_(ground ? ground->field_width_m : 1.4)
{
    if (ground) ground_estimator_.emplace(detector_config_, std::move(*ground));
    // 장애물 거리 계산과 동일한 기존 보정값을 사용한다. 별도 보정 설정을 만들지 않는다.
    lane_camera_matrix_ = cv::Mat(3, 3, CV_64F, detector_config_.camera_matrix.data()).clone();
    lane_distortion_ = cv::Mat(1, static_cast<int>(detector_config_.distortion_coefficients.size()),
                               CV_64F, detector_config_.distortion_coefficients.data()).clone();
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
void VisionPipeline::complete(const cv::Mat &frame, VisionFrameResult &result, const LaneResult &lanes)
{
    // 선은 원본 프레임과 함께 받은 YOLO 결과만 사용한다. HSV/Hough 재검출은 하지 않는다.
    result.lane = lanes;
    if (result.lane.mask.empty())
        result.lane.mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    // 현재 영상의 판 아래쪽 선분으로 행을 찾고, 이전 프레임과 연결해 위치·기울기를 평균낸다.
    result.rows = row_tracker_.smooth(
        estimate_row_lines(result.obstacles.colors, result.obstacles.detections, frame.size()), frame.size());
    // Both measurements use exactly this frame; a failure in one does not suppress the other.
    result.ground_geometry = ground_estimator_ ? ground_estimator_->estimate(result.lane, frame.size()) :
        FieldGeometryResult{};
    result.crossing_geometry = estimate_field_geometry(
        result.obstacles, result.lane, result.rows, frame.size(), detector_config_);
    const auto selected = boundary_selector_.select(result.ground_geometry, result.crossing_geometry);
    result.geometry = selected.geometry;
    result.boundary_source = selected.source;
    result.ground_left_median_m = selected.ground_median_m;
    result.crossing_left_median_m = selected.crossing_median_m;
    // 장애물 배열과 같은 순서로 바닥 투영 결과를 저장한다. 계산 불가한 항목은 nullopt로 남는다.
    result.ground_projections.clear();
    for (const auto &detection : result.obstacles.detections)
        result.ground_projections.push_back(project_to_ground(detection, camera_height_m_));
}

// 행 추적 이력을 비운다. 영상 끊김 감시 코드가 호출한다.
void VisionPipeline::reset_tracking()
{
    row_tracker_.reset();
    boundary_selector_.reset();
    if (ground_estimator_) ground_estimator_->clear_pending_reference();
}

cv::Mat VisionPipeline::bev_preview(const cv::Mat &raw, const LaneResult &lanes) const
{
    return ground_estimator_ ? ground_estimator_->preview(raw, lanes) : cv::Mat{};
}

} // namespace robot_vision
