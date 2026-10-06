// 파일 역할: 판 하단 선분, 행 기준선과 시간에 따른 행 추적 인터페이스를 정의한다.

#pragma once

#include <array>
#include <deque>
#include <vector>

#include <opencv2/core.hpp>

#include "robot_vision/distance_estimator.hpp"

namespace robot_vision {

// 판 아래쪽에서 실제로 관측한 선분. 사각형 검출로 얻었는지도 함께 저장한다.
struct BottomSegment {
  cv::Point2f first;
  cv::Point2f last;
  bool from_square{false};
};

// 영상 폭으로 연장한 행 기준선과 그 선을 뒷받침하는 관측 선분 목록.
struct RowLine {
  cv::Point first;
  cv::Point last;
  std::vector<BottomSegment> observed;
  int square_support{0};
};

// Debug overlay only: extend visible square bases or lower color-mask edges.
// 색상 마스크와 검출 판의 밑변으로 현재 영상의 행 후보를 찾는다.
std::vector<RowLine> estimate_row_lines(
    const std::array<ColorDebug, 2> &colors,
    const std::vector<Detection> &detections,
    const cv::Size &image_size);

// A track is matched by image height and slope, never by obstacle color or slot.
// Only rows visible in the current frame are returned; old rows are not drawn.
// 행의 중앙 높이와 기울기를 프레임 사이에 연결해 흔들림을 줄인다.
class RowLineTracker {
 public:
  // 이력 프레임 수와 추적 대응 기준을 설정한다.
  RowLineTracker(int history_frames = 5, double match_y_px = 35.0,
                 double match_slope = 0.18, int max_missing_frames = 3);
  // 현재 보이는 행만 이전 이력과 평균내어 반환한다.
  std::vector<RowLine> smooth(const std::vector<RowLine> &rows,
                              const cv::Size &image_size);
  // 저장된 추적을 전부 버린다.
  void reset();

 private:
  // 한 행의 최근 (중앙 높이, 기울기) 기록과 마지막 관측 후 지난 프레임 수.
  struct Track {
    std::deque<cv::Vec2d> samples;  // center y, slope
    int missing_frames{0};
  };
  int history_frames_, max_missing_frames_;
  double match_y_px_, match_slope_;
  cv::Size previous_size_{};
  std::vector<Track> tracks_;
};

}  // namespace robot_vision
