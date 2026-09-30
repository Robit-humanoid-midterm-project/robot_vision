#include "robot_vision/row_line_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

#include <opencv2/imgproc.hpp>

namespace robot_vision {
namespace {

constexpr double kMinLengthPx = 20.0;
constexpr double kMaxSlope = 0.35;

struct Candidate {
  BottomSegment segment;
  double slope;
  double center_y;
  double weight;
};

void append_candidate(std::vector<Candidate> &out, const cv::Point2f &a,
                      const cv::Point2f &b, bool from_square, double center_x) {
  const double dx = b.x - a.x;
  const double dy = b.y - a.y;
  if (std::abs(dx) < kMinLengthPx || std::abs(dy / dx) > kMaxSlope) return;
  const double slope = dy / dx;
  const double center_y = a.y + slope * (center_x - a.x);
  const double weight = std::abs(dx) * (from_square ? 2.0 : 1.0);
  out.push_back({{a, b, from_square}, slope, center_y, weight});
}

bool color_above_background_below(const cv::Mat &component,
                                  const cv::Point2f &a,
                                  const cv::Point2f &b) {
  int supporting = 0;
  for (int i = 1; i <= 7; ++i) {
    const double fraction = static_cast<double>(i) / 8.0;
    const int x = cvRound(a.x + fraction * (b.x - a.x));
    const int y = cvRound(a.y + fraction * (b.y - a.y));
    if (x < 0 || x >= component.cols || y < 4 || y + 4 >= component.rows) continue;
    bool above = false, below = false;
    for (int offset = 2; offset <= 4; ++offset) {
      above = above || component.at<uint8_t>(y - offset, x) != 0;
      below = below || component.at<uint8_t>(y + offset, x) != 0;
    }
    if (above && !below) ++supporting;
  }
  return supporting >= 5;
}

}  // namespace

std::vector<RowLine> estimate_row_lines(
    const std::array<ColorDebug, 2> &colors,
    const std::vector<Detection> &detections,
    const cv::Size &image_size) {
  if (image_size.width <= 0 || image_size.height <= 0) return {};
  std::vector<Candidate> candidates;
  const double center_x = (image_size.width - 1) * 0.5;
  for (const auto &detection : detections) {
    append_candidate(candidates, detection.corners[3], detection.corners[2],
                     true, center_x);
  }
  for (const auto &color : colors) {
    const cv::Mat &mask = color.cleaned_mask;
    if (mask.empty() || mask.size() != image_size) continue;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours) {
      if (cv::contourArea(contour) < 500.0) continue;
      const cv::Rect bounds = cv::boundingRect(contour);
      if (bounds.width < 30 || bounds.height < 20) continue;
      cv::Mat component = cv::Mat::zeros(mask.size(), CV_8UC1);
      cv::drawContours(component, std::vector<std::vector<cv::Point>>{contour},
                       0, cv::Scalar(255), cv::FILLED);
      cv::bitwise_and(component, mask, component);
      const cv::Rect search = (cv::Rect(bounds.x - 6, bounds.y - 6,
                                        bounds.width + 12, bounds.height + 12) &
                               cv::Rect(0, 0, image_size.width, image_size.height));
      cv::Mat edges;
      cv::Canny(component(search), edges, 50, 120);
      std::vector<cv::Vec4i> lines;
      const double minimum_line = std::max(20.0, 0.08 * bounds.width);
      cv::HoughLinesP(edges, lines, 1.0, CV_PI / 180.0, 20,
                      minimum_line, 8.0);
      for (const auto &line : lines) {
        const cv::Point2f a(line[0] + search.x, line[1] + search.y);
        const cv::Point2f b(line[2] + search.x, line[3] + search.y);
        const double midpoint_y = (a.y + b.y) * 0.5;
        if (midpoint_y < bounds.y + 0.85 * bounds.height ||
            midpoint_y >= image_size.height - 4) continue;
        if (!color_above_background_below(component, a, b)) continue;
        append_candidate(candidates, a, b, false, center_x);
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              return a.center_y < b.center_y;
            });
  struct Group {
    double weighted_y{0};
    double weighted_slope{0};
    double total_weight{0};
    std::vector<BottomSegment> observed;
    int square_support{0};
  };
  std::vector<Group> groups;
  for (const auto &candidate : candidates) {
    auto match = groups.end();
    for (auto it = groups.begin(); it != groups.end(); ++it) {
      const double y = it->weighted_y / it->total_weight;
      const double slope = it->weighted_slope / it->total_weight;
      if (std::abs(candidate.center_y - y) < 18.0 &&
          std::abs(candidate.slope - slope) < 0.15) {
        match = it;
        break;
      }
    }
    if (match == groups.end()) {
      groups.emplace_back();
      match = std::prev(groups.end());
    }
    match->weighted_y += candidate.center_y * candidate.weight;
    match->weighted_slope += candidate.slope * candidate.weight;
    match->total_weight += candidate.weight;
    match->observed.push_back(candidate.segment);
    if (candidate.segment.from_square) ++match->square_support;
  }

  std::vector<RowLine> result;
  const cv::Rect frame(0, 0, image_size.width, image_size.height);
  for (const auto &group : groups) {
    const double y = group.weighted_y / group.total_weight;
    const double slope = group.weighted_slope / group.total_weight;
    cv::Point first(0, cvRound(y - slope * center_x));
    cv::Point last(image_size.width - 1,
                   cvRound(y + slope * (image_size.width - 1 - center_x)));
    if (!cv::clipLine(frame, first, last)) continue;
    result.push_back({first, last, group.observed, group.square_support});
  }
  return result;
}

}  // namespace robot_vision
