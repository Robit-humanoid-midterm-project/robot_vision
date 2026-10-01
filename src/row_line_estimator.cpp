#include "robot_vision/row_line_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>

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

  struct RankedRow {
    RowLine row;
    double center_y;
    double slope;
    double score;
  };
  std::vector<RankedRow> ranked;
  const cv::Rect frame(0, 0, image_size.width, image_size.height);
  for (const auto &group : groups) {
    const double y = group.weighted_y / group.total_weight;
    const double slope = group.weighted_slope / group.total_weight;
    cv::Point first(0, cvRound(y - slope * center_x));
    cv::Point last(image_size.width - 1,
                   cvRound(y + slope * (image_size.width - 1 - center_x)));
    if (!cv::clipLine(frame, first, last)) continue;
    double visible_length = 0;
    for (const auto &segment : group.observed)
      visible_length += cv::norm(segment.last - segment.first);
    // A detected square base is stronger evidence than a mask-only edge.
    const double score = 200.0 * group.square_support + visible_length;
    ranked.push_back({{first, last, group.observed, group.square_support},
                      y, slope, score});
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const RankedRow &a, const RankedRow &b) {
              return a.score > b.score;
            });
  std::vector<RankedRow> selected;
  for (const auto &candidate : ranked) {
    bool duplicate = false;
    for (const auto &existing : selected) {
      if (std::abs(candidate.center_y - existing.center_y) < 24.0 &&
          std::abs(candidate.slope - existing.slope) < 0.15) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) selected.push_back(candidate);
    if (selected.size() == 3) break;  // The field has only three obstacle rows.
  }
  std::sort(selected.begin(), selected.end(),
            [](const RankedRow &a, const RankedRow &b) {
              return a.center_y < b.center_y;
            });
  std::vector<RowLine> result;
  for (auto &item : selected) result.push_back(std::move(item.row));
  return result;
}

RowLineTracker::RowLineTracker(int history_frames, double match_y_px,
                               double match_slope, int max_missing_frames)
    : history_frames_(history_frames), max_missing_frames_(max_missing_frames),
      match_y_px_(match_y_px), match_slope_(match_slope) {
  if (history_frames < 1 || history_frames > 30 || match_y_px <= 0 ||
      match_slope <= 0 || max_missing_frames < 0 || max_missing_frames > 30)
    throw std::invalid_argument("Invalid row smoothing parameters");
}

void RowLineTracker::reset() {
  tracks_.clear();
  previous_size_ = {};
}

std::vector<RowLine> RowLineTracker::smooth(const std::vector<RowLine> &rows,
                                           const cv::Size &image_size) {
  if (image_size.width <= 0 || image_size.height <= 0)
    throw std::invalid_argument("Invalid row image size");
  if (previous_size_ != image_size) reset();
  previous_size_ = image_size;
  for (auto &track : tracks_) ++track.missing_frames;
  const double center_x = (image_size.width - 1) * 0.5;
  std::vector<bool> used(tracks_.size(), false);
  std::vector<RowLine> output;
  const cv::Rect frame(0, 0, image_size.width, image_size.height);
  for (const auto &row : rows) {
    const double dx = row.last.x - row.first.x;
    if (std::abs(dx) < 1.0) continue;
    const double slope = (row.last.y - row.first.y) / dx;
    const double y = row.first.y + slope * (center_x - row.first.x);
    size_t match = tracks_.size();
    double best_cost = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < tracks_.size(); ++i) {
      if (used[i] || tracks_[i].samples.empty() ||
          tracks_[i].missing_frames > max_missing_frames_ + 1) continue;
      const auto &last = tracks_[i].samples.back();
      const double dy = std::abs(y - last[0]);
      const double ds = std::abs(slope - last[1]);
      if (dy >= match_y_px_ || ds >= match_slope_) continue;
      const double cost = dy + ds * 100.0;
      if (cost < best_cost) { match = i; best_cost = cost; }
    }
    if (match == tracks_.size()) {
      if (tracks_.size() == 3) {
        auto oldest = std::max_element(tracks_.begin(), tracks_.end(),
            [](const Track &a, const Track &b) {
              return a.missing_frames < b.missing_frames;
            });
        match = static_cast<size_t>(oldest - tracks_.begin());
        tracks_[match] = {};
      } else {
        tracks_.emplace_back();
        used.push_back(false);
        match = tracks_.size() - 1;
      }
    }
    used[match] = true;
    auto &track = tracks_[match];
    track.missing_frames = 0;
    track.samples.emplace_back(y, slope);
    while (track.samples.size() > static_cast<size_t>(history_frames_))
      track.samples.pop_front();
    double mean_y = 0, mean_slope = 0;
    for (const auto &sample : track.samples) {
      mean_y += sample[0]; mean_slope += sample[1];
    }
    mean_y /= track.samples.size();
    mean_slope /= track.samples.size();
    RowLine smoothed = row;
    smoothed.first = {0, cvRound(mean_y - mean_slope * center_x)};
    smoothed.last = {image_size.width - 1,
                     cvRound(mean_y + mean_slope * (image_size.width - 1 - center_x))};
    if (cv::clipLine(frame, smoothed.first, smoothed.last)) output.push_back(std::move(smoothed));
  }
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
              [&](const Track &track) {
                return track.missing_frames > max_missing_frames_;
              }), tracks_.end());
  return output;
}

}  // namespace robot_vision
