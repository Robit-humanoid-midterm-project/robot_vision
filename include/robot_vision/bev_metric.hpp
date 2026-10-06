#pragma once

#include <cmath>
#include <optional>
#include <stdexcept>

#include <opencv2/core.hpp>

namespace robot_vision {

struct BevMetricPoint {
  double x_from_left_m{0};
  double lateral_from_robot_m{0};  // Right is positive; left is negative.
  double forward_from_robot_m{0};
};

// The measured BEV width fixes the horizontal scale independently of line detection.
class BevMetricScale {
 public:
  BevMetricScale(cv::Size image_size, double width_m, double near_forward_m,
                 double far_forward_m)
      : size_(image_size), width_m_(width_m), near_m_(near_forward_m), far_m_(far_forward_m) {
    if (size_.width < 2 || size_.height < 2 || !std::isfinite(width_m_) || width_m_ <= 0 ||
        !std::isfinite(near_m_) || !std::isfinite(far_m_) ||
        near_m_ < 0 || far_m_ <= near_m_)
      throw std::invalid_argument("Invalid BEV metric scale");
  }

  BevMetricPoint at(const cv::Point2f &pixel, double robot_x_pixel) const {
    if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) || pixel.x < 0 || pixel.y < 0 ||
        pixel.x > size_.width - 1 || pixel.y > size_.height - 1 ||
        !std::isfinite(robot_x_pixel) || robot_x_pixel < 0 || robot_x_pixel > size_.width - 1)
      throw std::out_of_range("BEV pixel is outside the measured rectangle");
    return {pixel.x * meters_per_horizontal_pixel(),
            (pixel.x - robot_x_pixel) * meters_per_horizontal_pixel(), forward_at(pixel.y)};
  }

  double forward_at(double pixel_y) const {
    if (!std::isfinite(pixel_y) || pixel_y < 0 || pixel_y > size_.height - 1)
      throw std::out_of_range("BEV row is outside the measured rectangle");
    return far_m_ - pixel_y * (far_m_ - near_m_) / (size_.height - 1);
  }

  double width_m() const { return width_m_; }
  double meters_per_horizontal_pixel() const { return width_m_ / (size_.width - 1); }
  double meters_per_vertical_pixel() const { return (far_m_ - near_m_) / (size_.height - 1); }

 private:
  cv::Size size_;
  double width_m_, near_m_, far_m_;
};

// A single white line can locate the robot's pixel origin using its measured
// distance from that line. Freeze the origin after consistent observations.
class BevRobotReference {
 public:
  BevRobotReference(cv::Size image_size, double bev_width_m, double line_gap_m,
                    double robot_from_left_line_m)
      : width_px_(image_size.width), meters_per_px_(bev_width_m / (image_size.width - 1)),
        line_gap_m_(line_gap_m), robot_from_left_m_(robot_from_left_line_m) {
    if (image_size.width < 2 || !std::isfinite(bev_width_m) || bev_width_m <= 0 ||
        !std::isfinite(line_gap_m_) || line_gap_m_ <= 0 ||
        !std::isfinite(robot_from_left_m_) || robot_from_left_m_ < 0 ||
        robot_from_left_m_ > line_gap_m_)
      throw std::invalid_argument("Invalid robot reference measurements");
  }

  void observe(std::optional<double> left_x, std::optional<double> right_x) {
    if (ready_ || (!left_x && !right_x)) return;
    if ((left_x && (!std::isfinite(*left_x) || *left_x < 0 || *left_x > width_px_ - 1)) ||
        (right_x && (!std::isfinite(*right_x) || *right_x < 0 || *right_x > width_px_ - 1))) return;
    const double from_left = left_x ? *left_x + robot_from_left_m_ / meters_per_px_ : 0.0;
    const double from_right = right_x ? *right_x - (line_gap_m_ - robot_from_left_m_) /
                                                  meters_per_px_ : 0.0;
    if (left_x && right_x && std::abs(from_left - from_right) > 15.0) return;
    const double candidate = left_x && right_x ? (from_left + from_right) / 2.0 :
                             left_x ? from_left : from_right;
    if (candidate < 0 || candidate > width_px_ - 1) return;
    if (samples_ > 0 && std::abs(candidate - robot_x_) > 12.0) reset();
    robot_x_ = (robot_x_ * samples_ + candidate) / (samples_ + 1);
    if (++samples_ >= 5) ready_ = true;
  }

  void reset() { robot_x_ = 0.0; samples_ = 0; ready_ = false; }
  bool ready() const { return ready_; }
  double robot_x() const { return robot_x_; }

 private:
  int width_px_;
  double meters_per_px_, line_gap_m_, robot_from_left_m_;
  double robot_x_{0.0};
  int samples_{0};
  bool ready_{false};
};

}  // namespace robot_vision
