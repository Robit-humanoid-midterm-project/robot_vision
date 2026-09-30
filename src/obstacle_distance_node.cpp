#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "robot_vision/distance_estimator.hpp"
#include "robot_vision/row_line_estimator.hpp"
#include "robot_vision/msg/obstacle_array.hpp"
#include "robot_vision/msg/obstacle_detection.hpp"

namespace robot_vision {
namespace {

using SteadyClock = std::chrono::steady_clock;

std::string two_decimals(double value) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(2) << value;
  return stream.str();
}

std::string signed_two_decimals(double value) {
  std::ostringstream stream;
  stream << std::showpos << std::fixed << std::setprecision(2) << value;
  return stream.str();
}

}  // namespace

class ObstacleDistanceNode final : public rclcpp::Node {
 public:
  ObstacleDistanceNode() : rclcpp::Node("obstacle_distance") {
    rcl_interfaces::msg::ParameterDescriptor fixed;
    fixed.read_only = true;
    fixed.description = "Set at launch; restart to change";
    image_topic_ = declare_parameter<std::string>("image_topic", "/camera1/camera/compressed_image", fixed);
    optical_frame_id_ = declare_parameter<std::string>("optical_frame_id", "camera1_optical_frame", fixed);
    viewer_ = declare_parameter<bool>("viewer", true, fixed) &&
              (std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr);
    show_preprocess_ = declare_parameter<bool>("show_preprocess", true, fixed);
    max_processing_fps_ = declare_parameter<double>("max_processing_fps", 15.0, fixed);
    image_timeout_s_ = declare_parameter<double>("image_timeout_s", 1.0, fixed);
    if (max_processing_fps_ <= 0 || image_timeout_s_ <= 0) {
      throw std::invalid_argument("Frame rate and timeout must be positive");
    }

    DetectorConfig config;
    config.obstacle_size_m = declare_parameter<double>("obstacle_size_m", config.obstacle_size_m, fixed);
    camera_height_m_ = declare_parameter<double>("camera_height_m", 0.75, fixed);
    if (!std::isfinite(camera_height_m_) || camera_height_m_ <= 0) {
      throw std::invalid_argument("camera_height_m must be positive and finite");
    }
    config.calibration_width = declare_parameter<int>("calibration_width", config.calibration_width, fixed);
    config.calibration_height = declare_parameter<int>("calibration_height", config.calibration_height, fixed);
    config.calibration_verified = declare_parameter<bool>("calibration_verified", false, fixed);
    config.camera_matrix = declare_parameter<std::vector<double>>("camera_matrix", config.camera_matrix, fixed);
    config.distortion_coefficients = declare_parameter<std::vector<double>>(
        "distortion_coefficients", config.distortion_coefficients, fixed);
    auto hsv_parameter = [&](const std::string &name, std::array<int, 3> defaults) {
      std::vector<int64_t> values{defaults[0], defaults[1], defaults[2]};
      values = declare_parameter<std::vector<int64_t>>(name, values, fixed);
      if (values.size() != 3) throw std::invalid_argument(name + " must contain three HSV values");
      return std::array<int, 3>{static_cast<int>(values[0]), static_cast<int>(values[1]),
                                static_cast<int>(values[2])};
    };
    config.red_lower_1 = hsv_parameter("red_lower_1", config.red_lower_1);
    config.red_upper_1 = hsv_parameter("red_upper_1", config.red_upper_1);
    config.red_lower_2 = hsv_parameter("red_lower_2", config.red_lower_2);
    config.red_upper_2 = hsv_parameter("red_upper_2", config.red_upper_2);
    config.blue_lower = hsv_parameter("blue_lower", config.blue_lower);
    config.blue_upper = hsv_parameter("blue_upper", config.blue_upper);
    config.min_area_px = declare_parameter<double>("min_area_px", config.min_area_px, fixed);
    config.min_edge_px = declare_parameter<double>("min_edge_px", config.min_edge_px, fixed);
    config.border_margin_px = declare_parameter<int>("border_margin_px", config.border_margin_px, fixed);
    config.min_fill_ratio = declare_parameter<double>("min_fill_ratio", config.min_fill_ratio, fixed);
    config.max_reprojection_error_px = declare_parameter<double>(
        "max_reprojection_error_px", config.max_reprojection_error_px, fixed);
    config.max_relative_reprojection_error = declare_parameter<double>(
        "max_relative_reprojection_error", config.max_relative_reprojection_error, fixed);
    config.min_distance_m = declare_parameter<double>("min_distance_m", config.min_distance_m, fixed);
    config.max_distance_m = declare_parameter<double>("max_distance_m", config.max_distance_m, fixed);
    calibration_verified_ = config.calibration_verified;
    debug_config_ = config;
    estimator_ = std::make_unique<DistanceEstimator>(std::move(config));

    const auto image_qos = rclcpp::SensorDataQoS().keep_last(1);
    output_pub_ = create_publisher<msg::ObstacleArray>("/vision/obstacles", rclcpp::QoS(1));
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("/vision/obstacle_debug", image_qos);
    mask_pub_ = create_publisher<sensor_msgs::msg::Image>("/vision/obstacle_mask", image_qos);
    preprocess_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/vision/obstacle_preprocess", image_qos);
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, image_qos,
        [this](sensor_msgs::msg::Image::ConstSharedPtr image) { on_image(image); });
    watchdog_ = create_wall_timer(std::chrono::milliseconds(200), [this] { watchdog(); });
    RCLCPP_INFO(get_logger(), "Input: %s; outputs: /vision/obstacles, /vision/obstacle_debug",
                image_topic_.c_str());
    if (!calibration_verified_) {
      RCLCPP_WARN(get_logger(), "UNVERIFIED legacy calibration: check measured distances.");
    }
  }

  ~ObstacleDistanceNode() override {
    if (viewer_) {
      cv::destroyWindow(window_name_);
      if (show_preprocess_) cv::destroyWindow(preprocess_window_name_);
    }
  }

 private:
  msg::ObstacleArray make_message(const builtin_interfaces::msg::Time &stamp,
                                  const std::string &status) const {
    msg::ObstacleArray array;
    array.header.stamp = stamp;
    array.header.frame_id = optical_frame_id_;
    array.calibration_verified = calibration_verified_;
    array.status = status;
    return array;
  }

  void show(const cv::Mat &image) const {
    if (viewer_) {
      cv::imshow(window_name_, image);
      cv::waitKey(1);
    }
  }

  cv::Mat preprocess_view(const DetectResult &result) const {
    cv::Mat canvas(610, 640, CV_8UC3, cv::Scalar(22, 22, 22));
    auto put = [&](const std::string &label, int x, int y, double scale = 0.46) {
      cv::putText(canvas, label, {x, y}, cv::FONT_HERSHEY_SIMPLEX,
                  scale, cv::Scalar(235, 235, 235), 1);
    };
    const auto &r = debug_config_;
    put("RED H " + std::to_string(r.red_lower_1[0]) + "-" +
        std::to_string(r.red_upper_1[0]) + ", " +
        std::to_string(r.red_lower_2[0]) + "-" +
        std::to_string(r.red_upper_2[0]) + "  S>= " +
        std::to_string(std::min(r.red_lower_1[1], r.red_lower_2[1])) +
        "  V>= " + std::to_string(std::min(r.red_lower_1[2], r.red_lower_2[2])), 8, 19);
    put("BLUE H " + std::to_string(r.blue_lower[0]) + "-" +
        std::to_string(r.blue_upper[0]) + "  S>= " +
        std::to_string(r.blue_lower[1]) + "  V>= " +
        std::to_string(r.blue_lower[2]), 8, 39);
    put("White = selected pixels | RAW = HSV | CLEAN = open + close", 8, 62, 0.42);
    for (int i = 0; i < 2; ++i) {
      const auto &color = result.colors[i];
      for (int stage = 0; stage < 2; ++stage) {
        const int x = stage * 320, y = 80 + i * 230;
        const cv::Mat &mask = stage == 0 ? color.raw_mask : color.cleaned_mask;
        put(std::string(i == 0 ? "RED " : "BLUE ") +
            (stage == 0 ? "RAW " : "CLEAN ") +
            std::to_string(stage == 0 ? color.raw_pixels : color.cleaned_pixels) + " px",
            x + 8, y + 16, 0.42);
        if (!mask.empty()) {
          cv::Mat scaled, rgb;
          cv::resize(mask, scaled, {320, 210}, 0, 0, cv::INTER_NEAREST);
          cv::cvtColor(scaled, rgb, cv::COLOR_GRAY2BGR);
          rgb.copyTo(canvas(cv::Rect(x, y + 20, 320, 210)));
        }
      }
    }
    put("RED components " + std::to_string(result.colors[0].candidate_contours) +
        " detected " + std::to_string(result.colors[0].detections) +
        " | BLUE components " + std::to_string(result.colors[1].candidate_contours) +
        " detected " + std::to_string(result.colors[1].detections), 8, 561, 0.42);
    int y = 584;
    for (const auto &detection : result.detections) {
      if (y > 604) break;
      put((detection.color == "red" ? "R" : "B") +
          std::string(" fit ") + two_decimals(detection.shape_fit) +
          " reproj " + two_decimals(detection.reprojection_error_px) + " px" +
          (detection.color_split_estimate ? " approx" : ""), 8, y, 0.42);
      y += 18;
    }
    return canvas;
  }

  void publish_image(const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr &publisher,
                     const cv::Mat &image, const std_msgs::msg::Header &header) {
    auto converted = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, image).toImageMsg();
    publisher->publish(*converted);
  }

  void watchdog() {
    const auto now = SteadyClock::now();
    if (have_image_ && std::chrono::duration<double>(now - last_received_).count() <= image_timeout_s_) {
      return;
    }
    if (have_empty_ && std::chrono::duration<double>(now - last_empty_).count() < image_timeout_s_) {
      return;
    }
    last_empty_ = now;
    have_empty_ = true;
    auto array = make_message(get_clock()->now(),
                              have_image_ ? "image_timeout" : "no_image");
    output_pub_->publish(array);
    cv::Mat canvas(480, 640, CV_8UC3, cv::Scalar::all(0));
    cv::putText(canvas, "NO CAMERA IMAGE", {30, 200}, cv::FONT_HERSHEY_SIMPLEX,
                0.8, cv::Scalar(0, 200, 255), 2);
    cv::putText(canvas, image_topic_, {15, 245}, cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(255, 255, 255), 1);
    publish_image(debug_pub_, canvas, array.header);
    cv::Mat preprocess(610, 640, CV_8UC3, cv::Scalar::all(0));
    cv::putText(preprocess, "NO CAMERA IMAGE", {40, 300},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 200, 255), 2);
    publish_image(preprocess_pub_, preprocess, array.header);
    show(canvas);
    if (viewer_ && show_preprocess_) {
      cv::imshow(preprocess_window_name_, preprocess);
      cv::waitKey(1);
    }
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr &image) {
    const auto now = SteadyClock::now();
    if (have_processed_ &&
        std::chrono::duration<double>(now - last_processed_).count() < 1.0 / max_processing_fps_) {
      return;
    }
    have_processed_ = true;
    last_processed_ = now;
    cv::Mat frame;
    DetectResult result;
    try {
      frame = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8)->image;
      result = estimator_->detect(frame);
    } catch (const cv::Exception &error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 3000, "Image processing failed: %s", error.what());
      return;
    } catch (const std::exception &error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 3000, "Image processing failed: %s", error.what());
      return;
    }
    have_image_ = true;
    last_received_ = now;
    auto array = make_message(image->header.stamp, result.status);
    cv::Mat canvas = frame.clone();
    const auto row_lines = estimate_row_lines(result.colors, result.detections,
                                              frame.size());
    for (const auto &row : row_lines) {
      const cv::Point2d span = row.last - row.first;
      const double length = cv::norm(span);
      if (length > 0) {
        for (double offset = 0; offset < length; offset += 20.0) {
          const double end = std::min(offset + 12.0, length);
          cv::line(canvas,
                   row.first + cv::Point(cvRound(span.x * offset / length),
                                         cvRound(span.y * offset / length)),
                   row.first + cv::Point(cvRound(span.x * end / length),
                                         cvRound(span.y * end / length)),
                   cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
        }
      }
      for (const auto &segment : row.observed) {
        cv::line(canvas, segment.first, segment.last,
                 cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
      }
    }
    bool any_split = false;
    for (const auto &detection : result.detections) {
      msg::ObstacleDetection item;
      item.color = detection.color;
      item.position.x = detection.position[0];
      item.position.y = detection.position[1];
      item.position.z = detection.position[2];
      item.distance_m = detection.distance_m;
      if (const auto ground = project_to_ground(detection, camera_height_m_)) {
        item.ground_distance_m = ground->radial_m;
        item.ground_distance_valid = true;
        item.forward_distance_m = ground->forward_m;
        item.forward_distance_valid = true;
      }
      item.color_split_estimate = detection.color_split_estimate;
      item.reprojection_error_px = detection.reprojection_error_px;
      const cv::Scalar color = item.color == "red" ? cv::Scalar(0, 50, 255) : cv::Scalar(255, 160, 0);
      std::vector<cv::Point> polygon;
      for (size_t i = 0; i < 4; ++i) {
        item.corners[i].x = static_cast<float>(detection.corners[i].x);
        item.corners[i].y = static_cast<float>(detection.corners[i].y);
        item.corners[i].z = 0;
        polygon.emplace_back(cvRound(detection.corners[i].x), cvRound(detection.corners[i].y));
      }
      array.detections.push_back(item);
      any_split = any_split || item.color_split_estimate;
      cv::polylines(canvas, std::vector<std::vector<cv::Point>>{polygon}, true, color, 2);
      int min_x = canvas.cols, min_y = canvas.rows;
      for (const auto &point : polygon) {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
      }
      const int x = std::clamp(min_x, 2, std::max(2, canvas.cols - 300));
      const int y = std::clamp(min_y, 70, std::max(70, canvas.rows - 82));
      const std::vector<std::string> labels{
          (item.color == "red" ? "RED" : "BLUE") + std::string("  BOTTOM ") +
              (item.color_split_estimate ? "~" : "") + two_decimals(item.distance_m) + " m",
          "X FORWARD " + (item.forward_distance_valid ?
              two_decimals(item.forward_distance_m) + " m" : "--"),
          "Y GROUND " + (item.ground_distance_valid ?
              two_decimals(item.ground_distance_m) + " m" : "--"),
          "L/R " + signed_two_decimals(item.position.x) + " m"};
      for (size_t i = 0; i < labels.size(); ++i) {
        const cv::Point origin{x, y + static_cast<int>(i) * 19};
        cv::putText(canvas, labels[i], origin, cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 3);
        cv::putText(canvas, labels[i], origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
      }
    }
    const std::string banner = result.status == "image_size_mismatch" ?
        "SIZE MISMATCH - NO RANGE" :
        (!calibration_verified_ ? "ESTIMATE - CALIBRATION NOT VERIFIED" :
                                  "40 cm face - camera optical coordinates");
    cv::rectangle(canvas, {0, 0}, {canvas.cols, 46}, cv::Scalar(20, 20, 20), cv::FILLED);
    cv::putText(canvas, banner, {8, 18}, cv::FONT_HERSHEY_SIMPLEX,
                0.48, cv::Scalar(0, 220, 255), 1);
    const std::string note = any_split ? "~ = touching colors, check distance" :
                                         "BOTTOM = lens to lower edge midpoint";
    cv::putText(canvas, "Detected: " + std::to_string(result.detections.size()) +
                " | row lines: " + std::to_string(row_lines.size()) + " | " + note,
                {8, 37}, cv::FONT_HERSHEY_SIMPLEX, 0.39, cv::Scalar(255, 255, 255), 1);
    cv::putText(canvas, "GREEN = visible base    YELLOW DASH = estimated row",
                {8, 62}, cv::FONT_HERSHEY_SIMPLEX, 0.39, cv::Scalar(0, 0, 0), 3);
    cv::putText(canvas, "GREEN = visible base    YELLOW DASH = estimated row",
                {8, 62}, cv::FONT_HERSHEY_SIMPLEX, 0.39, cv::Scalar(255, 255, 255), 1);
    output_pub_->publish(array);
    publish_image(debug_pub_, canvas, image->header);
    publish_image(mask_pub_, result.mask_preview, image->header);
    const cv::Mat preprocess = preprocess_view(result);
    publish_image(preprocess_pub_, preprocess, image->header);
    show(canvas);
    if (viewer_ && show_preprocess_) {
      cv::imshow(preprocess_window_name_, preprocess);
      cv::waitKey(1);
    }
  }

  std::string image_topic_, optical_frame_id_;
  const std::string window_name_ = "Obstacle distance - RED / BLUE - 40cm";
  const std::string preprocess_window_name_ = "Obstacle preprocess - RED / BLUE";
  bool viewer_{false}, show_preprocess_{true}, calibration_verified_{false};
  DetectorConfig debug_config_;
  double max_processing_fps_{15}, image_timeout_s_{1}, camera_height_m_{0.75};
  std::unique_ptr<DistanceEstimator> estimator_;
  rclcpp::Publisher<msg::ObstacleArray>::SharedPtr output_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_, mask_pub_, preprocess_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::TimerBase::SharedPtr watchdog_;
  bool have_image_{false}, have_processed_{false}, have_empty_{false};
  SteadyClock::time_point last_received_{}, last_processed_{}, last_empty_{};
};

}  // namespace robot_vision

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<robot_vision::ObstacleDistanceNode>();
    rclcpp::spin(node);
  } catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("obstacle_distance"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
