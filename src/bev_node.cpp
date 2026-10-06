// 파일 역할: 왜곡 보정 영상에서 바닥 네 점을 선택하고 위에서 보는 BEV 영상을 미리 확인한다.
// 입력한 전방 거리와 검출한 좌우 선 간격으로 미터 좌표를 미리 확인한다.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <yaml-cpp/yaml.h>
#include "robot_vision/bev_line_detector.hpp"
#include "robot_vision/bev_metric.hpp"

namespace {
namespace fs = std::filesystem;
constexpr char kWindow[] = "Camera / BEV / White mask / Detected lines";

// YAML 배열의 길이와 유한한 숫자인지 검사하고 double 목록으로 반환한다.
std::vector<double> read_numbers(const YAML::Node &node, std::size_t expected, const char *name) {
  if (!node || !node.IsSequence() || node.size() != expected) {
    throw std::runtime_error(std::string(name) + " has the wrong number of values");
  }
  std::vector<double> values;
  for (const auto &item : node) {
    const double value = item.as<double>();
    if (!std::isfinite(value)) throw std::runtime_error(std::string(name) + " must be finite");
    values.push_back(value);
  }
  return values;
}

class BevPreview {
 public:
  // 설정 파일에서 영상 크기·보정값·BEV 크기를 읽고 왜곡 보정 맵을 한 번 만든다.
  // 저장된 네 점이 있으면 변환 행렬도 만들고 클릭 콜백을 창에 연결한다.
  explicit BevPreview(fs::path config_path) : config_path_(fs::absolute(std::move(config_path))) {
    const YAML::Node root = YAML::LoadFile(config_path_.string());
    const auto camera = root["obstacle_distance"]["ros__parameters"];
    const auto bev = root["camera_bev_preview"]["ros__parameters"];
    if (!camera || !bev) throw std::runtime_error("Both camera and BEV YAML sections are required");
    image_topic_ = camera["image_topic"].as<std::string>();
    size_ = {camera["calibration_width"].as<int>(), camera["calibration_height"].as<int>()};
    if (size_.width < 2 || size_.height < 2) throw std::runtime_error("Invalid calibration image size");
    const auto width_height = read_numbers(bev["bev_size"], 2, "bev_size");
    bev_size_ = {static_cast<int>(width_height[0]), static_cast<int>(width_height[1])};
    if (bev_size_.width < 2 || bev_size_.height < 2 || width_height[0] != bev_size_.width ||
        width_height[1] != bev_size_.height) throw std::runtime_error("Invalid BEV output size");
    metric_scale_.emplace(bev_size_, bev["bev_width_m"].as<double>(),
                          bev["near_edge_forward_m"].as<double>(),
                          bev["far_edge_forward_m"].as<double>());
    metric_reference_.emplace(bev_size_, bev["bev_width_m"].as<double>(),
                              bev["field_width_m"].as<double>(),
                              bev["robot_x_from_left_m"].as<double>());
    const auto intrinsics = read_numbers(camera["camera_matrix"], 9, "camera_matrix");
    const auto distortion = read_numbers(camera["distortion_coefficients"], 5, "distortion_coefficients");
    camera_matrix_ = cv::Mat(3, 3, CV_64F, const_cast<double *>(intrinsics.data())).clone();
    distortion_ = cv::Mat(1, 5, CV_64F, const_cast<double *>(distortion.data())).clone();
    // 같은 카메라 보정값을 반복 사용하므로 픽셀 이동 맵을 미리 계산해 매 프레임의 작업을 줄인다.
    cv::initUndistortRectifyMap(camera_matrix_, distortion_, cv::Mat(), camera_matrix_,
                                size_, CV_32FC1, map_x_, map_y_);
    if (bev["line_white_min_value"]) line_config_.white_min_value = bev["line_white_min_value"].as<int>();
    if (bev["line_white_max_saturation"])
      line_config_.white_max_saturation = bev["line_white_max_saturation"].as<int>();
    if (bev["line_min_grass_support"])
      line_config_.min_grass_support = bev["line_min_grass_support"].as<double>();
    if (camera["lane_grass_hue_min"])
      line_config_.grass_hue_min = camera["lane_grass_hue_min"].as<int>();
    if (camera["lane_grass_hue_max"])
      line_config_.grass_hue_max = camera["lane_grass_hue_max"].as<int>();
    if (camera["lane_grass_min_saturation"])
      line_config_.grass_min_saturation = camera["lane_grass_min_saturation"].as<int>();
    const auto stored = bev["source_points"];
    if (!stored || !stored.IsSequence() || (stored.size() != 0 && stored.size() != 8)) {
      throw std::runtime_error("source_points must be [] or eight x/y numbers");
    }
    if (stored.size() == 8) {
      const auto xy = read_numbers(stored, 8, "source_points");
      for (int i = 0; i < 4; ++i) points_.emplace_back(xy[2*i], xy[2*i+1]);
      update_homography();
    }
    cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(kWindow, &BevPreview::mouse_callback, this);
  }

  // 설정에서 읽은 카메라 입력 토픽 이름을 반환한다.
  const std::string &topic() const { return image_topic_; }

  // 입력 크기를 보정 기준과 비교하고, 점 선택으로 멈춘 상태가 아니면 왜곡 보정 영상을 갱신한다.
  void receive(const cv::Mat &raw) {
    if (raw.size() != size_) {
      throw std::runtime_error("Camera image size differs from calibration_width/calibration_height");
    }
    if (frozen_) return;
    const auto now = std::chrono::steady_clock::now();
    if (last_frame_time_ != std::chrono::steady_clock::time_point{} &&
        now - last_frame_time_ < std::chrono::milliseconds(66)) return;
    last_frame_time_ = now;
    raw_ = raw.clone();
    cv::remap(raw, undistorted_, map_x_, map_y_, cv::INTER_LINEAR);
    frame_pending_ = true;
  }

  // 선택한 점·좌표와 BEV 미리보기를 표시하고 키 입력을 처리한다.
  // S는 저장 후 실시간 복귀, R은 재선택, Q/ESC는 종료를 의미한다.
  bool draw() {
    if (!raw_.empty() && frame_pending_) {
      frame_pending_ = false;
      const int cell_w = std::max(size_.width, bev_size_.width);
      const int cell_h = std::max(size_.height, bev_size_.height);
      cv::Mat dashboard = cv::Mat::zeros(cell_h * 2, cell_w * 2, CV_8UC3);
      cv::Mat display = raw_.clone();
      for (std::size_t i = 0; i < points_.size(); ++i) {
        // Stored points are undistorted; project them back onto the raw view for display.
        std::vector<cv::Point3f> object{{float((points_[i].x - camera_matrix_.at<double>(0,2)) /
                                              camera_matrix_.at<double>(0,0)),
                                          float((points_[i].y - camera_matrix_.at<double>(1,2)) /
                                              camera_matrix_.at<double>(1,1)), 1.0f}};
        std::vector<cv::Point2f> projected;
        cv::projectPoints(object, cv::Vec3d(0,0,0), cv::Vec3d(0,0,0),
                          camera_matrix_, distortion_, projected);
        const cv::Point point(cvRound(projected[0].x), cvRound(projected[0].y));
        cv::circle(display, point, 5, {0, 0, 255}, cv::FILLED);
        const auto label = std::to_string(i+1);
        cv::putText(display, label, point + cv::Point(7, -7), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, {0, 255, 255}, 2);
      }
      display.copyTo(dashboard(cv::Rect(0, 0, size_.width, size_.height)));
      auto title = [](cv::Mat image, const std::string &label) {
        cv::rectangle(image, {0, 0}, {image.cols, 34}, {0, 0, 0}, cv::FILLED);
        cv::putText(image, label, {10, 24}, cv::FONT_HERSHEY_SIMPLEX,
                    0.65, {255, 255, 255}, 2);
      };
      cv::Mat raw_tile = dashboard(cv::Rect(0, 0, cell_w, cell_h));
      title(raw_tile, "RAW CAMERA (click 4 points)");
      if (!homography_.empty()) {
        cv::Mat bird_view;
        // 왜곡 보정된 영상을 선택한 바닥 사각형 기준으로 위에서 보는 형태로 변환한다.
        cv::warpPerspective(undistorted_, bird_view, homography_, bev_size_);
        bird_view.copyTo(dashboard(cv::Rect(cell_w, 0, bev_size_.width, bev_size_.height)));
        const auto detected = line_filter_.update(
          robot_vision::detect_bev_lines(bird_view, line_config_));
        cv::Mat white_view;
        cv::cvtColor(detected.white_mask, white_view, cv::COLOR_GRAY2BGR);
        white_view.copyTo(dashboard(cv::Rect(0, cell_h, bev_size_.width, bev_size_.height)));
        cv::Mat marked = bird_view.clone();
        const auto draw_line = [&](const robot_vision::BevLine &line, const char *label,
                                   const cv::Scalar &color) {
          if (!line.valid) return;
          cv::line(marked, line.top, line.bottom, color, 4, cv::LINE_AA);
          cv::putText(marked, label,
                      {cvRound(line.top.x) + 8, std::max(58, cvRound(line.top.y) + 20)},
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        };
        draw_line(detected.left, "LEFT", {0, 255, 0});
        draw_line(detected.right, "RIGHT", {0, 165, 255});
        marked.copyTo(dashboard(cv::Rect(cell_w, cell_h, bev_size_.width, bev_size_.height)));
        const float reference_y = (bev_size_.height - 1) * 0.5f;
        const double forward_m = metric_scale_->forward_at(reference_y);
        const auto line_x_at_reference = [&](const robot_vision::BevLine &line) -> std::optional<float> {
          if (!line.valid || line.held || reference_y < std::min(line.top.y, line.bottom.y) ||
              reference_y > std::max(line.top.y, line.bottom.y) ||
              std::abs(line.bottom.y - line.top.y) < 1.0f) return std::nullopt;
          const float x = line.top.x + (reference_y - line.top.y) *
                            (line.bottom.x - line.top.x) / (line.bottom.y - line.top.y);
          if (x < 0 || x > bev_size_.width - 1) return std::nullopt;
          return x;
        };
        const auto left_x = line_x_at_reference(detected.left);
        const auto right_x = line_x_at_reference(detected.right);
        metric_reference_->observe(left_x, right_x);
        const auto lateral_distance = [&](float line_x) -> std::string {
          std::ostringstream output;
          output << std::fixed << std::setprecision(2)
                 << std::abs(metric_scale_->at({line_x, reference_y}, metric_reference_->robot_x())
                             .lateral_from_robot_m) << "m";
          return output.str();
        };
        cv::rectangle(dashboard, {cell_w, cell_h + bev_size_.height - 79},
                      {cell_w + bev_size_.width, cell_h + bev_size_.height},
                      {0, 0, 0}, cv::FILLED);
        std::ostringstream depth_label;
        depth_label << "AT " << std::fixed << std::setprecision(2) << forward_m
                    << "m FORWARD  |  BEV WIDTH ~" << metric_scale_->width_m() << "m"
                    << (metric_reference_->ready() ? " LOCKED" : " WAIT LINE");
        cv::putText(dashboard, depth_label.str(),
                    {cell_w + 10, cell_h + bev_size_.height - 56},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 2);
        cv::putText(dashboard,
                    "LEFT: " + (metric_reference_->ready() && left_x ? lateral_distance(*left_x) : "--") +
                    "   RIGHT: " + (metric_reference_->ready() && right_x ? lateral_distance(*right_x) : "--"),
                    {cell_w + 10, cell_h + bev_size_.height - 33},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 2);
        const auto status = [](const robot_vision::BevLine &line) {
          return !line.valid ? "MISSING" : (line.held ? "HELD" : "FOUND");
        };
        cv::putText(dashboard,
                    std::string("L: ") + status(detected.left) +
                    "   R: " + status(detected.right),
                    {cell_w + 10, cell_h + bev_size_.height - 12},
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, {0, 0, 255}, 2);
      }
      title(dashboard(cv::Rect(cell_w, 0, cell_w, cell_h)), "BEV");
      title(dashboard(cv::Rect(0, cell_h, cell_w, cell_h)), "WHITE PREPROCESS");
      title(dashboard(cv::Rect(cell_w, cell_h, cell_w, cell_h)), "DETECTED LINES");
      cv::putText(dashboard, "S: save   R: reset   Q: quit", {8, cell_h - 10},
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 255}, 2);
      cv::imshow(kWindow, dashboard);
    }
    const int key = cv::waitKey(1) & 0xff;
    if (key == 'q' || key == 27) return false;
    if (key == 'r') {
      points_.clear();
      homography_.release();
      line_filter_.reset();
      metric_reference_->reset();
      frozen_ = false;
      frame_pending_ = true;
    }
    if (key == 's' && !homography_.empty()) {
      save_points();
      line_filter_.reset();
      metric_reference_->reset();
      frozen_ = false;
      frame_pending_ = true;
      std::cout << "Saved source_points to " << config_path_ << std::endl;
    }
    return true;
  }

 private:
  // OpenCV의 마우스 이벤트를 해당 BevPreview 객체의 click() 호출로 전달한다.
  static void mouse_callback(int event, int x, int y, int, void *self) {
    if (event == cv::EVENT_LBUTTONDOWN) static_cast<BevPreview *>(self)->click(x, y);
  }

  // 먼 왼쪽 → 먼 오른쪽 → 가까운 오른쪽 → 가까운 왼쪽 순서의 네 점을 기록한다.
  // 첫 클릭부터 영상을 멈춰 네 점이 모두 같은 프레임에서 선택되게 한다.
  void click(int x, int y) {
    if (undistorted_.empty() || points_.size() >= 4 || x < 0 || y < 0 ||
        x >= size_.width || y >= size_.height) return;
    frozen_ = true;  // Every point comes from the same frame.
    frame_pending_ = true;
    std::vector<cv::Point2f> rectified;
    cv::undistortPoints(std::vector<cv::Point2f>{{float(x), float(y)}}, rectified,
                        camera_matrix_, distortion_, cv::noArray(), camera_matrix_);
    points_.push_back(rectified[0]);
    std::cout << "Point " << points_.size() << ": [" << rectified[0].x << ", "
              << rectified[0].y << "]" << std::endl;
    if (points_.size() == 4) {
      try { update_homography(); }
      catch (const std::exception &e) { std::cerr << e.what() << "; press R to retry\n"; }
    }
  }

  // 유효한 볼록 사각형의 네 점을 BEV 출력 사각형으로 대응시켜 원근 변환 행렬을 만든다.
  void update_homography() {
    if (points_.size() != 4) return;
    for (const auto &p : points_) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 ||
          p.x >= size_.width || p.y >= size_.height)
        throw std::runtime_error("BEV point is outside the undistorted image");
    }
    if (!cv::isContourConvex(points_) || std::abs(cv::contourArea(points_)) < 100)
      throw std::runtime_error("Select a non-crossing convex quadrilateral");
    std::vector<cv::Point2f> destination{{0, 0}, {float(bev_size_.width - 1), 0},
      {float(bev_size_.width - 1), float(bev_size_.height - 1)}, {0, float(bev_size_.height - 1)}};
    homography_ = cv::getPerspectiveTransform(points_, destination);
  }

  // YAML의 BEV source_points 값만 바꾸고 나머지 설정과 주석은 보존한다.
  // 임시 파일에 완전히 쓴 뒤 원본을 교체해 저장 중 실패에 대비한다.
  void save_points() const {
    // Only replace the value on the source_points line; preserve all other YAML and comments.
    std::ifstream input(config_path_);
    if (!input) throw std::runtime_error("Cannot open BEV config for saving");
    std::string line, document;
    bool in_bev = false, replaced = false;
    while (std::getline(input, line)) {
      if (line == "camera_bev_preview:") in_bev = true;
      else if (!line.empty() && line[0] != ' ' && line[0] != '#') in_bev = false;
      if (in_bev && line.find("    source_points:") == 0) {
        if (replaced) throw std::runtime_error("Duplicate source_points in BEV section");
        std::ostringstream value;
        value << "    source_points: [";
        for (std::size_t i = 0; i < points_.size(); ++i) {
          if (i) value << ", ";
          value << points_[i].x << ", " << points_[i].y;
        }
        value << "]";
        line = value.str();
        replaced = true;
      }
      document += line + '\n';
    }
    if (!replaced) throw std::runtime_error("source_points line not found in BEV YAML section");
    // 다른 YAML 값은 건드리지 않고 완성된 내용만 마지막에 파일 이름 변경으로 반영한다.
    const auto temporary = fs::path(config_path_.string() + ".tmp");
    try {
      std::ofstream output(temporary, std::ios::trunc);
      output.exceptions(std::ios::failbit | std::ios::badbit);
      output << document;
      output.close();
      fs::permissions(temporary, fs::status(config_path_).permissions());
      fs::rename(temporary, config_path_);
    } catch (...) {
      std::error_code ignored;
      fs::remove(temporary, ignored);
      throw;
    }
  }

  fs::path config_path_;
  std::string image_topic_;
  cv::Size size_, bev_size_;
  cv::Mat camera_matrix_, distortion_, map_x_, map_y_, raw_, undistorted_, homography_;
  std::vector<cv::Point2f> points_;
  robot_vision::BevLineConfig line_config_;
  robot_vision::BevLineTemporalFilter line_filter_;
  std::optional<robot_vision::BevMetricScale> metric_scale_;
  std::optional<robot_vision::BevRobotReference> metric_reference_;
  std::chrono::steady_clock::time_point last_frame_time_{};
  bool frozen_{false}, frame_pending_{false};
};
}  // namespace

// 저장 이미지(--image) 또는 ROS 카메라 영상 중 입력 방식을 선택해 미리보기를 실행한다.
int main(int argc, char **argv) {
  try {
    fs::path config = fs::path(ament_index_cpp::get_package_share_directory("robot_vision")) /
                      "config/obstacle_distance.yaml";
    std::string image_file;
    std::vector<char *> ros_args{argv[0]};
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--config" && i+1 < argc) config = argv[++i];
      else if (arg == "--image" && i+1 < argc) image_file = argv[++i];
      else ros_args.push_back(argv[i]);
    }
    BevPreview preview(config);
    // 저장 이미지 모드는 ROS 영상 수신 없이 파일 한 장으로 점 선택을 수행한다.
    if (!image_file.empty()) {
      const cv::Mat image = cv::imread(image_file);
      if (image.empty()) throw std::runtime_error("Cannot read camera image: " + image_file);
      preview.receive(image);
      while (preview.draw()) preview.receive(image);
    } else {
      int ros_argc = static_cast<int>(ros_args.size());
      rclcpp::init(ros_argc, ros_args.data());
      auto node = std::make_shared<rclcpp::Node>("camera_bev_preview");
      // 실시간 모드에서는 최신 한 프레임만 받아 OpenCV 영상으로 변환한다.
      auto subscription = node->create_subscription<sensor_msgs::msg::CompressedImage>(
        preview.topic(), rclcpp::SensorDataQoS().keep_last(1),
        [&preview](const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
          const cv::Mat frame = cv::imdecode(msg->data, cv::IMREAD_COLOR);
          if (!frame.empty()) preview.receive(frame);
        });
      RCLCPP_INFO(node->get_logger(), "Waiting for %s (sensor_msgs/CompressedImage)",
                  preview.topic().c_str());
      while (rclcpp::ok() && preview.draw()) {
        try { rclcpp::spin_some(node); }
        catch (const rclcpp::exceptions::RCLError &) {
          if (!rclcpp::ok()) break;
          throw;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
      }
      rclcpp::shutdown();
    }
    cv::destroyAllWindows();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "BEV error: " << e.what() << '\n';
    cv::destroyAllWindows();
    return 1;
  }
}
