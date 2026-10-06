// 파일 역할: 왜곡 보정 영상에서 바닥 네 점을 선택하고 위에서 보는 BEV 영상을 미리 확인한다.
// 실측 경기장 좌표나 로봇 제어 토픽을 계산하는 도구는 아니다.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>

namespace {
namespace fs = std::filesystem;
constexpr char kWindow[] = "Undistorted - select 4 points";
constexpr char kBevWindow[] = "BEV - pixels only";

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
    const auto intrinsics = read_numbers(camera["camera_matrix"], 9, "camera_matrix");
    const auto distortion = read_numbers(camera["distortion_coefficients"], 5, "distortion_coefficients");
    cv::Mat k(3, 3, CV_64F, const_cast<double *>(intrinsics.data()));
    cv::Mat d(1, 5, CV_64F, const_cast<double *>(distortion.data()));
    // 같은 카메라 보정값을 반복 사용하므로 픽셀 이동 맵을 미리 계산해 매 프레임의 작업을 줄인다.
    cv::initUndistortRectifyMap(k, d, cv::Mat(), k, size_, CV_32FC1, map_x_, map_y_);
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
    if (!frozen_) cv::remap(raw, undistorted_, map_x_, map_y_, cv::INTER_LINEAR);
  }

  // 선택한 점·좌표와 BEV 미리보기를 표시하고 키 입력을 처리한다.
  // S는 저장 후 실시간 복귀, R은 재선택, Q/ESC는 종료를 의미한다.
  bool draw() {
    if (!undistorted_.empty()) {
      cv::Mat display = undistorted_.clone();
      for (std::size_t i = 0; i < points_.size(); ++i) {
        const cv::Point point(cvRound(points_[i].x), cvRound(points_[i].y));
        cv::circle(display, point, 5, {0, 0, 255}, cv::FILLED);
        const auto label = std::to_string(i+1) + ": " + std::to_string(point.x) + "," + std::to_string(point.y);
        cv::putText(display, label, point + cv::Point(7, -7), cv::FONT_HERSHEY_SIMPLEX,
                    0.45, {0, 255, 255}, 1);
      }
      cv::putText(display, "Click: far-left, far-right, near-right, near-left",
                  {8, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 255, 255}, 1);
      cv::putText(display, "S: save + live  R: reset  Q: quit",
                  {8, 42}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 255}, 1);
      cv::imshow(kWindow, display);
      if (!homography_.empty()) {
        cv::Mat bird_view;
        // 왜곡 보정된 영상을 선택한 바닥 사각형 기준으로 위에서 보는 형태로 변환한다.
        cv::warpPerspective(undistorted_, bird_view, homography_, bev_size_);
        cv::imshow(kBevWindow, bird_view);
      }
    }
    const int key = cv::waitKey(10) & 0xff;
    if (key == 'q' || key == 27) return false;
    if (key == 'r') {
      points_.clear();
      homography_.release();
      frozen_ = false;
      cv::imshow(kBevWindow, cv::Mat::zeros(bev_size_, CV_8UC3));
    }
    if (key == 's' && !homography_.empty()) {
      save_points();
      frozen_ = false;
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
    points_.emplace_back(x, y);
    std::cout << "Point " << points_.size() << ": [" << x << ", " << y << "]" << std::endl;
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
  cv::Mat map_x_, map_y_, undistorted_, homography_;
  std::vector<cv::Point2f> points_;
  bool frozen_{false};
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
      auto subscription = node->create_subscription<sensor_msgs::msg::Image>(
        preview.topic(), rclcpp::SensorDataQoS().keep_last(1),
        [&preview](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
          preview.receive(cv_bridge::toCvCopy(msg, "bgr8")->image);
        });
      RCLCPP_INFO(node->get_logger(), "Waiting for %s (sensor_msgs/Image)", preview.topic().c_str());
      while (rclcpp::ok() && preview.draw()) {
        try { rclcpp::spin_some(node); }
        catch (const rclcpp::exceptions::RCLError &) {
          if (!rclcpp::ok()) break;
          throw;
        }
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
