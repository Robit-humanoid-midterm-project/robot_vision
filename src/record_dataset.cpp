// Save the existing Insta360 ROS image stream as raw-camera training video.
// The camera device is owned by insta360_usb_cam; this node only subscribes.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

namespace fs = std::filesystem;

class DatasetRecorder : public rclcpp::Node {
 public:
  DatasetRecorder(const fs::path &output, const std::string &topic, double fps, bool show_preview)
      : Node("robot_vision_dataset_recorder"), output_(fs::absolute(output)), fps_(fps),
        show_preview_(show_preview) {
    if (output_.extension() != ".avi" || !std::isfinite(fps_) || fps_ <= 0)
      throw std::invalid_argument("Output must end in .avi and FPS must be positive");
    if (fs::exists(output_))
      throw std::runtime_error("Refusing to overwrite existing recording: " + output_.string());
    fs::create_directories(output_.parent_path());
    timestamps_.open(output_.string() + ".timestamps.csv", std::ios::out | std::ios::trunc);
    if (!timestamps_) throw std::runtime_error("Cannot create timestamp file");
    timestamps_ << "frame,ros_time_ns\n";
    subscription_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        topic, rclcpp::SensorDataQoS().keep_last(30),
        [this](sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) { receive(msg); });
    if (show_preview_) {
      cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
      cv::Mat waiting(480, 640, CV_8UC3, cv::Scalar::all(0));
      cv::putText(waiting, "Waiting for camera...  Q: stop", {35, 240},
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, {255, 255, 255}, 2);
      cv::imshow(kWindow, waiting);
      cv::waitKey(1);
    }
    RCLCPP_INFO(get_logger(), "Waiting for %s; recording to %s", topic.c_str(),
                output_.string().c_str());
  }

  ~DatasetRecorder() override {
    writer_.release();
    timestamps_.close();
    if (show_preview_) {
      try { cv::destroyWindow(kWindow); }
      catch (const cv::Exception &) {}  // The user may have closed the window.
    }
    RCLCPP_INFO(get_logger(), "Saved %zu frames to %s", frames_, output_.string().c_str());
  }

  // Only the preview contains overlays; the video writer receives clean camera frames.
  bool show_frame() {
    if (!show_preview_) return true;
    if (!latest_preview_.empty()) {
      cv::Mat display = latest_preview_.clone();
      cv::rectangle(display, {0, 0}, {640, 47}, {0, 0, 0}, cv::FILLED);
      cv::circle(display, {20, 23}, 8, {0, 0, 255}, cv::FILLED);
      cv::putText(display, "REC " + output_.filename().string() + "  #" +
                    std::to_string(frames_) + "  Q: stop", {38, 30},
                  cv::FONT_HERSHEY_SIMPLEX, 0.58, {255, 255, 255}, 2);
      cv::imshow(kWindow, display);
    }
    const int key = cv::waitKey(1) & 0xff;
    if (key == 'q' || key == 'Q' || key == 27) return false;
    return cv::getWindowProperty(kWindow, cv::WND_PROP_VISIBLE) >= 1;
  }

 private:
  void receive(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
    try {
      const cv::Mat frame = cv::imdecode(msg->data, cv::IMREAD_COLOR);
      if (frame.empty()) throw std::runtime_error("Cannot decode camera JPEG");
      if (frame.cols != 640 || frame.rows != 480) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                              "Expected calibrated 640x480 images, got %dx%d; frame skipped",
                              frame.cols, frame.rows);
        return;
      }
      if (!writer_.isOpened()) {
        writer_.open(output_.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps_,
                     {640, 480}, true);
        if (!writer_.isOpened()) throw std::runtime_error("Cannot open MJPG AVI video writer");
        RCLCPP_INFO(get_logger(), "Recording started: 640x480, nominal %.1f FPS", fps_);
        started_ = std::chrono::steady_clock::now();
      }
      writer_.write(frame);
      if (show_preview_) latest_preview_ = frame.clone();
      const auto stamp_ns = static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
                            static_cast<int64_t>(msg->header.stamp.nanosec);
      timestamps_ << frames_ << ',' << stamp_ns << '\n';
      ++frames_;
      if (frames_ % 150 == 0) {
        timestamps_.flush();
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_).count();
        RCLCPP_INFO(get_logger(), "Recorded %zu frames (received %.1f FPS)", frames_,
                    seconds > 0 ? frames_ / seconds : 0.0);
      }
    } catch (const std::exception &error) {
      RCLCPP_ERROR(get_logger(), "Recording stopped: %s", error.what());
      rclcpp::shutdown();
    }
  }

  fs::path output_;
  double fps_;
  cv::VideoWriter writer_;
  std::ofstream timestamps_;
  std::size_t frames_{0};
  bool show_preview_;
  cv::Mat latest_preview_;
  std::chrono::steady_clock::time_point started_{};
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
  static constexpr const char *kWindow = "Insta360 recording preview";
};

int main(int argc, char **argv) {
  fs::path output;
  std::string topic = "/camera1/camera/compressed_image";
  double fps = 30.0;
  bool show_preview = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--output" && i + 1 < argc) output = argv[++i];
    else if (arg == "--topic" && i + 1 < argc) topic = argv[++i];
    else if (arg == "--fps" && i + 1 < argc) fps = std::stod(argv[++i]);
    else if (arg == "--no-preview") show_preview = false;
    else {
      std::cerr << "Usage: record_dataset --output PATH.avi [--topic ROS_TOPIC] [--fps 30] [--no-preview]\n";
      return 2;
    }
  }
  if (output.empty()) {
    std::cerr << "Usage: record_dataset --output PATH.avi [--topic ROS_TOPIC] [--fps 30] [--no-preview]\n";
    return 2;
  }
  try {
    rclcpp::init(argc, argv);
    auto recorder = std::make_shared<DatasetRecorder>(output, topic, fps, show_preview);
    while (rclcpp::ok() && recorder->show_frame()) {
      rclcpp::spin_some(recorder);
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    recorder.reset();
    if (rclcpp::ok()) rclcpp::shutdown();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Recording failed: " << error.what() << '\n';
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
}
