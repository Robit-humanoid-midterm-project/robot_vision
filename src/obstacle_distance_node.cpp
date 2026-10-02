#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
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

#include "humanoid_interfaces/msg/vision_data.hpp"
#include "robot_vision/distance_estimator.hpp"
#include "robot_vision/lane_line_estimator.hpp"
#include "robot_vision/msg/lane_line.hpp"
#include "robot_vision/msg/obstacle_array.hpp"
#include "robot_vision/msg/obstacle_detection.hpp"
#include "robot_vision/row_line_estimator.hpp"

namespace robot_vision
{
namespace
{

using SteadyClock = std::chrono::steady_clock;

std::string two_decimals(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

std::string signed_two_decimals(double value)
{
    std::ostringstream stream;
    stream << std::showpos << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

// 판별로 네 값을 묶어 표시한다. 바닥/전방 값은 기존 바닥 투영 결과를 사용한다.
// OpenCV 기본 글꼴은 한글을 지원하지 않아 화면에는 영문 항목명을 사용한다.
void draw_distance_label(cv::Mat &canvas, const Detection &detection, const std::optional<GroundProjection> &ground,
                         std::vector<cv::Rect> &occupied)
{
    if (canvas.empty() || !std::isfinite(detection.distance_m))
        return;
    cv::Point2d anchor = detection.corners[0];
    for (const auto &corner : detection.corners)
    {
        anchor.x = std::min(anchor.x, corner.x);
        anchor.y = std::min(anchor.y, corner.y);
    }
    if (!std::isfinite(anchor.x) || !std::isfinite(anchor.y))
        return;
    const std::vector<std::string> lines{detection.color + (detection.color_split_estimate ? " (est.)" : ""),
                                         "Range: " + two_decimals(detection.distance_m) + " m",
                                         "Ground: " + (ground ? two_decimals(ground->radial_m) + " m" : "N/A"),
                                         "Forward: " + (ground ? two_decimals(ground->forward_m) + " m" : "N/A"),
                                         "Lateral: " + signed_two_decimals(detection.position[0]) + " m"};
    constexpr double scale = 0.43;
    constexpr int line_height = 18, padding = 5;
    int width = 0, baseline = 0;
    for (const auto &line : lines)
        width = std::max(width, cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline).width);
    width = std::min(width + 2 * padding, canvas.cols);
    const int height = std::min(int(lines.size()) * line_height + 2 * padding, canvas.rows);
    const int min_y = std::min(70, canvas.rows - height);
    cv::Rect box(std::clamp(cvRound(anchor.x), 0, canvas.cols - width),
                 std::clamp(cvRound(anchor.y) - height - 5, min_y, canvas.rows - height), width, height);
    // 붙어 있는 판들의 정보창이 서로 덮이지 않도록 빈 자리를 찾는다.
    const auto overlaps = [&](const cv::Rect &candidate) {
        return std::any_of(occupied.begin(), occupied.end(),
                           [&](const cv::Rect &other) { return (candidate & other).area() > 0; });
    };
    if (overlaps(box))
    {
        bool placed = false;
        for (int y = min_y; y <= canvas.rows - height && !placed; y += line_height)
        {
            for (int x = 0; x <= canvas.cols - width; x += 20)
            {
                const cv::Rect candidate(x, y, width, height);
                if (!overlaps(candidate))
                {
                    box = candidate;
                    placed = true;
                    break;
                }
            }
        }
    }
    occupied.push_back(box);
    const cv::Scalar color = detection.color == "red" ? cv::Scalar(80, 110, 255) : cv::Scalar(255, 210, 80);
    // 정보창을 옮겨도 어느 판의 값인지 연결선으로 확인할 수 있다.
    cv::line(canvas, {box.x + box.width / 2, box.y + box.height / 2}, {cvRound(anchor.x), cvRound(anchor.y)}, color, 1,
             cv::LINE_AA);
    cv::rectangle(canvas, box, cv::Scalar(20, 20, 20), cv::FILLED);
    for (size_t i = 0; i < lines.size(); ++i)
        cv::putText(canvas, lines[i], {box.x + padding, box.y + padding + int(i + 1) * line_height - 4},
                    cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

} // namespace

class ObstacleDistanceNode final : public rclcpp::Node
{
  public:
    ObstacleDistanceNode() : rclcpp::Node("obstacle_distance")
    {
        rcl_interfaces::msg::ParameterDescriptor fixed;
        fixed.read_only = true;
        fixed.description = "Set at launch; restart to change";
        image_topic_ = declare_parameter<std::string>("image_topic", "/camera1/camera/compressed_image", fixed);
        optical_frame_id_ = declare_parameter<std::string>("optical_frame_id", "camera1_optical_frame", fixed);
        viewer_ = declare_parameter<bool>("viewer", true, fixed) &&
                  (std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr);
        show_preprocess_ = declare_parameter<bool>("show_preprocess", true, fixed);
        viewer_fullscreen_ = declare_parameter<bool>("viewer_fullscreen", true, fixed);
        max_processing_fps_ = declare_parameter<double>("max_processing_fps", 15.0, fixed);
        image_timeout_s_ = declare_parameter<double>("image_timeout_s", 1.0, fixed);
        if (max_processing_fps_ <= 0 || image_timeout_s_ <= 0)
        {
            throw std::invalid_argument("Frame rate and timeout must be positive");
        }

        DetectorConfig config;
        config.obstacle_size_m = declare_parameter<double>("obstacle_size_m", config.obstacle_size_m, fixed);
        camera_height_m_ = declare_parameter<double>("camera_height_m", 0.60, fixed);
        if (!std::isfinite(camera_height_m_) || camera_height_m_ <= 0)
        {
            throw std::invalid_argument("camera_height_m must be positive and finite");
        }
        config.calibration_width = declare_parameter<int>("calibration_width", config.calibration_width, fixed);
        config.calibration_height = declare_parameter<int>("calibration_height", config.calibration_height, fixed);
        config.calibration_verified = declare_parameter<bool>("calibration_verified", false, fixed);
        config.camera_matrix = declare_parameter<std::vector<double>>("camera_matrix", config.camera_matrix, fixed);
        config.distortion_coefficients =
            declare_parameter<std::vector<double>>("distortion_coefficients", config.distortion_coefficients, fixed);
        auto hsv_parameter = [&](const std::string &name, std::array<int, 3> defaults) {
            std::vector<int64_t> values{defaults[0], defaults[1], defaults[2]};
            values = declare_parameter<std::vector<int64_t>>(name, values, fixed);
            if (values.size() != 3)
                throw std::invalid_argument(name + " must contain three HSV values");
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
        config.near_min_edge_px = declare_parameter<double>("near_min_edge_px", config.near_min_edge_px, fixed);
        config.near_min_fill_ratio =
            declare_parameter<double>("near_min_fill_ratio", config.near_min_fill_ratio, fixed);
        config.near_max_reprojection_error_px =
            declare_parameter<double>("near_max_reprojection_error_px", config.near_max_reprojection_error_px, fixed);
        config.near_max_relative_reprojection_error = declare_parameter<double>(
            "near_max_relative_reprojection_error", config.near_max_relative_reprojection_error, fixed);
        config.max_reprojection_error_px =
            declare_parameter<double>("max_reprojection_error_px", config.max_reprojection_error_px, fixed);
        config.max_relative_reprojection_error =
            declare_parameter<double>("max_relative_reprojection_error", config.max_relative_reprojection_error, fixed);
        config.min_distance_m = declare_parameter<double>("min_distance_m", config.min_distance_m, fixed);
        config.max_distance_m = declare_parameter<double>("max_distance_m", config.max_distance_m, fixed);
        lane_config_.white_max_saturation = declare_parameter<int>("lane_white_max_saturation", 85, fixed);
        lane_config_.white_min_value = declare_parameter<int>("lane_white_min_value", 175, fixed);
        lane_config_.grass_hue_min = declare_parameter<int>("lane_grass_hue_min", 30, fixed);
        lane_config_.grass_hue_max = declare_parameter<int>("lane_grass_hue_max", 95, fixed);
        lane_config_.grass_min_saturation = declare_parameter<int>("lane_grass_min_saturation", 45, fixed);
        lane_config_.roi_top_fraction = declare_parameter<double>("lane_roi_top_fraction", 0.18, fixed);
        lane_config_.reference_y_fraction = declare_parameter<double>("lane_reference_y_fraction", 0.85, fixed);
        lane_config_.candidate_min_bottom_y_fraction =
            declare_parameter<double>("lane_candidate_min_bottom_y_fraction", 0.70, fixed);
        lane_config_.min_abs_dx_per_dy = declare_parameter<double>("lane_min_abs_dx_per_dy", 0.35, fixed);
        lane_config_.max_abs_dx_per_dy = declare_parameter<double>("lane_max_abs_dx_per_dy", 2.3, fixed);
        lane_config_.min_observed_height_fraction =
            declare_parameter<double>("lane_min_observed_height_fraction", 0.18, fixed);
        lane_config_.max_fit_error_px = declare_parameter<double>("lane_max_fit_error_px", 8.0, fixed);
        lane_config_.group_tolerance_px = declare_parameter<double>("lane_group_tolerance_px", 18.0, fixed);
        lane_config_.min_grass_support = declare_parameter<double>("lane_min_grass_support", 0.60, fixed);
        lane_config_.bottom_outer_fraction = declare_parameter<double>("lane_bottom_outer_fraction", 0.35, fixed);
        calibration_verified_ = config.calibration_verified;
        debug_config_ = config;
        estimator_ = std::make_unique<DistanceEstimator>(std::move(config));
        const int row_smoothing_frames = declare_parameter<int>("row_smoothing_frames", 5, fixed);
        const double row_match_y_px = declare_parameter<double>("row_match_y_px", 35.0, fixed);
        const double row_match_slope = declare_parameter<double>("row_match_slope", 0.18, fixed);
        const int row_track_max_missing_frames = declare_parameter<int>("row_track_max_missing_frames", 3, fixed);
        row_tracker_ = std::make_unique<RowLineTracker>(row_smoothing_frames, row_match_y_px, row_match_slope,
                                                        row_track_max_missing_frames);

        const auto image_qos = rclcpp::SensorDataQoS().keep_last(1);
        master_pub_ = create_publisher<humanoid_interfaces::msg::VisionData>("vision2master", rclcpp::QoS(10));
        output_pub_ = create_publisher<msg::ObstacleArray>("/vision/obstacles", rclcpp::QoS(1));
        lane_pub_ = create_publisher<msg::LaneLine>("/vision/lane_line", rclcpp::QoS(1));
        lane_mask_pub_ = create_publisher<sensor_msgs::msg::Image>("/vision/lane_mask", image_qos);
        slot_pub_ = create_publisher<sensor_msgs::msg::Image>("/vision/slot_map", image_qos);
        debug_pub_ = create_publisher<sensor_msgs::msg::Image>("/vision/obstacle_debug", image_qos);
        mask_pub_ = create_publisher<sensor_msgs::msg::Image>("/vision/obstacle_mask", image_qos);
        preprocess_pub_ = create_publisher<sensor_msgs::msg::Image>("/vision/obstacle_preprocess", image_qos);
        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic_, image_qos, [this](sensor_msgs::msg::Image::ConstSharedPtr image) { on_image(image); });
        watchdog_ = create_wall_timer(std::chrono::milliseconds(200), [this] { watchdog(); });
        RCLCPP_INFO(get_logger(), "Input: %s; outputs: /vision/obstacles, /vision/obstacle_debug",
                    image_topic_.c_str());
        if (!calibration_verified_)
        {
            RCLCPP_WARN(get_logger(), "UNVERIFIED legacy calibration: check measured distances.");
        }
        // timer = this->create_wall_timer(std::chrono::milliseconds(1000/15),
        // std::bind(&ObstacleDistanceNode::timer_callback, this));
    }

    ~ObstacleDistanceNode() override
    {
        if (viewer_)
        {
            cv::destroyWindow(window_name_);
            if (show_preprocess_)
                cv::destroyWindow(preprocess_window_name_);
        }
    }

  private:
    // rclcpp:TimeBase::SharedPtr timer;
    // void timer_callback()
    // {
    //     master_pub_->publish(message);
    // }
    // 임시 테스트 규약: obstacle_1[0]에 가장 가까운 판의 Range(m)를 보낸다.
    // 원래 y/x 상대거리 규약과 다르며, 미측정 필드는 모두 -1000으로 표시한다.
    void publish_master_test(const msg::ObstacleArray &array, bool frame_drop)
    {
        humanoid_interfaces::msg::VisionData message;
        message.timestamp = rclcpp::Time(array.header.stamp).seconds();
        message.frame_drop = frame_drop ? 1.0 : 0.0;
        message.camera_x = message.camera_y = -1000.0;
        message.left_x_1_dist = message.right_x_2_dist = message.theta = -1000.0;
        message.section_1.fill(-1000.0);
        message.section_2.fill(-1000.0);
        message.section_3.fill(-1000.0);
        message.section_1_detected = message.section_2_detected = message.section_3_detected = 0.0;
        message.nearest_line.fill(-1000.0);
        message.obstacle_1.fill(-1000.0);
        message.obstacle_2.fill(-1000.0);
        message.obstacle_3.fill(-1000.0);
        message.confidence = -1000.0;
        double closest = std::numeric_limits<double>::infinity();
        if (!frame_drop)
        {
            for (const auto &obstacle : array.detections)
            {
                if (std::isfinite(obstacle.distance_m) && obstacle.distance_m > 0)
                    closest = std::min(closest, obstacle.distance_m);
            }
        }
        if (std::isfinite(closest))
            message.obstacle_1[0] = closest;
        master_pub_->publish(message);
    }

    msg::ObstacleArray make_message(const builtin_interfaces::msg::Time &stamp, const std::string &status) const
    {
        msg::ObstacleArray array;
        array.header.stamp = stamp;
        array.header.frame_id = optical_frame_id_;
        array.calibration_verified = calibration_verified_;
        array.status = status;
        return array;
    }

    cv::Mat make_slot_view(const DetectResult &result, const std::vector<RowLine> &rows, const cv::Size &size) const
    {
        cv::Mat panel(600, 320, CV_8UC3, cv::Scalar(24, 27, 32));
        auto text = [&](const std::string &value, int x, int y, double scale = 0.48) {
            cv::putText(panel, value, {x, y}, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(225, 225, 225), 1,
                        cv::LINE_AA);
        };
        text("TOP VIEW - PREVIEW", 25, 30, 0.62);
        text("GOAL / FAR", 100, 64);
        text("L", 66, 94);
        text("C", 154, 94);
        text("R", 242, 94);
        // Screen thirds and visible-row order are only a visualization heuristic.
        // Do not publish this as field occupancy. Missing rows must not renumber slots.
        std::array<int, 9> slots{};
        if (rows.size() == 3 && size.width > 0)
        {
            for (const auto &candidate : result.image_candidates)
            {
                const double x = candidate.center.x;
                const double y = candidate.bounds.br().y - 1;
                int best = -1;
                double error = 25.0;
                for (size_t r = 0; r < rows.size(); ++r)
                {
                    const auto &row = rows[r];
                    const double dx = row.last.x - row.first.x;
                    if (std::abs(dx) < 1)
                        continue;
                    const double line_y = row.first.y + (x - row.first.x) * (row.last.y - row.first.y) / dx;
                    if (std::abs(y - line_y) < error)
                    {
                        error = std::abs(y - line_y);
                        best = static_cast<int>(r);
                    }
                }
                if (best < 0)
                    continue;
                const int col = std::clamp(static_cast<int>(3 * x / size.width), 0, 2);
                const int value = candidate.color == "red" ? 1 : 2;
                auto &slot = slots[best * 3 + col];
                slot = slot == 0 ? value : 3; // Multiple regions: ambiguous, never guess.
            }
        }
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                const cv::Rect cell(32 + c * 88, 112 + r * 116, 80, 104);
                cv::rectangle(panel, cell, cv::Scalar(65, 70, 78), cv::FILLED);
                cv::rectangle(panel, cell, cv::Scalar(140, 145, 150), 1);
                const int value = slots[r * 3 + c];
                if (value == 1 || value == 2)
                {
                    cv::rectangle(panel, {cell.x + 17, cell.y + 28, 46, 46},
                                  value == 1 ? cv::Scalar(45, 60, 235) : cv::Scalar(235, 140, 35), cv::FILLED);
                }
                else
                    text("?", cell.x + 29, cell.y + 63, 0.9);
            }
            text(std::to_string(3 - r), 9, 174 + r * 116);
        }
        text("START / ROBOT", 91, 494);
        text("Gray / ? = unknown", 28, 527);
        text("Image-based guess, not field pose", 14, 554, 0.40);
        text(rows.size() == 3 ? "3 visible rows: provisional layout" : "Need 3 visible rows to place markers", 14, 579,
             0.39);
        return panel;
    }

    void show_dashboard(const cv::Mat &raw, const cv::Mat &white_mask, const cv::Mat &obstacle_mask,
                        const cv::Mat &annotated)
    {
        if (!viewer_)
            return;
        if (!window_initialized_)
        {
            cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
            cv::resizeWindow(window_name_, 1520, 626);
            cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN,
                                  viewer_fullscreen_ ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
            window_initialized_ = true;
        }
        // Keep the annotated image large so contours stay readable.
        // The header sits outside the image and does not cover its status banner.
        cv::Mat dashboard(626, 1520, CV_8UC3, cv::Scalar(18, 18, 18));
        cv::Mat large_annotated, small_raw, combined_mask;
        cv::resize(annotated, large_annotated, {800, 600}, 0, 0, cv::INTER_CUBIC);
        cv::resize(raw, small_raw, {400, 300}, 0, 0, cv::INTER_AREA);
        if (!obstacle_mask.empty() && obstacle_mask.type() == CV_8UC3 && obstacle_mask.size() == raw.size())
        {
            combined_mask = obstacle_mask.clone();
        }
        else
        {
            combined_mask = cv::Mat::zeros(raw.size(), CV_8UC3);
        }
        if (!white_mask.empty() && white_mask.type() == CV_8UC1 && white_mask.size() == raw.size())
        {
            combined_mask.setTo(cv::Scalar::all(255), white_mask);
        }
        cv::resize(combined_mask, combined_mask, {400, 300}, 0, 0, cv::INTER_NEAREST);
        large_annotated.copyTo(dashboard(cv::Rect(0, 26, 800, 600)));
        combined_mask.copyTo(dashboard(cv::Rect(800, 26, 400, 300)));
        small_raw.copyTo(dashboard(cv::Rect(800, 326, 400, 300)));
        auto label = [&](const std::string &name, int x, int y) {
            cv::putText(dashboard, name, {x + 6, y + 19}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1,
                        cv::LINE_AA);
        };
        label("ANNOTATED RESULT", 0, 0);
        label("OPENCV MASK", 800, 0);
        cv::rectangle(dashboard, {800, 326}, {990, 352}, cv::Scalar(20, 20, 20), cv::FILLED);
        label("RAW CAMERA", 800, 326);
        const cv::Mat slot_panel = slot_view_.empty() ? make_slot_view(DetectResult{}, {}, {}) : slot_view_;
        slot_panel.copyTo(dashboard(cv::Rect(1200, 26, 320, 600)));
        label("3 x 3 LAYOUT", 1200, 0);
        cv::imshow(window_name_, dashboard);
        const int key = cv::waitKey(1) & 0xff;
        if (key == 'f' || key == 'F' || key == 27)
        {
            viewer_fullscreen_ = key == 27 ? false : !viewer_fullscreen_;
            cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN,
                                  viewer_fullscreen_ ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
            if (!viewer_fullscreen_)
                cv::resizeWindow(window_name_, 1520, 626);
        }
    }

    cv::Mat preprocess_view(const DetectResult &result) const
    {
        cv::Mat canvas(610, 640, CV_8UC3, cv::Scalar(22, 22, 22));
        auto put = [&](const std::string &label, int x, int y, double scale = 0.46) {
            cv::putText(canvas, label, {x, y}, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(235, 235, 235), 1,
                        cv::LINE_AA);
        };
        const auto &r = debug_config_;
        put("RED H " + std::to_string(r.red_lower_1[0]) + "-" + std::to_string(r.red_upper_1[0]) + ", " +
                std::to_string(r.red_lower_2[0]) + "-" + std::to_string(r.red_upper_2[0]) +
                "  S>= " + std::to_string(std::min(r.red_lower_1[1], r.red_lower_2[1])) +
                "  V>= " + std::to_string(std::min(r.red_lower_1[2], r.red_lower_2[2])),
            8, 19);
        put("BLUE H " + std::to_string(r.blue_lower[0]) + "-" + std::to_string(r.blue_upper[0]) +
                "  S>= " + std::to_string(r.blue_lower[1]) + "  V>= " + std::to_string(r.blue_lower[2]),
            8, 39);
        put("White = selected pixels | RAW = HSV | CLEAN = open + close", 8, 62, 0.42);
        for (int i = 0; i < 2; ++i)
        {
            const auto &color = result.colors[i];
            for (int stage = 0; stage < 2; ++stage)
            {
                const int x = stage * 320, y = 80 + i * 230;
                const cv::Mat &mask = stage == 0 ? color.raw_mask : color.cleaned_mask;
                put(std::string(i == 0 ? "RED " : "BLUE ") + (stage == 0 ? "RAW " : "CLEAN ") +
                        std::to_string(stage == 0 ? color.raw_pixels : color.cleaned_pixels) + " px",
                    x + 8, y + 16, 0.42);
                if (!mask.empty())
                {
                    cv::Mat scaled, rgb;
                    cv::resize(mask, scaled, {320, 210}, 0, 0, cv::INTER_NEAREST);
                    cv::cvtColor(scaled, rgb, cv::COLOR_GRAY2BGR);
                    rgb.copyTo(canvas(cv::Rect(x, y + 20, 320, 210)));
                }
            }
        }
        put("RED components " + std::to_string(result.colors[0].candidate_contours) + " detected " +
                std::to_string(result.colors[0].detections) + " | BLUE components " +
                std::to_string(result.colors[1].candidate_contours) + " detected " +
                std::to_string(result.colors[1].detections),
            8, 561, 0.42);
        int y = 584;
        for (const auto &detection : result.detections)
        {
            if (y > 604)
                break;
            put((detection.color == "red" ? "R" : "B") + std::string(" fit ") + two_decimals(detection.shape_fit) +
                    " reproj " + two_decimals(detection.reprojection_error_px) + " px" +
                    (detection.color_split_estimate ? " approx" : ""),
                8, y, 0.42);
            y += 18;
        }
        return canvas;
    }

    void publish_image(const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr &publisher, const cv::Mat &image,
                       const std_msgs::msg::Header &header)
    {
        auto converted = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, image).toImageMsg();
        publisher->publish(*converted);
    }

    void watchdog()
    {
        const auto now = SteadyClock::now();
        if (have_image_ && std::chrono::duration<double>(now - last_received_).count() <= image_timeout_s_)
        {
            return;
        }
        if (have_empty_ && std::chrono::duration<double>(now - last_empty_).count() < image_timeout_s_)
        {
            return;
        }
        row_tracker_->reset();
        last_empty_ = now;
        have_empty_ = true;
        auto array = make_message(get_clock()->now(), have_image_ ? "image_timeout" : "no_image");
        slot_view_ = make_slot_view(DetectResult{}, {}, {});
        publish_image(slot_pub_, slot_view_, array.header);
        publish_master_test(array, true);
        output_pub_->publish(array);
        lane_pub_->publish(msg::LaneLine().set__header(array.header));
        cv::Mat canvas(480, 640, CV_8UC3, cv::Scalar::all(0));
        cv::putText(canvas, "NO CAMERA IMAGE", {30, 200}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 200, 255), 2,
                    cv::LINE_AA);
        cv::putText(canvas, image_topic_, {15, 245}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
        publish_image(debug_pub_, canvas, array.header);
        publish_image(lane_mask_pub_, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0)), array.header);
        cv::Mat preprocess(610, 640, CV_8UC3, cv::Scalar::all(0));
        cv::putText(preprocess, "NO CAMERA IMAGE", {40, 300}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 200, 255), 2,
                    cv::LINE_AA);
        publish_image(preprocess_pub_, preprocess, array.header);
        if (viewer_ && show_preprocess_)
            cv::imshow(preprocess_window_name_, preprocess);
        show_dashboard(canvas, cv::Mat::zeros(canvas.size(), CV_8UC1), cv::Mat::zeros(canvas.size(), CV_8UC3), canvas);
    }

    void on_image(const sensor_msgs::msg::Image::ConstSharedPtr &image)
    {
        const auto now = SteadyClock::now();
        if (have_processed_ && std::chrono::duration<double>(now - last_processed_).count() < 1.0 / max_processing_fps_)
        {
            return;
        }
        have_processed_ = true;
        last_processed_ = now;
        cv::Mat frame;
        DetectResult result;
        try
        {
            frame = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8)->image;
            result = estimator_->detect(frame);
        }
        catch (const cv::Exception &error)
        {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 3000, "Image processing failed: %s", error.what());
            return;
        }
        catch (const std::exception &error)
        {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 3000, "Image processing failed: %s", error.what());
            return;
        }
        have_image_ = true;
        last_received_ = now;
        auto array = make_message(image->header.stamp, result.status);
        cv::Mat canvas = frame.clone();
        LaneResult lane;
        try
        {
            cv::Mat color_exclusion;
            cv::cvtColor(result.mask_preview, color_exclusion, cv::COLOR_BGR2GRAY);
            cv::dilate(color_exclusion, color_exclusion, cv::getStructuringElement(cv::MORPH_RECT, {7, 7}));
            lane = estimate_lane_line(frame, lane_config_, color_exclusion);
        }
        catch (const std::exception &error)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "Lane processing failed: %s", error.what());
            lane.mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        msg::LaneLine lane_message;
        lane_message.header = image->header;
        if (lane.best.valid)
        {
            const auto &line = lane.best;
            lane_message.valid = true;
            lane_message.side = line.side;
            lane_message.top.x = line.top.x;
            lane_message.top.y = line.top.y;
            lane_message.bottom.x = line.bottom.x;
            lane_message.bottom.y = line.bottom.y;
            lane_message.reference_y_px = line.reference_y_px;
            lane_message.line_x_at_reference_px = line.line_x_at_reference_px;
            lane_message.pixel_separation_px = line.pixel_separation_px;
            lane_message.observed_y_min = line.observed_y_min;
            lane_message.observed_y_max = line.observed_y_max;
            lane_message.grass_support = static_cast<float>(line.grass_support);
            lane_message.fit_error_px = static_cast<float>(line.fit_error_px);
            const cv::Scalar color = line.side == "left" ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
            const cv::Point2f delta = line.bottom - line.top;
            const double length = cv::norm(delta);
            if (length > 0)
            {
                for (double offset = 0; offset < length; offset += 18.0)
                {
                    const double end = std::min(offset + 10.0, length);
                    cv::line(canvas, line.top + delta * (offset / length), line.top + delta * (end / length), color, 2,
                             cv::LINE_AA);
                }
            }
            for (const auto &segment : line.observed_segments)
                cv::line(canvas, {segment[0], segment[1]}, {segment[2], segment[3]}, color, 3, cv::LINE_AA);
            const int ref_y = cvRound(line.reference_y_px);
            cv::circle(canvas, {cvRound(line.line_x_at_reference_px), ref_y}, 6, color, cv::FILLED, cv::LINE_AA);
        }
        lane_pub_->publish(lane_message);
        cv::Mat lane_mask_bgr;
        cv::cvtColor(lane.mask, lane_mask_bgr, cv::COLOR_GRAY2BGR);
        publish_image(lane_mask_pub_, lane_mask_bgr, image->header);
        const auto row_lines =
            row_tracker_->smooth(estimate_row_lines(result.colors, result.detections, frame.size()), frame.size());
        slot_view_ = make_slot_view(result, row_lines, frame.size());
        publish_image(slot_pub_, slot_view_, image->header);
        for (const auto &row : row_lines)
        {
            const cv::Point2d span = row.last - row.first;
            const double length = cv::norm(span);
            if (length > 0)
            {
                for (double offset = 0; offset < length; offset += 20.0)
                {
                    const double end = std::min(offset + 12.0, length);
                    cv::line(canvas,
                             row.first +
                                 cv::Point(cvRound(span.x * offset / length), cvRound(span.y * offset / length)),
                             row.first + cv::Point(cvRound(span.x * end / length), cvRound(span.y * end / length)),
                             cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
                }
            }
            for (const auto &segment : row.observed)
            {
                cv::line(canvas, segment.first, segment.last, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            }
        }
        // Draw observed color contours independently of metric pose acceptance.
        for (const auto &candidate : result.image_candidates)
        {
            const cv::Scalar color = candidate.color == "red" ? cv::Scalar(0, 80, 255) : cv::Scalar(255, 180, 0);
            const std::vector<std::vector<cv::Point>> contours{candidate.contour};
            cv::drawContours(canvas, contours, -1, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
            cv::drawContours(canvas, contours, -1, color, 2, cv::LINE_AA);
        }
        std::vector<cv::Rect> distance_labels;
        for (const auto &detection : result.detections)
        {
            const auto ground = project_to_ground(detection, camera_height_m_);
            draw_distance_label(canvas, detection, ground, distance_labels);
            msg::ObstacleDetection item;
            item.color = detection.color;
            item.position.x = detection.position[0];
            item.position.y = detection.position[1];
            item.position.z = detection.position[2];
            item.distance_m = detection.distance_m;
            if (ground)
            {
                item.ground_distance_m = ground->radial_m;
                item.ground_distance_valid = true;
                item.forward_distance_m = ground->forward_m;
                item.forward_distance_valid = true;
            }
            item.color_split_estimate = detection.color_split_estimate;
            item.reprojection_error_px = detection.reprojection_error_px;
            for (size_t i = 0; i < 4; ++i)
            {
                item.corners[i].x = static_cast<float>(detection.corners[i].x);
                item.corners[i].y = static_cast<float>(detection.corners[i].y);
                item.corners[i].z = 0;
            }
            array.detections.push_back(item);
        }
        cv::rectangle(canvas, {0, 0}, {canvas.cols, 46}, cv::Scalar(20, 20, 20), cv::FILLED);
        cv::putText(canvas, "COLOR CONTOURS", {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 220, 255), 1,
                    cv::LINE_AA);
        cv::putText(canvas,
                    "Visible regions: " + std::to_string(result.image_candidates.size()) +
                        " | row lines: " + std::to_string(row_lines.size()),
                    {8, 37}, cv::FONT_HERSHEY_SIMPLEX, 0.39, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        const std::string lane_note = lane.best.valid ? ("LANE " + lane.best.side) : "LANE not detected";
        cv::putText(canvas, lane_note + " | GREEN/YELLOW = base/row", {8, 62}, cv::FONT_HERSHEY_SIMPLEX, 0.39,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(canvas, lane_note + " | GREEN/YELLOW = base/row", {8, 62}, cv::FONT_HERSHEY_SIMPLEX, 0.39,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        publish_master_test(array, false);
        output_pub_->publish(array);
        publish_image(debug_pub_, canvas, image->header);
        publish_image(mask_pub_, result.mask_preview, image->header);
        const cv::Mat preprocess = preprocess_view(result);
        publish_image(preprocess_pub_, preprocess, image->header);
        if (viewer_ && show_preprocess_)
            cv::imshow(preprocess_window_name_, preprocess);
        show_dashboard(frame, lane.mask, result.mask_preview, canvas);
    }

    std::string image_topic_, optical_frame_id_;
    const std::string window_name_ = "Robot vision - RAW / OPENCV / RESULT";
    const std::string preprocess_window_name_ = "Obstacle preprocess - RED / BLUE";
    bool viewer_{false}, show_preprocess_{true}, viewer_fullscreen_{true};
    bool window_initialized_{false}, calibration_verified_{false};
    cv::Mat slot_view_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr slot_pub_;
    DetectorConfig debug_config_;
    LaneConfig lane_config_;
    double max_processing_fps_{15}, image_timeout_s_{1}, camera_height_m_{0.60};
    std::unique_ptr<DistanceEstimator> estimator_;
    std::unique_ptr<RowLineTracker> row_tracker_;
    rclcpp::Publisher<humanoid_interfaces::msg::VisionData>::SharedPtr master_pub_;
    rclcpp::Publisher<msg::ObstacleArray>::SharedPtr output_pub_;
    rclcpp::Publisher<msg::LaneLine>::SharedPtr lane_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr lane_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_, mask_pub_, preprocess_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::TimerBase::SharedPtr watchdog_;
    bool have_image_{false}, have_processed_{false}, have_empty_{false};
    SteadyClock::time_point last_received_{}, last_processed_{}, last_empty_{};
};

} // namespace robot_vision

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    try
    {
        auto node = std::make_shared<robot_vision::ObstacleDistanceNode>();
        rclcpp::spin(node);
    }
    catch (const std::exception &error)
    {
        RCLCPP_FATAL(rclcpp::get_logger("obstacle_distance"), "%s", error.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
