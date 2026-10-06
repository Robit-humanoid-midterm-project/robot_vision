#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "humanoid_interfaces/msg/vision_data.hpp"
#include "robot_vision/frame_rate_limiter.hpp"
#include "robot_vision/vision_pipeline.hpp"
#include "robot_vision/vision_viewer.hpp"
#include "robot_vision/vision_message_builder.hpp"
#include "robot_vision/lane_line_estimator.hpp"
#include "robot_vision/msg/lane_line.hpp"
#include "robot_vision/msg/obstacle_array.hpp"
#include "robot_vision/msg/obstacle_detection.hpp"

namespace robot_vision
{
namespace
{

using SteadyClock = std::chrono::steady_clock;

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
        max_processing_fps_ = declare_parameter<double>("max_processing_fps", 30.0, fixed);
        image_timeout_s_ = declare_parameter<double>("image_timeout_s", 1.0, fixed);
        if (!std::isfinite(max_processing_fps_) || !std::isfinite(image_timeout_s_) ||
            max_processing_fps_ <= 0 || image_timeout_s_ <= 0)
        {
            throw std::invalid_argument("Frame rate and timeout must be positive");
        }

        frame_limiter_ = std::make_unique<FrameRateLimiter>(max_processing_fps_);

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
        lane_config_.min_abs_dx_per_dy = declare_parameter<double>("lane_min_abs_dx_per_dy", 0.1, fixed);
        lane_config_.max_abs_dx_per_dy = declare_parameter<double>("lane_max_abs_dx_per_dy", 2.3, fixed);
        lane_config_.min_observed_height_fraction =
            declare_parameter<double>("lane_min_observed_height_fraction", 0.18, fixed);
        lane_config_.max_fit_error_px = declare_parameter<double>("lane_max_fit_error_px", 8.0, fixed);
        lane_config_.group_tolerance_px = declare_parameter<double>("lane_group_tolerance_px", 18.0, fixed);
        lane_config_.min_grass_support = declare_parameter<double>("lane_min_grass_support", 0.70, fixed);
        lane_config_.bottom_outer_fraction = declare_parameter<double>("lane_bottom_outer_fraction", lane_config_.bottom_outer_fraction, fixed);
        calibration_verified_ = config.calibration_verified;
        const int row_smoothing_frames = declare_parameter<int>("row_smoothing_frames", 5, fixed);
        const double row_match_y_px = declare_parameter<double>("row_match_y_px", 35.0, fixed);
        const double row_match_slope = declare_parameter<double>("row_match_slope", 0.18, fixed);
        const int row_track_max_missing_frames = declare_parameter<int>("row_track_max_missing_frames", 3, fixed);
        pipeline_ = std::make_unique<VisionPipeline>(config, lane_config_, camera_height_m_,
            RowTrackerConfig{row_smoothing_frames, row_match_y_px, row_match_slope, row_track_max_missing_frames});
        display_ = std::make_unique<VisionViewer>(config, viewer_, show_preprocess_, viewer_fullscreen_);

        const auto image_qos = rclcpp::SensorDataQoS().keep_last(1);
        master_pub_ = create_publisher<humanoid_interfaces::msg::VisionData>("vision2master", rclcpp::QoS(10));
        output_pub_ = create_publisher<msg::ObstacleArray>("/vision/obstacles", rclcpp::QoS(1));
        lane_pub_ = create_publisher<msg::LaneLine>("/vision/lane_line", rclcpp::QoS(1));
        lane_mask_pub_ = create_publisher<sensor_msgs::msg::Image>("/vision/lane_mask", image_qos);
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
    }

  private:
    static bool has_image_subscribers(const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr &publisher)
    {
        return publisher->get_subscription_count() + publisher->get_intra_process_subscription_count() > 0;
    }

    void record_performance(SteadyClock::time_point start, SteadyClock::time_point calculated,
                            SteadyClock::time_point rendered, SteadyClock::time_point finished)
    {
        ++stats_frames_;
        stats_calculate_ms_ += std::chrono::duration<double, std::milli>(calculated - start).count();
        stats_debug_ms_ += std::chrono::duration<double, std::milli>(rendered - calculated).count();
        stats_display_ms_ += std::chrono::duration<double, std::milli>(finished - rendered).count();
        const double elapsed = std::chrono::duration<double>(finished - stats_start_).count();
        if (elapsed < 5.0)
            return;
        const double fps = stats_frames_ / elapsed;
        display_->set_processing_fps(fps);
        RCLCPP_INFO(get_logger(), "Vision %.1f FPS; calculate %.1f ms; debug/publish %.1f ms; display %.1f ms (5s mean)",
            fps, stats_calculate_ms_ / stats_frames_, stats_debug_ms_ / stats_frames_, stats_display_ms_ / stats_frames_);
        stats_start_ = finished;
        stats_frames_ = 0;
        stats_calculate_ms_ = stats_debug_ms_ = stats_display_ms_ = 0.0;
    }

    void publish_image(const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr &publisher, const cv::Mat &image,
                       const std_msgs::msg::Header &header)
    {
        if (!has_image_subscribers(publisher))
            return;
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
        pipeline_->reset_tracking();
        last_empty_ = now;
        have_empty_ = true;
        auto array = make_obstacle_array(get_clock()->now(), optical_frame_id_, calibration_verified_,
                                         have_image_ ? "image_timeout" : "no_image");
        master_pub_->publish(make_master_message(array, true));
        output_pub_->publish(array);
        lane_pub_->publish(msg::LaneLine().set__header(array.header));
        const auto canvas = display_->no_image_view(image_topic_);
        publish_image(debug_pub_, canvas, array.header);
        publish_image(lane_mask_pub_, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0)), array.header);
        const auto preprocess = display_->no_image_preprocess();
        publish_image(preprocess_pub_, preprocess, array.header);
        display_->show(canvas, cv::Mat::zeros(canvas.size(), CV_8UC1),
                       cv::Mat::zeros(canvas.size(), CV_8UC3), canvas, preprocess);
    }

    void on_image(const sensor_msgs::msg::Image::ConstSharedPtr &image)
    {
        const auto now = SteadyClock::now();
        if (!frame_limiter_->should_process(now))
            return;
        cv::Mat frame;
        VisionFrameResult result;
        try
        {
            frame = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8)->image;
            result = pipeline_->detect_obstacles(frame);
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
        pipeline_->complete(frame, result);
        if (!result.lane_error.empty())
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                                 "Lane processing failed: %s", result.lane_error.c_str());

        const auto calculated = SteadyClock::now();
        lane_pub_->publish(make_lane_message(image->header, result.lane));
        auto array = make_obstacle_array(image->header.stamp, optical_frame_id_,
                                         calibration_verified_, result.obstacles.status);
        append_obstacle_detections(array, result);
        // Deliver control data before debug rendering and local GUI work.
        master_pub_->publish(make_master_message(array, false,
            result.geometry.left_distance_m, result.geometry.right_distance_m));
        output_pub_->publish(array);

        if (has_image_subscribers(lane_mask_pub_)) {
            cv::Mat lane_mask_bgr;
            cv::cvtColor(result.lane.mask, lane_mask_bgr, cv::COLOR_GRAY2BGR);
            publish_image(lane_mask_pub_, lane_mask_bgr, image->header);
        }
        cv::Mat canvas, preprocess;
        if (viewer_ || has_image_subscribers(debug_pub_)) {
            canvas = display_->annotate(frame, result);
            publish_image(debug_pub_, canvas, image->header);
        }
        publish_image(mask_pub_, result.obstacles.mask_preview, image->header);
        if ((viewer_ && show_preprocess_) || has_image_subscribers(preprocess_pub_)) {
            preprocess = display_->preprocess_view(result.obstacles);
            publish_image(preprocess_pub_, preprocess, image->header);
        }
        const auto rendered = SteadyClock::now();
        display_->show(frame, result.lane.mask, result.obstacles.mask_preview, canvas, preprocess);
        record_performance(now, calculated, rendered, SteadyClock::now());
    }

    std::string image_topic_, optical_frame_id_;
    bool viewer_{false}, show_preprocess_{true}, viewer_fullscreen_{true};
    bool calibration_verified_{false};
    LaneConfig lane_config_;
    double max_processing_fps_{30}, image_timeout_s_{1}, camera_height_m_{0.60};
    std::unique_ptr<FrameRateLimiter> frame_limiter_;
    std::unique_ptr<VisionPipeline> pipeline_;
    std::unique_ptr<VisionViewer> display_;
    rclcpp::Publisher<humanoid_interfaces::msg::VisionData>::SharedPtr master_pub_;
    rclcpp::Publisher<msg::ObstacleArray>::SharedPtr output_pub_;
    rclcpp::Publisher<msg::LaneLine>::SharedPtr lane_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr lane_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_, mask_pub_, preprocess_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::TimerBase::SharedPtr watchdog_;
    bool have_image_{false}, have_empty_{false};
    SteadyClock::time_point last_received_{}, last_empty_{};
    SteadyClock::time_point stats_start_{SteadyClock::now()};
    size_t stats_frames_{0};
    double stats_calculate_ms_{0.0}, stats_debug_ms_{0.0}, stats_display_ms_{0.0};
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
