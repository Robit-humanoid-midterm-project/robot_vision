std_msgs/Header header
# False until calibration is verified at the configured reference resolution.
bool calibration_verified
# ok, unverified_calibration, image_size_mismatch, no_image, image_timeout
string status
robot_vision/ObstacleDetection[] detections
