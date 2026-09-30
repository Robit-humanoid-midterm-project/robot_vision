# Coordinates refer to the BOTTOM EDGE MIDPOINT of the visible 0.4 m square face.
string color
# Camera optical coordinates, meters: x right, y down, z forward.
geometry_msgs/Point position
# Euclidean distance from camera optical center to bottom-edge midpoint, meters.
float64 distance_m
# Ground-plane distance from the point directly below the lens to the base midpoint.
# Valid only if the base touches the same level floor and camera height is accurate.
float64 ground_distance_m
bool ground_distance_valid
# Forward component of ground distance along the fixed camera heading (X on screen).
# This is camera-relative, not an absolute field coordinate.
float64 forward_distance_m
bool forward_distance_valid
# True when a touching same-colored object required approximate color separation.
bool color_split_estimate
float64 reprojection_error_px
# Pixel coordinates in the source image, cyclic order.
geometry_msgs/Point32[4] corners
