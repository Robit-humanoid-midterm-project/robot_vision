# All coordinates refer to the CENTER of the visible 0.4 m square face.
string color
# Camera optical coordinates, meters: x right, y down, z forward.
geometry_msgs/Point position
# Euclidean distance from camera optical center to face center, meters.
float64 distance_m
# True when a touching same-colored object required approximate color separation.
bool color_split_estimate
float64 reprojection_error_px
# Pixel coordinates in the source image, cyclic order.
geometry_msgs/Point32[4] corners
