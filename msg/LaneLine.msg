std_msgs/Header header
bool valid
# "left", "right", or empty when no reliable line is visible.
string side
# All coordinates are pixels in the source image, not meters.
geometry_msgs/Point32 top
geometry_msgs/Point32 bottom
float32 reference_y_px
float32 line_x_at_reference_px
# Unsigned horizontal separation between image center and line at reference_y_px.
float32 pixel_separation_px
int32 observed_y_min
int32 observed_y_max
float32 grass_support
float32 fit_error_px
