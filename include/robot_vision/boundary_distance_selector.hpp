#pragma once

#include <deque>
#include "robot_vision/field_geometry.hpp"

namespace robot_vision {
enum class BoundarySource { none, ground, crossing };

struct BoundarySelection {
    FieldGeometryResult geometry;
    BoundarySource source{BoundarySource::none};
    std::optional<double> ground_median_m, crossing_median_m;
};

// Each current valid source contributes one left-coordinate sample.
// Ground wins when both are valid. The other distance is always width-left.
class BoundaryDistanceSelector {
 public:
    explicit BoundaryDistanceSelector(double field_width_m = 1.4, double left_offset_m = 0.0);
    BoundarySelection select(const FieldGeometryResult &ground, const FieldGeometryResult &crossing);
    void reset();
 private:
    std::optional<double> left_coordinate(const FieldGeometryResult &value) const;
    static double update(std::deque<double> &history, double value);
    double width_m_, left_offset_m_;
    std::deque<double> ground_history_, crossing_history_;
};
} // namespace robot_vision
