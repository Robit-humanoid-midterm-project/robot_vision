#include "robot_vision/boundary_distance_selector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace robot_vision {
BoundaryDistanceSelector::BoundaryDistanceSelector(double field_width_m, double left_offset_m)
    : width_m_(field_width_m), left_offset_m_(left_offset_m)
{
    if (!std::isfinite(width_m_) || width_m_ <= 0 || !std::isfinite(left_offset_m_))
        throw std::invalid_argument("Field width must be finite and positive");
}

std::optional<double> BoundaryDistanceSelector::left_coordinate(const FieldGeometryResult &value) const
{
    if (value.status != CrossingStatus::measured) return std::nullopt;
    const auto valid = [this](double d) { return std::isfinite(d) && d >= 0 && d <= width_m_; };
    if (valid(value.left_distance_m)) return value.left_distance_m;
    if (valid(value.right_distance_m)) return width_m_ - value.right_distance_m;
    return std::nullopt;
}

double BoundaryDistanceSelector::update(std::deque<double> &history, double value)
{
    history.push_back(value);
    if (history.size() > 3) history.pop_front();
    std::vector<double> sorted(history.begin(), history.end());
    std::sort(sorted.begin(), sorted.end());
    const auto mid = sorted.size()/2;
    return sorted.size()%2 ? sorted[mid] : (sorted[mid-1]+sorted[mid])*0.5;
}

BoundarySelection BoundaryDistanceSelector::select(
    const FieldGeometryResult &ground, const FieldGeometryResult &crossing)
{
    BoundarySelection output;
    // Preserve an explanation for the current failed frame, never a previous distance.
    output.geometry = ground.ground_based ? ground : crossing;
    output.geometry.left_distance_m = output.geometry.right_distance_m = -1000;
    const auto g = left_coordinate(ground), c = left_coordinate(crossing);
    if (g) output.ground_median_m = update(ground_history_, *g);
    if (c) output.crossing_median_m = update(crossing_history_, *c);
    if (!g && !c) {
        reset();
        return output;
    }
    output.source = g ? BoundarySource::ground : BoundarySource::crossing;
    output.geometry = g ? ground : crossing;
    const double left = g ? *output.ground_median_m : *output.crossing_median_m;
    // Apply once after source selection/smoothing; keep diagnostic medians raw.
    const double corrected = std::clamp(left + left_offset_m_, 0.0, width_m_);
    output.geometry.left_distance_m = corrected;
    output.geometry.right_distance_m = width_m_ - corrected;
    return output;
}

void BoundaryDistanceSelector::reset()
{
    ground_history_.clear();
    crossing_history_.clear();
}
} // namespace robot_vision
