#pragma once

#include <opencv2/core/mat.hpp>
#include <optional>

#include "VCCollection.hpp"

// Forward declarations
class QuadSurface;
class VCCollection;
namespace cv {
template <typename _Tp> class Mat_;
}  // namespace cv

struct SurfaceMetricsResult {
    std::optional<float> in_surface_frac_valid;
    std::optional<float> surface_missing_fraction;
    std::optional<float> winding_valid_fraction;
};

SurfaceMetricsResult calc_point_winding_metrics(const VCCollection& collection, QuadSurface* surface, const cv::Mat_<float>& winding, int z_min, int z_max);
SurfaceMetricsResult calc_point_metrics(const VCCollection& collection, QuadSurface* surface, int z_min, int z_max);
