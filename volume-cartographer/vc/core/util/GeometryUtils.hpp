#pragma once

#include <opencv2/core.hpp>

// Geometry and interpolation utility functions for Volume Cartographer
// This file consolidates duplicated utility functions from across the codebase

namespace vc::utils {

// ============================================================================
// Vector operations
// ============================================================================

/// Normalize a 3D vector
cv::Vec3f normed(const cv::Vec3f& v);

/// Component-wise minimum of two 2D vectors
cv::Vec2f vmin(const cv::Vec2f& a, const cv::Vec2f& b);

/// Component-wise maximum of two 2D vectors
cv::Vec2f vmax(const cv::Vec2f& a, const cv::Vec2f& b);

// ============================================================================
// Distance functions
// ============================================================================

/// Squared distance between two 3D points
float sdist(const cv::Vec3f& a, const cv::Vec3f& b);

/// Target distance (used in QuadSurface and SurfaceMeta)
float tdist(const cv::Vec3f& a, const cv::Vec3f& b, float td);

/// Sum of target distances
float tdist_sum(const cv::Vec3f& p, const std::vector<cv::Vec3f>& tgts, const std::vector<float>& tds);

// ============================================================================
// Bilinear interpolation
// ============================================================================

/// Bilinear interpolation at a floating-point location in a 2D grid
/// Template allows use with different element types (cv::Vec3f, cv::Vec2f, etc.)
template <typename E>
E at_int(const cv::Mat_<E>& points, cv::Vec2f p);

// ============================================================================
// Location validation
// ============================================================================

/// Check if a location is valid (not -1 and within bounds)
/// l is [y, x] format
template<typename T, int C>
bool loc_valid(const cv::Mat_<cv::Vec<T,C>>& m, const cv::Vec2d& l);

/// Check if a location is valid (not -1 and within bounds)
/// l is [x, y] format
template<typename T, int C>
bool loc_valid_xy(const cv::Mat_<cv::Vec<T,C>>& m, const cv::Vec2d& l);

/// Check if location is valid based on NaN values
/// l is [x, y] format
bool loc_valid_nan_xy(const cv::Mat_<cv::Vec3f>& m, const cv::Vec2f& l);

// ============================================================================
// Search and optimization
// ============================================================================

/// Find minimum location using gradient descent
/// out: resulting 3D point, loc: 2D location in grid, tgt: target point
/// z_search: whether to search in z direction
void min_loc(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f& loc, cv::Vec3f& out,
             cv::Vec3f tgt, bool z_search = true);

/// Templated version for different element types
template<typename E>
void min_loc_t(const cv::Mat_<E>& points, cv::Vec2f& loc, E& out, E tgt, bool z_search = true);

/// Multi-target version of min_loc
void min_loc_multi(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f& loc, cv::Vec3f& out,
                   const std::vector<cv::Vec3f>& tgts, const std::vector<float>& tds,
                   bool z_search = true, void* plane = nullptr);

// ============================================================================
// Coordinate system transformations
// ============================================================================
// NOTE: Volume Cartographer uses 3 coordinate systems:
// - Nominal (voxel volume) coordinates
// - Internal relative (ptr) coords (where _center is at 0/0)
// - Internal absolute (_points) coordinates where the upper left corner is at 0/0

/// Convert from nominal to internal coordinates
inline cv::Vec3f internal_loc(const cv::Vec3f& nominal, const cv::Vec3f& internal, const cv::Vec2f& scale)
{
    return internal + cv::Vec3f(nominal[0]*scale[0], nominal[1]*scale[1], nominal[2]);
}

/// Convert from internal to nominal coordinates
inline cv::Vec3f nominal_loc(const cv::Vec3f& nominal, const cv::Vec3f& internal, const cv::Vec2f& scale)
{
    return nominal + cv::Vec3f(internal[0]/scale[0], internal[1]/scale[1], internal[2]);
}

// ============================================================================
// Normal computation
// ============================================================================

/// Compute normal vector at a location in a point grid
cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f>& points, const cv::Vec3f& loc);

} // namespace vc::utils
