#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>

// Geometry utility functions
cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc);


// Bilinear interpolation at fractional coordinates
[[nodiscard, gnu::pure]] cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &p) noexcept;
[[nodiscard, gnu::pure]] float at_int(const cv::Mat_<float> &points, const cv::Vec2f& p) noexcept;
[[nodiscard, gnu::pure]] cv::Vec3d at_int(const cv::Mat_<cv::Vec3d> &points, const cv::Vec2f& p) noexcept;

// Check if location is valid (not -1) and within bounds
// l is [y, x]!
[[nodiscard, gnu::pure]] bool loc_valid(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l) noexcept;
[[nodiscard, gnu::pure]] bool loc_valid(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l) noexcept;
[[nodiscard, gnu::pure]] bool loc_valid(const cv::Mat_<float> &m, const cv::Vec2d &l) noexcept;

// Check if location is valid (not -1) and within bounds
// l is [x, y]!
[[nodiscard, gnu::pure]] bool loc_valid_xy(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l) noexcept;
[[nodiscard, gnu::pure]] bool loc_valid_xy(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l) noexcept;
[[nodiscard, gnu::pure]] bool loc_valid_xy(const cv::Mat_<float> &m, const cv::Vec2d &l) noexcept;


[[gnu::pure]] float tdist(const cv::Vec3f &a, const cv::Vec3f &b, float t_dist) noexcept;
[[gnu::pure]] float tdist_sum(const cv::Vec3f &v, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds) noexcept;

cv::Mat_<cv::Vec3f> clean_surface_outliers(
    const cv::Mat_<cv::Vec3f>& points,
    float distance_threshold = 5.0f,
    bool print_stats = false);