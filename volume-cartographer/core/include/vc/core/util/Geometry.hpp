#pragma once

#include <opencv2/core.hpp>

// Geometry utility functions

// Bilinear interpolation at fractional coordinates
cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &p);
float at_int(const cv::Mat_<float> &points, const cv::Vec2f& p);
cv::Vec3d at_int(const cv::Mat_<cv::Vec3d> &points, const cv::Vec2f& p);

// Check if location is valid (not -1) and within bounds
// l is [y, x]!
bool loc_valid(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l);
bool loc_valid(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l);
bool loc_valid(const cv::Mat_<float> &m, const cv::Vec2d &l);

// Check if location is valid (not -1) and within bounds
// l is [x, y]!
bool loc_valid_xy(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l);
bool loc_valid_xy(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l);
bool loc_valid_xy(const cv::Mat_<float> &m, const cv::Vec2d &l);