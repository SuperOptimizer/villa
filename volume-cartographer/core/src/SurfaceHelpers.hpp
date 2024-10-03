#pragma once

#include <opencv2/core.hpp>


cv::Mat_<cv::Vec3f> upsample_with_grounding(const cv::Mat_<cv::Vec3f> &small, const cv::Size &tgt_size, const cv::Mat_<cv::Vec3f> &points, double sx, double sy);
cv::Mat_<cv::Vec3f> derive_regular_region_largesteps(const cv::Mat_<cv::Vec3f> &points, int seed_x, int seed_y, float step_size, int w, int h);